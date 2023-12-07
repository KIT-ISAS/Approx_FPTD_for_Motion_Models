"""Utilities and plot function for first-passage / ejection distribution models.

"""
import os
from absl import logging

from abc import ABC, abstractmethod

from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import scipy.stats
from scipy.stats import rv_histogram
from scipy.spatial import distance_matrix
# from ot.lp import emd2  # TODO: Wieder anstellen

from abstract_hitting_time_distributions import AbstractHittingTimeDistribution
from abstract_hitting_location_distributions import AbstractHittingLocationDistribution


class AbstractHittingEvaluator(ABC):
    """A base class that handles the evaluations."""

    def __init__(self,
                 process_name,
                 x_predTo,
                 t_predicted,
                 t_L = 0.0,
                 get_example_tracks_fn=None,
                 save_results=False,
                 result_dir=None,
                 for_paper=True,
                 fig_width=0.34 * 505.89 * 1 / 72,  # factor_of_textwidth * textwidth_in_pt * pt_to_inches
                 font_size=6,
                 paper_scaling_factor=2,
                 no_show=False):
        """Initializes the evaluator.

         Format get_example_tracks_fn:

            t --> x_tracks

         where

            - t is a float representing the time,
            - x_tracks is a np.array of shape [num_time_steps, num_tracks] containing the x-positions of the tracks.

        :param process_name: A string, the name of the process, appears in headers.
        :param x_predTo: A float, the position of the boundary.
        :param t_predicted: A float, the deterministic time of arrival.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param get_example_tracks_fn: A function that draws example paths from the model.
        :param save_results: A Boolean, whether to save the plots.
        :param result_dir: A string, the directory where to save the plots.
        :param for_paper: A Boolean, whether to use a publication (omit headers, etc.).
        :param fig_width: A float, the width of the figures in inches.
        :param font_size: An integer, the font size in point.
        :param paper_scaling_factor: A float, a scaling factor to be applied to the figure and fonts if _for_paper is
            true.
        :param no_show: A Boolean, whether to show the plots (False).
        """
        # sanity checks
        if get_example_tracks_fn is not None and not callable(get_example_tracks_fn):
            raise ValueError('get_example_tracks_fn must be either None or a callable.')

        self._x_predTo = x_predTo
        self._t_predicted = t_predicted
        self._t_L = t_L

        self._get_example_tracks = get_example_tracks_fn

        self._process_name = process_name
        self._process_name_save = process_name.lower().replace(" ", "_")
        self.save_results = save_results
        if result_dir is not None and not os.path.exists(result_dir):
            os.makedirs(result_dir)
        self._result_dir = result_dir
        self._for_paper = for_paper
        self.no_show = no_show

        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # get color cycle

        if self._for_paper:
            basename = os.path.basename(os.path.normpath(result_dir))
            self._process_name_save = basename.lower().replace(" ", "_")

            # style
            # see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
            # use plt.style.available to print a list of all available styles
            plt.style.use('seaborn-paper')

            # figure size
            mpl.rcParams["figure.figsize"] = [paper_scaling_factor * fig_width,
                                              paper_scaling_factor * fig_width * 5.5 / 7]

            # font size
            mpl.rcParams.update({'font.size': paper_scaling_factor * font_size,
                                 'legend.fontsize': paper_scaling_factor * font_size,
                                 'xtick.labelsize': paper_scaling_factor * font_size,
                                 'ytick.labelsize': paper_scaling_factor * font_size,
                                 })

            # latex appearance
            # See, e.g.,  https://matplotlib.org/stable/gallery/userdemo/pgf_preamble_sgskip.html
            latex_pream = "\n".join([
                r"\usepackage{amsmath}",  # enable more math
                r"\usepackage{mathtools}",
                r"\usepackage{amssymb}",
                r"\usepackage{newtxtext,newtxmath}",  # setup font to Times (font in IEEEtran)
                r"\usepackage{accents}",  # underline
                r"\usepackage{mleftright}",  # \mleft, \mright
            ])
            plt.rcParams.update({
                # "pgf.texsystem": "pdflatex",
                "text.usetex": True,  # use inline maths for ticks
                "pgf.rcfonts": False,  # don't setup fonts from rc parameters
                "font.family": "serif",  # use serif/main font for text elements
                "font.serif": "Times",
                # "font.serif": "STIXGeneral",   # this comes very close to Times or even is the same for most of the
                # things (to use it, uncomment properties 'pgf.texsystem' and 'pgf.rcfonts')
                'text.latex.preamble': latex_pream,  # for pdf, png, ...
                "pgf.preamble": latex_pream,  # for pgf plots
            })

            # other appearance options
            plt.rcParams['legend.loc'] = 'upper left'  # legend location
            # plt.rcParams.update({'xtick.direction': 'inout',
            #                      'ytick.direction': 'inout'})  # move the ticks also inside the plot

            # 3 options to avoid cutting off the labels when reducing the figure size.
            plt.rcParams['savefig.bbox'] = 'tight'  # figsize (without labels) as defined add extra space for the labels
            # mpl.rcParams["figure.autolayout"] = True  # figure with labels! gets reduced to figsize
            # mpl.rcParams["figure.constrained_layout.use"] = True  # same as before, but more powerful function, use this one

            # options for saving
            mpl.rcParams['savefig.dpi'] = 100  # dots per inch
            plt.rcParams["savefig.pad_inches"] = 0.0  # remove space outside the labels
            mpl.rcParams['figure.dpi'] = 20  # takes a long time to show the figures otherwise

    @property
    @abstractmethod
    def plot_points(self):
        """The x-coordinates of the plot for which a y-value should be displayed.

        :returns: A np.array of shape [n_plot_points], the x-coordinates of the plot for which a y-value should be
            displayed.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @staticmethod
    @abstractmethod
    def remove_not_arriving_samples(samples, return_indices=False):
        """Returns a copy of samples with removed samples that stem from particles that did not arrive at the
        boundary.

        The method relies on fallback value of max(t_samples) + 1 in the t_samples and np.nan in the y_samples and all
        other samples for those particles that did not arrive.

        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param return_indices: A Boolean, whether to return only the indices of the samples to be removed

        :returns: A np.array of shape [num_reduced_samples] containing samples (if return_indices is False) or a Boolean
            np.array of shape[num_samples] representing a mask for the arriving samples (if return_indices is True).
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def _distribute_bins_in_plot_range(self, samples, no_of_bins=100):
        """Calculates the number of requires samples so that there are no_of_bin in between self.plot_points.min() and
        self.plot_points.max().

        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param no_of_bins: An integer, the desired number of bins in the visible area.

        :returns: An integer, the number of plot points required.
        """
        return int(no_of_bins * (max(samples) - min(samples)) / (max(self.plot_points) - min(self.plot_points)))

    def check_approaches_ls(func):
        """A decorator for functions that process a list of hitting time or location distributions.

        Assure that only distributions of the same type are in the list.

        :param func: A callable, the function to be decorated.

        :returns: A callable, the decorator.
        """
        @wraps(func)
        def check_approaches_in_approaches_ls_of_same_type(self, approaches_ls, *args, **kwargs):
            if hasattr(self, "_plot_t") and not all(
                    [isinstance(a, AbstractHittingTimeDistribution) for a in approaches_ls]):
                raise ValueError(
                    'The distributions in approaches_ls must be all hitting time distributions.')

            if hasattr(self, "_plot_y") and not all(
                    [isinstance(a, AbstractHittingLocationDistribution) for a in approaches_ls]):
                raise ValueError(
                    'The distributions in approaches_ls must be all hitting location distributions.')

            return func(self, approaches_ls, *args, **kwargs)

        return check_approaches_in_approaches_ls_of_same_type

    @check_approaches_ls
    def compare_moments(self, approaches_ls, prefix='temporal'):
        """Compares the means and standard deviations of different hitting time or location distributions and print them
        to stdout.

        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution or
            AbstractHittingLocationDistribution for the same process to be  compared.
        :param prefix: A string, a prefix for logging to stdout.
        """
        mu_ls = []
        stddev_ls = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            mu = hit_stats['EV']
            stddev = hit_stats['STDDEV']
            logging.info("{0} of {1}: {2}".format(prefix.capitalize(), approach.name, np.round(mu, 6)))
            logging.info("{0} stddev of {1}: {2}".format(prefix.capitalize(), approach.name, np.round(stddev, 6)))
            mu_ls.append(mu)
            stddev_ls.append(stddev)

        rel_devs = np.zeros((len(approaches_ls), len(approaches_ls)))
        for i in range(len(approaches_ls)):
            for j in range(len(approaches_ls)):
                rel_devs[i, j] = 100 * np.abs(stddev_ls[i] - stddev_ls[j]) / np.max([stddev_ls[i], stddev_ls[j]])

        logging.info(
            'Pairwise relative deviations of {} stddevs in percent: \n{}'.format(prefix, np.round(rel_devs, 2)))

    @check_approaches_ls
    def compare_skewness(self, approaches_ls):
        """Compares the skewness of different hitting time or location distributions and print them to stdout.

        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution or
            AbstractHittingLocationDistribution for the same process to be  compared.
        """
        skew_t_ls = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            skew_t = hit_stats['SKEW']
            logging.info("Temporal skewness of {0}: {1}".format(approach.name, np.round(skew_t, 6)))
            skew_t_ls.append(skew_t)

        rel_devs = np.zeros((len(approaches_ls), len(approaches_ls)))
        for i in range(len(approaches_ls)):
            for j in range(len(approaches_ls)):
                rel_devs[i, j] = 100 * np.abs(skew_t_ls[i] - skew_t_ls[j]) / np.max([skew_t_ls[i], skew_t_ls[j]])

        logging.info('Pairwise relative deviations of temporal skewness in percent: \n{}'.format(np.round(rel_devs, 2)))

    @check_approaches_ls
    def compare_wasserstein_distances(self, approaches_ls, samples, bins=500):
        """Compares the Wasserstein distance (using the euclidean distance on the feature axis) of different hitting
        time or location distributions with the MC-solution.

        Note that the Wasserstein distance in not suitable to compare normalized and non-normalized densities since it
        requires that the total moved mass stays constant (conservation of mass).

        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution or
            AbstractHittingLocationDistribution for the same process to be  compared.
        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param bins: An integer, the number of bins to use to represent the histogram.
        """
        bins = self._distribute_bins_in_plot_range(samples, no_of_bins=bins)
        mc_hist_values, left_edges = np.histogram(samples, bins=bins, density=True)

        hist_midpoints = (left_edges[1:] + left_edges[:-1]) / 2  # left edges also contain the righthand-most edge
        # (right edge of the last bin)
        mc_hist_values /= np.sum(mc_hist_values)  # norm them, they must sum up to one for emd2 (this is only valid
        # because bins are equidistant)

        wasserstein_distances = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            hist_values = np.array([hit_stats['PDF'](t) for t in self.plot_points])
            M = distance_matrix(hist_midpoints[:, None], self.plot_points[:, None])
            hist_values /= np.sum(hist_values)  # norm them, they must sum up to one for emd2 (this is only valid
            # because plot_t is equidistant)
            wasserstein_distances.append(emd2(mc_hist_values, hist_values, M))

        logging.info(
            'Wasserstein distances compared against MC histogram (in plot range!): \n{}'.format(wasserstein_distances))

    @check_approaches_ls
    def compare_hellinger_distances(self, approaches_ls, samples,bins=500):
        """Compares the Hellinger distance of different hitting time or location distributions with the MC-solution.

        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution or
            AbstractHittingLocationDistribution for the same process to be  compared.
        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param bins: An integer, the number of bins to use to represent the histogram.
        """
        bins = self._distribute_bins_in_plot_range(samples, no_of_bins=bins)
        mc_hist = np.histogram(samples, bins=bins, density=False)
        mc_dist = rv_histogram(mc_hist, density=True)

        hellinger_distances = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            hist_values = np.array([hit_stats['PDF'](t) for t in self.plot_points])
            mc_hist_values = np.array([mc_dist.pdf(t) for t in self.plot_points])
            hellinger_distances.append(
                1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(mc_hist_values) - np.sqrt(hist_values)) ** 2)))

        logging.info(
            'Hellinger distances compared against MC histogram (in plot range!): \n{}'.format(hellinger_distances))

    @check_approaches_ls
    def compare_first_wasserstein_distances(self, approaches_ls, samples, bins=500):
        """Compares the first Wasserstein distance of different hitting time or location distributions with the
        MC-solution.

        Note that the Wasserstein distance in not suitable to compare normalized and non-normalized densities since it
        requires that the total moved mass stays constant (conservation of mass).

        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution or
            AbstractHittingLocationDistribution for the same process to be  compared.
        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param bins: An integer, the number of bins to use to represent the histogram.
        """
        bins = self._distribute_bins_in_plot_range(samples, no_of_bins=bins)
        mc_hist_values, left_edges = np.histogram(samples, bins=bins, density=True)

        wasserstein_distances = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            hist_values = np.array([hit_stats['PDF'](t) for t in self.plot_points])
            wasserstein_distances.append(scipy.stats.wasserstein_distance(mc_hist_values, hist_values))

        logging.info('First Wasserstein distances compared against MC histogram (in plot range!): \n{}'.format(
            wasserstein_distances))

    @check_approaches_ls
    def compare_kolmogorov_distances(self, approaches_ls, samples, bins=500):
        """Compares the Kolmogorov distance of different hitting time or location distributions with the MC-solution.

        The Kolmogorov distance is defined as the maximum deviation in CDF.

        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution or
            AbstractHittingLocationDistribution for the same process to be  compared.
        :param samples: A np.array of shape [num_samples] containing the first-passage times of the particles.
        :param bins: An integer, the number of bins to use to represent the histogram.
        """
        bins = self._distribute_bins_in_plot_range(samples, no_of_bins=bins)
        mc_hist = np.histogram(samples, bins=bins, density=False)
        mc_dist = rv_histogram(mc_hist, density=True)

        kolmogorv_distances = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            cdf_values = np.array([hit_stats['CDF'](t) for t in self.plot_points])
            mc_cdf_values = np.array([mc_dist.cdf(t) for t in self.plot_points])
            kolmogorv_distances.append(np.nanmax(np.abs(cdf_values - mc_cdf_values)))

        logging.info(
            'Kolmogorov distances compared against MC histogram (in plot range!): \n{}'.format(kolmogorv_distances))

    def _plot_sample_histogram(self, samples, x_label='Time t in s'):
        """Plots a histogram of the samples from the Monte Carlo simulations.

        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param x_label: A string, the x_label of the plot.
        """
        fig, ax1 = plt.subplots(figsize=[19.20, 10.80], dpi=100)
        y_hist, x_hist, _ = ax1.hist(samples, bins=100, density=False, color=[0.8, 0.8, 0.8])
        plt.xlabel(x_label)
        plt.ylabel('Number of Samples')
        if not self.no_show:
            plt.show()
        plt.close()

    @abstractmethod
    def plot_sample_histogram(self, samples, x_label='Time t in s'):
        """Plots a histogram of the samples from the Monte Carlo simulations.

        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param x_label: A string, the x_label of the plot.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def plot_example_tracks(self, N=5, dt=0.0001, plot_x_predTo=True):
        """Plots example paths (with time evolving along the x-axis and the x-position along the y-axis).

        :param N: An integer, the number of tracks to plot.
        :param dt: A float, the time increment between two consecutive points.
        :param plot_x_predTo: A Boolean, whether to plot the boundary as a horizontal line.
        """
        plot_t = np.arange(self._t_L, 1.2 * self._t_predicted, dt)

        one_direction_tracks = self._get_example_tracks(plot_t, N)

        plt.figure(figsize=[19.20, 10.80], dpi=100)
        plt.plot(plot_t, one_direction_tracks)

        if plot_x_predTo:
            plt.hlines(self._x_predTo, xmin=plot_t[0], xmax=plot_t[-1], color='black', linestyle='dashdot',
                       label='a (Boundary)')

        plt.gca().set_xlim(plot_t[0], plot_t[-1])
        plt.title('Example Tracks')
        plt.xlabel('Time in s')
        plt.ylabel('Traveled distance in mm')
        plt.legend()
        if not self.no_show:
            plt.show()
        plt.close()

    def _plot_mean_and_stddev_over_time(self,
                                        ax,
                                        ev_fn,
                                        var_fn,
                                        dt=0.0001,
                                        show_example_tracks=False,
                                        plot_x_predTo=True):
        """Plots mean and standard deviation (and example tracks) over time.

         Format ev_fn, var_fn:

            t  --> ev respectively var

         where:
            - t is a float representing the time,
            - ev respectively var is a float representing the process expectation respectively variance at t.

        :param ax: A plt.axis object.
        :param ev_fn: A callable, the mean function of the process.
        :param var_fn: A callable, the variance function of the process.
        :param dt: A float, the time increment between two consecutive points.
        :param show_example_tracks: A Boolean, whether to additionally show some paths.
        :param plot_x_predTo: A Boolean, whether to plot the boundary as a horizontal line.
        """
        # sanity checks
        if not callable(ev_fn):
            raise ValueError("ev_fn must be callable.")
        if not callable(var_fn):
            raise ValueError("var_fn must be callable.")

        plot_t = np.arange(self._t_L, 1.2 * self._t_predicted, dt)

        ev_plot = ev_fn(plot_t)
        sigma_plot = np.sqrt(var_fn(plot_t))

        if show_example_tracks:
            one_direction_tracks = self._get_example_tracks(plot_t)
            ax.plot(plot_t, one_direction_tracks)
        ax.plot(plot_t, ev_plot, label="EV", color='black')
        ax.fill_between(plot_t, ev_plot - sigma_plot, ev_plot + sigma_plot,
                        color="gray",
                        alpha=0.2,
                        label='Stddev')

        if plot_x_predTo:
            ax.hlines(self._x_predTo, xmin=plot_t[0], xmax=plot_t[-1], color='black', linestyle='dashdot',
                      label='a (Boundary)')

        if not self._for_paper:
            plt.title('Expected Value and Variance over Time for ' + self._process_name)

        ax.set_xlim(plot_t[0], plot_t[-1])
        ax.set_xlabel('Time in s')
        ax.set_ylabel('Location in mm')
        ax.legend()

    def plot_mean_and_stddev_over_time(self, ev_fn, var_fn, dt=0.0001, show_example_tracks=False, plot_x_predTo=True):
        """Plots mean and standard deviation (and example tracks) over time.

         Format ev_fn, var_fn:

            t  --> ev respectively var

         where:
            - t is a float representing the time,
            - ev respectively var is a float representing the process expectation respectively variance at t.
.
        :param ev_fn: A callable, the mean function of the process.
        :param var_fn: A callable, the variance function of the process.
        :param dt: A float, the time increment between two consecutive points.
        :param show_example_tracks: A Boolean, whether to additionally show some paths.
        :param plot_x_predTo: A Boolean, whether to plot the boundary as a horizontal line.
        """
        fig, ax = plt.subplots()

        self._plot_mean_and_stddev_over_time(ax, ev_fn, var_fn, dt, show_example_tracks, plot_x_predTo)

        if self.save_results:
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_mean_and_stddev_over_time.pdf'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_mean_and_stddev_over_time.png'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_mean_and_stddev_over_time.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    @check_approaches_ls
    def plot_quantile_functions(self, approaches_ls, q_min=0.005, q_max=0.995, y_label='Time t in s'):
        """Plots the quantile functions of the different approaches.

        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution or
            AbstractHittingLocationDistribution for the same process to be  compared.
        :param q_min: A float, the smallest value of the confidence plot range.
        :param q_max: A float, the highest value of the confidence plot range.
        :param y_label: As string, the y_label of the plot.
        """
        plot_q = np.arange(q_min, q_max, 0.01)

        lines = []
        labels = []

        fig, ax = plt.subplots()
        for i, approach in enumerate(approaches_ls):
            hit_stats = approach.get_statistics()
            if 'PPF' in hit_stats.keys():
                plot_quant = [hit_stats['PPF'](q) for q in plot_q]
                ax.plot(plot_q, plot_quant, color=self.color_cycle[i])
                # add legend manually since it fails sometimes
                lines += [Line2D([0], [0], color=self.color_cycle[i], linewidth=3)]
                labels += [approach.name]

        plt.legend(lines, labels)
        if not self._for_paper:
            plt.title('Quantile Function (Inverse CDF)')
        plt.xlabel('Confidence level')
        plt.ylabel(y_label)

        if self.save_results:
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_ppf.pdf'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_ppf.png'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_ppf.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    # to make them accessible from outside (and suppress ugly IDE warnings), needs to be done at the end of the class
    check_approaches_ls = staticmethod(check_approaches_ls)

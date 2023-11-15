"""Utilities and plot function for first-passage time models.

"""
import os
from absl import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import numpy as np
import scipy.stats
from scipy.stats import rv_histogram
from scipy.stats import norm
from scipy.spatial import distance_matrix
from ot.lp import emd2

from evaluators.hitting_model_evaluator import AbstractHittingModelEvaluator


class HittingTimeEvaluator(AbstractHittingModelEvaluator):
    """A class that handles the evaluations for hitting time models."""

    def __init__(self,
                 process_name,
                 x_predTo,
                 plot_t,
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
        """Initialize the evaluator.

        :param process_name: String, name of the process, appears in headers.
        :param x_predTo: A float, position of the boundary.
        :param plot_t: A np.array of shape [n_plot_points], point in time, when a point in the plot should be displayed.
        :param t_predicted: A float, the deterministic time of arrival.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param get_example_tracks_fn: A function that draws example paths from the model.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        :param fig_width: A float, width of the figures in inches.
        :param font_size: An integer, the font size in point.
        :param paper_scaling_factor: A scaling factor to be applied to the figure and fonts if for_paper is true.
        :param no_show: Boolean, whether to show the plots (False).
        """
        super().__init__(process_name=process_name,
                         x_predTo=x_predTo,
                         t_predicted=t_predicted,
                         t_L=t_L,
                         get_example_tracks_fn=get_example_tracks_fn,
                         save_results=save_results,
                         result_dir=result_dir,
                         for_paper=for_paper,
                         fig_width=fig_width,
                         font_size=font_size,
                         paper_scaling_factor=paper_scaling_factor,
                         no_show=no_show,
                         )

        self.plot_t = plot_t

    @staticmethod
    def compare_skewness(approaches_ls):
        """Compare the skewness of different first passage time approaches and print them to stdout.

        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        """
        skew_t_ls = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            skew_t = hit_stats['SKEW']
            logging.info("Temporal skewness of {0}: {1}".format(approach.name, round(skew_t, 6)))
            skew_t_ls.append(skew_t)

        rel_devs = np.zeros((len(approaches_ls), len(approaches_ls)))
        for i in range(len(approaches_ls)):
            for j in range(len(approaches_ls)):
                rel_devs[i, j] = 100 * np.abs(skew_t_ls[i] - skew_t_ls[j]) / np.max([skew_t_ls[i], skew_t_ls[j]])

        logging.info('Pairwise relative deviations of temporal skewness in percent: \n{}'.format(np.round(rel_devs, 2)))

    def compare_wasserstein_distances(self, t_samples, approaches_ls, bins=500):  # TODO: Move them to base class?
        """Compares the Wasserstein distance (using the euclidean distance on the feature axis) of different first
        passage time approaches with the MC-solution.

        Note that the Wasserstein distance in not suitable to compare normalized and non-normalized densities since it
        requires that the total moved mass stays constant (conservation of mass).

        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared against the
            MC histogram.
        :param bins: An integer, the number of bins to use to represent the histogram.
        """
        bins = self._distribute_bins_in_plot_range(t_samples, self.plot_t)
        mc_hist_values, left_edges = np.histogram(t_samples, bins=bins, density=True)

        hist_midpoints = (left_edges[1:] + left_edges[:-1]) / 2  # left edges also contain the righthand-most edge
        # (right edge of the last bin)
        mc_hist_values /= np.sum(mc_hist_values)  # norm them, they must sum up to one for emd2 (this is only valid
        # because bins are equidistant)

        wasserstein_distances = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            hist_values = np.array([hit_stats['PDF'](t) for t in self.plot_t])
            M = distance_matrix(hist_midpoints[:, None], self.plot_t[:, None])
            hist_values /= np.sum(hist_values)  # norm them, they must sum up to one for emd2 (this is only valid
            # because plot_t is equidistant)
            wasserstein_distances.append(emd2(mc_hist_values, hist_values, M))

        logging.info(
            'Wasserstein distances compared against MC histogram (in plot range!): \n{}'.format(wasserstein_distances))

    def compare_hellinger_distances(self, t_samples, approaches_ls, bins=500):
        """Compares the Hellinger distance of different first passage time approaches with the MC-solution.

        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared against the
            MC histogram.
        :param bins: An integer, the number of bins to use to represent the histogram.
        """
        bins = self._distribute_bins_in_plot_range(t_samples, self.plot_t)
        mc_hist = np.histogram(t_samples, bins=bins, density=False)
        mc_dist = rv_histogram(mc_hist, density=True)

        hellinger_distances = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            hist_values = np.array([hit_stats['PDF'](t) for t in self.plot_t])
            mc_hist_values = np.array([mc_dist.pdf(t) for t in self.plot_t])
            hellinger_distances.append(
                1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(mc_hist_values) - np.sqrt(hist_values)) ** 2)))

        logging.info(
            'Hellinger distances compared against MC histogram (in plot range!): \n{}'.format(hellinger_distances))

    def compare_first_wasserstein_distances(self, t_samples, approaches_ls, bins=500):
        """Compares the first Wasserstein distance of different first passage time approaches with the MC-solution.

        Note that the Wasserstein distance in not suitable to compare normalized and non-normalized densities since it
        requires that the total moved mass stays constant (conservation of mass).

        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared against the
            MC histogram.
        :param bins: An integer, the number of bins to use to represent the histogram.
        """
        bins = self._distribute_bins_in_plot_range(t_samples, self.plot_t)
        mc_hist_values, left_edges = np.histogram(t_samples, bins=bins, density=True)

        wasserstein_distances = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            hist_values = np.array([hit_stats['PDF'](t) for t in self.plot_t])
            wasserstein_distances.append(scipy.stats.wasserstein_distance(mc_hist_values, hist_values))

        logging.info('First Wasserstein distances compared against MC histogram (in plot range!): \n{}'.format(
            wasserstein_distances))

    def compare_kolmogorv_distances(self, t_samples, approaches_ls, bins=500):
        """Compares the Kolmogorov distance of different first passage time approaches with the MC-solution.

        The Kolmogorov distance is defined as the maximum deviation in CDF.

        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared against the
            MC histogram.
        :param bins: An integer, the number of bins to use to represent the histogram.
        """
        bins = self._distribute_bins_in_plot_range(t_samples, self.plot_t)
        mc_hist = np.histogram(t_samples, bins=bins, density=False)
        mc_dist = rv_histogram(mc_hist, density=True)

        kolmogorv_distances = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            cdf_values = np.array([hit_stats['CDF'](t) for t in self.plot_t])
            mc_cdf_values = np.array([mc_dist.cdf(t) for t in self.plot_t])
            kolmogorv_distances.append(np.nanmax(np.abs(cdf_values - mc_cdf_values)))

        logging.info(
            'Kolmogorov distances compared against MC histogram (in plot range!): \n{}'.format(kolmogorv_distances))

    def _plot_first_hitting_time_distributions(self, ax1, t_samples, approaches_ls, plot_hist_for_all_particles=True):
        """Plots the first passage time distribution.

        :param ax1: A plt.axis object.
        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        :param plot_hist_for_all_particles: Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        """
        if not plot_hist_for_all_particles:
            # check if there are default values (particles that did not arrive) in the array
            if max(t_samples) - int(max(t_samples)) == 0.0:
                #  there are default values, remove them from array
                t_samples = t_samples[t_samples != max(t_samples)]
            y_hist, x_hist, _ = ax1.hist(t_samples,
                                         bins=self._distribute_bins_in_plot_range(t_samples, self.plot_t),
                                         # we want to have 100 samples in the plot window
                                         density=True,
                                         histtype='stepfilled',  # no space between the bars
                                         color=[0.8, 0.8, 0.8],
                                         )
            y_hist_max = y_hist.max()
            # sums up to 1 (sum(y_hist * np.diff(x_hist))=1) not including particles that did not arrive
        else:
            y_hist, x_hist, _ = ax1.hist(t_samples,
                                         bins=self._distribute_bins_in_plot_range(t_samples, self.plot_t),
                                         # we want to have 100 samples in the plot window
                                         density=True,
                                         histtype='stepfilled',  # no space between the bars
                                         color=[0.8, 0.8, 0.8],
                                         )
            y_hist_max = y_hist[:-1].max()
            # sums up to 1 (sum(y_hist * np.diff(x_hist))=1) while also including particles that do not not arrive at
            # the boundary

        ax1.vlines(self.t_predicted, 0, 350, color='black', label="Deterministic Prediction")

        ax2 = ax1.twinx()
        for i, approach in enumerate(approaches_ls):
            hit_stats = approach.get_statistics()
            if 'CDF' in hit_stats.keys():
                plot_f = [hit_stats['CDF'](t) for t in self.plot_t]
                ax2.plot(self.plot_t, plot_f, color=self.color_cycle[i], label=approach.name)
            if 'RAW_CDF' in hit_stats.keys():
                plot_f = [hit_stats['RAW_CDF'](t) for t in self.plot_t]
                ax2.plot(self.plot_t, plot_f, color=self.color_cycle[i], alpha=0.5)
            if 'PDF' in hit_stats.keys():
                plot_f = [hit_stats['PDF'](t) for t in self.plot_t]
                ax1.plot(self.plot_t, plot_f, color=self.color_cycle[i], label=approach.name)
            if 'PDFVALUES' in hit_stats.keys():
                ax1.plot(hit_stats['PDFVALUES'][0], hit_stats['PDFVALUES'][1], color=self.color_cycle[i],
                         label=approach.name)
            if 'EV' in hit_stats.keys():
                ax2.vlines(hit_stats['EV'], 0, 1, color=self.color_cycle[i], linestyle='dashed', label=approach.name)
                if 'STDDEV' in hit_stats.keys():
                    ax2.vlines([hit_stats['EV'] - hit_stats['STDDEV'], hit_stats['EV'] + hit_stats['STDDEV']], 0, 1,
                               color=self.color_cycle[i], linestyle='dashdot', label=approach.name)
            if 'Median' in hit_stats.keys():
                ax2.vlines(hit_stats['Median'], 0, 1, color=self.color_cycle[i], linestyle='dashed',
                           label=approach.name)
            if 'FirstQuantile' in hit_stats.keys():
                ax2.vlines(hit_stats['FirstQuantile'], 0, 1, color=self.color_cycle[i], linestyle='dotted',
                           label=approach.name)
            if 'ThirdQuantile' in hit_stats.keys():
                ax2.vlines(hit_stats['ThirdQuantile'], 0, 1, color=self.color_cycle[i], linestyle='dotted',
                           label=approach.name)

        # add legend manually since it fails sometimes
        legend_elements = [Line2D([0], [0], color=c, linewidth=3, label=approach.name) for c, approach in
                           zip(self.color_cycle, approaches_ls)]
        legend_elements.append(Patch(facecolor=[0.8, 0.8, 0.8], label='MC simulation'))
        ax2.legend(handles=legend_elements)

        ax1.set_ylim(0, 1.4 * y_hist_max)  # leave some space for labels
        ax2.set_ylim(0, 1.05)
        ax1.set_xlim(self.plot_t[0], self.plot_t[-1])
        ax1.set_xlabel("Time in s")
        ax1.set_ylabel("PDF")
        ax2.set_ylabel("CDF")

    def plot_first_hitting_time_distributions(self, t_samples, approaches_ls, plot_hist_for_all_particles=True):
        """Plots the first passage time distribution.

        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        :param plot_hist_for_all_particles: Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        """
        fig, ax1 = plt.subplots()

        self._plot_first_hitting_time_distributions(ax1, t_samples, approaches_ls, plot_hist_for_all_particles)

        if not self.for_paper:
            plt.title("Distribution of First Passage Time for " + self.process_name)
        #plt.legend()
        if self.save_results:
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_fptd.pdf'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_fptd.png'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_fptd.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    def plot_fptd_and_paths_in_one(self, ev_fn, var_fn, t_samples, approaches_ls, plot_hist_for_all_particles=True):
        """Creates a stacked plot of two subplots. The upper one is the first passage time distribution and the lower
        one is the plot of paths over time.

        :param ev_fn: The mean function of the process.
        :param var_fn: The variance function of the process.
        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        :param plot_hist_for_all_particles: Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        """
        fig, axes = plt.subplots(nrows=2,
                                 figsize=[mpl.rcParams["figure.figsize"][0], 2*mpl.rcParams["figure.figsize"][1]],
                                 tight_layout=True,  # avoid overlapping labels
                                 )

        self._plot_first_hitting_time_distributions(axes[0],
                                                    t_samples,
                                                    approaches_ls,
                                                    plot_hist_for_all_particles)

        self._plot_mean_and_stddev_over_time(axes[1],
                                             ev_fn,
                                             var_fn,
                                             show_example_tracks=True)

        if self.save_results:
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_fptd_and_example_paths.pdf'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_fptd_and_example_paths.png'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_fptd_and_example_paths.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    def _plot_returning_probs(self, mc_hist_func, approaches_ls, t_range=None):
        """Plots the estimated returning probabilities and compares it with the MC solution.

        :param mc_hist_func: A function f(plot_t) -> mc_return_plot_t, mc_return_plot_probs_values with
            mc_return_plot_t, mc_return_plot_probs_values both np.arrays of shape [num_evaluated_times] with
            mc_return_plot_t containing the times where the MC function for the return probabilities was evaluated
            and mc_return_plot_probs_values its values.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        :param t_range: None or a list of length 2, the (min, max) time for the plots.
        """
        if t_range is None:
            tmax = [approach.get_statistics()['t_max'] for approach in approaches_ls if
                    't_max' in approach.get_statistics().keys()]
            if len(tmax) == 0:
                raise ValueError(
                    'If no t_range is given, at least one approach must be of class EngineeringApproxHittingTimeModel.')
            t_range = [self.t_predicted - 0.3 * (self.t_predicted - self.t_L), 10 * tmax[0]]

        plot_t = np.arange(t_range[0], t_range[1], 0.001)
        mc_return_plot_t, mc_return_plot_probs_values = mc_hist_func(plot_t)

        fig, ax = plt.subplots()
        ax.fill_between(mc_return_plot_t, mc_return_plot_probs_values, color=[0.8, 0.8, 0.8], label='MC simulation')

        for i, approach in enumerate(approaches_ls):
            hit_stats = approach.get_statistics()
            if 'ReturningProbs' in hit_stats.keys():
                plot_prob = [hit_stats['ReturningProbs'](t) for t in plot_t]
                ax.plot(plot_t, plot_prob, label=approach.name, color=self.color_cycle[i])
                if 't_max' in hit_stats.keys():
                    ax.vlines(hit_stats['t_max'], 0,
                              1.05 * max(np.max(mc_return_plot_probs_values), np.max(plot_prob)),
                              color='black',
                              linestyle='dashed',
                              label=r'$t_c$')
            if 'TrueReturningProbs' in hit_stats.keys():
                plot_prob = [hit_stats['TrueReturningProbs'](t) for t in plot_t]
                ax.plot(plot_t, plot_prob, label=approach.name + ' (true)', color='red', linestyle='dotted')

        ax.legend()
        ax.set_xlim(plot_t[0], plot_t[-1])
        ax.set_xlabel("Time in s")
        ax.set_ylabel(
            r"Returning probability $\mathbb{P}\mleft( \boldsymbol{T}_a < t, \boldsymbol{x}(t) \le a \mright)$" if self.for_paper else 'Returning probability')

        if not self.for_paper:
            plt.title("Returning probabilities for " + self.process_name)
        if self.save_results:
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_return_probs.pdf'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_return_probs.png'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_return_probs.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    def plot_returning_probs_from_fptd_histogram(self, ev_fn, var_fn, t_samples, approaches_ls, bins=1000, t_range=None):
        """Plots the estimated returning probabilities and compares it with the MC solution. The MC solution is based
        on the MC FPTD and the process density.

        Note: For too few samples tracks, the result might be very noise.

        :param ev_fn: The mean function of the process.
        :param var_fn: The variance function of the process.
        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param t_range: None or a list of length 2, the (min, max) time for the plots.
        """
        def mc_hist_func(plot_t):
            hist = np.histogram(t_samples, bins=bins, density=True)
            mc_fptd = rv_histogram(hist, density=True)
            p_x_t_greater_x_predTo = lambda t: 1 - norm(loc=ev_fn(t), scale=np.sqrt(var_fn(t))).cdf(self.x_predTo)
            mc_return_plot_probs_values = np.array([mc_fptd.cdf(t) - p_x_t_greater_x_predTo(t) for t in plot_t])
            return plot_t, mc_return_plot_probs_values

        self._plot_returning_probs(mc_hist_func, approaches_ls, t_range)

    def plot_returning_probs_from_sample_paths(self, fraction_of_returns, dt, approaches_ls, t_range=None):
        """Plots the estimated returning probabilities and compares it with the MC solution. The MC solution is based
        on counting example tracks.

        :param fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
        :param dt: A float, time increment.
        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        :param t_range: None or a list of length 2, the (min, max) time for the plots.
        """
        def mc_hist_func(plot_t):
            # mc_return_plot_t = np.arange(self.t_L, self.t_L + len(fraction_of_returns) * dt, step=dt)  # do not use,
            # this can result in an undesired length for large intervals and small step size
            mc_return_plot_t = np.linspace(self.t_L, self.t_L + len(fraction_of_returns) * dt,
                                           num=len(fraction_of_returns),
                                           endpoint=False)
            in_plot_range = np.logical_and(mc_return_plot_t >= plot_t[0], mc_return_plot_t <= plot_t[-1])
            return mc_return_plot_t[in_plot_range], fraction_of_returns[in_plot_range]

        self._plot_returning_probs(mc_hist_func, approaches_ls, t_range)

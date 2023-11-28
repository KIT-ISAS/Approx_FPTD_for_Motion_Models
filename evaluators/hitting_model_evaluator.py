"""Utilities and plot function for first-passage / ejection distribution models.

"""
import os
from absl import logging

from abc import ABC

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


class AbstractHittingModelEvaluator(ABC):
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
        """Initialize the evaluator.

        :param process_name: String, name of the process, appears in headers.
        :param x_predTo: A float, position of the boundary.
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
        self.x_predTo = x_predTo  # TODO: Als private deklarieren was Sinn macht (auch in den anderen Klassen des Repos)
        self.t_predicted = t_predicted
        self.t_L = t_L

        self.get_example_tracks = get_example_tracks_fn

        self.process_name = process_name
        self.process_name_save = process_name.lower().replace(" ", "_")
        self.save_results = save_results
        if result_dir is not None and not os.path.exists(result_dir):
            os.makedirs(result_dir)
        self.result_dir = result_dir
        self.for_paper = for_paper
        self.no_show = no_show

        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # get color cycle

        if self.for_paper:
            basename = os.path.basename(os.path.normpath(result_dir))
            self.process_name_save = basename.lower().replace(" ", "_")

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

    def _distribute_bins_in_plot_range(self, samples, plot_points,
                                       no_of_bins=100):  # TODO: Static oder no_of_bins in init
        return int(no_of_bins * (max(samples) - min(samples)) / (
                plot_points[-1] - plot_points[0]))

    @staticmethod
    def _remove_not_arriving_samples(samples):
        samples = samples[np.isfinite(samples)] #  there are default values, remove them from array
        if max(samples) - int(max(samples)) == 0.0:
            #  there are default values, remove them from array
            samples = samples[samples != max(samples)]
        return samples

    @staticmethod
    def compare_moments(approaches_ls, prefix='temporal'):
        """Compare the means and standard deviations of different hitting models and print them to stdout.

        :param approaches_ls: A list of first-passage time model objects for the same process to be compared. # TODO
        :param prefix: TODO
        """
        # TODO: Value erorrs

        mu_ls = []
        stddev_ls = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            mu = hit_stats['EV']
            stddev = hit_stats['STDDEV']
            logging.info("{0} of {1}: {2}".format(prefix.capitalize(), approach.name, round(mu, 6)))
            logging.info("{0} stddev of {1}: {2}".format(prefix.capitalize(), approach.name, round(stddev, 6)))
            mu_ls.append(mu)
            stddev_ls.append(stddev)

        rel_devs = np.zeros((len(approaches_ls), len(approaches_ls)))
        for i in range(len(approaches_ls)):
            for j in range(len(approaches_ls)):
                rel_devs[i, j] = 100 * np.abs(stddev_ls[i] - stddev_ls[j]) / np.max([stddev_ls[i], stddev_ls[j]])

        logging.info(
            'Pairwise relative deviations of {} stddevs in percent: \n{}'.format(prefix, np.round(rel_devs, 2)))

    def plot_sample_histogram(self, samples, x_label='Time t in s'):
        """Plot histograms of the samples from the Monte Carlo simulations.

        :param samples: A np.array of shape [N] containing samples.
        :param x_label: String, x_label of the plot.
        """
        fig, ax1 = plt.subplots(figsize=[19.20, 10.80], dpi=100)
        y_hist, x_hist, _ = ax1.hist(samples, bins=100, density=False, color=[0.8, 0.8, 0.8])
        plt.xlabel(x_label)
        plt.ylabel('Number of Samples')
        if not self.no_show:
            plt.show()
        plt.close()

    def plot_example_tracks(self, N=5, dt=0.0001, plot_x_predTo=True):
        """Plot example paths.

        :param N: Integer, number of tracks to plot.
        :param dt: A float, time increment between two consecutive points.
        """
        plot_t = np.arange(self.t_L, 1.2 * self.t_predicted, dt)

        one_direction_tracks = self.get_example_tracks(plot_t, N)

        plt.figure(figsize=[19.20, 10.80], dpi=100)
        plt.plot(plot_t, one_direction_tracks)

        if plot_x_predTo:
            plt.hlines(self.x_predTo, xmin=plot_t[0], xmax=plot_t[-1], color='black', linestyle='dashdot',
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

        :param ax: A plt.axis object.
        :param ev_fn: The mean function of the process.
        :param var_fn: The variance function of the process.
        :param dt: A float, time increment between two consecutive points.
        :param show_example_tracks: Boolean, whether to additionally show some paths.
        """
        plot_t = np.arange(self.t_L, 1.2 * self.t_predicted, dt)

        ev_plot = ev_fn(plot_t)
        sigma_plot = np.sqrt(var_fn(plot_t))

        if show_example_tracks:
            one_direction_tracks = self.get_example_tracks(plot_t)
            ax.plot(plot_t, one_direction_tracks)
        ax.plot(plot_t, ev_plot, label="EV", color='black')
        ax.fill_between(plot_t, ev_plot - sigma_plot, ev_plot + sigma_plot,
                        color="gray",
                        alpha=0.2,
                        label='Stddev')

        if plot_x_predTo:
            ax.hlines(self.x_predTo, xmin=plot_t[0], xmax=plot_t[-1], color='black', linestyle='dashdot',
                      label='a (Boundary)')

        if not self.for_paper:
            plt.title('Expected Value and Variance over Time for ' + self.process_name)

        ax.set_xlim(plot_t[0], plot_t[-1])
        ax.set_xlabel('Time in s')
        ax.set_ylabel('Location in mm')
        ax.legend()

    def plot_mean_and_stddev_over_time(self, ev_fn, var_fn, dt=0.0001, show_example_tracks=False, plot_x_predTo=True):
        """Plots mean and standard deviation (and example tracks) over time.

        :param ev_fn: The mean function of the process.
        :param var_fn: The variance function of the process.
        :param dt: A float, time increment between two consecutive points.
        :param show_example_tracks: Boolean, whether to additionally show some paths.
        """
        fig, ax = plt.subplots()

        self._plot_mean_and_stddev_over_time(ax, ev_fn, var_fn, dt, show_example_tracks, plot_x_predTo)

        if self.save_results:
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_mean_and_stddev_over_time.pdf'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_mean_and_stddev_over_time.png'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_mean_and_stddev_over_time.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    def plot_quantile_functions(self, approaches_ls, q_min=0.005, q_max=0.995, y_label='Time t in s'):
        """Plots the quantile functions of the different approaches.

        :param approaches_ls: A list of first-passage time model objects for the same process to be compared.
        :param q_min: A float, the smallest value of the confidence plot range.
        :param q_max: A float, the highest value of the confidence plot range.
        :param y_label:  String, y_label of the plot.
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
        if not self.for_paper:
            plt.title('Quantile Function (Inverse CDF)')
        plt.xlabel('Confidence level')
        plt.ylabel(y_label)

        if self.save_results:
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_ppf.pdf'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_ppf.png'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_ppf.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

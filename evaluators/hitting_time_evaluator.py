"""Utilities and plot function for first-passage time models.

"""
import os
from absl import logging

from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import numpy as np
from scipy.stats import rv_histogram
from scipy.stats import norm

from evaluators.hitting_evaluator import AbstractHittingEvaluator
from abstract_hitting_time_distributions import AbstractHittingTimeDistribution


class HittingTimeEvaluator(AbstractHittingEvaluator):
    """A class that handles the evaluations for hitting time distributions."""

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
        """Initializes the evaluator.

         Format get_example_tracks_fn:

             plot_t --> y_tracks

          where

            - x_tracks is a np.array of shape [num_time_steps, num_tracks] containing the x-positions of the tracks.

        :param process_name: A string, the name of the process, appears in headers.
        :param x_predTo: A float, the position of the boundary.
        :param plot_t: A np.array of shape [n_plot_points], point in time, when a point in the plot should be displayed.
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

        self._plot_t = plot_t

    @property
    def plot_points(self):
        """The x-coordinates of the plot for which a y-value should be displayed.

        :returns: A np.array of shape [n_plot_points], the x-coordinates of the plot for which a y-value should be
            displayed.
        """
        return self._plot_t

    @staticmethod
    def _remove_not_arriving_samples(t_samples):
        """Returns a copy of t_samples with removed samples that stem from particles that did not arrive at the
        boundary.

        The method relies on fallback value of max(t_samples) + 1 in the t_samples and np.nan in the y_samples and all
        other samples for those particles that did not arrive.

        :param t_samples: A np.array of shape [num_samples] containing samples.

        :returns: A np.array of shape [num_reduced_samples] containing samples.
        """
        t_samples = t_samples.copy()
        if max(t_samples) - int(max(t_samples)) == 0.0:
            #  there are default values, remove them from array
            t_samples = t_samples[t_samples != max(t_samples)]
        return t_samples

    def plot_sample_histogram(self, t_samples, x_label='Time t in s', plot_hist_for_all_particles=True):
        """Plots a histogram of the samples from the Monte Carlo simulations.

        :param t_samples: A np.array of shape [num_samples] containing sampled values.
        :param x_label: A string, the x_label of the plot.
        :param plot_hist_for_all_particles: Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        """
        if not plot_hist_for_all_particles:
            # check if there are default values (particles that did not arrive) in the array
            t_samples = self._remove_not_arriving_samples(t_samples)
        self._plot_sample_histogram(t_samples, x_label)

    def _plot_first_hitting_time_distributions(self, ax1, t_samples, approaches_ls, plot_hist_for_all_particles=True):
        """Plots the first-passage time distribution.

        :param ax1: A plt.axis object.
        :param t_samples: A np.array of shape [num_samples] containing the first-passage times of the particles.
        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution for the same process to be
            compared.
        :param plot_hist_for_all_particles: Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        """
        if not plot_hist_for_all_particles:
            # check if there are default values (particles that did not arrive) in the array
            t_samples = self._remove_not_arriving_samples(t_samples)
            y_hist, x_hist, _ = ax1.hist(t_samples,
                                         bins=self._distribute_bins_in_plot_range(t_samples),
                                         # we want to have 100 samples in the plot window
                                         density=True,
                                         histtype='stepfilled',  # no space between the bars
                                         color=[0.8, 0.8, 0.8],
                                         )
            y_hist_max = y_hist.max()
            # sums up to 1 (sum(y_hist * np.diff(x_hist))=1) not including particles that did not arrive
        else:
            y_hist, x_hist, _ = ax1.hist(t_samples,
                                         bins=self._distribute_bins_in_plot_range(t_samples),
                                         # we want to have 100 samples in the plot window
                                         density=True,
                                         histtype='stepfilled',  # no space between the bars
                                         color=[0.8, 0.8, 0.8],
                                         )
            y_hist_max = y_hist[:-1].max()
            # sums up to 1 (sum(y_hist * np.diff(x_hist))=1) while also including particles that do not not arrive at
            # the boundary

        ax1.vlines(self._t_predicted, 0, 350, color='black', label="Deterministic Prediction")

        ax2 = ax1.twinx()
        for i, approach in enumerate(approaches_ls):
            hit_stats = approach.get_statistics()
            if 'CDF' in hit_stats.keys():
                plot_f = [hit_stats['CDF'](t) for t in self._plot_t]
                ax2.plot(self._plot_t, plot_f, color=self.color_cycle[i], label=approach.name)
            if 'RAW_CDF' in hit_stats.keys():
                plot_f = [hit_stats['RAW_CDF'](t) for t in self._plot_t]
                ax2.plot(self._plot_t, plot_f, color=self.color_cycle[i], alpha=0.5)
            if 'PDF' in hit_stats.keys():
                plot_f = [hit_stats['PDF'](t) for t in self._plot_t]
                ax1.plot(self._plot_t, plot_f, color=self.color_cycle[i], label=approach.name)
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
        ax1.set_xlim(self._plot_t[0], self._plot_t[-1])
        ax1.set_xlabel("Time in s")
        ax1.set_ylabel("PDF")
        ax2.set_ylabel("CDF")

    @AbstractHittingEvaluator.check_approaches_ls
    def plot_first_hitting_time_distributions(self, t_samples, approaches_ls, plot_hist_for_all_particles=True):
        """Plots the first-passage time distribution.

        :param t_samples: A np.array of shape [num_samples] containing the first-passage times of the particles.
        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution for the same process to be
            compared.
        :param plot_hist_for_all_particles: A Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        """
        fig, ax1 = plt.subplots()

        self._plot_first_hitting_time_distributions(ax1, t_samples, approaches_ls, plot_hist_for_all_particles)

        if not self._for_paper:
            plt.title("Distribution of First Passage Time for " + self._process_name)
        #plt.legend()
        if self.save_results:
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_fptd.pdf'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_fptd.png'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_fptd.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    @AbstractHittingEvaluator.check_approaches_ls
    def plot_fptd_and_paths_in_one(self, ev_fn, var_fn, t_samples, approaches_ls, plot_hist_for_all_particles=True):
        """Creates a stacked plot of two subplots. The upper one is the first-passage time distribution and the lower
        one is the plot of paths over time.

         Format ev_fn, var_fn:

            t  --> ev respectively var

         where:
            - t is a float representing the time,
            - ev respectively var is a float representing the process expectation respectively variance at t.

        :param ev_fn: A callable, the mean function of the process.
        :param var_fn: A callable, the variance function of the process.
        :param t_samples: A np.array of shape [num_samples] containing the first-passage times of the particles.
        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution for the same process to be
            compared.
        :param plot_hist_for_all_particles: A Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        """
        # sanity checks
        if not callable(ev_fn):
            raise ValueError("ev_fn must be callable.")
        if not callable(var_fn):
            raise ValueError("var_fn must be callable.")

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
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_fptd_and_example_paths.pdf'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_fptd_and_example_paths.png'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_fptd_and_example_paths.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    def _plot_returning_probs(self, mc_hist_fn, approaches_ls, t_range=None):
        """Plots the estimated returning probabilities and compares it with the MC solution.

         Format mc_hist_fn:

            plot_t  --> mc_return_plot_t, mc_return_plot_probs_values

         where:
            - mc_return_plot_t is np.array of shape [num_evaluated_times] containing the times when the MC function
                for the return probabilities was evaluated,
            - mc_return_plot_probs_values is np.array of shape [num_evaluated_times] containing the corresponding return
                probabilities.

        :param mc_hist_fn: A callable, a histogram generating function.
        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution for the same process to be
            compared.
        :param t_range: None or a list of length 2, the (min, max) time for the plots.
        """
        # sanity checks
        if not callable(mc_hist_fn):
            raise ValueError("mc_hist_fn must be callable.")

        if t_range is None:
            tmax = [approach.get_statistics()['t_max'] for approach in approaches_ls if
                    't_max' in approach.get_statistics().keys()]
            if len(tmax) == 0:
                raise ValueError(
                    'If no t_range is given, at least one approach must be of class EngineeringApproxCVHittingTimeModel.')
            t_range = [self._t_predicted - 0.3 * (self._t_predicted - self._t_L), 10 * tmax[0]]

        plot_t = np.arange(t_range[0], t_range[1], 0.001)
        mc_return_plot_t, mc_return_plot_probs_values = mc_hist_fn(plot_t)

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
            r"Returning probability $\mathbb{P}\mleft( \boldsymbol{T}_a < t, \boldsymbol{x}(t) \le a \mright)$" if self._for_paper else 'Returning probability')

        if not self._for_paper:
            plt.title("Returning probabilities for " + self._process_name)
        if self.save_results:
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_return_probs.pdf'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_return_probs.png'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_return_probs.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    @AbstractHittingEvaluator.check_approaches_ls
    def plot_returning_probs_from_fptd_histogram(self,
                                                 ev_fn,
                                                 var_fn,
                                                 t_samples,
                                                 approaches_ls,
                                                 bins=1000,
                                                 t_range=None,
                                                 plot_hist_for_all_particles=True,
                                                 ):
        """Plots the estimated returning probabilities and compares it with the MC solution. The MC solution is based
        on the MC FPTD and the process density.

        Note: For too few samples tracks, the result might be very noise.

         Format ev_fn, var_fn:

            t  --> ev respectively var

         where:
            - t is a float representing the time,
            - ev respectively var is a float representing the process expectation respectively variance at t.

        :param ev_fn: A callable, the mean function of the process.
        :param var_fn: A callable, the variance function of the process.
        :param t_samples: A np.array of shape [num_samples] containing the first-passage times of the particles.
        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution for the same process to be
            compared.
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param t_range: None or a list of length 2, the (min, max) time for the plots.
        :param plot_hist_for_all_particles: A Boolean, whether to plot the histogram
            only for particles that arrive at the boundary (False).
        """
        # sanity checks
        if not callable(ev_fn):
            raise ValueError("ev_fn must be callable.")
        if not callable(var_fn):
            raise ValueError("var_fn must be callable.")

        if not plot_hist_for_all_particles:
            # check if there are default values (particles that did not arrive) in the array
            t_samples = self._remove_not_arriving_samples(t_samples)

        def mc_hist_func(plot_t):
            hist = np.histogram(t_samples, bins=bins, density=True)
            mc_fptd = rv_histogram(hist, density=True)
            p_x_t_greater_x_predTo = lambda t: 1 - norm(loc=ev_fn(t), scale=np.sqrt(var_fn(t))).cdf(self._x_predTo)
            mc_return_plot_probs_values = np.array([mc_fptd.cdf(t) - p_x_t_greater_x_predTo(t) for t in plot_t])
            return plot_t, mc_return_plot_probs_values

        self._plot_returning_probs(mc_hist_func, approaches_ls, t_range)

    @AbstractHittingEvaluator.check_approaches_ls
    def plot_returning_probs_from_sample_paths(self, fraction_of_returns, dt, approaches_ls, t_range=None):
        """Plots the estimated returning probabilities and compares it with the MC solution. The MC solution is based
        on counting example tracks.

        :param fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
        :param dt: A float, the time increment.
        :param approaches_ls: A list of child instances of AbstractHittingTimeDistribution for the same process to be
            compared.
        :param t_range: None or a list of length 2, the (min, max) time for the plots.
        """
        def mc_hist_func(plot_t):
            # mc_return_plot_t = np.arange(self._t_L, self._t_L + len(fraction_of_returns) * dt, step=dt)  # do not use,
            # this can result in an undesired length for large intervals and small step size
            mc_return_plot_t = np.linspace(self._t_L, self._t_L + len(fraction_of_returns) * dt,
                                           num=len(fraction_of_returns),
                                           endpoint=False)
            in_plot_range = np.logical_and(mc_return_plot_t >= plot_t[0], mc_return_plot_t <= plot_t[-1])
            return mc_return_plot_t[in_plot_range], fraction_of_returns[in_plot_range]

        self._plot_returning_probs(mc_hist_func, approaches_ls, t_range)

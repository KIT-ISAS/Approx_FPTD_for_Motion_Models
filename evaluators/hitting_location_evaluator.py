import os

from abc import ABC


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from evaluators.hitting_evaluator import AbstractHittingEvaluator
from abstract_hitting_location_distributions import AbstractHittingLocationDistribution


class HittingLocationEvaluator(AbstractHittingEvaluator):
    """A class that handles the evaluations for hitting location models."""

    def __init__(self,
                 process_name,
                 x_predTo,
                 t_predicted,
                 y_predicted,
                 plot_y,
                 t_L = 0.0,
                 get_example_tracks_fn=None,
                 save_results=False,
                 result_dir=None,
                 for_paper=True,
                 fig_width=0.34 * 505.89 * 1 / 72,  # factor_of_textwidth * textwidth_in_pt * pt_to_inches
                 font_size=6,
                 paper_font='Times',
                 paper_scaling_factor=2,
                 no_show=False,
                 time_unit='s',
                 length_unit='m'):
        """Initializes the evaluator.

          Format get_example_tracks_fn:

             plot_t --> y_tracks

          where

             - plot_t is anp.array of shape [n_plot_points], point in time, when a point in the plot should be displayed,
             - y_tracks is a np.array of shape [num_time_steps, num_tracks] containing the y-positions of the tracks.

         :param process_name: A string, the name of the process, appears in headers.
         :param x_predTo: A float, the position of the boundary.
         :param plot_y: A np.array of shape [n_plot_points_y], y-positions where a point in the plot should be displayed.
         :param t_predicted: A float, the deterministic time of arrival.
         :param y_predicted: A float, the deterministic location of arrival at the actuator array, i.e., the predicted
            y-position at the first-passage time.
         :param t_L: A float, the time of the last state/measurement (initial time).
         :param get_example_tracks_fn: A function that draws example paths from the model.
         :param save_results: A Boolean, whether to save the plots.
         :param result_dir: A string, the directory where to save the plots.
         :param for_paper: A Boolean, whether to use a publication (omit headers, etc.).
         :param fig_width: A float, the width of the figures in inches.
         :param font_size: An integer, the font size in point.
         :param paper_font: A string, the font to be used for the paper. Either "Times", "Helvetica" or "Default". Only
            relevant if for_paper is True.
         :param paper_scaling_factor: A float, a scaling factor to be applied to the figure and fonts if _for_paper is
             true.
        :param no_show: A Boolean, whether to show the plots (False).
        :param time_unit: A string, the time unit of the process (used for the plot labels).
        :param length_unit: A string, the location unit of the process (used for the plot labels).
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
                         paper_font=paper_font,
                         paper_scaling_factor=paper_scaling_factor,
                         no_show=no_show,
                         time_unit=time_unit,
                         length_unit=length_unit,
                         )

        self._y_predicted = y_predicted
        self._plot_y = plot_y

    @property
    def plot_points(self):
        """The x-coordinates of the plot for which a y-value should be displayed.

        :returns: A np.array of shape [n_plot_points], the x-coordinates of the plot for which a y-value should be
            displayed.
        """
        return self._plot_y

    @staticmethod
    def remove_not_arriving_samples(y_samples, return_indices=False):
        """Returns a copy of y_samples with removed samples that stem from particles that did not arrive at the
        boundary.

        The method relies on fallback value of max(t_samples) + 1 in the t_samples and np.nan in the y_samples and all
        other samples for those particles that did not arrive.

        :param y_samples: A np.array of shape [num_samples] containing samples.
        :param return_indices: A Boolean, whether to return only the indices of the samples to be removed

        :returns: A np.array of shape [num_reduced_samples] containing samples (if return_indices is False) or a Boolean
            np.array of shape[num_samples] representing a mask for the arriving samples (if return_indices is True).
        """
        if return_indices:
            return np.isfinite(y_samples)
        y_samples = y_samples.copy()
        y_samples = y_samples[np.isfinite(y_samples)]  # there are default values, remove them from array
        return y_samples

    def compare_moments(self, approaches_ls, prefix='spatial'):
        # change the defaults
        super().compare_moments(approaches_ls, prefix='spatial')

    def plot_sample_histogram(self, y_samples):
        """Plots a histogram of the samples from the Monte Carlo simulations.

        :param y_samples: A np.array of shape [num_samples] containing sampled values.

        """
        # check if there are default values (particles that did not arrive) in the array and remove them
        y_samples = self.remove_not_arriving_samples(y_samples)
        x_label = 'Location y in ' + self.length_unit
        self._plot_sample_histogram(y_samples, x_label)

    def plot_example_tracks(self, N=5, dt=0.0001, plot_x_predTo=False):
        # plot_x_predTo is always false
        super().plot_example_tracks(N, dt, plot_x_predTo=False)

    def plot_mean_and_stddev_over_time(self, ev_fn, var_fn, dt=0.0001, show_example_tracks=False, plot_x_predTo=False):
        # plot_x_predTo is always false
        super().plot_mean_and_stddev_over_time(ev_fn, var_fn, dt, show_example_tracks, plot_x_predTo=False)

    @AbstractHittingEvaluator.check_approaches_ls
    def plot_quantile_functions(self, approaches_ls, q_min=0.005, q_max=0.995):
        """Plots the quantile functions of the different approaches.

        :param approaches_ls: A list of child instances of  AbstractHittingLocationDistribution for the same process to
            be compared.
        :param q_min: A float, the smallest value of the confidence plot range.
        :param q_max: A float, the highest value of the confidence plot range.
        """
        y_label = 'Location y in ' + self.length_unit
        self._plot_quantile_functions(approaches_ls, q_min, q_max, y_label)

    def _plot_y_at_first_hitting_time_distributions(self,
                                                    ax1,
                                                    y_samples,
                                                    approaches_ls,
                                                    plot_moments=True,
                                                    plot_quantiles=True):
        """Plots the distribution of y at the first-passage time.

        :param ax1: A plt.axis object.
        :param y_samples: A np.array of shape [num_samples] containing the y-position at the first-passage times of the
            particles.
        :param approaches_ls: A list of child instances of AbstractHittingLocationDistribution for the same process to
            be compared.
        :param plot_moments: A Boolean, whether to plot the expected value and the standard deviation.
        :param plot_quantiles: A Boolean, whether to plot the median, the first and the third quantile.
        """
        # check if there are default values (particles that did not arrive) in the array and remove them
        y_samples = self.remove_not_arriving_samples(y_samples)

        y_hist, x_hist, _ = ax1.hist(y_samples,
                                     bins=self._distribute_bins_in_plot_range(y_samples),
                                     # we want to have 100 samples in the plot window
                                     density=True,
                                     histtype='stepfilled',  # no space between the bars
                                     color=[0.8, 0.8, 0.8],
                                     )

        ax1.vlines(self._y_predicted, 0, 350, color='black', label="Deterministic Prediction")

        ax2 = ax1.twinx()
        for i, approach in enumerate(approaches_ls):
            hit_stats = approach.get_statistics()
            if 'CDF' in hit_stats.keys():
                plot_f = [hit_stats['CDF'](y) for y in self._plot_y]
                ax2.plot(self._plot_y, plot_f, color=self.color_cycle[i], label=approach.name)
            if 'PDF' in hit_stats.keys():
                plot_f = [hit_stats['PDF'](y) for y in self._plot_y]
                ax1.plot(self._plot_y, plot_f, color=self.color_cycle[i], label=approach.name)
            if 'PDFVALUES' in hit_stats.keys():
                ax1.plot(hit_stats['PDFVALUES'][0], hit_stats['PDFVALUES'][1], color=self.color_cycle[i],
                         label=approach.name)  # TODO: Raus? Auch bei fptd?
            if plot_moments and 'EV' in hit_stats.keys():
                ax2.vlines(hit_stats['EV'], 0, 1, color=self.color_cycle[i], linestyle='dashed', label=approach.name)
                if 'STDDEV' in hit_stats.keys():
                    ax2.vlines([hit_stats['EV'] - hit_stats['STDDEV'], hit_stats['EV'] + hit_stats['STDDEV']], 0, 1,
                               color=self.color_cycle[i], linestyle='dashdot', label=approach.name)
            if plot_quantiles:
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

        ax1.set_ylim(0, 1.4 * y_hist.max())  # leave some space for labels
        ax2.set_ylim(0, 1.05)
        ax1.set_xlim(self._plot_y[0], self._plot_y[-1])
        ax1.set_xlabel("Location in " + self.length_unit)
        ax1.set_ylabel("PDF")
        ax2.set_ylabel("CDF")

    @AbstractHittingEvaluator.check_approaches_ls
    def plot_y_at_first_hitting_time_distributions(self,
                                                   approaches_ls,
                                                   y_samples,
                                                   plot_moments=True,
                                                   plot_quantiles=True):
        """Plots the distribution of y at the first-passage time.

        :param approaches_ls: A list of child instances of AbstractHittingLocationDistribution for the same process to
            be compared.
        :param y_samples: A np.array of shape [num_samples] containing the y-position at the first-passage times of the
            particles.
        :param plot_moments: A Boolean, whether to plot the expected value and the standard deviation.
        :param plot_quantiles: A Boolean, whether to plot the median, the first and the third quantile.
        """
        fig, ax1 = plt.subplots()

        self._plot_y_at_first_hitting_time_distributions(ax1, y_samples, approaches_ls, plot_moments, plot_quantiles)

        if not self._for_paper:
            plt.title("Distribution of Y at First Passage Time for " + self._process_name)
        if self.save_results:
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_y_at_ftp.pdf'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_y_at_ftp.png'))
        if not self.no_show:
            plt.show()
        plt.close()





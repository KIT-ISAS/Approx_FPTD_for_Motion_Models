import os

from abc import ABC

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from evaluators.hitting_model_evaluator import AbstractHittingModelEvaluator


class HittingLocationEvaluator(AbstractHittingModelEvaluator, ABC):
    """A class that handles the evaluations."""

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
                 paper_scaling_factor=2,
                 no_show=False):
        """Initialize the evaluator.

        :param process_name: String, name of the process, appears in headers.
        :param x_predTo: A float, position of the boundary.
        :param t_predicted: A float, the deterministic time of arrival.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param y_predicted: A float, the deterministic y-position at the deterministic time of arrival.
        :param plot_y: A np.array of shape [n_plot_points_y], y-positions where a point in the plot should be displayed.
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

        self.y_predicted = y_predicted
        self.plot_y = plot_y

    def compare_moments(self, approaches_ls, prefix='spatial'):
        # change the defaults
        super().compare_moments(approaches_ls, prefix='spatial')

    def plot_sample_histogram(self, samples, x_label='Location y in mm'):
        # change the defaults
        super().plot_sample_histogram(samples, x_label)

    def plot_example_tracks(self, N=5, dt=0.0001, plot_x_predTo=False):
        # plot_x_predTo is always false
        super().plot_example_tracks(N, dt, plot_x_predTo=False)

    def plot_mean_and_stddev_over_time(self, ev_fn, var_fn, dt=0.0001, show_example_tracks=False, plot_x_predTo=False):
        # plot_x_predTo is always false
        super().plot_mean_and_stddev_over_time(ev_fn, var_fn, dt, show_example_tracks, plot_x_predTo=False)

    def plot_quantile_functions(self, approaches_ls, q_min=0.005, q_max=0.995, y_label='Location y in mm'):
        # change the defaults
        super().plot_quantile_functions(approaches_ls, q_min, q_max, y_label)

    def _plot_y_at_first_hitting_time_distributions(self, ax1, y_samples, approaches_ls):
        """Plots the distribution of y at the first passage time.

        :param ax1: A plt.axis object.
        :param y_samples: A np.array of shape [N] containing the y-position at the first passage times of the particles.
        :param approaches_ls: A list of model objects for the same process to be compared.
        """
        y_hist, x_hist, _ = ax1.hist(y_samples,
                                     bins=self._distribute_bins_in_plot_range(y_samples, self.plot_y),
                                     # we want to have 100 samples in the plot window
                                     density=True,
                                     histtype='stepfilled',  # no space between the bars
                                     color=[0.8, 0.8, 0.8],
                                     )

        ax1.vlines(self.y_predicted, 0, 350, color='black', label="Deterministic Prediction")

        ax2 = ax1.twinx()
        for i, approach in enumerate(approaches_ls):
            hit_stats = approach.get_statistics()
            if 'EV' in hit_stats.keys():
                ax2.vlines(hit_stats['EV'], 0, 1, color=self.color_cycle[i], linestyle='dashed', label=approach.name)
                if 'STDDEV' in hit_stats.keys():
                    ax2.vlines([hit_stats['EV'] - hit_stats['STDDEV'], hit_stats['EV'] + hit_stats['STDDEV']], 0, 1,
                               color=self.color_cycle[i], linestyle='dashdot', label=approach.name)
            if 'CDF' in hit_stats.keys():
                plot_f = [hit_stats['CDF'](y) for y in self.plot_y]
                ax2.plot(self.plot_y, plot_f, color=self.color_cycle[i], label=approach.name)
            if 'PDF' in hit_stats.keys():
                plot_f = [hit_stats['PDF'](y) for y in self.plot_y]
                ax1.plot(self.plot_y, plot_f, color=self.color_cycle[i], label=approach.name)
            if 'PDFVALUES' in hit_stats.keys():
                ax1.plot(hit_stats['PDFVALUES'][0], hit_stats['PDFVALUES'][1], color=self.color_cycle[i],
                         label=approach.name)

        # add legend manually since it fails sometimes
        legend_elements = [Line2D([0], [0], color=c, linewidth=3, label=approach.name) for c, approach in
                           zip(self.color_cycle, approaches_ls)]
        ax2.legend(handles=legend_elements)

        ax1.set_ylim(0, 1.4 * y_hist.max())  # leave some space for labels
        ax2.set_ylim(0, 1.05)
        ax1.set_xlim(self.plot_y[0], self.plot_y[-1])
        ax1.set_xlabel("Location in mm")
        ax1.set_ylabel("PDF")
        ax2.set_ylabel("CDF")

    def plot_y_at_first_hitting_time_distributions(self, y_samples, approaches_ls):
        """Plots the distribution of y at the first passage time.

        :param y_samples: A np.array of shape [N] containing the y-position at the first passage times of the particles.
        :param approaches_ls: A list of model objects for the same process to be compared.
        """
        fig, ax1 = plt.subplots()

        self._plot_y_at_first_hitting_time_distributions(ax1, y_samples, approaches_ls)

        if not self.for_paper:
            plt.title("Distribution of Y at First Passage Time for " + self.process_name)
        #plt.legend()
        if self.save_results:
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_y_at_ftp.pdf'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_y_at_ftp.png'))
        if not self.no_show:
            plt.show()
        plt.close()





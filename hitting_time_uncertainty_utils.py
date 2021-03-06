"""
Utilities and Plot Function for First Passage Time Models.
"""
import os

from abc import ABC

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
from cycler import cycler


class HittingTimeEvaluator(ABC):
    """A class that handles the evaluations."""

    def __init__(self,
                 process_name,
                 x_predTo,
                 plot_t,
                 t_predicted,
                 t_L = 0.0,
                 y_predicted=None,
                 plot_y=None,
                 get_example_tracks_fn=None,
                 save_results=False,
                 result_dir=None,
                 for_paper=True,
                 fig_width=0.34*505.89*1/72,  # factor_of_textwidth*textwidth_in_pt*pt_to_inches
                 no_show=False):
        """Initialize the evaluator.

        :param process_name: String, name of the process, appears in headers.
        :param x_predTo: A float, position of the boundary.
        :param plot_t: A np.array of shape [n_plot_points], point in time where a point in the plot should be displayed.
        :param t_predicted: A float, the deterministic time of arrival.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param y_predicted: A float, the deterministic y-postition at the deterministic time of arrival.
        :param plot_y: A np.array of shape [n_plot_points_y], y-positions where a point in the plot should be displayed.
        :param get_example_tracks_fn: A function that draws example paths from the model.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        :param fig_width: A float, width of the figures in inches.
        :param no_show: Boolean, whether to show the plots (False).
        """
        self.x_predTo = x_predTo
        self.t_predicted = t_predicted
        self.y_predicted = y_predicted
        self.t_L = t_L
        self.plot_t = plot_t
        self.plot_y = plot_y
        #self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.get_example_tracks = get_example_tracks_fn

        self.process_name = process_name
        self.process_name_save = process_name.lower().replace(" ", "_")
        self.save_results = save_results
        if result_dir is not None and not os.path.exists(result_dir):
            os.makedirs(result_dir)
        self.result_dir = result_dir
        self.for_paper = for_paper
        self.no_show = no_show

        if self.for_paper:
            basename = os.path.basename(os.path.normpath(result_dir))
            self.process_name_save = basename.lower().replace(" ", "_")

            scaling_factor = 2
            mpl.rcParams.update({'font.size': scaling_factor*6})  # font size

            #plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
            #plt.style.use('bmh')
            #plt.style.use('seaborn')
            self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # 3 options to avoid cutting off the labels when reducing the figure size..
            plt.rcParams['savefig.bbox'] = 'tight'  # figsize (without labels) as defined add extra space for the labels
            # mpl.rcParams["figure.autolayout"] = True  # figure with labels! gets reduced to figsize
            # mpl.rcParams["figure.constrained_layout.use"] = True  # same as before, but more powerful function, use this one

            mpl.rcParams['savefig.dpi'] = 100  # dots per inch
            mpl.rcParams['figure.dpi'] = 20  # takes a long time to show the figures otherwiese

            # use them if scaling_factor = 1, but spines are not scaled properly then
            # mpl.rcParams['axes.linewidth'] = 0.25  # pt
            # mpl.rcParams['lines.linewidth'] = 0.5  # pt
            mpl.rcParams["figure.figsize"] = [scaling_factor*fig_width, scaling_factor*fig_width * 5 / 7]
            plt.rcParams['legend.loc'] = 'upper left'

    def compare_moments_temporal(self, approaches_ls):
        """Compare the means and standard deviations of different first passage time approaches and print them to stdout.

        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        """
        mu_t_ls = []
        stddev_t_ls = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            mu_t = hit_stats['EV']
            stddev_t = hit_stats['STDDEV']
            print("Temporal mean of {0}: {1}".format(approach.name, round(mu_t, 6)))
            print("Temporal stddev of {0}: {1}".format(approach.name, round(stddev_t, 6)))
            mu_t_ls.append(mu_t)
            stddev_t_ls.append(stddev_t)

        rel_devs = np.zeros((len(approaches_ls), len(approaches_ls)))
        for i in range(len(approaches_ls)):
            for j in range(len(approaches_ls)):
                rel_devs[i, j] = 100 * np.abs(stddev_t_ls[i] - stddev_t_ls[j]) / np.max([stddev_t_ls[i], stddev_t_ls[j]])

        print('Pairwise relative deviations of temporal stddevs in percent: \n', np.round(rel_devs, 2))

    def compare_skewness_temporal(self, approaches_ls):
        """Compare the skewness of different first passage time approaches and print them to stdout.

        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
        """
        skew_t_ls = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            skew_t = hit_stats['SKEW']
            print("Temporal skewness of {0}: {1}".format(approach.name, round(skew_t, 6)))
            skew_t_ls.append(skew_t)

        rel_devs = np.zeros((len(approaches_ls), len(approaches_ls)))
        for i in range(len(approaches_ls)):
            for j in range(len(approaches_ls)):
                rel_devs[i, j] = 100 * np.abs(skew_t_ls[i] - skew_t_ls[j]) / np.max([skew_t_ls[i], skew_t_ls[j]])

        print('Pairwise relative deviations of temporal skewness in percent: \n', np.round(rel_devs, 2))

    def compare_moments_spatial(self, approaches_ls):
        """Compare the means and standard deviations of different approaches for calculating the
         distribution in y at the first passage time and print them to stdout.

         :param approaches_ls: A list of model objects for the same process to be compared.
         """
        mu_y_ls = []
        stddev_y_ls = []
        for approach in approaches_ls:
            hit_stats = approach.get_statistics()
            mu_y = hit_stats['EV']
            stddev_y = hit_stats['STDDEV']
            print("Spatial mean of {0}: {1}".format(approach.name, round(mu_y, 6)))
            print("Spatial stddev of {0}: {1}".format(approach.name, round(stddev_y, 6)))
            mu_y_ls.append(mu_y)
            stddev_y_ls.append(stddev_y)

        rel_devs = np.zeros((len(approaches_ls), len(approaches_ls)))
        for i in range(len(approaches_ls)):
            for j in range(len(approaches_ls)):
                rel_devs[i, j] = 100 * np.abs(stddev_y_ls[i] - stddev_y_ls[j]) / np.max([stddev_y_ls[i], stddev_y_ls[j]])

        print('Pairwise relative deviations of spatial stddevs in percent: \n', np.round(rel_devs, 2))

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

    def plot_example_tracks(self, N=5, dt=0.0001):
        """Plot example paths.

        :param N: Integer, number of tracks to plot.
        :param dt: A float, time increment between two consecutive points.
        """

        plot_t = np.arange(self.t_L, 1.2 * self.t_predicted, dt)

        x_tracks = self.get_example_tracks(plot_t, N)

        plt.figure(figsize=[19.20, 10.80], dpi=100)
        plt.plot(plot_t, x_tracks)
        plt.hlines(self.x_predTo, xmin=plot_t[0], xmax=plot_t[-1], color='black', linestyle='dashdot',
                   label='a (Boundary)')

        ax.set_xlim(plot_t[0], plot_t[-1])
        plt.title('Example Tracks')
        plt.xlabel('Time in s')
        plt.ylabel('Traveled distance in mm')
        plt.legend()
        if not self.no_show:
            plt.show()
        plt.close()

    def _plot_mean_and_stddev_over_time(self, ax, ev_fn, var_fn, dt=0.0001, show_example_tracks=False):
        """Plots mean and standard deviation (and example tracks) over time.

        :param ax: A plt.axis object.
        :param ev_fn: The mean function of the process.
        :param var_fn: The variance function of the process.
        :param dt: A float, time increment between two consecutive points.
        :param show_example_tracks: Boolean, whether to additionally show some paths.
        """

        plot_t = np.arange(self.t_L, 1.2*self.t_predicted, dt)

        ev_x_plot = ev_fn(plot_t)
        sigma_x_plot = np.sqrt(var_fn(plot_t))

        if show_example_tracks:
            x_tracks = self.get_example_tracks(plot_t)
            ax.plot(plot_t, x_tracks)
        ax.plot(plot_t, ev_x_plot, label="EV", color='black')
        ax.fill_between(plot_t, ev_x_plot - sigma_x_plot, ev_x_plot + sigma_x_plot,
                        color="gray",
                        alpha=0.2,
                        label='Stddev')
        ax.hlines(self.x_predTo, xmin=plot_t[0], xmax=plot_t[-1], color='black', linestyle='dashdot', label='a (Boundary)')

        if not self.for_paper:
            plt.title('Expected Value and Variance over Time for ' + self.process_name)

        ax.set_xlim(plot_t[0], plot_t[-1])
        ax.set_xlabel('Time in s')
        ax.set_ylabel('Location in mm')
        ax.legend()

    def plot_mean_and_stddev_over_time(self, ev_fn, var_fn, dt=0.0001, show_example_tracks=False):
        """Plots mean and standard deviation (and example tracks) over time.

        :param ev_fn: The mean function of the process.
        :param var_fn: The variance function of the process.
        :param dt: A float, time increment between two consecutive points.
        :param show_example_tracks: Boolean, whether to additionally show some paths.
        """
        fig, ax = plt.subplots()

        self._plot_mean_and_stddev_over_time(ax, ev_fn, var_fn, dt, show_example_tracks)

        if self.save_results:
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_mean_and_stddev_over_time.pdf'))
            plt.savefig(os.path.join(self.result_dir, self.process_name_save + '_mean_and_stddev_over_time.png'))
        if not self.no_show:
            plt.show()
        plt.close()

    def plot_quantile_functions(self, approaches_ls, q_min=0.005, q_max=0.995, y_label='Time t in s'):
        """Plots the quantile functions of the different approaches.

        :param approaches_ls: A list of first passage time model objects for the same process to be compared.
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
        if not self.no_show:
            plt.show()
        plt.close()

    def _plot_y_at_first_hitting_time_distributions(self, ax1, y_samples, approaches_ls):
        """Plots the distribution of y at the first passage time.

        :param ax1: A plt.axis object.
        :param y_samples: A np.array of shape [N] containing the y-position at the first passage times of the particles.
        :param approaches_ls: A list of model objects for the same process to be compared.
        """
        y_hist, x_hist, _ = ax1.hist(y_samples,
                                     bins=int(100*(max(y_samples) - min(y_samples))/(self.plot_y[-1] - self.plot_y[0])),
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
                ax1.plot(hit_stats['PDFVALUES'][0], hit_stats['PDFVALUES'][1], color=self.color_cycle[i], label=approach.name)

        # add legend manually since it fails sometimes
        lines =[Line2D([0], [0], color=c, linewidth=3) for c in self.color_cycle]
        labels = [approach.name for approach in approaches_ls]
        ax2.legend(lines, labels)

        ax1.set_ylim(0, 1.4*y_hist.max())  # leave some space for labels
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
                                         bins=int(100 * (max(t_samples) - min(t_samples)) / (
                                                     self.plot_t[-1] - self.plot_t[0])),
                                         # we want to have 100 samples in the plot window
                                         density=True,
                                         histtype='stepfilled',  # no space between the bars
                                         color=[0.8, 0.8, 0.8],
                                         )
            y_hist_max = y_hist.max()
            # sums up to 1 (sum(y_hist * np.diff(x_hist))=1) not including particles that did not arrive
        else:
            y_hist, x_hist, _ = ax1.hist(t_samples,
                                         bins=int(100 * (max(t_samples) - min(t_samples)) / (
                                                     self.plot_t[-1] - self.plot_t[0])),
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
        lines = [Line2D([0], [0], color=c, linewidth=3) for c in self.color_cycle]
        labels = [approach.name for approach in approaches_ls]
        ax2.legend(lines, labels)

        ax1.set_ylim(0, 1.4 * y_hist_max)  # leave some space for labels
        ax2.set_ylim(0, 1.05)
        ax1.set_xlim(self.plot_t[0], self.plot_t[-1])
        # ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # ax1.set_yticks(np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(ax2.get_yticks())))
        # ax1.set_ylim(0, 1.05*ax1.get_yticks()[-1])
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
        if not self.no_show:
            plt.show()
        plt.close()




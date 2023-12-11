"""Utilities and plot function for first-passage time models with a particle extent.

"""
import os
from absl import logging

from abc import ABC, abstractmethod

from functools import wraps

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

import numpy as np
from scipy.stats import rv_histogram

from evaluators.hitting_evaluator import AbstractHittingEvaluator
from evaluators.hitting_time_evaluator import HittingTimeEvaluator
from evaluators.hitting_location_evaluator import HittingLocationEvaluator
from extent_models import AbstractHittingTimeWithExtentsModel, HittingLocationWithExtentsModel


class AbstractHittingEvaluatorWithExtents(AbstractHittingEvaluator, ABC):
    """A base class that handles the evaluations for hitting time or location models with extents."""

    def __init__(self, **kwargs):
        """Initializes the evaluator."""
        super().__init__(**kwargs)

    def _plot_interval_distributions_on_single_axis(self,
                                                    ax1,
                                                    lower_marginal_samples,
                                                    upper_marginal_samples,
                                                    approaches_ls,
                                                    plot_hist_for_all_particles=True,
                                                    plot_cdfs=False,
                                                    plot_back_and_front_distr=False,
                                                    prefix_ls=[' (front arrival)', ' (back arrival)'],
                                                    ):
        """Plots the (actually two-dimensional) temporal or spatial deflection window on the same axis (representing
        either time or y-location).
        
        :param ax1: A plt.axis object.
        :param lower_marginal_samples: A np.array of shape [num_samples] containing samples of the lower marginal 
            distribution, e.g., the front arrival times oder the particles' lowermost edge locations.
        :param upper_marginal_samples: A np.array of shape [num_samples] containing samples of the lower marginal 
            distribution, e.g., the back arrival times oder the particles' uppermost edge locations.
        :param approaches_ls: A list of child instances of AbstractHittingTimeWithExtentsModel or
            HittingLocationWithExtentsModel for the same process to be  compared.
        :param plot_hist_for_all_particles: A Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        :param plot_cdfs: A Boolean, whether to additionally plot the CDF on a new plot axis.
        :param plot_back_and_front_distr: A Boolean, whether to plot the marginal distributions for the front and back
            arrival locations. Only relevant for spatial deflection windows.
        :param prefix_ls: A list of length 2 containing strings, a prefix for the lower and upper marginal distribution.
        """
        if not plot_hist_for_all_particles:
            # check if there are default values (particles that did not arrive) in the array
            lower_marginal_samples = self.remove_not_arriving_samples(lower_marginal_samples)

            y_hist, x_hist, _ = ax1.hist(lower_marginal_samples,
                                         bins=self._distribute_bins_in_plot_range(lower_marginal_samples,
                                                                                  no_of_bins=200),
                                         # we want to have 200 samples in the plot window since the plot window is
                                         # larger than for the distributions without extents
                                         density=True,
                                         histtype='stepfilled',  # no space between the bars
                                         color=[0.8, 0.8, 0.8],
                                         )
            # sums up to 1 (sum(y_hist * np.diff(x_hist))=1) not including particles that did not arrive

            # check if there are default values (particles that did not arrive) in the array
            upper_marginal_samples = self.remove_not_arriving_samples(upper_marginal_samples)
            ax1.hist(upper_marginal_samples,
                     bins=self._distribute_bins_in_plot_range(upper_marginal_samples, no_of_bins=200),
                     density=True,
                     histtype='stepfilled',  # no space between the bars
                     color=[0.8, 0.8, 1.0],
                     alpha=0.6,
                     )
            y_hist_max = y_hist.max()

        else:
            y_hist, x_hist, _ = ax1.hist(lower_marginal_samples,
                                         bins=self._distribute_bins_in_plot_range(lower_marginal_samples,
                                                                                  no_of_bins=200),
                                         density=True,
                                         histtype='stepfilled',  # no space between the bars
                                         color=[0.8, 0.8, 0.8])
            ax1.hist(upper_marginal_samples,
                     bins=self._distribute_bins_in_plot_range(upper_marginal_samples, no_of_bins=200),
                     # we want to have 100 samples in the plot window
                     density=True,
                     histtype='stepfilled',  # no space between the bars
                     color=[0.8, 0.8, 1.0],
                     alpha=0.6,
                     )

            y_hist_max = y_hist[:-1].max()
            # sums up to 1 (sum(y_hist * np.diff(x_hist))=1) while also including particles that do not not arrive at
            # the boundary

        if plot_cdfs:

            lower_marginal_hist = np.histogram(lower_marginal_samples,
                                               bins=self._distribute_bins_in_plot_range(lower_marginal_samples,
                                                                                        no_of_bins=200),
                                               density=False)
            lower_marginal_density = rv_histogram(lower_marginal_hist, density=True)
            upper_marginal_hist = np.histogram(upper_marginal_samples,
                                               bins=self._distribute_bins_in_plot_range(upper_marginal_samples,
                                                                                        no_of_bins=200),
                                               density=False)
            upper_marginal_density = rv_histogram(upper_marginal_hist, density=True)

            ax2 = ax1.twinx()
            ax2.plot(self.plot_points, [lower_marginal_density.cdf(t) for t in self.plot_points],
                     color='black',
                     )
            ax2.plot(self.plot_points, [upper_marginal_density.cdf(t) for t in self.plot_points],
                     color='black',
                     )
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("CDF")

        for i, approach in enumerate(approaches_ls):
            lower_bound, upper_bound = approach.calculate_confidence_bounds(0.95)
            logging.info('Deflection windows for model {}: {}, {}'.format(approach.name, np.round(lower_bound, 4),
                                                                          np.round(upper_bound, 4)))
            ax1.vlines((lower_bound, upper_bound), ymin=0, ymax=1.4 * y_hist_max,
                       label=approach.name,
                       color=self.color_cycle[i])
            if hasattr(approach, 'front_arrival_distribution'):
                ax1.plot(self.plot_points, [approach.front_arrival_distribution.pdf(t) for t in self.plot_points],
                         label=approach.name,
                         color=self.color_cycle[i])
                if plot_cdfs:
                    ax2.plot(self.plot_points, [approach.front_arrival_distribution.cdf(t) for t in self.plot_points],
                             label=approach.name,
                             color=self.color_cycle[i])
            if hasattr(approach, 'back_arrival_distribution'):
                ax1.plot(self.plot_points, [approach.back_arrival_distribution.pdf(t) for t in self.plot_points],
                         label=approach.name,
                         color=self.color_cycle[i])
                if plot_cdfs:
                    ax2.plot(self.plot_points, [approach.back_arrival_distribution.cdf(t) for t in self.plot_points],
                             label=approach.name,
                             color=self.color_cycle[i])
            if hasattr(approach, 'max_y_distribution'):
                ax1.plot(self.plot_points, [approach.max_y_distribution.pdf(y) for y in self.plot_points],
                         label=approach.name,
                         color=self.color_cycle[i])
                if plot_back_and_front_distr:
                    ax1.plot(self.plot_points,
                             [approach.max_y_distribution.back_location_pdf(y) for y in self.plot_points],
                             label=approach.name,
                             linestyle='dashed',
                             color='red')
                    ax1.plot(self.plot_points,
                             [approach.max_y_distribution.front_location_pdf(y) for y in self.plot_points],
                             label=approach.name,
                             linestyle='dotted',
                             color=self.color_cycle[i])
                if plot_cdfs:
                    ax2.plot(self.plot_points, [approach.max_y_distribution.cdf(y) for y in self.plot_points],
                             label=approach.name,
                             color=self.color_cycle[i])
                    if plot_back_and_front_distr:
                        ax2.plot(self.plot_points,
                                 [approach.max_y_distribution.back_location_cdf(y) for y in self.plot_points],
                                 label=approach.name,
                                 linestyle='dashed',
                                 color='red')
                        ax2.plot(self.plot_points,
                                 [approach.max_y_distribution.front_location_cdf(y) for y in self.plot_points],
                                 label=approach.name,
                                 linestyle='dotted',
                                 color=self.color_cycle[i])
            if hasattr(approach, 'min_y_distribution'):
                ax1.plot(self.plot_points, [approach.min_y_distribution.pdf(y) for y in self.plot_points],
                         label=approach.name,
                         color=self.color_cycle[i])
                if plot_back_and_front_distr:
                    ax1.plot(self.plot_points,
                             [approach.min_y_distribution.back_location_pdf(y) for y in self.plot_points],
                             label=approach.name,
                             linestyle='dashed',
                             color='red')
                    ax1.plot(self.plot_points,
                             [approach.min_y_distribution.front_location_pdf(y) for y in self.plot_points],
                             label=approach.name,
                             linestyle='dotted',
                             color=self.color_cycle[i])
                if plot_cdfs:
                    ax2.plot(self.plot_points, [approach.min_y_distribution.cdf(y) for y in self.plot_points],
                             label=approach.name,
                             color=self.color_cycle[i])
                    if plot_back_and_front_distr:
                        ax2.plot(self.plot_points,
                                 [approach.min_y_distribution.back_location_cdf(y) for y in self.plot_points],
                                 label=approach.name,
                                 linestyle='dashed',
                                 color='red')
                        ax2.plot(self.plot_points,
                                 [approach.min_y_distribution.front_location_cdf(y) for y in self.plot_points],
                                 label=approach.name,
                                 linestyle='dotted',
                                 color=self.color_cycle[i])

        # add legend manually since it fails sometimes
        legend_elements = [Line2D([0], [0], color=c, linewidth=3, label=approach.name) for c, approach in
                           zip(self.color_cycle, approaches_ls)]
        legend_elements.append(Patch(facecolor=[0.8, 0.8, 0.8], label='MC simulation' + prefix_ls[0]))
        legend_elements.append(Patch(facecolor=[0.8, 0.8, 1], alpha=0.6, label='MC simulation' + prefix_ls[1]))
        if plot_cdfs:
            ax2.legend(handles=legend_elements)
        else:
            ax1.legend(handles=legend_elements)

        ax1.set_ylim(0, 1.65 * y_hist_max)  # leave some space for labels
        ax1.set_xlim(self.plot_points[0], self.plot_points[-1])
        ax1.set_ylabel("PDF")

    def _plot_joint_interval_distribution(self,
                                          ax1,
                                          lower_marginal_samples,
                                          upper_marginal_samples,
                                          approaches_ls,
                                          plot_hist_for_all_particles,
                                          use_independent_joint,
                                          marginal_x_axis=None,
                                          marginal_y_axis=None):
        """Plots the two-dimensional joint distribution of the temporal or spatial deflection windows.

        :param ax1: A plt.axis object.
        :param lower_marginal_samples: A np.array of shape [num_samples] containing samples of the lower marginal 
            distribution, e.g., the front arrival times oder the particles' lowermost edge locations.
        :param upper_marginal_samples: A np.array of shape [num_samples] containing samples of the lower marginal 
            distribution, e.g., the back arrival times oder the particles' uppermost edge locations.
        :param approaches_ls: A list of child instances of AbstractHittingTimeWithExtentsModel or
            HittingLocationWithExtentsModel for the same process to be  compared.
        :param plot_hist_for_all_particles: A Boolean, whether to plot the histogram
                only for particles that arrive at the boundary (False).
        :param use_independent_joint: A Boolean, whether to construct the joint distribution by multiplication of the
            marginal distributions (with mathematically is a simplification) or use the true two-dimensional histogram.
        :param marginal_x_axis: None or a plt.axis object. If given, the lower marginal distribution will be 
            additionally plotted on this axis.
        :param marginal_y_axis: None or a plt.axis object. If given, the upper marginal distribution will be 
            additionally plotted on this axis.
        """
        if not plot_hist_for_all_particles:
            # check if there are default values (particles that did not arrive) in the array
            # lower_marginal_samples and upper_marginal_samples are required to have the same shape for np.histogram2d
            valid_mask_lo = self.remove_not_arriving_samples(lower_marginal_samples, return_indices=True)
            valid_mask_hi = self.remove_not_arriving_samples(upper_marginal_samples, return_indices=True)
            combined_mask = np.logical_and(valid_mask_lo, valid_mask_hi)
            lower_marginal_samples = lower_marginal_samples.copy()[combined_mask]
            upper_marginal_samples = upper_marginal_samples.copy()[combined_mask]

        if marginal_x_axis is not None or marginal_y_axis is not None or use_independent_joint:
            # marginal densities
            lower_marginal_hist = np.histogram(lower_marginal_samples,
                                               bins=self._distribute_bins_in_plot_range(lower_marginal_samples,
                                                                                        no_of_bins=200),
                                               # we want to have 200 samples in the plot window since the plot window is
                                               # larger than for the distributions without extents
                                               density=False)
            lower_marginal_density = rv_histogram(lower_marginal_hist, density=True)
            upper_marginal_hist = np.histogram(upper_marginal_samples,
                                               bins=self._distribute_bins_in_plot_range(upper_marginal_samples,
                                                                                        no_of_bins=200),
                                               density=False)
            upper_marginal_density = rv_histogram(upper_marginal_hist, density=True)

        if not use_independent_joint:
            # joint distribution
            twod_hist, xedges, yedges, _ = ax1.hist2d(lower_marginal_samples,
                                                      upper_marginal_samples,
                                                      bins=[self._distribute_bins_in_plot_range(lower_marginal_samples,
                                                                                                no_of_bins=200),
                                                            self._distribute_bins_in_plot_range(upper_marginal_samples,
                                                                                                no_of_bins=200)],
                                                      density=True,
                                                      norm=mcolors.PowerNorm(0.3),
                                                      )

        else:
            # joint by multiplication
            # first get the edges from joint histogram
            twod_hist, xedges, yedges = np.histogram2d(lower_marginal_samples,
                                                       upper_marginal_samples,
                                                       bins=[self._distribute_bins_in_plot_range(lower_marginal_samples,
                                                                                                 no_of_bins=200),
                                                             self._distribute_bins_in_plot_range(upper_marginal_samples,
                                                                                                 no_of_bins=200)],
                                                       density=True,
                                                       )
            # calulate the joint
            z = lower_marginal_density.pdf(xedges)[:, None] @ upper_marginal_density.pdf(yedges)[None, :]
            ax1.pcolormesh(xedges, yedges, z.T,
                           norm=mcolors.PowerNorm(0.3),
                           )

        # marginal distributions
        if marginal_x_axis is not None:
            lower_mc_pdf_values = [lower_marginal_density.pdf(x) for x in self.plot_points]
            marginal_x_axis.plot(self.plot_points, lower_mc_pdf_values)
        if marginal_y_axis is not None:
            upper_mc_pdf_values = [upper_marginal_density.pdf(t) for t in self.plot_points]
            marginal_y_axis.plot(upper_mc_pdf_values, self.plot_points)

        ax1.plot([min(lower_marginal_samples), max(lower_marginal_samples)],
                 [min(lower_marginal_samples), max(lower_marginal_samples)],
                 color='black', label='Bisector')  # bisector

        for i, approach in enumerate(approaches_ls):
            ax1.vlines(approach.calculate_confidence_bounds(0.95)[0],
                       ymin=min(upper_marginal_samples),
                       ymax=max(upper_marginal_samples),
                       label=approach.name,
                       color=self.color_cycle[i])
            ax1.hlines(approach.calculate_confidence_bounds(0.95)[1],
                       xmin=min(lower_marginal_samples),
                       xmax=max(lower_marginal_samples),
                       label=approach.name,
                       color=self.color_cycle[i])
            if marginal_x_axis is not None:
                if hasattr(approach, 'front_arrival_distribution'):
                    pdf_values = [approach.front_arrival_distribution.pdf(t) for t in self.plot_points]
                    marginal_x_axis.plot(self.plot_points, pdf_values,
                                         label=approach.name,
                                         color=self.color_cycle[i])
                if hasattr(approach, 'min_y_distribution'):
                    pdf_values = [approach.min_y_distribution.pdf(y) for y in self.plot_points]
                    marginal_x_axis.plot(self.plot_points, pdf_values,
                                         label=approach.name,
                                         color=self.color_cycle[i])
                marginal_x_axis.vlines(approach.calculate_confidence_bounds(0.95)[0],
                                       ymin=0,
                                       ymax=max(lower_mc_pdf_values),
                                       label=approach.name,
                                       color=self.color_cycle[i])

            if marginal_y_axis is not None:
                if hasattr(approach, 'back_arrival_distribution'):
                    pdf_values = [approach.back_arrival_distribution.pdf(t) for t in self.plot_points]
                    marginal_y_axis.plot(pdf_values, self.plot_points,
                                         label=approach.name,
                                         color=self.color_cycle[i])
                if hasattr(approach, 'max_y_distribution'):
                    pdf_values = [approach.max_y_distribution.pdf(y) for y in self.plot_points]
                    marginal_y_axis.plot(pdf_values, self.plot_points,
                                         label=approach.name,
                                         color=self.color_cycle[i])
                marginal_y_axis.hlines(approach.calculate_confidence_bounds(0.95)[1],
                                       xmin=0,
                                       xmax=max(upper_mc_pdf_values),
                                       label=approach.name,
                                       color=self.color_cycle[i])

        # add legend manually since it fails sometimes
        legend_elements = [Line2D([0], [0], color=c, linewidth=3, label=approach.name) for c, approach in
                           zip(self.color_cycle, approaches_ls)]
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=3, label='Bisector'))
        y_legend_pos = 1.05 if marginal_y_axis is None else 1.35
        ax1.legend(handles=legend_elements, bbox_to_anchor=(y_legend_pos, 1.0), loc='upper left')
        ax1.set_xlim(max(min(xedges), self.plot_points[0]), min(max(xedges), self.plot_points[-1]))
        ax1.set_ylim(max(min(yedges), self.plot_points[0]), min(max(yedges), self.plot_points[-1]))
        ax1.set_aspect('equal')
        
    def _plot_calibration(self,
                          axes,
                          lower_marginal_samples,
                          upper_marginal_samples,
                          approaches_ls,
                          ):
        """Plots the calibration of the deflection windows, i.e., the ratio of expects hits vs. the "true" hits in the
        corresponding deflection window.
 
        :param axes: A plt.axis object.
        :param lower_marginal_samples: A np.array of shape [num_samples] containing samples of the lower marginal 
            distribution, e.g., the front arrival times oder the particles' lowermost edge locations.
        :param upper_marginal_samples: A np.array of shape [num_samples] containing samples of the lower marginal 
            distribution, e.g., the back arrival times oder the particles' uppermost edge locations.
        :param approaches_ls: A list of child instances of AbstractHittingTimeWithExtentsModel or
            HittingLocationWithExtentsModel for the same process to be  compared.
        """
        # check if there are default values (particles that did not arrive) in the array and remove them
        # lower_marginal_samples and upper_marginal_samples are required to have the same shape for np.histogram2d
        valid_mask_lo = self.remove_not_arriving_samples(lower_marginal_samples, return_indices=True)
        valid_mask_hi = self.remove_not_arriving_samples(upper_marginal_samples, return_indices=True)
        combined_mask = np.logical_and(valid_mask_lo, valid_mask_hi)
        lower_marginal_samples = lower_marginal_samples.copy()[combined_mask]
        upper_marginal_samples = upper_marginal_samples.copy()[combined_mask]

        # joint densities
        twod_hist, xedges, yedges, = np.histogram2d(lower_marginal_samples,
                                                    upper_marginal_samples,
                                                    bins=[self._distribute_bins_in_plot_range(lower_marginal_samples,
                                                                                              no_of_bins=500),
                                                          self._distribute_bins_in_plot_range(upper_marginal_samples,
                                                                                              no_of_bins=500)],
                                                    density=True,
                                                    )

        def get_prob_from_hist(t1, t2):
            valid_x = xedges[:-1] > t1
            valid_y = yedges[:-1] < t2
            bin_areas = np.diff(xedges)[:, None] @ np.diff(yedges)[None, :]
            return np.sum((twod_hist * bin_areas)[valid_x[:, None] @ valid_y[None, :]])

        # marginal densities
        front_hist = np.histogram(lower_marginal_samples,
                                  bins=self._distribute_bins_in_plot_range(lower_marginal_samples, no_of_bins=500),
                                  density=False)
        front_arrival_time_density = rv_histogram(front_hist, density=True)
        back_hist = np.histogram(upper_marginal_samples,
                                 bins=self._distribute_bins_in_plot_range(upper_marginal_samples, no_of_bins=500),
                                 density=False)
        back_arrival_time_density = rv_histogram(back_hist, density=True)

        ax1, ax2 = axes

        ax1.plot([0, 1],
                 [0, 1],
                 color='black', label='Bisector')  # bisector
        ax2.plot([0, 1],
                 [0, 1],
                 color='black', label='Bisector')  # bisector

        q_range = np.arange(0.01, 0.99, step=0.01)
        for i, approach in enumerate(approaches_ls):
            ratio = []
            marginal_ratio = []
            for q in q_range:
                lower_bound, upper_bound = approach.calculate_confidence_bounds(q)
                ratio.append(get_prob_from_hist(lower_bound, upper_bound))
                marginal_ratio.append(
                    back_arrival_time_density.cdf(upper_bound) - front_arrival_time_density.cdf(lower_bound))

            logging.info('Calibration mean error for model {}: {}'.format(approach.name,
                                                                          np.round(np.mean(np.abs(ratio - q_range)),
                                                                                   4)))
            logging.info('Marginal mean calibration error for model {}: {}'.format(approach.name,
                                                                                   np.round(np.mean(np.abs(
                                                                                       marginal_ratio - q_range)), 4)))

            ax1.plot(q_range, ratio, label=approach.name, color=self.color_cycle[i])
            ax2.plot(q_range, marginal_ratio, label=approach.name, color=self.color_cycle[i])

        # add legend manually since it fails sometimes
        legend_elements = [Line2D([0], [0], color=c, linewidth=3, label=approach.name) for c, approach in
                           zip(self.color_cycle, approaches_ls)]
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=3, label='Bisector (perfect calibration)'))

        for ax in axes:
            ax.legend(handles=legend_elements)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel('Ratio from MC')
            ax.set_xlabel('Confidence range')


class HittingTimeEvaluatorWithExtents(HittingTimeEvaluator, AbstractHittingEvaluatorWithExtents):
    """A class that handles the evaluations for hitting time models with a particle extent."""

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
                 paper_font='Times',
                 paper_scaling_factor=2,
                 no_show=False,
                 time_unit='s',
                 length_unit='m'):
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
        :param paper_font: A string, the font to be used for the paper. Either "Times", "Helvetica" or "Default". Only
            relevant if for_paper is True.
        :param fig_width: A float, the width of the figures in inches.
        :param font_size: An integer, the font size in point.
        :param paper_scaling_factor: A float, a scaling factor to be applied to the figure and fonts if _for_paper is
            true.
        :param no_show: A Boolean, whether to show the plots (False).
        :param time_unit: A string, the time unit of the process (used for the plot labels).
        :param length_unit: A string, the location unit of the process (used for the plot labels).
        """
        super().__init__(process_name=process_name,
                         x_predTo=x_predTo,
                         plot_t=plot_t,
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

    def check_approaches_ls(func):
        """A decorator for functions that process a list of hitting time with extents models.

        Assure that only distributions of the same type are in the list.

        :param func: A callable, the function to be decorated.

        :returns: A callable, the decorator.
        """
        @wraps(func)
        def check_approaches_in_approaches_ls_of_same_type(self, approaches_ls, *args, **kwargs):
            if not all([isinstance(a, AbstractHittingTimeWithExtentsModel) for a in approaches_ls]):
                raise ValueError(
                    'The distributions in approaches_ls must be all child instances of AbstractHittingTimeWithExtentsModel.')

            return func(self, approaches_ls, *args, **kwargs)

        return check_approaches_in_approaches_ls_of_same_type

    @check_approaches_ls
    def plot_first_arrival_interval_distribution_on_time_axis(self,
                                                              approaches_ls,
                                                              front_arrival_time_samples,
                                                              back_arrival_time_samples,
                                                              plot_hist_for_all_particles=True,
                                                              plot_cdfs=True,
                                                              ):
        """Plots the (actually two-dimensional) temporal deflection window on the time axis.

        :param approaches_ls: A list of child instances of AbstractHittingTimeWithExtentsModel.
        :param front_arrival_time_samples: A np.array of shape [num_samples] containing samples of the front arrival
            times.
        :param back_arrival_time_samples: A np.array of shape [num_samples] containing samples of the back arrival
            times.
        :param plot_hist_for_all_particles: A Boolean, whether to plot the histogram only for particles that arrive at
            the boundary (False).
        :param plot_cdfs: A Boolean, whether to additionally plot the CDF on a new plot axis.
        """
        fig, ax1 = plt.subplots()
        self._plot_interval_distributions_on_single_axis(ax1,
                                                         front_arrival_time_samples,
                                                         back_arrival_time_samples,
                                                         approaches_ls,
                                                         plot_hist_for_all_particles=plot_hist_for_all_particles,
                                                         plot_cdfs=plot_cdfs,
                                                         )
        ax1.set_xlabel("Time in " + self.time_unit)

        if not self._for_paper:
            plt.title(
                "Distribution of First Particle Front and Back Arrival Time for " + self._process_name)
        # plt.legend()
        if self.save_results:
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_first_arrival_interval_distr.pdf'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_first_arrival_interval_distr.png'))
            plt.savefig(os.path.join(self._result_dir, self._process_name_save + '_first_arrival_interval_distr.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    @check_approaches_ls
    def plot_joint_first_arrival_interval_distribution(self,
                                                       approaches_ls,
                                                       front_arrival_time_samples,
                                                       back_arrival_time_samples,
                                                       plot_marginals=True,
                                                       plot_hist_for_all_particles=True,
                                                       use_independent_joint=False,
                                                       ):
        """Plots the two-dimensional joint distribution of the temporal deflection windows.

        :param approaches_ls: A list of child instances of AbstractHittingTimeWithExtentsModel.
        :param front_arrival_time_samples: A np.array of shape [num_samples] containing samples of the front arrival
            times.
        :param back_arrival_time_samples: A np.array of shape [num_samples] containing samples of the back arrival
            times.
        :param plot_marginals: A Boolean, whether to plot the marginal distributions on separate x- and y-axis along
            with the joint distribution.
        :param plot_hist_for_all_particles: A Boolean, whether to plot the histogram only for particles that arrive at
            the boundary (False).
        :param use_independent_joint: A Boolean, whether to construct the joint distribution by multiplication of the
            marginal distributions (with mathematically is a simplification) or use the true two-dimensional histogram.
        """
        if not plot_marginals:
            fig, ax1 = plt.subplots()
            self._plot_joint_interval_distribution(ax1,
                                                   front_arrival_time_samples,
                                                   back_arrival_time_samples,
                                                   approaches_ls,
                                                   plot_hist_for_all_particles=plot_hist_for_all_particles,
                                                   use_independent_joint=use_independent_joint)

        else:
            # Create a Figure, which doesn't have to be square.
            fig = plt.figure()
            # Create the main axes, leaving 25% of the figure space at the top and on the
            # right to position marginals.
            gs = fig.add_gridspec(2, 2, top=0.75, right=0.75)
            ax1 = fig.add_subplot(gs[1, 0])
            # The main axes' aspect can be fixed.
            ax1.set(aspect=1)
            # Create marginal axes, which have 25% of the size of the main axes.  Note that
            # the inset axes are positioned *outside* (on the right and the top) of the
            # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
            # less than 0 would likewise specify positions on the left and the bottom of
            # the main axes.
            ax_histx = ax1.inset_axes([0, 1.05, 1, 0.25], sharex=ax1)
            ax_histy = ax1.inset_axes([1.05, 0, 0.25, 1], sharey=ax1)
            # ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            # ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            ax_histx.xaxis.set_ticklabels([])
            ax_histy.yaxis.set_ticklabels([])
            self._plot_joint_interval_distribution(ax1,
                                                   front_arrival_time_samples,
                                                   back_arrival_time_samples,
                                                   approaches_ls,
                                                   plot_hist_for_all_particles=plot_hist_for_all_particles,
                                                   use_independent_joint=use_independent_joint,
                                                   marginal_x_axis=ax_histx,
                                                   marginal_y_axis=ax_histy,
                                                   )
        ax1.set_xlabel('Front arrival time in ' + self.time_unit)
        ax1.set_ylabel('Back arrival time in ' + self.time_unit)

        if not self._for_paper:
            if not use_independent_joint:
                plt.title(
                    "Joint Distribution of First Particle Front and Back Arrival Time for " + self._process_name)
            else:
                plt.title(
                    "Joint Distribution (by Independence Assumption) of First Particle Front and Back Arrival Time for " + self._process_name)
        if self.save_results:
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_joint_first_arrival_interval_distr.pdf'))
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_joint_first_arrival_interval_distr.png'))
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_joint_first_arrival_interval_distr.pgf'))
        if not self.no_show:
            plt.show()

    @check_approaches_ls
    def plot_calibration(self,
                         approaches_ls,
                         front_arrival_time_samples,
                         back_arrival_time_samples,
                         ):
        """Plots the calibration of the deflection windows, i.e., the ratio of expects hits vs. the "true" hits in the
        corresponding deflection window.

        :param approaches_ls: A list of child instances of AbstractHittingTimeWithExtentsModel.
        :param front_arrival_time_samples: A np.array of shape [num_samples] containing samples of the front arrival
            times.
        :param back_arrival_time_samples: A np.array of shape [num_samples] containing samples of the back arrival
            times.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2)

        self._plot_calibration(axes,
                               front_arrival_time_samples,
                               back_arrival_time_samples,
                               approaches_ls,
                               )

        if not self._for_paper:
            fig.suptitle(
                "Calibration including Particle Extents for " + self._process_name)
            ax1, ax2 = axes
            ax1.set_title('Calibration w.r.t. Joint Distribution')
            ax2.set_title('Calibration w.r.t. Marginal Distributions')

        # plt.legend()
        if self.save_results:
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_calibration_with_extents.pdf'))
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_calibration_with_extents.png'))
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_calibration_with_extents.pgf'))
        if not self.no_show:
            plt.show()

    # to make them accessible from outside (and suppress ugly IDE warnings), needs to be done at the end of the class
    check_approaches_ls = staticmethod(check_approaches_ls)


class HittingLocationEvaluatorWithExtents(HittingLocationEvaluator, AbstractHittingEvaluatorWithExtents):
    """A class that handles the evaluations for hitting location models with a particle extent."""

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
                         y_predicted=y_predicted,
                         plot_y=plot_y,
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

    def check_approaches_ls(func):
        """A decorator for functions that process a list of hitting location with extents models.

        Assure that only distributions of the same type are in the list.

        :param func: A callable, the function to be decorated.

        :returns: A callable, the decorator.
        """
        @wraps(func)
        def check_approaches_in_approaches_ls_of_same_type(self, approaches_ls, *args, **kwargs):
            if not all([isinstance(a, HittingLocationWithExtentsModel) for a in approaches_ls]):
                raise ValueError(
                    'The distributions in approaches_ls must be all instances of HittingLocationWithExtentsModel.')

            return func(self, approaches_ls, *args, **kwargs)

        return check_approaches_in_approaches_ls_of_same_type

    @check_approaches_ls
    def plot_y_at_first_arrival_interval_distribution_on_y_axis(self,
                                                                approaches_ls,
                                                                min_y_samples,
                                                                max_y_samples,
                                                                plot_cdfs=True,
                                                                plot_back_and_front_distr=False,
                                                                ):
        """Plots the (actually two-dimensional) temporal deflection window on the time axis.

        :param approaches_ls: A list of instances of HittingLocationWithExtentsModel.
        :param min_y_samples: A np.array of shape [num_samples] containing samples of the particles' lowermost edge
            locations.
        :param max_y_samples: A np.array of shape [num_samples] containing samples of the particles' uppermost edge
            locations.
        :param plot_cdfs: A Boolean, whether to additionally plot the CDF on a new plot axis.
        :param plot_back_and_front_distr: A Boolean, whether to plot the marginal distributions for the front and back
            arrival locations.
        """
        fig, ax1 = plt.subplots()
        self._plot_interval_distributions_on_single_axis(ax1,
                                                         min_y_samples,
                                                         max_y_samples,
                                                         approaches_ls,
                                                         plot_hist_for_all_particles=False,  # always False
                                                         plot_cdfs=plot_cdfs,
                                                         plot_back_and_front_distr=plot_back_and_front_distr,
                                                         prefix_ls=[' (min of y)', ' (max of y)'])
        ax1.set_xlabel("Location in " + self.length_unit)

        if not self._for_paper:
            plt.title(
                "Distribution of the Maximum and Minimum of Y within the First Arrival Interval for " + self._process_name)
        # plt.legend()
        if self.save_results:
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_y_at_first_arrival_interval_distr.pdf'))
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_y_at_first_arrival_interval_distr.png'))
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_y_at_first_arrival_interval_distr.pgf'))
        if not self.no_show:
            plt.show()
        plt.close()

    @check_approaches_ls
    def plot_joint_y_at_first_arrival_interval_distribution(self,
                                                            approaches_ls,
                                                            min_y_samples,
                                                            max_y_samples,
                                                            plot_marginals=True,
                                                            use_independent_joint=False,
                                                            ):
        """Plots the two-dimensional joint distribution of the temporal deflection windows.

        :param approaches_ls: A list of instances of HittingLocationWithExtentsModel.
        :param min_y_samples: A np.array of shape [num_samples] containing samples of the particles' lowermost edge
            locations.
        :param max_y_samples: A np.array of shape [num_samples] containing samples of the particles' uppermost edge
            locations.
        :param plot_marginals: A Boolean, whether to plot the marginal distributions on separate x- and y-axis along
            with the joint distribution.
        :param use_independent_joint: A Boolean, whether to construct the joint distribution by multiplication of the
            marginal distributions (with mathematically is a simplification) or use the true two-dimensional histogram.
        """
        if not plot_marginals:
            fig, ax1 = plt.subplots()
            self._plot_joint_interval_distribution(ax1,
                                                   min_y_samples,
                                                   max_y_samples,
                                                   approaches_ls,
                                                   plot_hist_for_all_particles=True,  # always True
                                                   use_independent_joint=use_independent_joint,
                                                   )

        else:
            # Create a Figure, which doesn't have to be square.
            fig = plt.figure()
            # Create the main axes, leaving 25% of the figure space at the top and on the
            # right to position marginals.
            gs = fig.add_gridspec(2, 2, top=0.75, right=0.75)
            ax1 = fig.add_subplot(gs[1, 0])
            # The main axes' aspect can be fixed.
            ax1.set(aspect=1)
            # Create marginal axes, which have 25% of the size of the main axes.  Note that
            # the inset axes are positioned *outside* (on the right and the top) of the
            # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
            # less than 0 would likewise specify positions on the left and the bottom of
            # the main axes.
            ax_histx = ax1.inset_axes([0, 1.05, 1, 0.25], sharex=ax1)
            ax_histy = ax1.inset_axes([1.05, 0, 0.25, 1], sharey=ax1)
            # ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            # ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            ax_histx.xaxis.set_ticklabels([])
            ax_histy.yaxis.set_ticklabels([])
            self._plot_joint_interval_distribution(ax1,
                                                   min_y_samples,
                                                   max_y_samples,
                                                   approaches_ls,
                                                   plot_hist_for_all_particles=False,  # always False
                                                   use_independent_joint=use_independent_joint,
                                                   marginal_x_axis=ax_histx,
                                                   marginal_y_axis=ax_histy,
                                                   )
        ax1.set_xlabel('Min of y in ' + self.length_unit)
        ax1.set_ylabel('Max of y in ' + self.length_unit)

        if not self._for_paper:
            if not use_independent_joint:
                plt.title(
                    "Joint Distribution of the Minimum and Maximum of Y within the First Arrival Interval for " + self._process_name)
            else:
                plt.title(
                    "Joint Distribution (by Independence Assumption) of the Minimum and Maximum of Y within the First Arrival Interval for " + self._process_name)
        # plt.legend()
        if self.save_results:
            plt.savefig(
                os.path.join(self._result_dir,
                             self._process_name_save + '_joint_y_at_first_arrival_interval_distr.pdf'))
            plt.savefig(
                os.path.join(self._result_dir,
                             self._process_name_save + '_joint_y_at_first_arrival_interval_distr.png'))
            plt.savefig(
                os.path.join(self._result_dir,
                             self._process_name_save + '_joint_y_at_first_arrival_interval_distr.pgf'))
        if not self.no_show:
            plt.show()

    @check_approaches_ls
    def plot_calibration(self,
                         approaches_ls,
                         min_y_samples,
                         max_y_samples,
                         ):
        """Plots the calibration of the deflection windows, i.e., the ratio of expects hits vs. the "true" hits in the
        corresponding deflection window.

        :param approaches_ls: A list of instances of HittingLocationWithExtentsModel.
        :param min_y_samples: A np.array of shape [num_samples] containing samples of the particles' lowermost edge
            locations.
        :param max_y_samples: A np.array of shape [num_samples] containing samples of the particles' uppermost edge
            locations.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2)

        self._plot_calibration(axes,
                               min_y_samples,
                               max_y_samples,
                               approaches_ls,
                               )

        if not self._for_paper:
            fig.suptitle(
                "Calibration including Particle Extents for " + self._process_name)
            ax1, ax2 = axes
            ax1.set_title('Calibration w.r.t. Joint Distribution')
            ax2.set_title('Calibration w.r.t. Marginal Distributions')

        # plt.legend()
        if self.save_results:
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_calibration_with_extents.pdf'))
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_calibration_with_extents.png'))
            plt.savefig(
                os.path.join(self._result_dir, self._process_name_save + '_calibration_with_extents.pgf'))
        if not self.no_show:
            plt.show()

    # to make them accessible from outside (and suppress ugly IDE warnings), needs to be done at the end of the class
    check_approaches_ls = staticmethod(check_approaches_ls)

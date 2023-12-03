from absl import logging
from absl import app
from absl import flags

from timeit import time

import numpy as np
import tensorflow as tf


from cv_process import GaussTaylorCVHittingTimeDistribution, NoReturnCVHittingTimeDistribution
from sampler import get_example_tracks_lgssm as get_example_tracks
from cv_arrival_distributions.cv_utils import _get_system_matrices_from_parameters, create_ty_cv_samples_hitting_time
from cv_arrival_distributions.cv_hitting_location_distributions import MCCVHittingLocationDistribution
from extent_models import  HittingTimeWithExtentsModel, HittingTimeWithExtentsSimplifiedModel, HittingLocationWithExtentsModel

from evaluators.hitting_evaluator_with_extents import HittingTimeEvaluatorWithExtents, HittingLocationEvaluatorWithExtents


# Delete all FLAGS defined by CV process as we here not want them to be overwritten by the following flags.
for name in list(flags.FLAGS):
    if name in ['load_samples', 'save_samples', 'save_path', 'save_results', '_result_dir', 'no_show', '_for_paper',
                'measure_computational_times', 'verbosity_level']:
        delattr(flags.FLAGS, name)


flags.DEFINE_bool('load_samples', default=False,
                    help='Whether the samples should be loaded from a .npz  file.')
flags.DEFINE_bool('save_samples', default=False,
                    help='Whether the samples should be saved to a .npz  file.')
flags.DEFINE_string('save_path', default='/mnt/cv_with_system_noise_samples.npz',
                    help='The path to save the .npz  file.')
flags.DEFINE_bool('save_results', default=False,
                    help='Whether to save the results.')
flags.DEFINE_string('_result_dir', default='/mnt/results/',
                    help='The directory where to save the results.')
flags.DEFINE_bool('no_show', default=False,
                  help='Set this to True if you do not want to show evaluation graphics and only save them.')
flags.DEFINE_bool('_for_paper', default=False,
                  help='Boolean, whether to use the plots for publication (omit headers, etc.)..')
flags.DEFINE_bool('measure_computational_times', default=False,
                    help='Whether to measure the computational times.')

flags.DEFINE_string('verbosity_level', default='INFO', help='Verbosity options.')
flags.register_validator('verbosity_level',
                         lambda value: value in ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                         message='dataset_type must one of ' + str(['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']))

FLAGS = flags.FLAGS


def main(args):
    del args

    # Setup logging
    logging.set_verbosity(logging.FLAGS.verbosity_level)

    # Define system parameters
    # System noise
    S_w = 10
    # Covariance matrix at last timestep
    C_L = np.array([[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]])
    # Mean position at last timestep
    x_L = np.array([0.3, 6.2, 0.5, 0.0])
    # length and width of the particle
    particle_size = [0.08, 0.08]

    # Boundary position
    x_predTo = 0.6458623971412047
    # Last time step
    t_L = 0  # In principle, we could assume w.l.o.g. that _t_L = 0 (_t_L is just a location argument).

    # Run the experiment
    run_experiment(x_L, C_L, t_L, S_w, x_predTo,
                   particle_size=particle_size,
                   measure_computational_times=FLAGS.measure_computational_times,
                   load_samples=FLAGS.load_samples,
                   save_samples=FLAGS.save_samples,
                   save_path=FLAGS.save_path,
                   save_results=FLAGS.save_results,
                   result_dir=FLAGS._result_dir,
                   for_paper=FLAGS._for_paper,
                   no_show=FLAGS.no_show,
                   )


def run_experiment(x_L, C_L, t_L, S_w, x_predTo,
                   particle_size,
                   t_range_with_extents=None,
                   y_range_with_extents=None,
                   measure_computational_times=False,
                   load_samples=False,
                   save_samples=False,
                   save_path=None,
                   save_results=False,
                   result_dir=None,
                   no_show=False,
                   for_paper=False):
    """Runs an experiment including a comparison with Monte Carlo simulation with the given settings.

    The underlying process is a 2D (x, y) constant velocity (CV) model with independent components in x, y.
    Therefore, the state is [pos_x, velo_x, pos_y, velo_y].

    :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, velo_x, pos_y, velo_y].
    :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, the position of the boundary.
    :param length: TODO
    :param t_range_with_extents: A list of length 2 representing the plot limits for the first-passage time.
    :param y_range_with_extents: A list of length 2 representing the plot limits for the y component at the first
        passage time.
    :param measure_computational_times: A Boolean, whether to measure the computational times.
    :param load_samples: Boolean, whether to load the samples for the Monte Carlo simulation from file.
    :param save_samples: Boolean, whether to save the samples for the Monte Carlo simulation from file.
    :param save_path: String, path where to save the .npz file with the samples (suffix .npz).
    :param save_results: Boolean, whether to save the plots.
    :param result_dir: String, directory where to save the plots.
    :param no_show: Boolean, whether to show the plots (False).
    :param for_paper: Boolean, whether to use the plots for a publication (omit headers, etc.).
    """
    hitting_time_model_kwargs = {"x_L": x_L,
                                 "C_L": C_L,
                                 "S_w": S_w,
                                 "x_predTo": x_predTo,
                                 "_t_L": t_L}

    # Deterministic predictions
    t_predicted = t_L + (x_predTo - x_L[0]) / x_L[1]
    y_predicted = x_L[2] + (t_predicted - t_L) * x_L[3]

    # Plot settings
    if t_range_with_extents is None:
        t_range_with_extents = [t_predicted - 0.3 * (t_predicted - t_L) - 1 / 2 * particle_size[0] / x_L[1],
                                t_predicted + 0.3 * (t_predicted - t_L) + particle_size[0] / x_L[1]]
    if y_range_with_extents is None:
        y_range_with_extents = [0.7 * y_predicted - 1 / 2 * particle_size[1],
                                1.3 * y_predicted + 1 / 2 * particle_size[1]]
    plot_t = np.arange(t_range_with_extents[0], t_range_with_extents[1], 0.00001)
    plot_y = np.arange(y_range_with_extents[0], y_range_with_extents[1], 0.001)

    # Create base class
    hte = HittingTimeEvaluatorWithExtents('CV Process', x_predTo, plot_t, t_predicted, t_L,
                                          get_example_tracks_fn=get_example_tracks(x_L,
                                                                                   C_L,
                                                                                   S_w,
                                                                                   _get_system_matrices_from_parameters),
                                          # TODO: Nicht mehr privat
                                          save_results=save_results,
                                          result_dir=result_dir,
                                          no_show=no_show,
                                          for_paper=for_paper)

    # Create samples
    dt = 1 / 1000
    first_passage_statistics, first_arrival_interval_statistics = create_ty_cv_samples_hitting_time(
        x_L=x_L,
        C_L=C_L,
        S_w=S_w,
        x_predTo=x_predTo,
        t_L=t_L,
        length=particle_size[0],
        dt=dt,
        N=100000)
    _, y_samples, _ = first_passage_statistics
    t_samples_first_front_arrival, t_samples_first_back_arrival, y_min_samples, y_max_samples, y_samples_first_front_arrival, y_samples_first_back_arrival = first_arrival_interval_statistics

    # Set up the hitting time approaches
    taylor_model_with_extents = HittingTimeWithExtentsModel(particle_size[0], GaussTaylorCVHittingTimeDistribution,
                                                            hitting_time_model_kwargs,
                                                            name="Gauß-Taylor with extent")
    approx_model_with_extents = HittingTimeWithExtentsModel(particle_size[0], NoReturnCVHittingTimeDistribution,
                                                            hitting_time_model_kwargs,
                                                            name="No-return approx. with extent")
    simplified_taylor_model_with_extents = HittingTimeWithExtentsSimplifiedModel(particle_size[0], GaussTaylorCVHittingTimeDistribution,
                                                                                 hitting_time_model_kwargs,
                                                                                 name="Gauß-Taylor with extent (simplified)")
    simplified_approx_model_with_extents = HittingTimeWithExtentsSimplifiedModel(particle_size[0],
                                                                                 NoReturnCVHittingTimeDistribution,
                                                                                 hitting_time_model_kwargs,
                                                                                 name="No-return approx. with extent (simplified)")
    approaches_temp_ls = [taylor_model_with_extents,
        approx_model_with_extents,
        simplified_taylor_model_with_extents,
        simplified_approx_model_with_extents,
    ]

    # plot the distribution of the particle front and back arrival time at one axis (the time axis)
    hte.plot_first_arrival_interval_distribution_on_time_axis(t_samples_first_front_arrival,
                                                              t_samples_first_back_arrival,
                                                              approaches_temp_ls,
                                                              plot_hist_for_all_particles=True,
                                                              )

    # # plot the joint distribution of the particle front and back arrival time (2-dimensional distribution, heatmap)
    # hte.plot_joint_first_arrival_interval_distribution(t_samples_first_front_arrival,
    #                                                    t_samples_first_back_arrival,
    #                                                    approaches_temp_ls,
    #                                                    plot_hist_for_all_particles=True,
    #                                                    plot_marginals=False,
    #                                                    )

    # # plot a simplifies joint distribution (based on the marginals and independence assumption) of the particle front
    # # and back arrival time (2-dimensional distribution, heatmap)
    # hte.plot_joint_first_arrival_interval_distribution(t_samples_first_front_arrival,
    #                                                    t_samples_first_back_arrival,
    #                                                    approaches_temp_ls,
    #                                                    plot_hist_for_all_particles=True,
    #                                                    use_independent_joint=True,
    #                                                    )

    # plot the calibration
    # hte.plot_calibration(t_samples_first_front_arrival,
    #                      t_samples_first_back_arrival,
    #                      approaches_temp_ls,
    #                      )

    # Set up the hitting location approaches
    hitting_location_model_kwargs = {'S_w': hitting_time_model_kwargs['S_w']}
    # taylor_model_with_extents = HittingLocationWithExtentsModel(*particle_size,
    #                                                             TaylorCVHittingTimeModel,
    #                                                             TaylorCVHittingLocationModel,
    #                                                             htd_kwargs,
    #                                                             hld_kwargs,
    #                                                             name="Gauß-Taylor with extent",
    #                                                             )

    hitting_location_model_kwargs['y_range'] = y_range_with_extents
    y_samples_for_mc = y_samples.copy()
    y_samples_for_mc = y_samples_for_mc[np.isfinite(
        y_samples_for_mc)]  # TODO: Das mit der _remove_not_arriving_samples Funktion verbinden, wo das am besten hinmachen? in die AbstractMCHitting*Model jeweils getrennt für Location (immer) und für time (nach wahl)
    hitting_location_model_kwargs['y_samples'] = y_samples_for_mc
    taylor_model_with_extents = HittingLocationWithExtentsModel(*particle_size,
                                                                GaussTaylorCVHittingTimeDistribution,
                                                                MCCVHittingLocationDistribution,
                                                                hitting_time_model_kwargs,
                                                                hitting_location_model_kwargs,
                                                                name="Gauß-Taylor with extent",
                                                                )



    # Results for spatial uncertainties
    hte_spatial = HittingLocationEvaluatorWithExtents('CV Process', x_predTo, t_predicted, y_predicted, plot_y, t_L,
                                                      get_example_tracks_fn=get_example_tracks(x_L,
                                                                                               C_L,
                                                                                               S_w,
                                                                                               _get_system_matrices_from_parameters),
                                                      save_results=save_results,
                                                      result_dir=result_dir,
                                                      no_show=no_show,
                                                      for_paper=for_paper)

    approaches_spatial_ls = [taylor_model_with_extents]

    # plot the distribution of the particle front and back arrival time at one axis (the time axis)
    hte_spatial.plot_y_at_first_arrival_interval_distribution_on_y_axis(
        # min_y_samples=y_min_samples - particle_size[1] / 2,
        # may_y_samples=y_max_samples + particle_size[1] / 2,
        min_y_samples=np.min(np.vstack([y_samples_first_front_arrival, y_samples_first_back_arrival]), axis=0) - particle_size[1] / 2,
        max_y_samples=np.max(np.vstack([y_samples_first_front_arrival, y_samples_first_back_arrival]), axis=0) + particle_size[1] / 2,
        approaches_ls=approaches_spatial_ls,
        plot_hist_for_all_particles=False,
        # TODO: Das ist auch bisschen unschön, muss man ansstellen für das sehr unsichere Partikel, sonst geht der distribute_bins code nicht
    )

    # # plot the joint distribution of the particle front and back arrival time (2-dimensional distribution, heatmap)
    # hte_spatial.plot_joint_y_at_first_arrival_interval_distribution(y_min_samples - particle_size[1] / 2,
    #                                                                 y_max_samples + particle_size[1] / 2,
    #                                                                 approaches_spatial_ls,
    #                                                                 plot_hist_for_all_particles=True,
    #                                                                 plot_marginals=False,
    #                                                                 )
    # plot the calibration
    hte_spatial.plot_calibration(y_min_samples - particle_size[1] / 2,
                                 y_max_samples + particle_size[1] / 2,
                                 approaches_spatial_ls,
                                 )


if __name__ == "__main__":
    app.run(main)

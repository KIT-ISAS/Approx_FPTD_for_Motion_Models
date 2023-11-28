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

from evaluators.hitting_time_evaluator_with_extents import HittingTimeEvaluatorWithExtents, HittingLocationEvaluatorWithExtents


# Delete all FLAGS defined by CV process as we here not want them to be overwritten by the following flags.
for name in list(flags.FLAGS):
    if name in ['load_samples', 'save_samples', 'save_path', 'save_results', 'result_dir', 'no_show', 'for_paper',
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
flags.DEFINE_string('result_dir', default='/mnt/results/',
                    help='The directory where to save the results.')
flags.DEFINE_bool('no_show', default=False,
                  help='Set this to True if you do not want to show evaluation graphics and only save them.')
flags.DEFINE_bool('for_paper', default=False,
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
    t_L = 0  # In principle, we could assume w.l.o.g. that t_L = 0 (t_L is just a location argument).

    # Run the experiment
    run_experiment(x_L, C_L, t_L, S_w, x_predTo,
                   particle_size=particle_size,
                   measure_computational_times=FLAGS.measure_computational_times,
                   load_samples=FLAGS.load_samples,
                   save_samples=FLAGS.save_samples,
                   save_path=FLAGS.save_path,
                   save_results=FLAGS.save_results,
                   result_dir=FLAGS.result_dir,
                   for_paper=FLAGS.for_paper,
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
    :param x_predTo: A float, position of the boundary.
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
                                 "t_L": t_L}

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
    #                                                             hitting_time_model_kwargs,
    #                                                             hitting_location_model_kwargs,
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



def create_hitting_time_samples(initial_samples,
                                compute_x_next_func,
                                x_predTo,
                                length,
                                t_L=0.0,
                                N=100000,
                                dt=1 / 1000,
                                break_after_n_time_steps=1000,
                                break_min_time=None):  # TODO: Mit anderer Funktion verbinden bzw. die andere sampling function ersetzen
    """Monte Carlo approach to solve the first-passage time problem. Propagates particles through the motion model and
    determines time before arrival, and positions before and after the arrival as well as more statistics.

    :param initial_samples: A np.array of shape [num_samples, state_size], the initial particles samples at time step
        t_L.
    :param compute_x_next_func: A function propagating the samples to the next time step, i.e., the one time step
        transition function.
    :param x_predTo: A float, position of the boundary.
    :param length: TODO
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_after_n_time_steps: Integer, maximum number of time steps for the simulation.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least.
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        t_samples: A np.array of shape [N] containing the first-passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first-passage times of the particles.
        fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
        TODO
    """
    start_time = time.time()
    # Let the samples move to the boundary

    t = t_L
    ind = 0
    x_curr = initial_samples

    fraction_of_returns = []

    x_term = np.zeros(initial_samples.shape[0], dtype=bool)
    x_term_first_front_arrival = np.zeros(initial_samples.shape[0], dtype=bool)
    x_term_first_back_not_in = np.zeros(initial_samples.shape[0], dtype=bool)

    time_before_arrival = np.full(N, t_L, dtype=np.float64)
    time_before_first_front_arrival = np.full(N, t_L, dtype=np.float64)
    time_before_first_back_not_in = np.full(N, t_L, dtype=np.float64)

    x_before_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_before_arrival[:] = np.nan
    x_before_front_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_before_front_arrival[:] = np.nan
    x_before_first_back_not_in = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_before_first_back_not_in[:] = np.nan

    x_after_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_after_arrival[:] = np.nan
    x_after_front_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_after_front_arrival[:] = np.nan
    x_after_first_back_not_in = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_after_first_back_not_in[:] = np.nan

    y_min = np.empty((initial_samples.shape[0]))
    y_min[:] = np.inf
    y_max = np.empty((initial_samples.shape[0]))
    y_max[:] = - np.inf
    while True:
        if ind % 100 == 0:
            logging.info('Timestep {0}, x equals approx. {1}'.format(ind, x_curr[0, 0]))
        fraction_of_returns.append(np.sum(np.logical_and(x_curr[:, 0] < x_predTo, x_term)) / N)

        x_next = compute_x_next_func(x_curr)

        first_passage = np.logical_and(np.logical_not(x_term), x_next[:, 0] >= x_predTo)
        x_term[first_passage] = True
        x_before_arrival[first_passage] = x_curr[first_passage]
        x_after_arrival[first_passage] = x_next[first_passage]

        first_front_arrival = np.logical_and(np.logical_not(x_term_first_front_arrival), x_next[:, 0] + length/2 >= x_predTo)
        first_back_not_in = np.logical_and(np.logical_not(x_term_first_back_not_in), x_next[:, 0] - length/2 > x_predTo)
        x_term_first_front_arrival[first_front_arrival] = True
        x_term_first_back_not_in[first_back_not_in] = True
        
        x_before_front_arrival[first_front_arrival] = x_curr[first_front_arrival]
        x_after_front_arrival[first_front_arrival] = x_next[first_front_arrival]
        x_before_first_back_not_in[first_back_not_in] = x_curr[first_back_not_in]
        x_after_first_back_not_in[first_back_not_in] = x_next[first_back_not_in]

        particle_passes_mask = np.logical_and(x_term_first_front_arrival, np.logical_not(x_term_first_back_not_in))
        y_min[np.logical_and(particle_passes_mask, x_next[:, 2] < y_min)] = x_next[
            np.logical_and(particle_passes_mask, x_next[:, 2] < y_min), 2]
        y_max[np.logical_and(particle_passes_mask, x_next[:, 2] > y_max)] = x_next[
            np.logical_and(particle_passes_mask, x_next[:, 2] > y_max), 2]

        if break_min_time is None:
            if np.all(x_term_first_front_arrival) and np.all(x_term_first_back_not_in):
                break
        else:
            if t >= break_min_time and np.all(x_term_first_front_arrival) and np.all(x_term_first_back_not_in):
                break
        if break_after_n_time_steps is not None and ind >= break_after_n_time_steps:
            logging.info(
                'Warning: Sampling interrupted because {}. reached. Please adjust break_after_n_time_steps if you want to move the particles more timesteps.'.format(
                    break_after_n_time_steps))
            break
        x_curr = x_next
        t += dt
        time_before_first_front_arrival[np.logical_not(x_term_first_front_arrival)] = t
        time_before_first_back_not_in[np.logical_not(x_term_first_back_not_in)] = t
        ind += 1

    logging.info('MC time: {0}ms'.format(round(1000 * (time.time() - start_time))))

    first_passage_statistics = (
        time_before_arrival, x_before_arrival, x_after_arrival, x_term, np.array(fraction_of_returns))
    first_arrival_interval_statistics = (
        time_before_first_front_arrival, time_before_first_back_not_in, x_before_front_arrival, x_after_front_arrival,
        x_before_first_back_not_in, x_after_first_back_not_in, x_term_first_front_arrival, x_term_first_back_not_in,
        y_min,
        y_max)

    return first_passage_statistics, first_arrival_interval_statistics


def create_lgssm_hitting_time_samples(F,
                                      Q,
                                      x_L,
                                      C_L,
                                      x_predTo,
                                      length,
                                      t_L=0.0,
                                      N=100000,
                                      dt=1 / 1000,
                                      break_after_n_time_steps=1000,
                                      break_min_time=None):
    """Monte Carlo approach to solve the first-passage time problem. Propagates particles through the discrete-time
    LGSSM motion model and determines their first-passage at x_predTo as well as the location in y at the first-passage 
    by interpolating the positions between the last time before and the first time after the boundary.

    :param F: A np.array of shape [4, 4], the transition matrix of the LGSSM.
    :param Q: A np.array of shape [4, 4], the transition noise covariance matrix of the LGSSM.
    :param x_L: A np.array of shape [length_state] representing the expected value of the initial state. We use index L 
        here because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, ..., pos_y, ...].
    :param C_L: A np.array of shape [length_state, length_state] representing the covariance matrix of the initial state.
    :param x_predTo: A float, position of the boundary.
    :param length: TODO
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least.
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        t_samples: A np.array of shape [N] containing the first-passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first-passage times of the particles.
        fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
        TODO
    """
    initial_samples = np.random.multivariate_normal(mean=x_L, cov=C_L, size=N)

    mean_w = np.zeros(x_L.shape[0])

    def compute_x_next_func(x_curr):
        x_curr_tf = tf.convert_to_tensor(x_curr)
        x_next = tf.linalg.matvec(F, x_curr_tf).numpy()
        w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=N)
        x_next = x_next + w_k
        return x_next

    return create_hitting_time_samples(
        initial_samples,
        compute_x_next_func,
        x_predTo,
        length,
        t_L=t_L,
        N=N,
        dt=dt,
        break_after_n_time_steps=break_after_n_time_steps,
        break_min_time=break_min_time)


def create_ty_cv_samples_hitting_time(x_L,
                                      C_L,
                                      S_w,
                                      x_predTo,
                                      length,
                                      t_L=0.0,
                                      N=100000,
                                      dt=1 / 1000,
                                      break_after_n_time_steps=1000,
                                      break_min_time=None):
    """Monte Carlo approach to solve the first-passage time problem. Propagates particles through the 2D discrete-time
    CV motion model and determines their first-passage at x_predTo as well as the location in y at the first-passage by
    interpolating the positions between the last time before and the first time after the boundary.

    Note that particles that do not reach the boundary after break_after_n_time_steps time_steps are handled with a
    fallback value of max(t_samples) + 1 in the t_samples and np.nan in the y_samples.

    :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, velo_x, pos_y, velo_y].
    :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, position of the boundary.
    :param length: TODO
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_after_n_time_steps: Integer, maximum number of time steps for the simulation.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        t_samples: A np.array of shape [N] containing the first-passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first-passage times of the particles.
        fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
        TODO
    """
    F, Q = _get_system_matrices_from_parameters(dt, S_w)

    first_passage_statistics, first_arrival_interval_statistics = create_lgssm_hitting_time_samples(
        F,
        Q,
        x_L,
        C_L,
        x_predTo,
        length,
        t_L=t_L,
        N=N,
        dt=dt,
        break_after_n_time_steps=break_after_n_time_steps,
        break_min_time=break_min_time)

    # first for the first-passage statistics
    time_before_arrival, x_before_arrival, x_after_arrival, x_term, fraction_of_returns = first_passage_statistics

    # Linear interpolation to get time
    v_interpolated = (x_after_arrival[x_term, 1] - x_before_arrival[x_term, 1]) / (
            x_after_arrival[x_term, 0] - x_before_arrival[x_term, 0]) * (x_predTo - x_before_arrival[x_term, 0]) + \
                     x_before_arrival[x_term, 1]
    last_t = (x_predTo - x_before_arrival[x_term, 0]) / v_interpolated
    t_samples = time_before_arrival
    t_samples[x_term] = time_before_arrival[x_term] + last_t
    t_samples[np.logical_not(x_term)] = int(
        max(t_samples)) + 1  # default value for particles that do not arrive

    y_samples = x_before_arrival[:, 2]
    y_samples[x_term] = x_before_arrival[x_term, 2] + last_t * x_before_arrival[x_term, 3]
    y_samples[np.logical_not(x_term)] = np.nan  # default value for particles that do not arrive

    first_passage_statistics = t_samples, y_samples, fraction_of_returns

    # then for the first-interval statistics
    time_before_first_front_arrival, time_before_first_back_not_in, x_before_front_arrival, x_after_front_arrival, x_before_first_back_not_in, x_after_first_back_not_in, x_term_first_front_arrival, x_term_first_back_not_in, y_min_samples, y_max_samples = first_arrival_interval_statistics

    # Linear interpolation to get time
    v_interpolated_arrival = (x_after_front_arrival[x_term_first_front_arrival, 1] - x_before_front_arrival[
        x_term_first_front_arrival, 1]) / (
                                     x_after_front_arrival[x_term_first_front_arrival, 0] - x_before_front_arrival[
                                 x_term_first_front_arrival, 0]) * (
                                     x_predTo - x_before_front_arrival[x_term_first_front_arrival, 0]) + \
                             x_before_front_arrival[x_term_first_front_arrival, 1]
    delta_t_first_arrival = ((x_predTo - length / 2) - x_before_front_arrival[
        x_term_first_front_arrival, 0]) / v_interpolated_arrival
    t_samples_first_front_arrival = time_before_first_front_arrival  # TODO
    t_samples_first_front_arrival[x_term_first_front_arrival] = time_before_first_front_arrival[
                                                                    x_term_first_front_arrival] + delta_t_first_arrival
    t_samples_first_front_arrival[np.logical_not(x_term_first_front_arrival)] = int(
        max(t_samples_first_front_arrival)) + 1  # default value for particles that do not arrive

    v_interpolated_last_in = (x_after_first_back_not_in[x_term_first_back_not_in, 1] - x_before_first_back_not_in[
        x_term_first_back_not_in, 1]) / (
                                     x_after_first_back_not_in[x_term_first_back_not_in, 0] -
                                     x_before_first_back_not_in[x_term_first_back_not_in, 0]) * (
                                     (x_predTo + length / 2) - x_before_first_back_not_in[
                                 x_term_first_back_not_in, 0]) + \
                             x_before_first_back_not_in[x_term_first_back_not_in, 1]
    delta_t_last_in = ((x_predTo + length / 2) - x_before_first_back_not_in[
        x_term_first_back_not_in, 0]) / v_interpolated_last_in
    t_samples_first_back_arrival = time_before_first_back_not_in  # TODO
    t_samples_first_back_arrival[x_term_first_back_not_in] = time_before_first_back_not_in[
                                                                 x_term_first_back_not_in] + delta_t_last_in
    t_samples_first_back_arrival[np.logical_not(x_term_first_back_not_in)] = int(
        max(t_samples_first_back_arrival)) + 1  # default value for particles that do not arrive

    y_samples_first_front_arrival = x_before_front_arrival[:, 2]
    y_samples_first_front_arrival[x_term_first_front_arrival] = x_before_front_arrival[
                                                                    x_term_first_front_arrival, 2] + delta_t_first_arrival * \
                                                                x_before_front_arrival[x_term_first_front_arrival, 3]

    y_min_samples[np.logical_not(x_term_first_front_arrival)] = np.nan # default value for particles that do not arrive
    y_max_samples[np.logical_not(x_term_first_front_arrival)] = np.nan

    y_samples_first_front_arrival[
        np.logical_not(x_term_first_front_arrival)] = np.nan  # default value for particles that do not arrive

    y_samples_first_back_arrival = x_before_first_back_not_in[:, 2]
    y_samples_first_back_arrival[x_term_first_back_not_in] = x_before_first_back_not_in[
                                                                 x_term_first_back_not_in, 2] + delta_t_last_in * \
                                                             x_before_first_back_not_in[x_term_first_back_not_in, 3]
    y_samples_first_back_arrival[
        np.logical_not(x_term_first_back_not_in)] = np.nan  # default value for particles that do not arrive

    y_min_samples[y_samples_first_back_arrival < y_min_samples] = y_samples_first_back_arrival[
        y_samples_first_back_arrival < y_min_samples]
    y_max_samples[y_samples_first_back_arrival > y_min_samples] = y_samples_first_back_arrival[
        y_samples_first_back_arrival > y_min_samples]

    y_min_samples[y_samples_first_front_arrival < y_min_samples] = y_samples_first_front_arrival[
        y_samples_first_front_arrival < y_min_samples]
    y_max_samples[y_samples_first_front_arrival > y_min_samples] = y_samples_first_front_arrival[
        y_samples_first_front_arrival > y_min_samples]

    first_arrival_interval_statistics = (
        t_samples_first_front_arrival, t_samples_first_back_arrival, y_min_samples, y_max_samples,
        y_samples_first_front_arrival, y_samples_first_back_arrival)

    return first_passage_statistics, first_arrival_interval_statistics


if __name__ == "__main__":
    app.run(main)

"""
############################################ ca_process_main.py  ###########################################
Authors: Marcel Reith-Braun (ISAS, marcel.reith-braun@kit.edu), Jakob Thumm
#######################################################################################################
Calculates approximate first-passage time distributions for a constant acceleration model using different
approaches.

usage:
 - run docker container - tested with tensorflow/approx_fptd:2.8.0-gpu image:
    $ docker run -u $(id -u):$(id -g) \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
            tensorflow/approx_fptd:2.8.0-gpu
 - within container:
     $   python3 /mnt/ca_process_main.py \\
requirements:
  - Required tensorflow/approx_fptd:2.8.0-gpu image: See corresponding dockerfile.
  - Volume mounts: Specify a path </path/to/repo/> that points to the repo.
"""

from absl import logging
from absl import app
from absl import flags

import numpy as np

from ca_arrival_distributions.ca_hitting_time_distributions import NoReturnCAHittingTimeDistribution, \
    GaussTaylorCAHittingTimeDistribution, UniformCAHittingTimeDistribution, MCCAHittingTimeDistribution
from ca_arrival_distributions.ca_hitting_location_distributions import GaussTaylorCAHittingLocationDistribution, \
    SimpleGaussCAHittingLocationDistribution, UniformCAHittingLocationDistribution, \
    BayesMixtureCAHittingLocationDistribution, BayesianCAHittingLocationDistribution, MCCAHittingLocationDistribution
from extent_models import HittingTimeWithExtentsModel, HittingTimeWithExtentsSimplifiedModel, \
    HittingLocationWithExtentsModel
from evaluators.hitting_time_evaluator import HittingTimeEvaluator
from evaluators.hitting_location_evaluator import HittingLocationEvaluator
from evaluators.hitting_evaluator_with_extents import HittingTimeEvaluatorWithExtents, \
    HittingLocationEvaluatorWithExtents
from ca_arrival_distributions.ca_utils import get_system_matrices_from_parameters, create_ty_ca_samples_hitting_time
from sampler import get_example_tracks_lgssm as get_example_tracks
from timer import measure_computation_times


flags.DEFINE_bool('load_samples', default=False,
                    help='Whether the samples should be loaded from a .npz  file.')
flags.DEFINE_bool('save_samples', default=False,
                    help='Whether the samples should be saved to a .npz  file.')
flags.DEFINE_string('save_path', default='/mnt/ca_with_system_noise_samples.npz',
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
                  help='Whether to measure the computational times. This is only considered if with_extends is False.')
flags.DEFINE_bool('with_extents', default=False,
                  help='Whether to run experiments based on a point-based (False) or extent-based representation of a particle (True).')

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
    S_w = 1000
    # Covariance matrix at last timestep
    C_L = np.array([[2E-7, 2E-5, 8E-4, 0, 0, 0], [2E-5, 3E-3, 1.5E-1, 0, 0, 0], [8E-4, 1.5E-1, 1.3E1, 0, 0, 0],
                     [0, 0, 0, 2E-7, 2E-5, 8E-4], [0, 0, 0, 2E-5, 3E-3, 1.5E-1], [0, 0, 0, 8E-4, 1.5E-1, 1.3E1]])
    # Mean position at last timestep
    x_L = np.array([0.3, 6.2, 4.4, 0.5, 0.2, 2.8])
    # length and width of the particle
    particle_size = [0.08, 0.08]

    # Boundary position
    x_predTo = 0.6458623971412047
    # Last time step
    t_L = 0

    # Run the experiment
    if not FLAGS.with_extents:
        run_experiment(x_L, C_L, t_L, S_w, x_predTo,
                       measure_computational_times=FLAGS.measure_computational_times,
                       load_samples=FLAGS.load_samples,
                       save_samples=FLAGS.save_samples,
                       save_path=FLAGS.save_path,
                       save_results=FLAGS.save_results,
                       result_dir=FLAGS.result_dir,
                       for_paper=FLAGS.for_paper,
                       no_show=FLAGS.no_show,
                       )
    else:
        run_experiment_with_extent(x_L, C_L, t_L, S_w, x_predTo,
                                   particle_size=particle_size,
                                   load_samples=FLAGS.load_samples,
                                   save_samples=FLAGS.save_samples,
                                   save_path=FLAGS.save_path,
                                   save_results=FLAGS.save_results,
                                   result_dir=FLAGS.result_dir,
                                   for_paper=FLAGS.for_paper,
                                   no_show=FLAGS.no_show,
                                   )


def run_experiment(x_L, C_L, t_L, S_w, x_predTo,
                   t_range=None,
                   y_range=None,
                   measure_computational_times=False,
                   load_samples=False,
                   save_samples=False,
                   save_path=None,
                   save_results=False,
                   result_dir=None,
                   no_show=False,
                   for_paper=False):
    """Runs an experiment including a comparison with Monte Carlo simulation with the given settings.

    The underlying process is a 2D (x, y) constant acceleration (CA) model with independent components in x, y.
    Therefore, the state is [pos_x, velo_x, acc_x, pos_y, velo_y, acc_y].

    :param x_L: A np.array of shape [6] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, velo_x, acc_x, pos_y, velo_y, acc_y].
    :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, the position of the boundary.
    :param t_range: A list of length 2 representing the plot limits for the first-passage time.
    :param y_range: A list of length 2 representing the plot limits for the y component at the first-passage time.
    :param measure_computational_times: A Boolean, whether to measure the computational times.
    :param load_samples: Boolean, whether to load the samples for the Monte Carlo simulation from file.
    :param save_samples: Boolean, whether to save the samples for the Monte Carlo simulation from file.
    :param save_path: String, path where to save the .npz file with the samples (suffix .npz).
    :param save_results: Boolean, whether to save the plots.
    :param result_dir: String, directory where to save the plots.
    :param no_show: Boolean, whether to show the plots (False).
    :param for_paper: Boolean, whether to use the plots for a publication (omit headers, etc.).
    """
    # Deterministic predictions
    ca_temporal_point_predictor = lambda pos_l, v_l, a_l, x_predTo: - v_l[..., 0] / a_l[..., 0] + np.sign(a_l[..., 0]) * \
                                                                    np.sqrt((v_l[..., 0] / a_l[..., 0]) ** 2 + 2 / a_l[
                                                                        ..., 0] * (
                                                                                    x_predTo - pos_l[..., 0]))  # TODO: Stimmt das eigentlich?, die formel nutzen wir auch für wnca
    ca_spatial_point_predictor = lambda pos_l, v_l, a_l, dt_pred: dt_pred * v_l[..., 1] + 1 / 2 * dt_pred ** 2 * a_l[
        ..., 1]
    # t_predicted = t_L - x_L[1] / x_L[2] + np.sign(x_L[2]) * \
    #               np.sqrt((x_L[1] / x_L[2]) ** 2 + 2 / x_L[2] * (x_predTo - x_L[0]))
    # y_predicted = x_L[3] + (t_predicted - t_L) * x_L[4] + 1 / 2 * (t_predicted - t_L) ** 2 * x_L[5]
    t_predicted = t_L + ca_temporal_point_predictor(x_L[[0, 3]], x_L[[1, 4]], x_L[[2, 5]], x_predTo)
    y_predicted = x_L[3] + ca_spatial_point_predictor(x_L[[0, 3]], x_L[[1, 4]], x_L[[2, 5]], dt_pred=t_predicted - t_L)

    # Plot settings
    if t_range is None:
        t_range = [t_predicted - 0.2 * (t_predicted - t_L), t_predicted + 0.2 * (t_predicted - t_L)]
    if y_range is None:
        y_range = [0.9 * y_predicted, 1.1 * y_predicted]
    plot_t = np.arange(t_range[0], t_range[1], 0.00001)
    plot_y = np.arange(y_range[0], y_range[1], 0.001)

    # Create samples
    dt = 1 / 1000
    if not load_samples:
        first_passage_statistics, _ = create_ty_ca_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L, dt=dt)
        t_samples, y_samples, fraction_of_returns = first_passage_statistics

        if save_samples:
            np.savez(save_path, name1=t_samples, name2=y_samples)
            logging.info("Saved samples.")
    else:
        data = np.load(save_path)
        t_samples = data['name1']
        y_samples = data['name2']
        fraction_of_returns = data['name3']

    logging.info('Evaluations for hitting time distributions.')

    # Create class for evaluations
    hte = HittingTimeEvaluator('CA Process', x_predTo, plot_t, t_predicted, t_L,
                               get_example_tracks_fn=get_example_tracks(x_L,
                                                                        C_L,
                                                                        S_w,
                                                                        get_system_matrices_from_parameters),
                               save_results=save_results,
                               result_dir=result_dir,
                               no_show=no_show,
                               for_paper=for_paper)

    # Show example histogram
    # hte.plot_sample_histogram(t_samples)

    # Show example tracks and visualize uncertainties over time
    # hte.plot_example_tracks(N=5)
    ev_fn = lambda t: x_L[0] + x_L[1] * (t - t_L) + x_L[2] / 2 * (t - t_L) ** 2
    var_fn = lambda t: C_L[0, 0] + (C_L[0, 1] + C_L[1, 0]) * (t - t_L) \
                       + (1 / 2 * C_L[0, 2] + 1 / 2 * C_L[2, 0] + C_L[1, 1]) * (t - t_L) ** 2 \
                       + (1 / 2 * C_L[1, 2] + 1 / 2 * C_L[2, 1]) * (t - t_L) ** 3 \
                       + 1 / 4 * C_L[2, 2] * (t - t_L) ** 4 + 1 / 20 * S_w * (t - t_L) ** 5
    hte.plot_mean_and_stddev_over_time(ev_fn, var_fn, show_example_tracks=True)

    # Set up the hitting time approaches
    gauss_taylor_htd = GaussTaylorCAHittingTimeDistribution(x_L, C_L, S_w, x_predTo, t_L,
                                                            point_predictor=ca_temporal_point_predictor)
    no_return_htd = NoReturnCAHittingTimeDistribution(x_L, C_L, S_w, x_predTo, t_L)
    uniform_htd = UniformCAHittingTimeDistribution(x_L, x_predTo, t_L,
                                                   point_predictor=ca_temporal_point_predictor,
                                                   window_length=0.08 / x_L[1],  # length / x-velocity
                                                   )
    mc_htd = MCCAHittingTimeDistribution(x_L, C_L, S_w, x_predTo, t_L, t_range,
                                         t_samples=t_samples)

    # Results for temporal uncertainties
    approaches_temp_ls = [gauss_taylor_htd, no_return_htd, uniform_htd, mc_htd]

    logging.info('MAX CDF: {} at {}'.format(no_return_htd.q_max, no_return_htd.t_max))
    no_return_htd.plot_valid_regions(theta=t_predicted, save_results=save_results, result_dir=result_dir,
                                     for_paper=True,
                                     no_show=no_show)
    # approx_model.plot_valid_regions(save_results=save_results, result_dir=_result_dir, for_paper=True, no_show_no_show)
    logging.info('tau_max: {}'.format(no_return_htd.trans_dens_ppf(t_predicted)[0]))
    logging.info('Mass inside invalid region: {}'.format(
        1 - no_return_htd.cdf(t_predicted + no_return_htd.trans_dens_ppf(t_predicted)[0])))
    logging.info('Approximate returning probs after a crossing until time t_max: {}'.format(
        no_return_htd.get_statistics()['ReturningProbs'](no_return_htd.t_max)))

    # Plot the quantile functions
    hte.plot_quantile_functions(approaches_temp_ls)
    # Calculate moments and compare the results
    hte.compare_moments(approaches_temp_ls)
    # Calculate the skewness and compare the results
    hte.compare_skewness(approaches_temp_ls)

    # Calculate wasserstein distance and compare results
    hte.compare_wasserstein_distances(approaches_temp_ls, t_samples)
    # Calculate the Hellinger distance
    hte.compare_hellinger_distances(approaches_temp_ls, t_samples)
    # Calculate the first wasserstein distance
    hte.compare_first_wasserstein_distances(approaches_temp_ls, t_samples)
    # Calculate the kolmogorov distance
    hte.compare_kolmogorov_distances(approaches_temp_ls, t_samples)

    # Plot histogram of samples and hitting time distributions
    hte.plot_first_hitting_time_distributions(approaches_temp_ls, t_samples, plot_hist_for_all_particles=True)
    hte.plot_fptd_and_paths_in_one(approaches_temp_ls, ev_fn, var_fn, t_samples, plot_hist_for_all_particles=True)
    # Plot histogram of samples for returning distribution and estimated returning distribution
    # hte.plot_returning_probs_from_fptd_histogram(ev_fn, var_fn, t_samples, approaches_temp_ls)   # this is too noisy
    hte.plot_returning_probs_from_sample_paths(approaches_temp_ls, fraction_of_returns, dt)

    logging.info('Evaluations for the distributions in y at the first hitting time.')

    # Create class for evaluations
    hle = HittingLocationEvaluator('CA Process', x_predTo, t_predicted, y_predicted, plot_y, t_L,
                                   get_example_tracks_fn=get_example_tracks(x_L,
                                                                            C_L,
                                                                            S_w,
                                                                            get_system_matrices_from_parameters,
                                                                            return_component_ind=3,
                                                                            # y-position
                                                                            ),
                                   save_results=save_results,
                                   result_dir=result_dir,
                                   for_paper=for_paper,
                                   no_show=no_show,
                                   )

    # Show example histogram
    hte.plot_sample_histogram(y_samples, x_label='y-Coordinate')

    # Set up the hitting location approaches
    htd_for_hld = no_return_htd  # we use the same hitting time distribution for all approaches except the uniform and
    # MC approach
    gauss_taylor_hld = GaussTaylorCAHittingLocationDistribution(htd_for_hld, S_w,
                                                                point_predictor=ca_spatial_point_predictor)
    simple_gauss_hld = SimpleGaussCAHittingLocationDistribution(htd_for_hld, S_w,
                                                                point_predictor=ca_spatial_point_predictor)
    bayes_mixture_hld = BayesMixtureCAHittingLocationDistribution(htd_for_hld, S_w)
    bayesian_hld = BayesianCAHittingLocationDistribution(htd_for_hld, S_w)

    uniform_hld = UniformCAHittingLocationDistribution(uniform_htd,
                                                       point_predictor=ca_spatial_point_predictor,
                                                       window_length=0.08,  # width
                                                       )
    mc_hld = MCCAHittingLocationDistribution(mc_htd, S_w, y_range, y_samples=y_samples)

    # Results for spatial uncertainties
    approaches_spatial_ls = [gauss_taylor_hld, simple_gauss_hld, bayes_mixture_hld, bayesian_hld, uniform_hld, mc_hld]

    # Plot the quantile functions
    hle.plot_quantile_functions(approaches_spatial_ls)
    # Calculate moments and compare the results
    hle.compare_moments(approaches_spatial_ls)

    # Calculate wasserstein distance and compare results
    hle.compare_wasserstein_distances(approaches_spatial_ls, y_samples)
    # Calculate the Hellinger distance
    hle.compare_hellinger_distances(approaches_spatial_ls, y_samples)
    # Calculate the first wasserstein distance
    hle.compare_first_wasserstein_distances(approaches_spatial_ls, y_samples)
    # Calculate the kolmogorov distance
    hle.compare_kolmogorov_distances(approaches_spatial_ls, y_samples)

    # Plot histogram of samples and hitting time distributions
    hle.plot_y_at_first_hitting_time_distributions(approaches_spatial_ls, y_samples)

    if measure_computational_times:
        logging.info('Measuring computational time for ca process hitting time distributions.')
        model_class_ls = [MCCAHittingTimeDistribution, GaussTaylorCAHittingTimeDistribution,
                          NoReturnCAHittingTimeDistribution]
        model_attributes_ls = [[x_L, C_L, S_w, x_predTo, t_L, t_range]] + [
            [x_L, C_L, S_w, x_predTo, t_L, ca_temporal_point_predictor]] + [[x_L, C_L, S_w, x_predTo, t_L]]
        measure_computation_times(model_class_ls, model_attributes_ls, t_range=t_range)

        logging.info('Measuring computational time for ca process hitting location distributions.')
        model_class_ls = [MCCAHittingLocationDistribution, GaussTaylorCAHittingLocationDistribution,
                          BayesMixtureCAHittingLocationDistribution, BayesianCAHittingLocationDistribution]
        model_attributes_ls = [[mc_hld, S_w, y_range]] + [[htd_for_hld, S_w], ca_spatial_point_predictor] + 2 * [
            [htd_for_hld, S_w]]
        measure_computation_times(model_class_ls, model_attributes_ls, t_range=t_range)


def run_experiment_with_extent(x_L, C_L, t_L, S_w, x_predTo,
                               particle_size,
                               t_range_with_extents=None,
                               y_range_with_extents=None,
                               load_samples=False,
                               save_samples=False,
                               save_path=None,
                               save_results=False,
                               result_dir=None,
                               no_show=False,
                               for_paper=False):
    """Runs an experiment including a comparison with Monte Carlo simulation with the given settings for the
    extent-based representation of the particles.

    The underlying process is a 2D (x, y) constant acceleration (CA) model with independent components in x, y.
    Therefore, the state is [pos_x, velo_x, acc_x, pos_y, velo_y, acc_y].

    :param x_L: A np.array of shape [6] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, velo_x, acc_x, pos_y, velo_y, acc_y].
    :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, the position of the boundary.
    :param particle_size: A list of length 2 representing the length and width (in transport direction and
        perpendicular) of the particle.
    :param t_range_with_extents: A list of length 2 representing the plot limits for the first-passage time.
    :param y_range_with_extents: A list of length 2 representing the plot limits for the y component at the first
        passage time.
    :param load_samples: Boolean, whether to load the samples for the Monte Carlo simulation from file.
    :param save_samples: Boolean, whether to save the samples for the Monte Carlo simulation from file.
    :param save_path: String, path where to save the .npz file with the samples (suffix .npz).
    :param save_results: Boolean, whether to save the plots.
    :param result_dir: String, directory where to save the plots.
    :param no_show: Boolean, whether to show the plots (False).
    :param for_paper: Boolean, whether to use the plots for a publication (omit headers, etc.).
    """
    # Deterministic predictions
    ca_temporal_point_predictor = lambda pos_l, v_l, a_l, x_predTo: - v_l[..., 0] / a_l[..., 0] + np.sign(a_l[..., 0]) * \
                                                                    np.sqrt((v_l[..., 0] / a_l[..., 0]) ** 2 + 2 / a_l[
                                                                        ..., 0] * (
                                                                                    x_predTo - pos_l[..., 0]))
    ca_spatial_point_predictor = lambda pos_l, v_l, a_l, dt_pred: dt_pred * v_l[..., 1] + 1 / 2 * dt_pred ** 2 * a_l[
        ..., 1]
    # t_predicted = t_L - x_L[1] / x_L[2] + np.sign(x_L[2]) * \
    #               np.sqrt((x_L[1] / x_L[2]) ** 2 + 2 / x_L[2] * (x_predTo - x_L[0]))
    # y_predicted = x_L[3] + (t_predicted - t_L) * x_L[4] + 1 / 2 * (t_predicted - t_L) ** 2 * x_L[5]
    t_predicted = t_L + ca_temporal_point_predictor(x_L[[0, 3]], x_L[[1, 4]], x_L[[2, 5]], x_predTo)
    y_predicted = x_L[3] + ca_spatial_point_predictor(x_L[[0, 3]], x_L[[1, 4]], x_L[[2, 5]], dt_pred=t_predicted - t_L)

    # Plot settings
    if t_range_with_extents is None:
        t_range_with_extents = [t_predicted - 0.2 * (t_predicted - t_L) - 1 / 2 * particle_size[0] / x_L[1],
                                t_predicted + 0.2 * (t_predicted - t_L) + particle_size[0] / x_L[1]]
    if y_range_with_extents is None:
        y_range_with_extents = [0.9 * y_predicted - 1 / 2 * particle_size[1],
                                1.1 * y_predicted + 1 / 2 * particle_size[1]]
    plot_t = np.arange(t_range_with_extents[0], t_range_with_extents[1], 0.00001)
    plot_y = np.arange(y_range_with_extents[0], y_range_with_extents[1], 0.001)

    # Create samples
    dt = 1 / 1000
    if not load_samples:
        first_passage_statistics, first_arrival_interval_statistics = create_ty_ca_samples_hitting_time(x_L, C_L, S_w,
                                                                                                        x_predTo, t_L,
                                                                                                        length=
                                                                                                        particle_size[
                                                                                                            0],
                                                                                                        dt=dt)
        t_samples, y_samples, fraction_of_returns = first_passage_statistics
        t_samples_first_front_arrival, t_samples_first_back_arrival, y_min_samples, y_max_samples, y_samples_first_front_arrival, y_samples_first_back_arrival = first_arrival_interval_statistics

        if save_samples:
            np.savez(save_path,
                     name1=t_samples,
                     name2=y_samples,
                     name3=fraction_of_returns,
                     name4=t_samples_first_front_arrival,
                     name5=t_samples_first_back_arrival,
                     name6=y_min_samples,
                     name7=y_max_samples,
                     name8=y_samples_first_front_arrival,
                     name9=y_samples_first_back_arrival)
            logging.info("Saved samples.")
    else:
        data = np.load(save_path)
        t_samples = data['name1']
        y_samples = data['name2']
        fraction_of_returns = data['name3']
        t_samples_first_front_arrival = data['name4']
        t_samples_first_back_arrival = data['name5']
        y_min_samples = data['name6']
        y_max_samples = data['name7']
        y_samples_first_front_arrival = data['name8']
        y_samples_first_back_arrival = data['name9']

    logging.info('Evaluations for hitting time models with extents.')

    # Create class for evaluations
    hte = HittingTimeEvaluatorWithExtents('CA Process', x_predTo, plot_t, t_predicted, t_L,
                                          get_example_tracks_fn=get_example_tracks(x_L,
                                                                                   C_L,
                                                                                   S_w,
                                                                                   get_system_matrices_from_parameters),
                                          save_results=save_results,
                                          result_dir=result_dir,
                                          no_show=no_show,
                                          for_paper=for_paper)

    # Set up the hitting time approaches
    hitting_time_distr_kwargs = {"x_L": x_L,
                                 "C_L": C_L,
                                 "S_w": S_w,
                                 "x_predTo": x_predTo,
                                 "t_L": t_L}

    gauss_taylor_htwe = HittingTimeWithExtentsModel(particle_size[0], GaussTaylorCAHittingTimeDistribution,
                                                    dict(hitting_time_distr_kwargs,
                                                         point_predictor=ca_temporal_point_predictor),
                                                    name="Gauß-Taylor with extent")
    no_return_htwe = HittingTimeWithExtentsModel(particle_size[0], NoReturnCAHittingTimeDistribution,
                                                 hitting_time_distr_kwargs,
                                                 name="No-return approx. with extent")
    uniform_htwe = HittingTimeWithExtentsModel(particle_size[0], UniformCAHittingTimeDistribution,
                                               dict(x_L=hitting_time_distr_kwargs["x_L"],
                                                    x_predTo=hitting_time_distr_kwargs["x_predTo"],
                                                    t_L=hitting_time_distr_kwargs["t_L"],
                                                    point_predictor=ca_temporal_point_predictor,
                                                    window_length=0),  # TODO: macht das sinn?
                                               name="Uniform with extent")
    mc_htwe = HittingTimeWithExtentsModel(particle_size[0], MCCAHittingTimeDistribution,
                                          dict(hitting_time_distr_kwargs,
                                               t_samples=t_samples,
                                               t_range=t_range_with_extents),
                                          name="MC simulation with extent")
    simplified_taylor_htwe = HittingTimeWithExtentsSimplifiedModel(particle_size[0],
                                                                   GaussTaylorCAHittingTimeDistribution,
                                                                   dict(hitting_time_distr_kwargs,
                                                                        point_predictor=ca_temporal_point_predictor),
                                                                   name="Gauß-Taylor with extent (simplified)")

    # Results for temporal uncertainties
    approaches_temp_ls = [gauss_taylor_htwe,
                          no_return_htwe,
                          uniform_htwe,
                          mc_htwe,
                          simplified_taylor_htwe,
                          ]

    # plot the distribution of the particle front and back arrival time at one axis (the time axis)
    hte.plot_first_arrival_interval_distribution_on_time_axis(approaches_temp_ls,
                                                              t_samples_first_front_arrival,
                                                              t_samples_first_back_arrival,
                                                              plot_hist_for_all_particles=True,
                                                              plot_cdfs=True,
                                                              )

    # # plot the joint distribution of the particle front and back arrival time (2-dimensional distribution, heatmap)
    # hte.plot_joint_first_arrival_interval_distribution(approaches_temp_ls,
    #                                                    t_samples_first_front_arrival,
    #                                                    t_samples_first_back_arrival,
    #                                                    plot_hist_for_all_particles=True,
    #                                                    plot_marginals=False,
    #                                                    )

    # plot a simplifies joint distribution (based on the marginals and independence assumption) of the particle front
    # and back arrival time (2-dimensional distribution, heatmap)
    hte.plot_joint_first_arrival_interval_distribution(approaches_temp_ls,
                                                       t_samples_first_front_arrival,
                                                       t_samples_first_back_arrival,
                                                       plot_hist_for_all_particles=True,
                                                       use_independent_joint=True,
                                                       )

    # plot the calibration
    hte.plot_calibration(approaches_temp_ls,
                         t_samples_first_front_arrival,
                         t_samples_first_back_arrival,

                         )

    logging.info('Evaluations for hitting location models with extents.')

    # Create class for evaluations
    hle = HittingLocationEvaluatorWithExtents('CA Process', x_predTo, t_predicted, y_predicted, plot_y, t_L,
                                              get_example_tracks_fn=get_example_tracks(x_L,
                                                                                       C_L,
                                                                                       S_w,
                                                                                       get_system_matrices_from_parameters,
                                                                                       return_component_ind=3,
                                                                                       # y-position
                                                                                       ),
                                              save_results=save_results,
                                              result_dir=result_dir,
                                              for_paper=for_paper,
                                              no_show=no_show,
                                              )

    # Set up the hitting location approaches
    htwe_model_for_hlwe_model = gauss_taylor_htwe  # we use the same hitting time distribution for all approaches except
    # the uniform and MC approach
    hitting_location_distr_kwargs = {'S_w': hitting_time_distr_kwargs['S_w']}
    gauss_taylor_hlwe = HittingLocationWithExtentsModel(particle_size[1],
                                                        htwe_model_for_hlwe_model,
                                                        GaussTaylorCAHittingLocationDistribution,
                                                        dict(hitting_location_distr_kwargs,
                                                             point_predictor=ca_spatial_point_predictor),
                                                        name="Gauß-Taylor with extent",
                                                        )
    simple_gauss_hlwe = HittingLocationWithExtentsModel(particle_size[1],
                                                        htwe_model_for_hlwe_model,
                                                        SimpleGaussCAHittingLocationDistribution,
                                                        dict(hitting_location_distr_kwargs,
                                                             point_predictor=ca_spatial_point_predictor),
                                                        name="Simple Gauss with extent",
                                                        )
    bayes_mixture_hlwe = HittingLocationWithExtentsModel(particle_size[1],
                                                         htwe_model_for_hlwe_model,
                                                         BayesMixtureCAHittingLocationDistribution,
                                                         hitting_location_distr_kwargs,
                                                         name="Bayes-Mixture with extent",
                                                         )
    bayesian_hlwe = HittingLocationWithExtentsModel(particle_size[1],
                                                    htwe_model_for_hlwe_model,
                                                    BayesianCAHittingLocationDistribution,
                                                    hitting_location_distr_kwargs,
                                                    name="Bayesian with extent",
                                                    )

    uniform_hlwe = HittingLocationWithExtentsModel(particle_size[1],
                                                   uniform_htwe,
                                                   UniformCAHittingLocationDistribution,
                                                   dict(point_predictor=ca_spatial_point_predictor,
                                                        window_length=0),  # TODO: Macht das Sinn?
                                                   name="MC with extent",
                                                   )
    y_samples_for_mc = hle.remove_not_arriving_samples(y_samples)
    mc_hlwe = HittingLocationWithExtentsModel(particle_size[1],
                                              htwe_model_for_hlwe_model,
                                              MCCAHittingLocationDistribution,
                                              dict(hitting_location_distr_kwargs,
                                                   y_range=y_range_with_extents,
                                                   y_samples=y_samples_for_mc),
                                              name="MC with extent",
                                              )

    # Results for spatial uncertainties
    approaches_spatial_ls = [gauss_taylor_hlwe,
                             simple_gauss_hlwe,
                             # bayes_mixture_hlwe,  # TODO: PPF fehlt!
                             # bayesian_hlwe,
                             uniform_hlwe,
                             mc_hlwe]

    # plot the distribution of the particle front and back arrival time at one axis (the time axis)
    hle.plot_y_at_first_arrival_interval_distribution_on_y_axis(
        approaches_ls=approaches_spatial_ls,
        min_y_samples=y_min_samples - particle_size[1] / 2,
        max_y_samples=y_max_samples + particle_size[1] / 2,
        # If one wants to instead use the min/max of the front or back arrival (note that is assumed for the model that
        # min_y = min(front, back) and max_y = max(front, back))
        # min_y_samples=np.min(np.vstack([y_samples_first_front_arrival, y_samples_first_back_arrival]), axis=0) -
        #               particle_size[1] / 2,
        # max_y_samples=np.max(np.vstack([y_samples_first_front_arrival, y_samples_first_back_arrival]), axis=0) +
        #               particle_size[1] / 2,
        plot_cdfs=True,
    )

    # plot the joint distribution of the particle front and back arrival time (2-dimensional distribution, heatmap)
    hle.plot_joint_y_at_first_arrival_interval_distribution(approaches_spatial_ls,
                                                            y_min_samples - particle_size[1] / 2,
                                                            y_max_samples + particle_size[1] / 2,
                                                            plot_marginals=False,
                                                            )
    # plot the calibration
    hle.plot_calibration(approaches_spatial_ls,
                         y_min_samples - particle_size[1] / 2,
                         y_max_samples + particle_size[1] / 2,
                         )


if __name__ == "__main__":
    app.run(main)

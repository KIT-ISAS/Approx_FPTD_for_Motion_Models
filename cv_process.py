"""
############################################ cv_process.py  ###########################################
Authors: Marcel Reith-Braun (ISAS, marcel.reith-braun@kit.edu), Jakob Thumm
#######################################################################################################
Calculates approximate first-passage time distributions for a constant velocity model using different
approaches.

usage:
 - run docker container - tested with tensorflow/approx_fptd:2.8.0-gpu image:
    $ docker run -u $(id -u):$(id -g) \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
            tensorflow/approx_fptd:2.8.0-gpu
 - within container:
     $   python3 /mnt/cv_process.py \\
requirements:
  - Required packages/tensorflow/approx_fptd:2.8.0-gpu image: See corresponding dockerfile.
  - Volume mounts: Specify a path </path/to/repo/> that points to the repo.
"""


from absl import logging
from absl import app
from absl import flags

import numpy as np

from evaluators.hitting_time_evaluator import HittingTimeEvaluator
from cv_arrival_distributions.cv_hitting_time_distributions import GaussTaylorCVHittingTimeDistribution, NoReturnCVHittingTimeDistribution, MCCVHittingTimeDistribution
from evaluators.hitting_location_evaluator import HittingLocationEvaluator
from cv_arrival_distributions.cv_hitting_location_distributions import GaussTaylorCVHittingLocationDistribution, SimpleGaussCVHittingLocationDistribution, MCCVHittingLocationDistribution
from sampler import get_example_tracks_lgssm as get_example_tracks
from cv_arrival_distributions.cv_utils import get_system_matrices_from_parameters, create_ty_cv_samples_hitting_time
from timer import measure_computation_times


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
    x_L = np.array([0.3, 6.2, 0.5, 0.2])

    # Boundary position
    x_predTo = 0.6458623971412047
    # Last time step
    t_L = 0  # In principle, we could assume w.l.o.g. that _t_L = 0 (_t_L is just a location argument).

    # Run the experiment
    run_experiment(x_L, C_L, t_L, S_w, x_predTo,
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

    The underlying process is a 2D (x, y) constant velocity (CV) model with independent components in x, y.
    Therefore, the state is [pos_x, velo_x, pos_y, velo_y].

    :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, velo_x, pos_y, velo_y].
    :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
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
    t_predicted = t_L + (x_predTo - x_L[0]) / x_L[1]
    y_predicted = x_L[2] + (t_predicted - t_L) * x_L[3]

    # Plot settings
    if t_range is None:
        t_range = [t_predicted - 0.3*(t_predicted - t_L), t_predicted + 0.3*(t_predicted - t_L)]
    if y_range is None:
        y_range = [0.7 * y_predicted, 1.3 * y_predicted]
    plot_t = np.arange(t_range[0], t_range[1], 0.00001)
    plot_y = np.arange(y_range[0], y_range[1], 0.001)

    # Create base class
    hte = HittingTimeEvaluator('CV Process', x_predTo, plot_t, t_predicted, t_L,
                               get_example_tracks_fn=get_example_tracks(x_L,
                                                                        C_L,
                                                                        S_w,
                                                                        get_system_matrices_from_parameters),
                               save_results=save_results,
                               result_dir=result_dir,
                               no_show=no_show,
                               for_paper=for_paper)

    # Create samples
    dt = 1 / 1000
    if not load_samples:
        t_samples, y_samples, fraction_of_returns = create_ty_cv_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L, dt=dt)
        if save_samples:
            np.savez(save_path, name1=t_samples, name2=y_samples, name3=fraction_of_returns)
            logging.info("Saved samples.")
    else:
        data = np.load(save_path)
        t_samples = data['name1']
        y_samples = data['name2']
        fraction_of_returns = data['name3']
    hte.plot_sample_histogram(t_samples)
    # hte.plot_sample_histogram(y_samples, x_label='y-Coordinate')

    # Show example tracks and visualize uncertainties over time
    # hte.plot_example_tracks(N=5)
    ev_fn = lambda t: x_L[0] + x_L[1] * (t - t_L)
    var_fn = lambda t: C_L[0, 0] + 2 * C_L[1, 0] * (t - t_L) + C_L[1, 1] * (t - t_L)**2 + S_w * pow(t - t_L, 3)/3
    hte.plot_mean_and_stddev_over_time(ev_fn, var_fn, show_example_tracks=True)

    # Set up the hitting time approaches
    cv_temporal_point_predictor = lambda pos_l, v_l: (x_predTo - pos_l) / v_l
    cv_spatial_point_predictor = lambda pos_l, v_l, dt:  v_l * dt
    taylor_model = GaussTaylorCVHittingTimeDistribution(x_L, C_L, S_w, x_predTo, t_L, point_predictor=cv_temporal_point_predictor)
    approx_model = NoReturnCVHittingTimeDistribution(x_L, C_L, S_w, x_predTo, t_L)
    mc_model = MCCVHittingTimeDistribution(x_L, C_L, S_w, x_predTo, t_L, t_range, t_samples=t_samples)

    # Results for temporal uncertainties
    logging.info('MAX CDF: {} at {}'.format(approx_model.q_max, approx_model.t_max))
    approx_model.plot_valid_regions(theta=t_predicted, save_results=save_results, result_dir=result_dir, for_paper=True,
                                    no_show=no_show)
    # approx_model.plot_valid_regions(save_results=save_results, _result_dir=_result_dir, _for_paper=True, no_show_no_show)
    logging.info('tau_max: {}'.format(approx_model.trans_dens_ppf(t_predicted)[0]))
    logging.info('Mass inside invalid region: {}'.format(
        1 - approx_model.cdf(t_predicted + approx_model.trans_dens_ppf(t_predicted)[0])))
    logging.info('Approximate returning probs after a crossing until time t_max: {}'.format(
        approx_model.get_statistics()['ReturningProbs'](approx_model.t_max)))

    approaches_temp_ls = [taylor_model, approx_model]

    # Plot the quantile functions
    # hte.plot_quantile_functions(approaches_temp_ls)
    # # Calculate moments and compare the results
    # hte.compare_moments(approaches_temp_ls)
    # # Calculate the skewness and compare the results
    # hte.compare_skewness(approaches_temp_ls)
    #
    # # Calculate wasserstein distance and compare results
    # hte.compare_wasserstein_distances(t_samples, approaches_temp_ls)
    # # Calculate the Hellinger distance
    # hte.compare_hellinger_distances(t_samples, approaches_temp_ls)
    # # Calculate the first wasserstein distance
    # hte.compare_first_wasserstein_distances(t_samples, approaches_temp_ls)
    # # Calculate the kolmogorov distance
    # hte.compare_kolmogorv_distances(t_samples, approaches_temp_ls)
    #
    # # Plot histogram of samples and hitting time distributions
    # hte.plot_first_hitting_time_distributions(t_samples, approaches_temp_ls, plot_hist_for_all_particles=True)
    # hte.plot_fptd_and_paths_in_one(ev_fn, var_fn, t_samples, approaches_temp_ls, plot_hist_for_all_particles=True)
    # # Plot histogram of samples for returning distribution and estimated returning distribution
    # # hte.plot_returning_probs_from_fptd_histogram(ev_fn, var_fn, t_samples, approaches_temp_ls)   # this is too noisy
    # hte.plot_returning_probs_from_sample_paths(fraction_of_returns, dt, approaches_temp_ls)

    # Set up the hitting location approaches
    spatial_taylor_model = GaussTaylorCVHittingLocationDistribution(taylor_model, S_w, point_predictor=cv_spatial_point_predictor)
    spatial_simple_gauss_model = SimpleGaussCVHittingLocationDistribution(approx_model, S_w, point_predictor=cv_spatial_point_predictor)
    # spatial_proj_model = ProjectionCVHittingLocationModel(approx_model, S_w)
    spatial_mc_model = MCCVHittingLocationDistribution(mc_model, S_w, y_range, y_samples=y_samples)

    # Results for spatial uncertainties
    hte_spatial = HittingLocationEvaluator('CV Process', x_predTo, t_predicted, y_predicted, plot_y, t_L,
                                   get_example_tracks_fn=get_example_tracks(x_L,
                                                                            C_L,
                                                                            S_w,
                                                                            get_system_matrices_from_parameters),
                                   save_results=save_results,
                                   result_dir=result_dir,
                                   no_show=no_show,
                                   for_paper=for_paper)

    approaches_spatial_ls = [ #spatial_taylor_model,
    #                          spatial_simple_gauss_model,
                             # spatial_proj_model,
                             spatial_mc_model]

    # Plot the quantile functions
    hte_spatial.plot_quantile_functions(approaches_spatial_ls)
    # Calculate moments and compare the results
    hte_spatial.compare_moments(approaches_spatial_ls)

    # Plot histogram of samples and hitting time distributions
    hte_spatial.plot_y_at_first_hitting_time_distributions(y_samples, approaches_spatial_ls)


    if measure_computational_times:  # TODO: Auch für spatial?
        logging.info('Measuring computational time for cv process.')
        model_class_ls = [MCCVHittingTimeDistribution, GaussTaylorCVHittingTimeDistribution, NoReturnCVHittingTimeDistribution]
        model_attributes_ls = [[x_L, C_L, S_w, x_predTo,  t_L, t_range]] + 2 * [[x_L, C_L, S_w, x_predTo,  t_L]]
        measure_computation_times(model_class_ls, model_attributes_ls, t_range=t_range)


if __name__ == "__main__":
    app.run(main)

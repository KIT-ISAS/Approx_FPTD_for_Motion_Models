"""
####################################### wnca_experiments_main.py ######################################
Authors: Marcel Reith-Braun (ISAS, marcel.reith-braun@kit.edu), Jakob Thumm
#######################################################################################################
Defines and executes experiments (first-passage time problems) for the white-noise constant acceleration model.
See wnca_process_main.py for details.

usage:
 - run docker container - tested with tensorflow/approx_fptd:2.8.0-gpu image:
    $ docker run -u $(id -u):$(id -g) \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
            tensorflow/approx_fptd:2.8.0-gpu
 - within container:
     $   python3 /mnt/wnca_experiments_main.py \\
requirements:
  - Required packages/tensorflow/approx_fptd:2.8.0-gpu image: See corresponding dockerfile.
  - Volume mounts: Specify a path </path/to/repo/> that points to the repo.
"""

from absl import logging
from absl import app
from absl import flags

import numpy as np

from wnca_process_main import run_experiment, run_experiment_with_extent
from experiments_runner import get_experiments_by_name, add_defaults, convert_to_numpy, store_config
from kalman_filter import KalmanFilter
from cv_arrival_distributions.cv_utils import get_system_matrices_from_parameters
from abstract_distributions import AbstractArrivalDistribution


# Delete all FLAGS defined by CV process as we here not want them to be overwritten by the following flags.
for name in list(flags.FLAGS):
    if name in ['load_samples', 'save_samples', 'save_path', 'save_results', 'result_dir', 'no_show', 'for_paper',
                'measure_computational_times', 'with_extents', 'verbosity_level']:
        delattr(flags.FLAGS, name)

flags.DEFINE_bool('load_samples', default=False,
                  help='Whether the samples should be loaded from a .npz  file.')
flags.DEFINE_bool('save_samples', default=False,
                  help='Whether the samples should be saved to a .npz  file.')
flags.DEFINE_string('save_dir', default='/mnt/',
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


# Experiments config
# List of dictionaries describing experiments
# The syntax of each dictionary is:
"""
{
    # Experiment name
    "experiment_name": "CV_Long_Track_Sw1",
    # Process parameters
    "x_L": [0.3, 6.2, 0.5, 0.2],
    "C_L": [[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]],
    "t_L": 0,
    "S_w": 1,
    # Boundary
    "x_predTo": 0.6458623971412047,
    # Particle size (only required for experiments with extents)
        "particle_size": [0.08, 0.08],
    # Plot settings (optional)
    "t_range": [t_min, t_max], (floats, defaults by cv_process)
    "y_range": [y_min, y_max], (floats, defaults by cv_process)
    "t_range_with_extents": [t_min, t_max], (floats, defaults by cv_process_with_extents)
    "y_range_with_extents": [y_min, y_max], (floats, defaults by cv_process_with_extents)
    # Units (optional)
    "time_unit": "s",
    "length_unit": "m",
    # Paths and directories (optional)
    "save_path": ..., (string, default by main function)
    "results_dir": ..., (string, default by main function)
}
"""

track_measurements = np.array([[1101.069, 34.743],
                               [1100.751, 52.036],
                               [1100.551, 69.456],
                               [1100.299, 86.986],
                               [1099.915, 105.336],
                               [1099.435, 124.121],
                               [1099.266, 142.945],
                               [1098.795, 162.083],
                               [1098.679, 181.177],
                               [1098.278, 200.815],
                               [1097.677, 220.970],
                               [1097.237, 241.073],
                               [1096.600, 261.273],
                               [1096.152, 281.885],
                               [1095.759, 302.818],
                               [1095.557, 323.968],
                               [1095.059, 345.553],
                               [1094.912, 367.113],
                               [1094.650, 389.180],
                               [1094.209, 411.721],
                               [1094.004, 434.886],
                               [1093.541, 458.248],
                               [1093.039, 481.609],
                               [1092.477, 505.452],
                               [1092.147, 529.755],
                               [1091.831, 554.447],
                               [1091.383, 579.074],
                               [1091.083, 604.125],
                               [1090.960, 629.309],
                               [1090.581, 655.141],
                               [1090.146, 681.447],
                               [1089.600, 707.745],
                               [1088.869, 734.235],
                               [1088.378, 761.158],
                               [1088.141, 788.505],
                               [1087.688, 816.160],
                               [1087.000, 843.933],
                               [1086.563, 872.073],
                               [1086.205, 900.417],
                               [1085.885, 929.273]])

experiments_config = [
    {
        # Experiment name
        "experiment_name": "WNCA IOSB",
        # Process parameters
        "measurements": np.fliplr(track_measurements),
        "dt": 1,
        "S_w": 0.005,
        "a_c": 0.3,
        "S_v": 10,
        # "init_state_mean": np.array([30, 20, 1000, 0]),
        "init_state_mean": np.array([track_measurements[0, 1], 18, track_measurements[0, 0], 0]),
        "init_state_cov": np.diag([1, 60, 1, 60]),
        # Boundary
        "x_predFrom": 800 - (5.8 + 4 / 2) * 30,  # = 566
        # x_predTo - (5.8 frames + half length i frames) * velo (in pixel / frame)  # 530,
        "x_predTo": 800,
        # Particle size
        "particle_size": [4, 58],
        # Plot settings (optional)
        "t_range": [34, 35],
        "y_range": [1050, 1130],
        "t_range_with_extents": [33.9, 35.1],
        "y_range_with_extents": [1040, 1140],
        # Units (optional)
        "time_unit": "frames",
        "length_unit": "pixels",
        # Factors for changing the units (no config settings, just for information)
        # "pixel_to_mm": 1/2.88,
        # "frame_to_ms": 4,
    }, {
        # Experiment name
        "experiment_name": "WNCA IOSB denorm",
        # Process parameters
        "measurements": np.fliplr(track_measurements) * 1 / 2.88,
        "dt": 1 * 4,
        "S_w": 0.005 * (1 / 2.88) ** 2 / 4 ** 3,
        "a_c": 0.3 * (1 / 2.88) / 4 ** 2,
        "S_v": 10 * (1 / 2.88) ** 2,
        # "init_state_mean": np.array([30, 20, 1000, 0]),
        "init_state_mean": np.array(
            [track_measurements[0, 1] * 1 / 2.88,
             18 * (1 / 2.88) / 4,
             track_measurements[0, 0] * 1 / 2.88,
             0]),
        "init_state_cov": np.diag(
            [1 * (1 / 2.88) ** 2,
             60 * (1 / 2.88) ** 2 / 4,
             1 * (1 / 2.88) ** 2,
             60 * (1 / 2.88) ** 2 / 4]),
        # Boundary
        "x_predFrom": (800 - (5.8 + 4 / 2) * 30) * 1 / 2.88,  # = 566
        # x_predTo - (5.8 frames + half length i frames) * velo (in pixel / frame)  # 530,
        "x_predTo": 800 * 1 / 2.88,
        # Particle size
        "particle_size": [4 * 1 / 2.88, 58 * 1 / 2.88],
        # Plot settings (optional)
        "t_range": [34 * 4, 35 * 4],
        "y_range": [1050 * 1 / 2.88, 1130 * 1 / 2.88],
        "t_range_with_extents": [33.9 * 4, 35.1 * 4],
        "y_range_with_extents": [1040 * 1 / 2.88, 1140 * 1 / 2.88],
        # Units (optional)
        "time_unit": "ms",
        "length_unit": "mm",
        # Factors for changing the units (no config settings, just for information)
        # "pixel_to_mm": 1/2.88,
        # "frame_to_ms": 4,
    }
]


def estimate_states_from_measurements(measurements,
                                      dt,
                                      S_w,
                                      a_c,
                                      S_v,
                                      init_state_mean,
                                      init_state_cov,
                                      ):

    F, Q = get_system_matrices_from_parameters(dt, S_w)
    F = np.block([[F, np.zeros((2, 2))], [np.zeros((2, 2)), F]])
    Q = np.block([[Q, np.zeros((2, 2))], [np.zeros((2, 2)), Q]])
    C_v = np.diag([S_v, S_v])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    u = a_c * np.array([0.5 * dt ** 2, dt, 0, 0])

    # reshape the arrays to the required format
    F = AbstractArrivalDistribution.batch_atleast_3d(F)
    Q = AbstractArrivalDistribution.batch_atleast_3d(Q)
    C_v = AbstractArrivalDistribution.batch_atleast_3d(C_v)
    u = np.atleast_2d(u)
    init_state_mean = np.atleast_2d(init_state_mean)
    init_state_cov = AbstractArrivalDistribution.batch_atleast_3d(init_state_cov)
    measurements = AbstractArrivalDistribution.batch_atleast_3d(measurements)

    def system_model(state_mean, state_cov, k):
        return np.matmul(F, state_mean[:, :, np.newaxis]).squeeze(-1) + u, \
               np.matmul(np.matmul(F, state_cov), F.transpose((0, 2, 1))) + Q

    kf = KalmanFilter(system_model, init_state_mean, init_state_cov)

    for k in range(1, measurements.shape[1]):
        kf.predict_own_state()
        kf.update_own_state(measurements[:, k, :], H, C_v)
    # for k in range(0, measurements.shape[1]):
        # kf.update_own_state(measurements[:, k, :], H, C_v)
        # if k < measurements.shape[1] - 1:
        #     kf.predict_own_state()

    return np.squeeze(kf.state_mean), np.squeeze(kf.state_cov), k * dt


def main(args):
    del args

    # setup logging
    logging.set_verbosity(logging.FLAGS.verbosity_level)

    # define the experiments to execute by name
    experiments_name_list = ['WNCA IOSB denorm']

    # get the configs
    experiments_list = get_experiments_by_name(experiments_name_list, experiments_config)

    # add the defaults (if necessary)
    add_defaults(experiments_list, FLAGS)

    # run the experiments and store the configs
    for i, config in enumerate(experiments_list):

        if "measurements" in config.keys():  # Overwrite all other state components
            # shorten measurements
            measurements = config["measurements"]
            measurements = measurements[measurements[:, 0] < config["x_predFrom"], :]

            x_L, C_l, t_L = estimate_states_from_measurements(measurements,
                                                              config["dt"],
                                                              config["S_w"],
                                                              config["a_c"],
                                                              config["S_v"],
                                                              config["init_state_mean"],
                                                              config["init_state_cov"])

            config["x_L"] = x_L.tolist()
            config["C_L"] = C_l.tolist()
            config["t_L"] = t_L
            del config['measurements']
            del config['dt']
            del config['S_v']
            del config['init_state_mean']
            del config['init_state_cov']
            del config['x_predFrom']

        store_config(config, FLAGS.save_dir)
        convert_to_numpy(config)  # convert the configs entries to numpy arrays
        logging.info('Running experiment {}.'.format(config['experiment_name']))
        del config['experiment_name']  # name cannot be passed to run_experiment

        if not FLAGS.with_extents:
            if 't_range_with_extents' in config.keys():
                del config['t_range_with_extents']  # t_range_with_extents cannot be passed to run_experiment
            if 'y_range_with_extents' in config.keys():
                del config['y_range_with_extents']
            run_experiment(**config,
                           for_paper=FLAGS.for_paper,
                           measure_computational_times=True if FLAGS.measure_computational_times and i == 0 else False)
            # by default, takes the first defined experiment for measuring the computational times.
        else:
            if 't_range' in config.keys():
                del config['t_range']
            if 'y_range' in config.keys():
                del config['y_range']
            run_experiment_with_extent(**config,
                                       for_paper=FLAGS.for_paper)


if __name__ == "__main__":
    app.run(main)

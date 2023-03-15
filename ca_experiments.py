"""
########################################### ca_experiments.py #########################################
Authors: Marcel Reith-Braun (ISAS, marcel.reith-braun@kit.edu), Jakob Thumm
#######################################################################################################
Defines and executes experiments (first passage time problems) for the constant acceleration model.
See ca_process.py for details.

usage:
 - run docker container - tested with tensorflow/approx_fptd:2.8.0-gpu image:
    $ docker run -u $(id -u):$(id -g) \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
           tensorflow/approx_fptd:2.8.0-gpu
 - within container:
     $   python3 /mnt/ca_experiments.py \\
requirements:
  - Required packages/tensorflow/approx_fptd:2.8.0-gpu image: See corresponding dockerfile.
  - Volume mounts: Specify a path </path/to/repo/> that points to the repo.
"""

from absl import logging
from absl import app
from absl import flags

from ca_process import run_experiment
from experiments_runner import get_experiments_by_name, add_defaults, convert_to_numpy, store_config


# Delete all FLAGS defined by CV process as we here not want them to be overwritten by the following flags.
for name in list(flags.FLAGS):
    if name in ['load_samples', 'save_samples', 'save_path', 'save_results', 'result_dir', 'no_show', 'for_paper',
                'measure_computational_times', 'verbosity_level']:
        delattr(flags.FLAGS,name)


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
                    help='Whether to measure the computational times (using the first defined experiment).')

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
    "experiment_name": "Long_Track_Sw1",
    # Process parameters
    "x_L": [0.3, 6.2, 0.5, 0.2],
    "C_L": [[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]],
    "t_L": 0,
    "S_w": 1,
    # Boundary
    "x_predTo": 0.6458623971412047,
    # Plot settings (optional)
    "t_range": [t_min, t_max], (floats, defaults by cv_process)
    "y_range": [y_min, y_max], (floats, defaults by cv_process)
    # Paths and directories (optional)
    "save_path": ..., (string, default by main function)
    "results_dir": ..., (string, default by main function)
}
"""

experiments_config = [
    {
        # Experiment name
        "experiment_name": "CA_Sw1000",
        # Process parameters
        "x_L": [0.3, 6.2, 4.4, 0.5, 0.2, 2.8],
        "C_L": [[2E-7, 2E-5, 8E-4, 0, 0, 0], [2E-5, 3E-3, 1.5E-1, 0, 0, 0], [8E-4, 1.5E-1, 1.3E1, 0, 0, 0],
                [0, 0, 0, 2E-7, 2E-5, 8E-4], [0, 0, 0, 2E-5, 3E-3, 1.5E-1], [0, 0, 0, 8E-4, 1.5E-1, 1.3E1]],
        "t_L": 0,
        "S_w": 1000,
        # Boundary
        "x_predTo": 0.6458623971412047,
        # Plot settings (optional)
        "t_range": [0.048, 0.062],
        "y_range": [0.47, 0.56]
    }, {
        # Experiment name
        "experiment_name": "CA_Sw1000_denorm",
        # Process parameters
        "x_L": [29.0304, 599.9616, 425.7792,  64.96, 25.984, 363.776],
        "C_L": [[1.87280916e-03, 1.87280916e-01, 7.49123666e+00, 0, 0, 0],
                [1.87280916e-01, 2.80921375e+01, 1.40460687e+03, 0, 0, 0],
                [7.49123666e+00, 1.40460687e+03, 1.21732596e+05, 0, 0, 0],
                [0, 0, 0, 3.37584128e-03, 3.37584128e-01, 1.35033651e+01],
                [0, 0, 0, 3.37584128e-01, 5.06376192e+01, 2.53188096e+03],
                [0, 0, 0, 1.35033651e+01, 2.53188096e+03, 2.19429683e+05]],
        "t_L": 0,
        "S_w": 9364045.824,
        # Boundary
        "x_predTo": 62.5,
        # Plot settings (optional)
        "t_range": [0.048, 0.062],
        "y_range": [62, 72]
    }, {
        # Experiment name
        "experiment_name": "CA_Sw100000",
        # Process parameters
        "x_L": [0.3, 6.2, 4.4, 0.5, 0.2, 2.8],
        "C_L": [[2E-7, 2E-5, 8E-4, 0, 0, 0], [2E-5, 3E-3, 1.5E-1, 0, 0, 0], [8E-4, 1.5E-1, 1.3E1, 0, 0, 0],
                [0, 0, 0, 2E-7, 2E-5, 8E-4], [0, 0, 0, 2E-5, 3E-3, 1.5E-1], [0, 0, 0, 8E-4, 1.5E-1, 1.3E1]],
        "t_L": 0,
        "S_w": 100000,
        # Boundary
        "x_predTo": 0.6458623971412047,
        # Plot settings (optional)
        "t_range": [0.035, 0.07],
        "y_range": [0.4, 0.64]
    }, {
        # Experiment name
        "experiment_name": "CA_Sw100000_denorm",
        # Process parameters
        "x_L": [29.0304, 599.9616, 425.7792,  64.96, 25.984, 363.776],
        "C_L": [[1.87280916e-03, 1.87280916e-01, 7.49123666e+00, 0, 0, 0],
                [1.87280916e-01, 2.80921375e+01, 1.40460687e+03, 0, 0, 0],
                [7.49123666e+00, 1.40460687e+03, 1.21732596e+05, 0, 0, 0],
                [0, 0, 0, 3.37584128e-03, 3.37584128e-01, 1.35033651e+01],
                [0, 0, 0, 3.37584128e-01, 5.06376192e+01, 2.53188096e+03],
                [0, 0, 0, 1.35033651e+01, 2.53188096e+03, 2.19429683e+05]],
        "t_L": 0,
        "S_w": 936404582.4,
        # Boundary
        "x_predTo": 62.5,
        # Plot settings (optional)
        "t_range": [0.025, 0.1],
        "y_range": [50, 85]
    }, {
        # Experiment name
        "experiment_name": "CA_Sw1000_negative_acceleration",
        # Process parameters
        "x_L": [0.3, 6.2, -8.0, 0.5, 0.2, 2.8],
        "C_L": [[2E-7, 2E-5, 8E-4, 0, 0, 0], [2E-5, 3E-3, 1.5E-1, 0, 0, 0], [8E-4, 1.5E-1, 1.3E1, 0, 0, 0],
                [0, 0, 0, 2E-7, 2E-5, 8E-4], [0, 0, 0, 2E-5, 3E-3, 1.5E-1], [0, 0, 0, 8E-4, 1.5E-1, 1.3E1]],
        "t_L": 0,
        "S_w": 1000,
        # Boundary
        "x_predTo": 0.6458623971412047,
    }, {
        # Experiment name
        "experiment_name": "CA_Sw1000_negative_acceleration_denorm",
        # Process parameters
        "x_L": [29.0304, 599.9616, -775,  64.96, 25.984, 363.776],
        "C_L": [[1.87280916e-03, 1.87280916e-01, 7.49123666e+00, 0, 0, 0],
                [1.87280916e-01, 2.80921375e+01, 1.40460687e+03, 0, 0, 0],
                [7.49123666e+00, 1.40460687e+03, 1.21732596e+05, 0, 0, 0],
                [0, 0, 0, 3.37584128e-03, 3.37584128e-01, 1.35033651e+01],
                [0, 0, 0, 3.37584128e-01, 5.06376192e+01, 2.53188096e+03],
                [0, 0, 0, 1.35033651e+01, 2.53188096e+03, 2.19429683e+05]],
        "t_L": 0,
        "S_w": 9364045.824,
        # Boundary
        "x_predTo": 62.5,
        # Plot settings (optional)
        "t_range": [0.051, 0.066],
        "y_range": [62, 72]
    },  {
        # Experiment name
        "experiment_name": "CA_Sw1000_high_acceleration",
        # Process parameters
        "x_L": [0.3, 6.2, 10.0, 0.5, 0.2, 2.8],
        "C_L": [[2E-7, 2E-5, 8E-4, 0, 0, 0], [2E-5, 3E-3, 1.5E-1, 0, 0, 0], [8E-4, 1.5E-1, 1.3E1, 0, 0, 0],
                [0, 0, 0, 2E-7, 2E-5, 8E-4], [0, 0, 0, 2E-5, 3E-3, 1.5E-1], [0, 0, 0, 8E-4, 1.5E-1, 1.3E1]],
        "t_L": 0,
        "S_w": 1000,
        # Boundary
        "x_predTo": 0.6458623971412047,
        # Plot settings (optional)
        "t_range": [0.0475, 0.06],
        "y_range": [0.48, 0.56]
    }, {
        # Experiment name
        "experiment_name": "CA_Sw1000_high_acceleration_denorm",
        # Process parameters
        "x_L": [29.0304, 599.9616, 870, 64.96, 25.984, 363.776],
        "C_L": [[1.87280916e-03, 1.87280916e-01, 7.49123666e+00, 0, 0, 0],
                [1.87280916e-01, 2.80921375e+01, 1.40460687e+03, 0, 0, 0],
                [7.49123666e+00, 1.40460687e+03, 1.21732596e+05, 0, 0, 0],
                [0, 0, 0, 3.37584128e-03, 3.37584128e-01, 1.35033651e+01],
                [0, 0, 0, 3.37584128e-01, 5.06376192e+01, 2.53188096e+03],
                [0, 0, 0, 1.35033651e+01, 2.53188096e+03, 2.19429683e+05]],
        "t_L": 0,
        "S_w": 9364045.824,
        # Boundary
        "x_predTo": 62.5,
        # Plot settings (optional)
        "t_range": [0.049, 0.06],
        "y_range": [62, 72]
    },
]


def main(args):
    del args

    # setup logging
    logging.set_verbosity(logging.FLAGS.verbosity_level)

    # define the experiments to execute by name
    experiments_name_list = ['CA_Sw1000_denorm', 'CA_Sw100000_denorm', 'CA_Sw1000_negative_acceleration_denorm', 'CA_Sw1000_high_acceleration_denorm']
    # experiments_name_list = ['CA_Sw100000_denorm', 'CA_Sw1000_negative_acceleration_denorm',
    #                          'CA_Sw1000_high_acceleration_denorm']

    # get the configs
    experiments_list = get_experiments_by_name(experiments_name_list, experiments_config)

    # add the defaults (if necessary)
    add_defaults(experiments_list, FLAGS)

    # run the experiments and store the configs
    for i, config in enumerate(experiments_list):
        store_config(config, FLAGS.save_results)
        convert_to_numpy(config)  # convert the configs entries to numpy arrays
        logging.info('Running experiment {}.'.format(config['experiment_name']))
        del config['experiment_name']  # name cannot be passed to run_experiment
        run_experiment(**config,
                       for_paper=FLAGS.for_paper,
                       measure_computational_times=True if FLAGS.measure_computational_times and i == 0 else False)
        # by default, takes the first defined experiment for measuring the computational times.


if __name__ == "__main__":
    app.run(main)

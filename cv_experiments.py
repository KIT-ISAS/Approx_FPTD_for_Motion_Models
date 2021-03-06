'''
########################################### cv_experiments.py #########################################
Authors: Marcel Reith-Braun (ISAS, marcel.reith-braun@kit.edu), Jakob Thumm
#######################################################################################################
Defines and executes experiments (first passage time problems) for the constant velocity model.
See cv_process.py for details.

usage:
 - run docker container - tested with tracksort_neural:2.1.0-gpu-py3 image:
    $ docker run -u $(id -u):$(id -g) \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
            tensorflow/tracksort_neural:2.1.0-gpu-py3
 - within container:
     $   python3 /mnt/cv_experiments.py \\
requirements:
  - Required packages/tracksort_neural:2.1.0-gpu-py3 image: See corresponding dockerfile.
  - Volume mounts: Specify a path </path/to/repo/> that points to the repo.
'''

import os
import json

from absl import logging
from absl import app
from absl import flags

import numpy as np

from cv_process import run_experiment


# Delete all FLAGS defined by CV process as we here not want them to be overwritten by the following flags.
for name in list(flags.FLAGS):
    if name in ['load_samples', 'save_samples', 'save_dir', 'save_results', 'result_dir', 'verbosity_level']:
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
                    help='Whether to show the plots.')

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
        "experiment_name": "Long_Track_Sw1",
        # Process parameters
        "x_L": [0.3, 6.2, 0.5, 0.2],
        "C_L": [[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]],
        "t_L": 0,
        "S_w": 1,
        # Boundary
        "x_predTo": 0.6458623971412047,
        # Plot settings (optional)
        "t_range": [0.05, 0.065],
        "y_range": [0.46, 0.56],
    }, {
        # Experiment name
        "experiment_name": "Long_Track_Sw1_denorm",
        # Process parameters
        "x_L": [29.0304,  599.9616, 64.96,  25.984],
        "C_L": [[1.87280916e-03, 1.87280916e-01, 0, 0],
                [1.87280916e-01, 5.61842749e+01, 0, 0],
                [0, 0, 3.37584128e-03, 3.37584128e-01],
                [0, 0, 3.37584128e-01, 1.01275238e+02]],
        "t_L": 0,
        "S_w": 9364.045824,
        # Boundary
        "x_predTo": 62.5,
        # Plot settings (optional)
        "t_range": [0.05, 0.062],
        "y_range": [62, 71]
    }, {
        # Experiment name
        "experiment_name": "Long_Track_Sw10",
        # Process parameters
        "x_L": [0.3, 6.2, 0.5, 0.2],
        "C_L": [[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]],
        "t_L": 0,
        "S_w": 10,
        # Boundary
        "x_predTo": 0.6458623971412047,
    },  {
        # Experiment name
        "experiment_name": "Long_Track_Sw10_denorm",
        # Process parameters
        "x_L": [29.0304, 599.9616, 64.96, 25.984],
        "C_L": [[1.87280916e-03, 1.87280916e-01, 0, 0],
                [1.87280916e-01, 5.61842749e+01, 0, 0],
                [0, 0, 3.37584128e-03, 3.37584128e-01],
                [0, 0, 3.37584128e-01, 1.01275238e+02]],
        "t_L": 0,
        "S_w": 93640.45824,
        # Boundary
        "x_predTo": 62.5,
        "t_range": [0.04, 0.08],
        "y_range": [55, 77]
    },  {
        # Experiment name
        "experiment_name": "Long_Track_Sw100",
        # Process parameters
        "x_L": [0.3, 6.2, 0.5, 0.2],
        "C_L": [[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]],
        "t_L": 0,
        "S_w": 100,
        # Boundary
        "x_predTo": 0.6458623971412047,
        # Plot settings (optional)
        "t_range": [0.0, 0.25],
        "y_range": [0.2, 1.3],
    },  {
        # Experiment name
        "experiment_name": "Long_Track_Sw100_denorm",
        # Process parameters
        "x_L": [29.0304, 599.9616, 64.96, 25.984],
        "C_L": [[1.87280916e-03, 1.87280916e-01, 0, 0],
                [1.87280916e-01, 5.61842749e+01, 0, 0],
                [0, 0, 3.37584128e-03, 3.37584128e-01],
                [0, 0, 3.37584128e-01, 1.01275238e+02]],
        "t_L": 0,
        "S_w": 9364054.5824,
        # Boundary
        "x_predTo": 62.5,
        # Plot settings (optional)
        "t_range": [0.0, 0.25],
        "y_range": [0, 150],
    },  {
        # Experiment name
        "experiment_name": "Long_Track_Sw300",
        # Process parameters
        "x_L": [0.3, 6.2, 0.5, 0.2],
        "C_L": [[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]],
        "t_L": 0,
        "S_w": 300,
        # Boundary
        "x_predTo": 0.6458623971412047,
        # Plot settings (optional)
        "t_range": [0.0, 0.25],
        "y_range": [0.0, 1.5],
    }, {
        # Experiment name
        "experiment_name": "Long_Track_High_Initial_Noise",
        # Process parameters
        "x_L": [0.3, 6.2, 0.5, 0.2],
        "C_L": [[3.2E-3, 5.9E-4, 0, 0], [5.9E-4, 5.3, 0, 0], [0, 0, 3.2E-5, 5.9E-3], [0, 0, 5.9E-3, 5.3]],
        "t_L": 0,
        "S_w": 10,
        # Boundary
        "x_predTo": 0.6458623971412047,
    }, {
        # Experiment name
        "experiment_name": "Long_Track_High_tL",
        # Process parameters
        "x_L": [0.3, 6.2, 0.5, 0.2],
        "C_L": [[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]],
        "t_L": 5,
        "S_w": 10,
        # Boundary
        "x_predTo": 0.6458623971412047,
    }, {
        # Experiment name
        "experiment_name": "Short_Track_Sw18",
        # Process parameters
        "x_L": [0.3, 6.2, 0.5, 0.2],
        "C_L": [[3.2E-7, 5.9E-5, 0, 0], [5.9E-5, 5.3E-2, 0, 0], [0, 0, 3.2E-7, 5.9E-5], [0, 0, 5.9E-5, 5.3E-2]],
        "t_L": 0,
        "S_w": 18,
        # Boundary
        "x_predTo": 0.6458623971412047,
        # Plot settings (optional)
    }, {
        # Experiment name
        "experiment_name": "Long_Track_Sw10_slow",
        # Process parameters
        "x_L": [0.3, 3.2, 0.5, 0.2],
        "C_L": [[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]],
        "t_L": 0,
        "S_w": 10,
        # Boundary
        "x_predTo": 0.6458623971412047,
        # Plot settings (optional)
        "t_range": [0.07, 0.3],
        "y_range": [0.0, 1.5],
    },  {
        # Experiment name
        "experiment_name": "Long_Track_Sw1_slow_denorm",
        # Process parameters
        "x_L": [29.0304, 400, 64.96, 25.984],
        "C_L": [[1.87280916e-03, 1.87280916e-01, 0, 0],
                [1.87280916e-01, 5.61842749e+01, 0, 0],
                [0, 0, 3.37584128e-03, 3.37584128e-01],
                [0, 0, 3.37584128e-01, 1.01275238e+02]],
        "t_L": 0,
        "S_w": 9364.045824,
        # Boundary
        "x_predTo": 62.5,
        # Plot settings (optional)
        "t_range": [0.07, 0.105],
        "y_range": [60, 75],
    },
]


def main(args):
    del args

    # setup logging
    logging.set_verbosity(logging.FLAGS.verbosity_level)

    # define the experiments to execute by name
    #experiments_name_list = ['Long_Track_Sw1', 'Long_Track_Sw100', 'Long_Track_Sw300']
    experiments_name_list = ['Long_Track_Sw1_denorm', 'Long_Track_Sw10_denorm', 'Long_Track_Sw100_denorm', 'Long_Track_Sw1_slow_denorm']
    #experiments_name_list = ['Long_Track_Sw100_denorm']

    # get the configs
    experiments_list = get_experiments_by_name(experiments_name_list)

    # add the defaults (if necessary)
    add_defaults(experiments_list)

    # run the experiments and store the configs
    for config in experiments_list:
        store_config(config)
        convert_to_numpy(config)  # convert the configs entries to numpy arrays
        logging.info('Running experiment {}.'.format(config['experiment_name']))
        del config['experiment_name']  # name cannot be passed to run_experiment
        run_experiment(**config)


def get_experiments_by_name(names_ls):
    """Creates a list with the experiments from experiments_config that are named in names_ls.

    :param names_ls: A list of names of experiments on experiments_config that should be used for evaluations.

    :return:
        experiments_list: A list of configs for the chosen experiments.
    """
    experiments_list = []
    for idx, elem in enumerate(experiments_config):
        if elem['experiment_name'] in names_ls:
            experiments_list.append(elem)
    if not experiments_list:
        raise ValueError('No experiments passed for evaluation.')

    return experiments_list


def add_defaults(experiments_list):
    """Add defaults to the configs.

    :param experiments_list: A list of configs for the chosen experiments.
    """
    for experiment_config in experiments_list:
        if (FLAGS.save_samples or FLAGS.load_samples) and "save_path" not in experiment_config.keys():
            experiment_config['save_path'] = os.path.join(FLAGS.save_dir, experiment_config['experiment_name'] + '.npz')
            experiment_config['save_samples'] = FLAGS.save_samples
        if FLAGS.result_dir is not None and "result_dir" not in experiment_config.keys():
            result_dir = os.path.join(FLAGS.result_dir, experiment_config['experiment_name'])
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            experiment_config['result_dir'] = result_dir
            experiment_config['save_results'] = FLAGS.save_results
        if FLAGS.load_samples and "load_samples" not in experiment_config.keys():
            experiment_config['load_samples'] = FLAGS.load_samples
        experiment_config['no_show'] = FLAGS.no_show


def convert_to_numpy(config):
    """Converts list in configs to np.arrays.

    :param config: A dict describing an experiment's config.
    """
    config['x_L'] = np.array(config['x_L'])
    config['C_L'] = np.array(config['C_L'])


def store_config(config):
    """Saves the config to file.

    :param config: A dict describing an experiment's config.
    """
    if FLAGS.save_results:
        config_file = os.path.join(config['result_dir'], "config.json")
        with open(config_file, 'w') as outfile:
            json.dump(config, outfile, indent=4)


if __name__ == "__main__":
    app.run(main)

"""Functions used to run a list of first-passage time experiments for different CV or CA processes.

"""

import os
import json

import numpy as np


def get_experiments_by_name(names_ls, experiments_config):
    """Creates a list with the experiments from experiments_config that are named in names_ls.

    :param names_ls: A list of names of experiments on experiments_config that should be used for evaluations. Keys to
         experiments_config.
    :param experiments_config: A dict containing the descriptions of all defined experiments.

    :returns:
        experiments_list: A list of configs for the chosen experiments.
    """
    experiments_list = []
    for idx, elem in enumerate(experiments_config):
        if elem['experiment_name'] in names_ls:
            experiments_list.append(elem)
    if not experiments_list:
        raise ValueError('No experiments passed for evaluation.')

    return experiments_list


def add_defaults(experiments_list, flags):
    """Add defaults to the configs.

    :param experiments_list: A list of configs for the chosen experiments.
    :param flags: An abseil.flags.FLAGS object.
    """
    for experiment_config in experiments_list:
        if (flags.save_samples or flags.load_samples) and "save_path" not in experiment_config.keys():
            experiment_config['save_path'] = os.path.join(flags.save_dir, experiment_config['experiment_name'] + '.npz')
            experiment_config['save_samples'] = flags.save_samples
        if flags._result_dir is not None and "_result_dir" not in experiment_config.keys():
            result_dir = os.path.join(flags._result_dir, experiment_config['experiment_name'])
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            experiment_config['_result_dir'] = result_dir
            experiment_config['save_results'] = flags.save_results
        if flags.load_samples and "load_samples" not in experiment_config.keys():
            experiment_config['load_samples'] = flags.load_samples
        experiment_config['no_show'] = flags.no_show


def convert_to_numpy(config):
    """Converts list in configs to np.arrays.

    :param config: A dict describing an experiment's config.
    """
    config['x_L'] = np.array(config['x_L'])
    config['C_L'] = np.array(config['C_L'])


def store_config(config, save_results):
    """Saves the config to file.

    :param config: A dict describing an experiment's config.
    :param save_results: A Boolean, whether to save the results.
    """
    if save_results:
        config_file = os.path.join(config['_result_dir'], "config.json")
        with open(config_file, 'w') as outfile:
            json.dump(config, outfile, indent=4)

"""Functions used to measure teh computational times for the approaches to solve first-passage time problems.

"""

from absl import logging
from timeit import time

import numpy as np


def measure_computation_times(model_class_ls, model_attributes_ls, t_range, num_runs=10):  # TODO: Das auch für CA? und die anderen?
    """Measure the computational times required for calculating the PDF, CDF, PPF, EV, Var.

    Note that all times are measured including the initialization times for the respective classes. Thus, calling
    function from already build instances may be much faster.

    :param num_runs: An integer, the number of runs to average the computational times.
    """

    def measure_comp_times(model_class, model_attributes, function_attribute, function_values=None):
        """A general function for measuring computational times.

        :param model_class:
        :param model_attributes:
        :param function_attribute:
        :param function_values:
        :return:
        """
        comp_times = []
        if function_values is not None:
            for v in function_values:
                start_time = time.time()
                model_instance = model_class(**model_attributes)
                getattr(model_instance, function_attribute)(v)
                comp_times.append(1000 * (time.time() - start_time))
        else:
            for i in range(num_runs):
                start_time = time.time()
                model_instance = model_class(**model_attributes)
                getattr(model_instance, function_attribute)
                comp_times.append(1000 * (time.time() - start_time))
        logging.info('Computational time {0} {1} (means, stddev): {2}ms, {3}ms'.format(function_attribute,
                                                                                       model_instance.name,
                                                                                       np.mean(comp_times),
                                                                                       np.std(comp_times)))
        return np.array(comp_times)

    # for pdf & cdf
    t_values = np.random.uniform(low=t_range[0], high=t_range[1], size=num_runs)
    for model_class, model_attributes in zip(model_class_ls, model_attributes_ls):
        measure_comp_times(model_class, model_attributes, 'pdf', t_values)
        measure_comp_times(model_class, model_attributes, 'cdf', t_values)

    # for ppf
    q_values = np.random.uniform(low=0.0, high=1.0, size=num_runs)
    for model_class, model_attributes in zip(model_class_ls, model_attributes_ls):
        measure_comp_times(model_class, model_attributes, 'pdf', q_values)

    # for ev and var
    for model_class, model_attributes in zip(model_class_ls, model_attributes_ls):
        measure_comp_times(model_class, model_attributes, 'ev')
        measure_comp_times(model_class, model_attributes, 'var')


def measure_computation_times_lgssm(x_L, C_L, t_L, S_w, x_predTo, num_runs=10):
    """Measure the computational times required for calculating the PDF, CDF, PPF, EV, Var.

    Note that all times are measured including the initialization times for the respective classes. Thus, calling
    function from already build instances may be much faster.

    :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, vel_x, pos_y, vel_y].
    :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, position of the boundary.
    :param num_runs: An integer, the number of runs to average the computational times.
    """



    # build a general function for measuring computational times

    def measure_comp_times(model_class, model_attributes, attribute, values=None):
        comp_times = []
        if values is not None:
            for v in values:
                start_time = time.time()
                model_instance = model_class(x_L, C_L, S_w, x_predTo, t_L)
                getattr(model_instance, attribute)(v)
                comp_times.append(1000 * (time.time() - start_time))
        else:
            start_time = time.time()
            model_instance = model_class(x_L, C_L, S_w, x_predTo, t_L)
            getattr(model_instance, attribute)
            comp_times.append(1000 * (time.time() - start_time))
        logging.info('Computational time {0} {1} (means, stddev): {2}ms, {3}ms'.format(attribute,
                                                                                       model_instance.name,
                                                                                       np.mean(comp_times),
                                                                                       np.std(comp_times)))
        return np.array(comp_times)

    # for pdf & cdf
    theta_t = (x_predTo - x_L[0]) / x_L[1] + t_L
    t_values = np.random.uniform(low=t_L, high=1.5 * theta_t, size=num_runs)

    measure_comp_times(MCHittingTimeModel, 'pdf', t_values)  # TODO: t_range einfügen, überall wo MC models vorkommen!
    measure_comp_times(TaylorHittingTimeModel, 'pdf', t_values)
    measure_comp_times(EngineeringApproxHittingTimeModel, 'pdf', t_values)
    measure_comp_times(MCHittingTimeModel, 'cdf', t_values)
    measure_comp_times(TaylorHittingTimeModel, 'cdf', t_values)
    measure_comp_times(EngineeringApproxHittingTimeModel, 'cdf', t_values)

    # for ppf
    q_values = np.random.uniform(low=0.0, high=1.0, size=num_runs)

    measure_comp_times(MCHittingTimeModel, 'ppf', q_values)
    measure_comp_times(TaylorHittingTimeModel, 'ppf', q_values)
    measure_comp_times(EngineeringApproxHittingTimeModel, 'ppf', q_values)

    # for ev and var
    measure_comp_times(MCHittingTimeModel, 'ev')
    measure_comp_times(TaylorHittingTimeModel, 'ev')
    measure_comp_times(EngineeringApproxHittingTimeModel, 'ev')
    measure_comp_times(MCHittingTimeModel, 'var')
    measure_comp_times(TaylorHittingTimeModel, 'var')
    measure_comp_times(EngineeringApproxHittingTimeModel, 'var')
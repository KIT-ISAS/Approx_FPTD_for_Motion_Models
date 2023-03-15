"""Functions used to measure teh computational times for the approaches to solve first-passage time problems.

"""

from absl import logging
from timeit import time

import numpy as np


def measure_computation_times(model_class_ls, model_attributes_ls, t_range, num_runs=10):
    """Measure the computational times required for calculating the PDF, CDF, PPF, EV, Var.

    Note that all times are measured including the initialization times for the respective classes. Thus, calling
    function from already build instances may be much faster.

    :param model_class_ls: A list of AbstractHittingTimeModel class, the classes which computational times should be
        measured.
    :param model_attributes_ls: A list of lists containing args to initialize classes in model_class_ls.
    :param t_range: A list of length 2 representing the plot limits for the first passage time (used to find suitable
        values to insert in the pdf or cdf methods).
    :param num_runs: An integer, the number of runs to average the computational times.
    """

    def measure_comp_times(model_class, model_attributes, function_attribute, function_values=None):
        """A general function for measuring computational times.

        :param model_class: A AbstractHittingTimeModel class, the class which computational times should be measured.
        :param model_attributes: A list or dict, args to initialize model_class.
        :param function_attribute: A string, the name of the method or property of the class to call for measuring the
            computational times
        :param function_values: None or a float, value for method call. Use None if calling a property.
        """
        comp_times = []
        if function_values is not None:
            for v in function_values:
                start_time = time.time()
                model_instance = model_class(*model_attributes)
                getattr(model_instance, function_attribute)(v)
                comp_times.append(1000 * (time.time() - start_time))
        else:
            for i in range(num_runs):
                start_time = time.time()
                model_instance = model_class(*model_attributes)
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

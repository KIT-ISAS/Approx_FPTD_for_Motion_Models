"""

TODO

"""

import os
from abc import ABC, abstractmethod

from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from absl import logging
from scipy.stats import norm, uniform, rv_histogram


class AbstractArrivalDistribution(ABC):
    """A base class for all arrival distributions.

    This includes

        - temporal and spatial arrival distributions (arrival time & arrival location),
        - hitting (first-passage) time approximations (for the CV and CA model, and the wiener process with drift)
            as well as generic distributions (Uniform, Normal, GaussianMixture) that can be instantiated with given
            parameter values (point estimates, corresponding variances, weights, etc.).

    Arrival distributions are one-dimensional distribution with batch shape support, i.e., each component of the
    distribution is a single, mutually independent one-dimensional distribution.
    """
    def __init__(self, name='AbstractArrivalDistribution'):
        """Initializes the distribution.

        :param name: String, the (default) name for the distribution.
        """
        self.name = name

    @property
    @abstractmethod
    def batch_size(self):
        """The batch size of the distribution.

        :returns: An integer, the batch size.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def ev(self):
        """The expected value of the distribution.

        :returns: A float or a np.array of shape [batch_size], the expected value.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def var(self):
        """The variance of the distribution.

        :returns: A float or a np.array of shape [batch_size], the variance.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def third_central_moment(self):
        """The third central moment of the distribution.

        :returns: A float or a np.array of shape [batch_size], the third central moment.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def stddev(self):
        """The standard deviation of the distribution.

        :returns: A float or a np.array of shape [batch_size], the standard deviation.
        """
        return np.sqrt(self.var)

    @property
    def second_moment(self):
        """The second moment of the distribution.

        :returns: A float or a np.array of shape [batch_size], the second moment.
        """
        return self.var + self.ev ** 2

    @property
    def third_moment(self):
        """The third moment of the distribution.

        :returns: A float or a np.array of shape [batch_size], the third moment.
        """
        return self.third_central_moment + 3 * self.ev * self.var + self.ev ** 3

    @property
    def skew(self):
        """The skewness of the distribution.

        :returns: A float or a np.array of shape [batch_size], the skewness.
        """
        return self.third_central_moment / self.stddev ** 3

    @abstractmethod
    def pdf(self, x):
        """The probability density function (PDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the PDF for x:
            - If the distribution is scalar (batch_size = 1)
                - and x is scalar, then returns a float,
                - and x is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and x is scalar, then returns a np.array of shape [batch_size],
                - and x is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def cdf(self, x):
        """The cumulative distribution function (CDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the CDF for x:
            - If the distribution is scalar (batch_size = 1)
                - and x is scalar, then returns a float,
                - and x is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and x is scalar, then returns a np.array of shape [batch_size],
                - and x is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the distribution.

        :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the distribution.

        :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @staticmethod
    def batch_atleast_3d(arys):
        """A replacement for np.atleast_3d that returns the array with new axis 0 (if a new axis is added) instead at
        axis -1.

        View inputs as arrays with at least three dimensions.

        :param arys: arys1, arys2, …array_like
            One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have three
            or more dimensions are preserved.

        :returns: res1, res2, …ndarray
            An array, or list of arrays, each with a.ndim >= 3. Copies are avoided where possible, and views with three
            or more dimensions are returned. For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1),
            and a 2-D array of shape (M, N) becomes a view of shape (1, M, N).
        """
        def _atleast_3d_dispatcher(*arys):
            return arys

        arys = _atleast_3d_dispatcher(arys)

        res = []
        for ary in arys:
            ary = np.asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1, 1, 1)
            elif ary.ndim == 1:
                result = ary[np.newaxis, :, np.newaxis]
            elif ary.ndim == 2:
                result = ary[np.newaxis, :, :]
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res

    def batch_size_one_function(func):
        """A decorator for functions that only support a batch size of 1.

        Assure that the batch size of the class equals 1 when calling the decorated function.

        :param func: A callable, the function to be decorated.

        :returns: A callable, the decorator.
        """
        @wraps(func)
        def assure_batch_size_one(self, *args, **kwargs):
            if self.batch_size != 1:
                raise ValueError(
                    "The function {} does not support batch-wise processing, i.e., batch_size must be 1.".format(
                        func.__name__))
            return func(self, *args, **kwargs)

        return assure_batch_size_one

    def check_setitem(setitem_func):
        """A decorator for setitem functions.

        Assures that the indexed class and the class from which the values to be taken are of the same type. Converts
        indices to slice.

        :param setitem_func: A callable, the setitem function to be decorated.

        :returns: A callable, the decorator.
        """
        @wraps(setitem_func)
        def check_same_type_convert_slices(self, indices, values):
            # sanity checks
            if type(values) != type(self):
                raise ValueError('When assigning values to {}, both instances must be of same class, but '
                                 'object to be assigned is of class {} and assigned values are of class {}.'.format(
                    self.__class__.__name__, type(self), type(values)))

            if not isinstance(indices, slice):
                indices = np.array(indices)  # e.g. if it is a list of integers or Booleans
            return setitem_func(self, indices, values)

        return check_same_type_convert_slices

    def check_density_input_dim(pdf_cdf_func):
        """A decorator for PDF or CDF functions.

        Assures that the parameter of the the CDF of PDF function satisfies the requirements on its shape if the batch
        size is not equal to 1.

        :param pdf_cdf_func: A callable, the PDF or CDF function to be decorated.

        :returns: A callable, the decorator.
        """
        @wraps(pdf_cdf_func)
        def check_input_valid(self, x):
            if self.batch_size > 1 and not np.isscalar(x) and x.ndim != 2:
                raise ValueError(
                    'If batch size > 1, to avoid ambiguities, only scalars and np.arrays of shape [batch_size, sample_size] are supported.')
            return pdf_cdf_func(self, x)

        return check_input_valid

    @batch_size_one_function
    def plot_quantile_function(self,
                               q_min=0.005,
                               q_max=0.995,
                               save_results=False,
                               result_dir=None,
                               for_paper=True,
                               y_label='Time in s'):
        """Plot the quantile function.
        :param q_min: A float, the smallest value of the confidence plot range.
        :param q_max: A float, the highest value of the confidence plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        :param y_label: String, the y_label of the plot.
        """
        plot_q = np.arange(q_min, q_max, 0.01)
        plot_quant = [self.ppf(q) for q in plot_q]
        plt.plot(plot_q, plot_quant)
        plt.xlabel('Confidence level')
        plt.ylabel(y_label)
        if not for_paper:
            plt.title('Quantile Function (Inverse CDF) for ' + self.name)

        if save_results:
            plt.savefig(result_dir + self.name + '_quantile_function.pdf')
            plt.savefig(result_dir + self.name + '_quantile_function.png')
        plt.show()

    @batch_size_one_function
    def plot_first_hitting_time_distribution(self, x_min=None, x_max=None, save_results=False, result_dir=None,
                                             for_paper=False, x_label='Time in s'):  # TODO Name
        """Plots the ejection distribution (PDF, CDF, Ev, Stddev).

        :param x_min: A float or np.array of shape [batch_size], the smallest value of the confidence plot range.
        :param x_max: A float or np.array of shape [batch_size], the highest value of the confidence plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        :param x_label: String, the x_label of the plot.
        """
        # TODO: Die plot functions müssen noch auf batch betrieb umgestellt werden, z. B. als abstract machen, docstrings
        # TODO: Welcher dieser Funktionnen verwenden

        fig, ax1 = plt.subplots()

        if x_min is None:
            x_min = self.ppf(0.0005)
        if x_max is None:
            x_max = self.ppf(0.9995)

        # sanity checks on t_min, t_max
        if np.ndim(self.ev) == 0:
            if np.ndim(x_min) != 0 or np.ndim(x_max) != 0:
                raise ValueError('Both x_min and x_max need to be scalars or of size 1.')
        elif x_min.shape[0] != self.ev.shape[0] or x_max.shape[0] != self.ev.shape[0]:
            raise ValueError('Both x_min and x_max need to be of shape [batch_size].')

        plot_t = np.linspace(x_min, x_max, num=1000).T  # shape [batch_size, n] or [n]

        # get the current color cycle
        color_cycle = plt.rcParams['axes.prop_cycle']

        ax2 = ax1.twinx()
        plot_f = self.cdf(plot_t)
        ax2.plot(plot_t.T, plot_f.T)
        plot_f = self.pdf(plot_t)
        # reset the color cycle
        ax1.set_prop_cycle(color_cycle)
        ax1.plot(plot_t.T, plot_f.T)
        # reset the color cycle
        ax1.set_prop_cycle(color_cycle)
        ax2.vlines(self.ev, 0, 1, linestyle='dashed', colors=color_cycle.by_key()['color'])
        ax2.vlines([self.ev - self.stddev], 0, 1,
                   colors=color_cycle.by_key()['color'], linestyle='dashdot')
        ax2.vlines([self.ev + self.stddev], 0, 1,
                   colors=color_cycle.by_key()['color'], linestyle='dashdot')

        # add legend manually
        lines = [mlines.Line2D([], [], color='black', linestyle='dashed'),
                 mlines.Line2D([], [], color='black', linestyle='dashdot')]
        labels = ['Mean', 'Mean +/- Stddev']
        ax2.legend(lines, labels)

        ax2.set_ylim(0, 1.05)
        ax1.set_ylim(0, None)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("PDF")
        ax2.set_ylabel("CDF")
        if not for_paper:
            plt.title("Ejection Distribution for " + self.name)

        if save_results:
            plt.savefig(os.path.join(result_dir, self.name + '_ejection_distr.pdf'))
            plt.savefig(os.path.join(result_dir, self.name + '_ejection_distr.png'))
        plt.show()

    def __getitem__(self, indices):
        """Takes elements along the batch shape. Use this for fancy indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to extract.

        :returns: A copy of the distribution with the extracted values.
        """
        if not isinstance(indices, slice):
            indices = np.array(indices)  # e.g. if it is a list or boolean

        obj = type(self).__new__(type(self))  # create a new object of same type
        obj._left_hand_indexing(indices, self)  # assign values to obj
        return obj

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self.name = values.name

    # to make them accessible from outside (and suppress ugly IDE warnings), needs to be done at the end of the class
    batch_size_one_function = staticmethod(batch_size_one_function)
    check_setitem = staticmethod(check_setitem)
    check_density_input_dim = staticmethod(check_density_input_dim)


class AbstractNormalArrivalDistribution(AbstractArrivalDistribution, ABC):
    """A base class for all Gaussian arrival distributions.

    This includes

        - temporal and spatial arrival distributions (arrival time & arrival location),
        - hitting (first-passage) time approximations (for the CV and CA model, and the wiener process with drift)
            as well as generic distributions that can be instantiated with given parameter values (point estimates,
            and corresponding variances).

    Arrival distributions are one-dimensional distribution with batch shape support, i.e., each component of the
    AbstractNormalArrivalDistribution is a single, mutually independent one-dimensional Gaussian distribution.
    """
    def __init__(self, ev, var, name='AbstractNormalArrivalDistribution', **kwargs):
        """Initializes the distribution.

        :param ev: A float or np.array of shape [batch_size], the expected value of the distribution.
        :param var: A float or np.array of shape [batch_size], the variance of the distribution.
        :param name: String, the (default) name for the distribution.
        """
        # sanity checks
        if not np.array_equal(np.atleast_1d(ev).shape, np.atleast_1d(var).shape):
            raise ValueError('Shapes of ev and var must be equal, or both must be floats.')

        super().__init__(name=name, **kwargs)

        self._ev = np.atleast_1d(ev)
        self._var = np.atleast_1d(var)

    @property
    def batch_size(self):
        """The batch size of the distribution.

        :returns: An integer, the batch size.
        """
        return len(self._ev)

    @property
    def ev(self):
        """The expected value of the distribution.

        :returns: A float or a np.array of shape [batch_size], the expected value.
        """
        return np.squeeze(self._ev)

    @property
    def var(self):
        """The variance of the distribution.

        :returns: A float or a np.array of shape [batch_size], the variance.
        """
        return np.squeeze(self._var)

    @property
    def third_central_moment(self):
        """The third central moment of the distribution.

        :returns: A float or a np.array of shape [batch_size], the third central moment.
        """
        return np.squeeze(np.zeros(self.batch_size))  # Gaussian third central moment

    @AbstractArrivalDistribution.check_density_input_dim
    def pdf(self, x):
        """The probability density function (PDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the PDF for x:
            - If the distribution is scalar (batch_size = 1)
                - and x is scalar, then returns a float,
                - and x is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and x is scalar, then returns a np.array of shape [batch_size],
                - and x is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        return np.squeeze(norm.pdf(x, loc=self.ev, scale=self.stddev))

    @AbstractArrivalDistribution.check_density_input_dim
    def cdf(self, x):
        """The cumulative distribution function (CDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the CDF for x:
            - If the distribution is scalar (batch_size = 1)
                - and x is scalar, then returns a float,
                - and x is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and x is scalar, then returns a np.array of shape [batch_size],
                - and x is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        return np.squeeze(norm.cdf(x, loc=self.ev, scale=self.stddev))

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the distribution.

        :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the distribution.

        :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
        """
        return np.squeeze(norm.ppf(q, loc=self.ev, scale=self.stddev))

    # def __getitem__(self, indices):
    #     """Takes elements along the batch shape. Use this for fancy indexing (e.g., new_distr = distr[:2]).
    #
    #     :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to extract.
    #
    #     :returns: A copy of the distribution with the extracted values.
    #     """
    #     if not isinstance(indices, slice):
    #         indices = np.array(indices)  # e.g. if it is a list or boolean
    #     return type(self)(self._ev[indices], self._var[indices], self.name)

    @AbstractArrivalDistribution.check_setitem
    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._ev[indices] = values.ev
        self._var[indices] = values.var

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._ev = values.ev[indices]
        self._var = values.var[indices]
        super()._left_hand_indexing(indices, values)


class AbstractUniformArrivalDistribution(AbstractArrivalDistribution, ABC):
    """A base class for all Uniform arrival distributions.

    This includes

        - temporal and spatial arrival distributions (arrival time & arrival location),
        - hitting (first-passage) time approximations (for the CV and CA model) as well as generic distributions that
            can be instantiated with given parameter values (point estimates).

    Arrival distributions are one-dimensional distribution with batch shape support, i.e., each component of the
    AbstractUniformArrivalDistribution is a single, mutually independent one-dimensional Uniform distribution.
    """
    def __init__(self, point_prediction, window_length, a=0.5, name='AbstractUniformArrivalDistribution', **kwargs):
        """Initializes the distribution.

        Illustration of the parameters point_prediction, window_length, and a.

                /\ p(x)                                          /\ p(x)
                |    ____w_____                                  |    ____w_____
                |   |         |                                  |   |          |
                ---------|---------->  x                         ----|-------------->  x
                point prediction (a = 0.5)                  point prediction (a = 0)

        :param point_prediction: A float or np.array of shape [batch_size], the point prediction.
        :param window_length: A float or np.array of shape [batch_size], the window length of the distribution.
        :param a: A float or np.array of shape [batch_size], the ratio of the window length, where the point prediction
            is located.
        :param name: String, the (default) name for the distribution.
        """
        self._point_prediction = np.atleast_1d(point_prediction)
        self._window_length = np.broadcast_to(window_length, shape=self.batch_size)  # this itself raises an error if not
            # compatible
        self._a = np.broadcast_to(a, shape=self.batch_size)  # this itself raises an error if not compatible
        super().__init__(name=name, **kwargs)

    @property
    def point_prediction(self):
        """The point prediction.

        :returns: A float or np.array of shape [batch_size], the point prediction.
        """
        return np.squeeze(self._point_prediction)

    @property
    def window_length(self):
        """The window length of the distribution.

        :returns: window_length: A float or np.array of shape [batch_size], the window length of the distribution.
        """
        return np.squeeze(self._window_length)

    @window_length.setter
    def window_length(self, value):
        """The setter of the window length. Along with self.a, window_length is a hyperparameter of the distribution and
         therefore we may want to adjust it after initializing.

        :param value: A float or np.array of shape [batch_size], the window length of the distribution.
        """
        self._window_length = np.broadcast_to(value, shape=self.batch_size)  # this itself raises an error if not
            # compatible

    @property
    def a(self):
        """The ratio of the window length, where the point prediction is located.

        :returns: A float or np.array of shape [batch_size], the ratio of the window length, where the point prediction
            is located.
        """
        return np.squeeze(self._a)

    @a.setter
    def a(self, value):
        """The setter of the ratio of the window length, where the point prediction is located. Along with
        self.window_length, it is a hyperparameter of the distribution and therefore we may want to adjust it after
        initializing.

        :param value: A float or np.array of shape [batch_size], the ratio of the window length, where the point prediction
            is located.
        """
        self._a = np.broadcast_to(value, shape=self.batch_size)  # this itself raises an error if not compatible

    @property
    def x_max(self):
        """The maximum value where that has nonzero probability mass.

        :returns: A float or np.array of shape [batch_size], the maximum value where that has nonzero probability mass.
        """
        return np.squeeze(self._point_prediction + self._a * self._window_length)

    @property
    def x_min(self):
        """The minimum value where that has nonzero probability mass.

        :returns: A float or np.array of shape [batch_size], the maximum value where that has nonzero probability mass.
        """
        return np.squeeze(self._point_prediction - self._a * self._window_length)

    @property
    def batch_size(self):
        """The batch size of the distribution.

        :returns: An integer, the batch size.
        """
        return len(self._point_prediction)

    @property
    def ev(self):
        """The expected value of the distribution.

        :returns: A float or a np.array of shape [batch_size], the expected value.
        """
        return np.squeeze(uniform.mean(loc=self.x_min, scale=self.x_max - self.x_min))

    @property
    def var(self):
        """The variance of the distribution.

        :returns: A float or a np.array of shape [batch_size], the variance.
        """
        return np.squeeze(uniform.var(loc=self.x_min, scale=self.x_max - self.x_min))

    @property
    def third_central_moment(self):
        """The third central moment of the distribution.

        :returns: A float or a np.array of shape [batch_size], the third central moment.
        """
        return np.squeeze(np.zeros(self.batch_size))  # Uniform third central moment

    @AbstractArrivalDistribution.check_density_input_dim
    def pdf(self, x):
        """The probability density function (PDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the PDF for x:
            - If the distribution is scalar (batch_size = 1)
                - and x is scalar, then returns a float,
                - and x is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and x is scalar, then returns a np.array of shape [batch_size],
                - and x is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        return np.squeeze(uniform.pdf(x, loc=self.x_min, scale=self.x_max - self.x_min))

    @AbstractArrivalDistribution.check_density_input_dim
    def cdf(self, x):
        """The cumulative distribution function (CDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the CDF for x:
            - If the distribution is scalar (batch_size = 1)
                - and x is scalar, then returns a float,
                - and x is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and x is scalar, then returns a np.array of shape [batch_size],
                - and x is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        return np.squeeze(uniform.cdf(x, loc=self.x_min, scale=self.x_max - self.x_min))

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the distribution.

        :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the distribution.

        :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
        """
        return np.squeeze(uniform.ppf(q, loc=self.x_min, scale=self.x_max - self.x_min))

    # def __getitem__(self, indices):
    #     """Takes elements along the batch shape. Use this for fancy indexing (e.g., new_distr = distr[:2]).
    #
    #     :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to extract.
    #
    #     :returns: A copy of the distribution with the extracted values.
    #     """
    #     if not isinstance(indices, slice):
    #         indices = np.array(indices)  # e.g. if it is a list or boolean
    #     return type(self)(self._point_prediction[indices], self._window_length[indices], self._a[indices], self.name)

    @AbstractArrivalDistribution.check_setitem
    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._point_prediction[indices] = values.point_prediction
        self._window_length[indices] = values.window_length
        self._a[indices] = values.a

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._point_prediction = values.point_prediction[indices]
        self._window_length = values.window_length[indices]
        self._a = values.a[indices]
        super()._left_hand_indexing(indices, values)


class AbstractMCArrivalDistribution(AbstractArrivalDistribution, ABC):
    """A base class for all Monte-Carlo arrival distributions.

    Wraps the histogram derived by a Monte-Carlo approach to a distribution using scipy.stats.rv_histogram.

    This includes

        - temporal and spatial arrival distributions (arrival time & arrival location),
        - hitting (first-passage) time approximations (for the CV and CA model, and the wiener process with drift).

    Note that in contrast the all other child classes of AbstractArrivalDistribution, this distribution does not
    support a batch dimension.
    """
    def __init__(self, samples, range, bins=100, name="AbstractMCArrivalDistribution", **kwargs):
        """Initialize the distribution.

        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param range: A list of length 2 representing the limits for the histogram (the number of bins within range will
            correspond to bins).
        :param bins: An integer, the number of bins to use to represent the histogram within range.
        :param name: String, the (default) name for the distribution
        """
        # sanity checks
        if samples.ndim != 1:
            raise ValueError(
                'samples must be a one-dimensional array. Note that {} does not support a batch dimension.'.format(
                    self.__class__.__name__))
        if not np.all(np.isfinite(samples)):
            raise ValueError('{} expects clean samples with all values being finite.'.format(self.__class__.__name__))

        super().__init__(name=name,
                         **kwargs)

        self._samples = samples
        self._range = range
        self._bins = bins
        self._density = self._build_distribution_from_samples(samples, range)

        # TODO: For perfect fit with histogram, adjust when plotting mc_distribution
        # fig, ax1 = plt.subplots()
        # _, b, _ = ax1.hist(samples, bins=bins, density=True)
        # ax1.plot((b[:-1] + b[1:])/2, [self._density.pdf(t) for t in (b[:-1] + b[1:])/2], color='b')
        # plt.show()

    @property
    def samples(self):
        """The Monte Carlo samples representing the distribution.

        :returns: A np.array of shape [num_samples] containing the samples.
        """
        return self._samples

    @property
    def batch_size(self):
        """The batch size of the distribution.

        :returns: An integer, the batch size.
        """
        return 1

    @property
    def ev(self):
        """The expected value of the distribution.

        :returns: A float, the expected value.
        """
        return self._density.mean()

    @property
    def var(self):
        """The variance of the distribution.

        :returns: A float, the variance.
        """
        return self._density.var()

    @property
    def third_central_moment(self):
        """The third central moment of the distribution.

        :returns: A float, the third central moment.
        """
        return self.third_moment - 3 * self.ev * self.var - self.ev ** 3

    @property
    def third_moment(self):
        """The third moment of the distribution.

        :returns: A float, the third moment.
        """
        return self._density.moment(3)

    def pdf(self, x):
        """The probability density function (PDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], the parameter of the distribution.

        :returns: A float or a np.array of shape [sample_size], the value of the PDF for x:
        """
        return self._density.pdf(x)

    def cdf(self, x):
        """The cumulative distribution function (CDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], the parameter of the distribution.

        :returns: A float or a np.array of shape [sample_size], the value of the CDF for x:
        """
        return self._density.cdf(x)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the distribution.

        :param q: A float in [0, 1], the confidence parameter of the distribution.

        :returns: A float, the value of the PPF for q.
        """
        return self._density.ppf(q)

    def _build_distribution_from_samples(self, samples, range):
        """Builds a distribution from the samples based on a scipy stats object.

        :param samples: A np.array of shape [num_samples] containing sampled values.
        :param range: A list of length 2 representing the limits for the histogram (the number of bins within range will
            correspond to bins).

        :returns: A scipy.stats.r_histogram object representing the distribution.
        """
        bins = int(self._bins * (max(samples) - min(samples)) / (range[-1] - range[0]))   # TODO: Auch mit max und min schreiben?
        # we want to have bins samples in the plot window
        hist = np.histogram(samples, bins=bins, density=False)
        return rv_histogram(hist, density=True)

    def __getitem__(self, indices):
        """Takes elements along the batch shape. Use this for fancy indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to extract.

        :returns: A copy of the distribution with the extracted values.
        """
        raise NotImplementedError('Fancy indexing not supported for {}.'.format(self.__class__.__name__))

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        raise NotImplementedError('Fancy indexing not supported for {}.'.format(self.__class__.__name__))


class AbstractGaussianMixtureArrivalDistribution(AbstractArrivalDistribution, ABC):
    """A base class for all Gaussian mixture arrival distributions.

    This includes

        - temporal and spatial arrival distributions (arrival time & arrival location),

    for generic distributions that can be instantiated with given parameter values (point estimates, corresponding
    variances, weights, etc.)

    One-dimensional Gaussian-mixture distribution with batch shape support, i.e., each component of the
    AbstractGaussianMixtureArrivalDistribution is a single, mutually independent one-dimensional Gaussian-mixture
    distribution.
    """
    def __init__(self, mus, sigmas, weights, name='AbstractEjectionGaussianMixtureDistribution', **kwargs):
        """Initializes the distribution.

        :param mus: A np.array of shape [component_size] or [batch_size, component_size], the means of the component
            Gaussian distributions.
        :param sigmas: A np.array of shape [component_size] or [batch_size, component_size], the standard deviations of
            the component Gaussian distributions.
        :param weights: A np.array of shape [component_size] or [batch_size, component_size], the weights
            (probabilities) for the component Gaussian distributions. Weights must be in [0, 1] and sum to 1.
        :param name: String, the (default) name for the distribution.
        """
        # sanity checks
        if not np.array_equal.reduce(
                (np.atleast_2d(mus).shape, np.atleast_2d(sigmas).shape, np.atleast_2d(weights).shape)):
            raise ValueError('Shapes of mus, sigmas, and weights must be equal.')
        if np.any(np.logical_or(weights < 0, weights > 1)):
            raise ValueError('Weights must be in [0, 1].')
        if np.any(np.logical_not(np.isclose(np.sum(weights, axis=1), 1))):
            raise ValueError('Weights must sum up to 1.')

        super().__init__(name=name,
                         **kwargs)

        self._mus = tf.constant(np.atleast_2d(mus), dtype=tf.float32)
        self._sigmas = tf.constant(np.atleast_2d(sigmas), dtype=tf.float32)
        self._weights = tf.constant(np.atleast_2d(weights), dtype=tf.float32)

        self._distr = self._build_dist()

        # for properties
        self._ev = None
        self._var = None

    @property
    def mus(self):
        """The means of the component Gaussians.

        :returns: A np.array of shape [component_size] or [batch_size, component_size], the means of the component
            Gaussian distributions.
        """
        return np.squeeze(self._mus.numpy(), axis=0) if self.batch_size == 1 else self._mus.numpy()

    @property
    def sigmas(self):
        """The standard deviations of the component Gaussians.

        :returns: A np.array of shape [component_size] or [batch_size, component_size], the standard deviations of the
            component Gaussian distributions.
        """
        return np.squeeze(self._sigmas.numpy(), axis=0) if self.batch_size == 1 else self._sigmas.numpy()

    @property
    def weights(self):
        """The weights (probabilities) for the component Gaussians.

        :returns: A np.array of shape [component_size] or [batch_size, component_size], the weights (probabilities) for
            the component Gaussian distributions.
        """
        return np.squeeze(self._weights.numpy(), axis=0) if self.batch_size == 1 else self._weights.numpy()

    @property
    def batch_size(self):
        """The batch size of the distribution.

        :returns: An integer, the batch size.
        """
        return len(self._mus.numpy())

    @property
    def ev(self):
        """The expected value of the distribution.

        :returns: A float or a np.array of shape [batch_size], the expected value.
        """
        if self._ev is None:
            self._ev = self._distr.mean().numpy()
        return np.squeeze(self._ev)

    @property
    def var(self):
        """The variance of the distribution.

        :returns: A float or a np.array of shape [batch_size], the variance.
        """
        if self._var is None:
            self._var = self._distr.variance().numpy()
        return np.squeeze(self._var)

    @property
    def third_central_moment(self):
        """The third central moment of the distribution.

        :returns: A float or a np.array of shape [batch_size], the third central moment.
        """
        return np.squeeze(self.third_moment - 3 * self.ev * self.var - self.ev ** 3)

    @property
    def third_moment(self):
        """The third moment of the distribution.

        :returns: A float or a np.array of shape [batch_size], the third moment.
        """
        return np.squeeze(np.dot(self._mus**3 + 3*self._mus * self._sigmas**2, self._weights))

    @AbstractArrivalDistribution.check_density_input_dim
    def pdf(self, x):
        """The probability density function (PDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the PDF for x:
            - If the distribution is scalar (batch_size = 1)
                - and x is scalar, then returns a float,
                - and x is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and x is scalar, then returns a np.array of shape [batch_size],
                - and x is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        return np.squeeze(self._distr.prob(x).numpy())

    @AbstractArrivalDistribution.check_density_input_dim
    def cdf(self, x):
        """The cumulative distribution function (CDF) of the distribution.

        :param x: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the CDF for x:
            - If the distribution is scalar (batch_size = 1)
                - and x is scalar, then returns a float,
                - and x is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and x is scalar, then returns a np.array of shape [batch_size],
                - and x is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        return np.squeeze(self._distr.cdf(x).numpy())

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the distribution.

        :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the distribution.

        :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
        """
        return np.squeeze(self._distr.quantile(q).numpy())

    def _build_dist(self):
        """Builds a Gaussian mixture distribution based on a tensorflow object.

        :returns: A tensorflow_probability.distributions.MixtureSameFamily object representing the Gaussian mixture.
        """
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=self._weights),
            components_distribution=tfd.Normal(
                loc=self._mus,  # One for each component.
                scale=self._sigmas),
            validate_args=True, allow_nan_stats=True)  # And same here.

    # def __getitem__(self, indices):
    #     """Takes elements along the batch shape. Use this for fancy indexing (e.g., new_distr = distr[:2]).
    #
    #     :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to extract.
    #
    #     :returns: A copy of the distribution with the extracted values.
    #     """
    #     if not isinstance(indices, slice):
    #         indices = np.array(indices)  # e.g. if it is a list or boolean
    #     return type(self)(self._mus[indices], self._sigmas[indices], self._weights, self.name)

    @AbstractArrivalDistribution.check_setitem
    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._mus[indices] = values.mus
        self._sigmas[indices] = values.sigmas
        self._weights[indices] = values.weights

        # delete properties
        self._ev = None
        self._var = None

        # reinitialize tfd
        self._distr = self._build_dist()

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._mus = values.mus[indices]
        self._sigmas = values.sigmas[indices]
        self._weights = values.weights[indices]
        # we do not assign ev and var since they are not too hard to calculate

        # reinitialize tfd
        self._distr = self._build_dist()
        super()._left_hand_indexing(indices, values)


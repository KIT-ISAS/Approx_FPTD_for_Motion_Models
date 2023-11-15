

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from absl import logging
from scipy.stats import norm, uniform, rv_histogram


class AbstractEjectionDistribution(ABC):

    # TODO: Über die Namen können wir uns noch unterhalten
    # TODO: Auf mehrer Files aufteilen, CA mit dazu machen!

    def __init__(self, name='AbstractEjectionDistribution'):
        """Initializes the distribution.

        :param name: String, (default) name for the model.
        """
        self.name = name

        # For properties  # TODO: Braucht man die?
        self._stddev = None
        self._second_moment = None
        self._third_moment = None
        self._skew = None

    @property
    @abstractmethod
    def ev(self):
        """The expected value of the first passage time distribution."""  # TODO. Die ganzen Docstrings sind noch falsch bzw. unangepasst and den allgemeinen nutzen!
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def var(self):
        """The variance of the first passage time distribution."""
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def stddev(self):
        """The standard deviation of the first passage time distribution."""
        return np.sqrt(self.var)

    @property
    def second_moment(self):
        """The second moment of the first passage time distribution."""
        return self.var + self.ev ** 2

    @property
    def third_moment(self):
        """The third moment of the first passage time distribution."""
        return self.third_central_moment + 3 * self.ev * self.var + self.ev ** 3

    @property
    def skew(self):
        """The skew of the first passage time distribution."""
        return self.third_central_moment / self.stddev ** 3

    @abstractmethod
    def pdf(self, t):
        # TODO: Mit x umschreiben!
        """The first passage time distribution (FPTD).
        :param t: A float or np.array, the time parameter of the distribution.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def cdf(self, t):
        """The CDF of the first passage time distribution.
        :param t: A float or np.array, the time parameter of the distribution.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.
        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    # @abstractmethod  # TODO: Support batch-wise processing and activate them again
    # def scale_params(self, length_scaling_factor, time_scaling_factor):
    #     """Scales the parameters of the distribution according to the scaling factor.
    #
    #     :param length_scaling_factor: Float, the scaling factor for lengths.
    #     :param time_scaling_factor: Float, the scaling factor for times.
    #     """
    #     # To be overwritten by subclass
    #     raise NotImplementedError('Call to abstract method.')
    #
    # @abstractmethod
    # def __getitem__(self, indices):
    #     """Takes elements along the batch shape. Use this for fancy indexing, e.g. new_distr = distr[:2].
    #
    #     :param indices: Slices, or list or np.array of integers or Booleans. The indices of the values to extract.
    #
    #     :returns out: A copy of the distribution with the extracted values.
    #     """
    #     # To be overwritten by subclass
    #     raise NotImplementedError('Call to abstract method.')
    #
    # @abstractmethod
    # def __setitem__(self, indices, values):
    #     """Places the elements along the batch shape at the given indices. Use this for fancy indexing,
    #         e.g. distr[:2] = old_distr.
    #
    #     :param indices: Slices, or list or np.array of integers or Booleans. The indices of the values to place.
    #     """
    #     # To be overwritten by subclass
    #     raise NotImplementedError('Call to abstract method.')


    # TODO: Eine Plot funktion um den Verlauf der Tracks zu sehen wäre noch sehr hilfreicg

    def plot_quantile_function(self, q_min=0.005, q_max=0.995, save_results=False, result_dir=None,
                               for_paper=True, y_label='Time in s'):
        """Plot the quantile function.
        :param q_min: A float, the smallest value of the confidence plot range.
        :param q_max: A float, the highest value of the confidence plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        :param y_label: String, the y_label of the plot.
        """
        # TODO: Die plot functions müssen noch auf batch betrieb umgestellt werden, z. B. als abstract machen, docstrings

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

    def plot_first_hitting_time_distribution(self, x_min=None, x_max=None, save_results=False, result_dir=None,
                                             for_paper=False, x_label='Time in s'):
        """Plots the ejection distribution (PDF, CDF, Ev, Stddev).
        :param x_min: A float or np.array of shape [batchsize], the smallest value of the confidence plot range.
        :param x_max: A float or np.array of shape [batchsize], the highest value of the confidence plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        :param x_label: String, the x_label of the plot.
        """
        # TODO: Die plot functions müssen noch auf batch betrieb umgestellt werden , z. B. als abstract machen, docstrings
        fig, ax1 = plt.subplots()

        if x_min is None:
            x_min = self.ppf(0.0005)
        if x_max is None:
            x_max = self.ppf(0.9995)

        # Sanity checks on t_min, t_max
        if np.ndim(self.ev) == 0:
            if np.ndim(x_min) != 0 or np.ndim(x_max) != 0:
                raise ValueError('Both x_min and x_max need to be scalars or of size 1.')
        elif x_min.shape[0] != self.ev.shape[0] or x_max.shape[0] != self.ev.shape[0]:
            raise ValueError('Both x_min and x_max need to be of shape [batchsize].')

        plot_t = np.linspace(x_min, x_max, num=1000).T  # shape [batchsize, n] or [n]

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


class AbstractEjectionNormalDistribution(AbstractEjectionDistribution, ABC):
    """One-dimensional Gaussian distribution with batch shape support, i.e., each component of the distribution is a
     single one-dimensional Gaussian distribution."""

    def __init__(self, ev, var, name='AbstractEjectionNormalDistribution', **kwargs):
        """Initializes the distribution.

        :param: ev: A np.array of shape [batch_size], the expected values of the distribution.
        :param: var: A np.array of shape [batch_size], the variance of the distribution.
        :param name: String, (default) name for the model.
        """
        super().__init__(name=name, **kwargs)

        self._ev = ev
        self._var = var

    @property
    def ev(self):
        """The expected value of the ejection distribution."""
        return self._ev

    @property
    def var(self):
        """The variance of the ejection distribution."""
        return self._var

    @property
    def third_central_moment(self):
        """The third central moment of the ejection distribution."""
        return 0  # Gaussian third central moment

    def pdf(self, x):
        """The PDF of the ejection distribution.
        :param x: A float or np.array, the parameter of the ejection distribution..
        """
        return norm.pdf(x, loc=self.ev, scale=self.stddev)

    def cdf(self, x):
        """The CDF of the first passage time distribution.
        :param x: A float or np.array, the parameter of the ejection distribution.
        """
        return norm.cdf(x, loc=self.ev, scale=self.stddev)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the ejection distribution.
        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        return norm.ppf(q, loc=self.ev, scale=self.stddev)

    def __getitem__(self, indices):
        """Takes elements along the batch shape. Use this for fancy indexing, e.g. new_distr = distr[:2].

        :param indices: Slices, or list or np.array of integers or Booleans. The indices of the values to extract.

        :returns out: A copy of the distribution with the extracted values.
        """
        if not isinstance(indices, slice):
            indices = np.array(indices)  # e.g. if it is a list or boolean
        return type(self)(self._ev[indices], self._var[indices], self.name)

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing,
            e.g. distr[:2] = old_distr.

        :param indices: Slices, or list or np.array of integers or Booleans. The indices of the values to place.
        """
        # sanity checks
        if type(values) != type(self):
            raise ValueError('When assigning values to {}, both instances must be of same class, but '
                             'object to be assigned is of class {} and assigned values are of class {}.'.format(self.name, type(self), type(values)))

        if not isinstance(indices, slice):
            indices = np.array(indices)  # e.g. if it is a list of integers or Booleans
        self._ev[indices] = values._ev
        self._var[indices] = values._var


class AbstractEjectionUniformDistribution(AbstractEjectionDistribution, ABC):
    """One-dimensional Uniform distribution with batch shape support, i.e., each component of the distribution is a
     single one-dimensional Uniform distribution."""

    def __init__(self, point_prediction, window_length, a=0.5, name='AbstractEjectionUniformDistribution', **kwargs):
        """Initializes the distribution.

                /\ p(x)                                         /\ p(x)
                |    ____w_____                                  |    ____w_____
                |   |         |                                  |   |          |
                ---------|---------->  x                         ----|-------------->  x
                point prediction (a = 0.5)                  point prediction (a = 0)

        :param point_prediction: A np.array of shape [batch_size], the point prediction.
        :param window_length: A float, the window length of the distribution.
        :param a: The ratio of the window length, where the point prediction is located.
        :param name: String, (default) name for the model.
        """
        # TODO: Docstrings anpassen überall im script
        self.point_prediction = point_prediction
        self.window_length = window_length
        self.a = a
        super().__init__(name=name, **kwargs)

    @property
    def x_max(self):
        return self.point_prediction+ self.a * self.window_length

    @property
    def x_min(self):
        return self.point_prediction - self.a * self.window_length

    @property
    def ev(self):
        """The expected value of the ejection distribution."""
        return uniform.mean(loc=self.x_min, scale=self.x_max - self.x_min)

    @property
    def var(self):
        """The variance of the ejection distribution."""
        return uniform.var(loc=self.x_min, scale=self.x_max - self.x_min)

    @property
    def third_central_moment(self):
        """The third central moment of the ejection distribution."""
        return 0  # Uniform third central moment

    def pdf(self, x):
        """The PDF of the ejection distribution.
        :param x: A float or np.array, the parameter of the distribution.
        """
        return uniform.pdf(x, loc=self.x_min, scale=self.x_max - self.x_min)

    def cdf(self, x):
        """The CDF of the ejection distribution.
        :param x: A float or np.array, the parameter of the distribution.
        """
        return uniform.cdf(x, loc=self.x_min, scale=self.x_max - self.x_min)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the ejection distribution.
        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        return uniform.ppf(q, loc=self.x_min, scale=self.x_max - self.x_min)

    def __getitem__(self, indices):
        """Takes elements along the batch shape. Use this for fancy indexing, e.g. new_distr = distr[:2].

        :param indices: Slices, or list or np.array of integers or Booleans. The indices of the values to extract.

        :returns out: A copy of the distribution with the extracted values.
        """
        if not isinstance(indices, slice):
            indices = np.array(indices)  # e.g. if it is a list or boolean
        return type(self)(self.point_prediction[indices], self.window_length, self.a, self.name)

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing,
            e.g. distr[:2] = old_distr.

        :param indices: Slices, or list or np.array of integers or Booleans. The indices of the values to place.
        """
        # sanity checks
        if type(values) != type(self):
            raise ValueError('When assigning values to {}, both instances must be of same class, but '
                             'object to be assigned is of class {} and assigned values are of class {}.'.format(self.name,
                                                                                                          type(self),
                                                                                                          type(values)))

        if not isinstance(indices, slice):
            indices = np.array(indices)  # e.g. if it is a list of integers or Booleans
        self.point_prediction[indices] = values.point_prediction


class AbstractMCDistribution(AbstractEjectionDistribution, ABC):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem to a distribution
     using scipy.stats.rv_histogram.
    """

    def __init__(self, samples, range, bins=100, name="MC simulation", **kwargs):
        """Initialize the model.

        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param t_range: A list of length 2 representing the limits for the first passage time histogram (the number of
            bins within t_range will correspond to bins).
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param name: String, name for the model.
        """
        super().__init__(name=name, **kwargs)

        self._samples = samples

        bins = int(bins * (max(samples) - min(samples)) / (range[-1] - range[0]))
        # we want to have bins samples in the plot window
        hist = np.histogram(self._samples, bins=bins, density=False)
        self._density = rv_histogram(hist, density=True)

    @property
    def samples(self):
        """The first passage times of the samples paths."""
        return self._samples

    @property
    def ev(self):
        """The expected value of the first passage time distribution."""
        return self._density.mean()

    @property
    def var(self):
        """The variance of the first passage time distribution."""
        return self._density.var()

    def pdf(self, t):
        """The first passage time distribution (FPTD).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        return self._density.pdf(t)

    def cdf(self, t):
        """The CDF of the first passage time distribution.

        :param t: A float or np.array, the time parameter of the distribution.
        """
        return self._density.pdf(t)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        return self._density.ppf(q)


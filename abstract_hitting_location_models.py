from absl import logging

from abc import ABC, abstractmethod

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm, uniform

from abstract_distributions import AbstractEjectionDistribution, AbstractEjectionNormalDistribution, AbstractEjectionUniformDistribution, AbstractMCDistribution


class AbstractHittingLocationModel(AbstractEjectionDistribution, ABC):
    """A base class for all hitting location models.

    These models calculate the distribution in y at the first passage time.
    """

    def __init__(self, hitting_time_model, S_w, name='DefaultName'):
        """Initialize the model.

        :param hitting_time_model, a HittingTimeModel object.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param name: String, (default) name for the model.
        """
        super().__init__(name=name)

        self._ht = hitting_time_model
        self._S_w = S_w

        # For properties
        self._ev = None
        self._var = None

    @property
    def third_central_moment(self):
        """The third central moment of the distribution in y at the first passage time."""
        # the third standardized moment and the third moment are zero -> skewness = 0
        return 0

    @abstractmethod
    def pdf(self, y):  # TODO: kann auch razs
        """The PDF in y at the first passage time.

        :param y: A float or np.array, the y parameter of the distribution.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def cdf(self, y):  # TODO: kann auch razs
        """The CDF in y at the first passage time.

        :param y: A float or np.array, the y parameter of the distribution.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def get_statistics(self):  # TODO: Evtl. doch in die distributions?
        """Get some statistics from the model as a dict."""
        hit_stats = {}
        hit_stats['PDF'] = self.pdf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['SKEW'] = self.skew  # TODO: Auch bei Hitting time models? wie ist es aktuell dort gelöst?
        return hit_stats


class AbstractTaylorHittingLocationModel(AbstractEjectionNormalDistribution, AbstractHittingLocationModel, ABC):
    """A purely Gaussian approximation for the distribution in y at the first passage time problem by simply using the
    mean and variance of the distribution in y.

    Note that the mean and variance can be calculated directly (an independently of the used model)
    with the given FPTD as done in the AbstractCVHittingLocationModel parent class.

    Compared with the TaylorHittingLocationModel, this model has the exact moments, but its shape may capture
    the underlying distribution less well.
    """

    def __init__(self, hitting_time_model, S_w, name='Gauß--Taylor approx.', **kwargs):  # TODO: alle rausnehmen?
        """Initialize the model.

        :param hitting_time_model, a HittingTimeModel object.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param name: String, name for the model.
        """
        super().__init__(hitting_time_model=hitting_time_model,
                         S_w=S_w,
                         name=name,
                         **kwargs,
                         )

    @property
    def ev(self):
        """The expected value of the distribution in y at the first passage time."""
        # overwrites the result of the base class (which is the true, theoretical moment)
        return self._ev

    @property
    def var(self):
        """The variance of the distribution in y at the first passage time."""
        # overwrites the result of the base class (which is the true, theoretical moment)
        return self._var

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'CDF': self.cdf,
                          'PPF': self.ppf,
                          })
        return hit_stats


class AbstractSimpleGaussHittingLocationModel(AbstractEjectionNormalDistribution, AbstractHittingLocationModel, ABC):
    """A purely Gaussian approximation for the distribution in y at the first passage time problem by simply using the
    mean and variance of the distribution in y.

    Note that the mean and variance can be calculated directly (an independently of the used model)
    with the given FPTD as done in the AbstractCVHittingLocationModel parent class.

    Compared with the TaylorHittingLocationModel, this model has the exact moments, but its shape may capture
    the underlying distribution less well.
    """

    def __init__(self, hitting_time_model, S_w, name='Gauß approx.', **kwargs):  # TODO: alle rausnehmen?
        """Initialize the model.

        :param hitting_time_model, a HittingTimeModel object.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param name: String, name for the model.
        """
        super().__init__(hitting_time_model=hitting_time_model,
                         S_w=S_w,
                         name=name,
                         **kwargs,
                         )

    def pdf(self, y):  # TODO: Kann dann raus
        """The PDF in y at the first passage time.

        :param y: A float or np.array, the y parameter of the distribution.
        """
        return norm.pdf(y, loc=self.ev, scale=self.stddev)

    def cdf(self, y):
        """The CDF in y at the first passage time.

        :param y: A float or np.array, the y parameter of the distribution.
        """
        return norm.cdf(y, loc=self.ev, scale=self.stddev)

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'CDF': self.cdf,
                          'PPF': self.ppf,
                          })
        return hit_stats


class AbstractUniformHittingLocationModel(AbstractEjectionUniformDistribution, AbstractHittingLocationModel, ABC):  # TODO: Die sehen von der intialisierung nicht aus wie die oberen, ändern!
    """One-dimensional Uniform distribution with batch shape support, i.e., each component of the distribution is a
     single one-dimensional Uniform distribution."""

    def __init__(self, point_prediction, window_length, a=0.5, name='AbstractEjectionUniformDistribution',
                 **kwargs):
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

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'CDF': self.cdf,
                          'PPF': self.ppf,
                          })
        return hit_stats


class AbstractMCHittingLocationModel(AbstractMCDistribution, AbstractHittingLocationModel, ABC):  # TODO: Die sehen von der intialisierung nicht aus wie die oberen, ändern!
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem to a distribution
     using scipy.stats.rv_histogram.
    """

    def __init__(self, y_samples, y_range, bins=100, name="MC simulation", **kwargs):  # TODO: alle rausnehmen?):
        """Initialize the model.

        :param t_samples: A np.array of shape [N] containing the first passage times of the particles.
        :param t_range: A list of length 2 representing the limits for the first passage time histogram (the number of
            bins within t_range will correspond to bins).
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param name: String, name for the model.
        """
        super().__init__(samples=y_samples,
                         range=y_range,
                         bins=bins,
                         name=name,
                         **kwargs)

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'CDF': self.cdf,
                          'PPF': self.ppf,
                          })
        return hit_stats


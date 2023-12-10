from absl import logging

from abc import ABC, abstractmethod

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm, uniform

from abstract_distributions import AbstractArrivalDistribution, AbstractNormalArrivalDistribution, AbstractUniformArrivalDistribution, AbstractMCArrivalDistribution
from abstract_hitting_time_distributions import AbstractHittingTimeDistribution


class AbstractHittingLocationDistribution(AbstractArrivalDistribution, ABC):
    """A base class for all hitting location distributions.

    These classes describe the distribution in y at the first-passage time.
    """
    def __init__(self, htd, name='AbstractHittingLocationDistribution'):
        """Initializes the distribution.

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param name: String, the (default) name for the distribution.
        """
        # sanity checks
        if not isinstance(htd, AbstractHittingTimeDistribution):
            raise ValueError('htd must be a child of AbstractHittingTimeDistribution.')

        super().__init__(name=name)

        self._htd = htd

    @property
    def batch_size(self):
        """The batch size of the distribution.

        :returns: An integer, the batch size.
        """
        return self._htd.batch_size

    @property
    def third_central_moment(self):
        """The third central moment of the distribution in y at the first-passage time.

        :returns: A float or a np.array of shape [batch_size], the third central moment.
        """
        # the third standardized moment and the third moment are zero -> skewness = 0
        return np.squeeze(np.zeros(self.batch_size))

    @abstractmethod
    def _ev_t(self, t):
        """The mean function of the motion model in y.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the mean in y at time t.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def _var_t(self, t):
        """The variance function of the motion model in y.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the variance in y at time t.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._htd[indices] = values.ht  # TODO: Lassen wir das so, ergibt das Sinn es so zu machen?
        super().__setitem__(indices, values)  # TODO: notwendig?

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._htd = values.ht[indices]  # TODO: Lassen wir das so, ergibt das Sinn es so zu machen?
        super()._left_hand_indexing(indices, values)

    def get_statistics(self):  # TODO: Evtl. doch in die distributions?
        """Get some statistics from the model as a dict."""
        hit_stats = {}  # TODO: Lassen wir die CDF,PPF hier nach wie vor raus? oder CDF, PPF auch bei ProjectionMethod implementieren?
        hit_stats['PDF'] = self.pdf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['SKEW'] = self.skew
        return hit_stats

    # TODO: Ist es nötig, wenn die htd gescalted wird, auch die self._htd zu scalene? Also ist das eine copy? Nee oder? Testen!


class AbstractGaussTaylorHittingLocationDistribution(AbstractNormalArrivalDistribution, AbstractHittingLocationDistribution, ABC):
    """A simple Gaussian approximation for the distribution in y at the first-passage time problem using a Taylor
    approximation and error propagation.

    Note that this method, although it may capture the shape of the distribution very well, does not have the exact
    moments as calculated by the parent class.
    """
    def __init__(self, name='AbstractGaussTaylorHittingLocationDistributio', **kwargs):
        """Initializes the distribution.

        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name,
                         **kwargs,
                         )

    @abstractmethod
    def _compute_var(self, htd, S_w):
        """Computes the variance of the distribution in y at the first-passage time based on error propagation.

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A np.array of shape [batch_size], the power spectral density (PSD).

        :returns: A np.array of shape [batch_size], the variance of the approximation.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._ev *= length_scaling_factor  # TODO. Das gilt aber auch nicht immer (gilt nur wenn nur ort gescaled). Wird das benötigt oder sowieso überschrieben?
        self._var *= length_scaling_factor**2

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'CDF': self.cdf,
                          'PPF': self.ppf,
                          })
        return hit_stats


class AbstractSimpleGaussHittingLocationDistribution(AbstractNormalArrivalDistribution, AbstractHittingLocationDistribution, ABC):
    """A purely Gaussian approximation for the distribution in y at the first-passage time problem by simply using the
    (theoretic) mean and variance of the distribution in y given the hitting time model.

    Note that the mean and variance can be calculated directly (and independently of the used approximation for the
    distribution of y at the first-passage time) with the given FPTD as done by the parent class.

    Compared with the GaussTaylorHittingLocationDistribution, this distribution uses the exact first and second moments,
    but its shape may capture the underlying distribution less well.
    """
    def __init__(self, name='AbstractSimpleGaussHittingLocationDistribution', **kwargs):
        """Initializes the distribution.

        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name,
                         **kwargs,
                         )

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._ev *= length_scaling_factor  # TODO. Das gilt aber auch nicht immer (gilt nur wenn nur ort gescaled). Wird das benötigt oder sowieso überschrieben?
        self._var *= length_scaling_factor**2

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'CDF': self.cdf,
                          'PPF': self.ppf,
                          })
        return hit_stats


class AbstractBayesMixtureHittingLocationDistribution(AbstractHittingLocationDistribution, ABC):  # TODO: Hier stimmt noch was nicht,
    """Mathematically exact way to solve the problem of finding the distribution in y at the first-passage time. Sets up
    the joint distribution of the process in y and the approximation for the given first-passage time distribution
    and performs a marginalization over the latter.

    The integration is done using a Riemann-sum-like approach by summing Gaussian random variables that represent
    the densities in y at different times weighted by the first-passage time probability in a (small) range around
    these times.
    """
    def __init__(self, t_min=None, t_max=None, n=100, name="AbstractBayesMixtureHittingLocationDistribution", **kwargs):
        """Initializes the distribution.

        :param t_min: A float or a np.array of shape [batch_size], the lower integration limit.
        :param t_max: A float or a np.array of shape [batch_size], the upper integration limit.
        :param n: An integer, the number of integration points to use.
        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name,
                         **kwargs,
                         )

        # self._t_min = np.atleast_1d(self._htd.ppf(0.0005) if t_min is None else t_min)
        # if t_max is None:
        #     if hasattr(self._htd, 'q_max'):
        #         self._t_max = np.atleast_1d(self._htd.t_max.copy())
        #         self._t_max[0.99995 < self._htd.q_max] = self._htd.ppf(0.99995)
        #     else:
        #         self._t_max = np.atleast_1d(self._htd.ppf(0.99995))
        # else:
        #     self._t_max = np.atleast_1d(t_max)

        self._t_min = np.atleast_1d(t_min).astype(float) if t_min is not None else None
        self._t_max = np.atleast_1d(t_max).astype(float) if t_max is not None else None

        self._n = n

        # for properties, shared variables
        self._weights = None
        self._locations = None  # shape [batch_size, n]
        self._scales = None  # shape [batch_size, n]

    @property
    def t_min(self):
        """The lower integration limit.

        :returns: A np.array of shape [batch_size], the lower integration limit.
        """
        if self._t_min is None:
            self._t_min = np.atleast_1d(self._htd.ppf(0.0005))
        return self._t_min

    @property
    def t_max(self):
        """The upper integration limit.

        :returns: A np.array of shape [batch_size], the upper integration limit.
        """
        if self._t_max is None:
            if hasattr(self._htd, 'q_max'):
                self._t_max = np.atleast_1d(self._htd.t_max.copy())
                self._t_max[0.99995 < self._htd.q_max] = self._htd.ppf(0.99995)
            else:
                self._t_max = np.atleast_1d(self._htd.ppf(0.99995))
        return self._t_max

    @property
    def n(self):
        """The number of integration points to use.

        :returns: An integer, the number of integration points to use.
        """
        return self._n

    @property
    def locations(self):
        """The location of the mixture components.

        :return: A np.array of shape [batch_size, n], the location of the mixture components.
        """
        if self._weights is None:
            self._locations, self._scales, self._weights = self._compute_locations_scales_and_weights()
        return self._locations

    @property
    def scales(self):
        """The scales of the mixture components.

        :return: A np.array of shape [batch_size, n], the scales of the mixture components.
        """
        if self._weights is None:
            self._locations, self._scales, self._weights = self._compute_locations_scales_and_weights()
        return self._scales

    @property
    def weights(self):
        """The integration weights, i.e., the probabilities of the mixture components.

        :returns: A np.array of shape [batch_size, n], the integration weights.
        """
        if self._weights is None:
            self._locations, self._scales, self._weights = self._compute_locations_scales_and_weights()
        return self._weights

    @property
    def locations_scales_weights_available(self):
        """Indicator that shows if the location, scales, and integration weights were already calculated.

           :returns: True if the location, scales, and integration weights were already calculated, False otherwise.
           """
        return self._locations is not None

    def _compute_locations_scales_and_weights(self):
        """Compute the locations, scales, and integration weights from the first-passage time CDF.

        :returns:
            locations: A np.array of shape [batch_size, n], the location of the mixture components.
            scales: A np.array of shape [batch_size, n], the scales of the mixture components.
            weights: A np.array of shape [batch_size, n], the integration weights, i.e., the probabilities of the
                mixture components.
        """
        delta_t = (self.t_max - self.t_min) / self._n
        t_k = np.array([self.t_min + k * delta_t for k in range(self._n + 1)]).T  # shape [batch_size, n + 1]

        cdf_tk = self._htd.cdf(t_k)  # shape [n + 1] or [batch_size, n + 1]
        cdf_tk_plus_one = cdf_tk[1:]
        weights = cdf_tk_plus_one - cdf_tk[:-1]  # shape [num_samples] or [batch_size, n]

        locations = self._ev_t(t_k[:, :-1])  # shape [batch_size, n]
        scales = np.sqrt(self._var_t(t_k[:, :-1]))  # shape [batch_size, n]

        return locations, scales, np.atleast_2d(weights)

    @AbstractArrivalDistribution.check_density_input_dim
    def pdf(self, y):
        """The PDF in y at the first-passage time.

        :param y: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the y parameter of the distribution.

        :returns: A float or a np.array, the value of the PDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        # y_mixture = 0
        # for k in range(self.n - 1):
        #     # This can be done in parallel
        #     t_k = self.t_min + k * self.delta_t
        #     t_k_plus_1 = self.t_min + (k + 1) * self.delta_t
        #     weight = self._htd.cdf(t_k_plus_1) - self._htd.cdf(t_k)
        #     p_y_t_k = norm.pdf(y, loc=self._ev_t(t_k), scale=np.sqrt(self._var_t(t_k)))
        #     y_mixture += weight * p_y_t_k
        # return y_mixture

        # p_y_t_k = norm.pdf(y, loc=self._locations, scale=self.scales)
        # return np.dot(self._weights, p_y_t_k)

        # we assume that y has shape [batch_size, sample_size], so transform it to that shape
        y = np.broadcast_to(y, shape=(self.batch_size, 1)) if np.isscalar(y) else np.atleast_2d(y).reshape(
            self.batch_size, -1)

        # y has shape [batch_size, sample_size], and locations/scales and weights are of shape [batch_size, n]
        dist_matrix = (y[:, :, None] - self.locations[:, None, :]) / self.scales[:, None, :]
        # shape [batch_size, sample_size, n]
        p_y_t_k = norm.pdf(dist_matrix) * 1 / self.scales[:, None, :]  # shape [batch_size, sample_size, n]
        # since p_y_t_k = norm.pdf(y, loc=self._ev_t(t_k), scale=np.sqrt(self._var_t(t_k))) does not work as required
        # here for 3D arrays.
        return np.squeeze(np.sum(self.weights[:, None, :] * p_y_t_k, axis=-1))  # shape [batch_size, sample_size]

    @AbstractArrivalDistribution.check_density_input_dim
    def cdf(self, y):
        """The cumulative distribution function (CDF) of the distribution in y at the first-passage time.

        :param y: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the y parameter of the distribution.

        :returns: A float or a np.array, the value of the CDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        # y_mixture = 0
        # for k in range(self.n - 1):
        #     # This can be done in parallel
        #     t_k = self.t_min + k * self.delta_t
        #     t_k_plus_1 = self.t_min + (k + 1) * self.delta_t
        #     weight = self._htd.cdf(t_k_plus_1) - self._htd.cdf(t_k)
        #     prob_y_t_k = norm.cdf(y, loc=self._ev_t(t_k), scale=np.sqrt(self._var_t(t_k)))
        #     y_mixture += weight * prob_y_t_k
        #
        # return y_mixture

        # prob_y_t_k = norm.cdf(y, loc=self._locations, scale=self.scales)
        # return np.dot(self._weights, prob_y_t_k)

        # we assume that y has shape [batch_size, sample_size], so transform it to that shape
        y = np.broadcast_to(y, shape=(self.batch_size, 1)) if np.isscalar(y) else np.atleast_2d(y).reshape(
            self.batch_size, -1)

        # y has shape [batch_size, sample_size], and locations/scales and weights are of shape [batch_size, n]
        dist_matrix = (y[:, :, None] - self.locations[:, None, :]) / self.scales[:, None, :]
        # shape [batch_size, sample_size, n]
        prob_y_t_k = norm.cdf(dist_matrix)  # shape [batch_size, sample_size, n]
        return np.squeeze(np.sum(self.weights[:, None, :] * prob_y_t_k, axis=-1))  # shape [batch_size, sample_size]

    def ppf(self, q, disable_double_check=False):
        """The quantile function / percent point function (PPF) of the distribution in y at the first-passage time.

         :param q: A float, the confidence parameter of the distribution, 0 <= q <= 1.
         :param disable_double_check: Boolean, if False, the results for the ppf are inserted into the cdf and
             double-checked. Disable this to save computation time.

         :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
         """
        # TODO
        raise NotImplementedError()

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        # sanity checks
        if self._n != values.n:
            raise ValueError(
                'When assigning values to {}, both instances must have the same parameter n'.format(
                    self.__class__.__name__))

        super().__setitem__(indices, values)

        self._t_min[indices] = values.t_min  # we force calculating this values since it's computationally not that demanding
        self._t_max[indices] = values.t_max

        if self.locations_scales_weights_available and values.locations_scales_weights_available:
            self._weights[indices] = values.weights
            self._locations[indices] = values.weights
            self._scales[indices] = values.weights
        else:
            self._weights = None
            self._locations = None
            self._scales = None

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._t_min = values.t_min[
            indices]  # we force calculating this values since it's computationally not that demanding
        self._t_max = values.t_max[indices]

        if values.locations_scales_weights_available:
            self._weights = values.weights[indices]
            self._locations = values.weights[indices]
            self._scales = values.weights[indices]
        super()._left_hand_indexing(indices, values)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according t scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        super().scale_params(length_scaling_factor, time_scaling_factor)
        self._t_min *= time_scaling_factor  # TODO: Stimmt das?
        self._t_max *= time_scaling_factor
        # Force recalculating all other privates
        self._weights = None
        self._locations = None
        self._scales = None

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = {}  # TODO: nicht so
        hit_stats['PDF'] = self.pdf
        hit_stats['CDF'] = self.cdf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['SKEW'] = self.skew
        return hit_stats


class AbstractBayesianHittingLocationDistribution(AbstractHittingLocationDistribution, ABC):
    """ Mathematically exact way to solve the problem of finding the distribution in y at the first-passage time. Sets up
    the joint distribution of the process in y and the approximation for the given first-passage time distribution
    and performs a marginalization over the latter.

    The integration is done using scipy's integrate quad function.
    """
    def __init__(self, t_min=None, t_max=None, name="AbstractBayesianHittingLocationDistribution", **kwargs):
        """Initializes the distribution.

        :param t_min: A float or a np.array of shape [batch_size], the lower integration limit.
        :param t_max: A float or a np.array of shape [batch_size], the upper integration limit.
        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name,
                         **kwargs,
                         )

        self._t_min = np.atleast_1d(t_min).astype(float) if t_min is not None else None
        self._t_max = np.atleast_1d(t_max).astype(float) if t_max is not None else None

    @property
    def t_min(self):
        """The lower integration limit.

        :returns: A float or a np.array of shape [batch_size], the lower integration limit.
        """
        if self._t_min is None:
            self._t_min = np.atleast_1d(self._htd.ppf(0.0005))
        return np.squeeze(self._t_min)

    @property
    def t_max(self):
        """The upper integration limit.

        :returns: A float or a np.array of shape [batch_size], the upper integration limit.
        """
        if self._t_max is None:
            if hasattr(self._htd, 'q_max'):
                self._t_max = np.atleast_1d(self._htd.t_max.copy())
                self._t_max[0.99995 < self._htd.q_max] = self._htd.ppf(0.99995)
            else:
                self._t_max = np.atleast_1d(self._htd.ppf(0.99995))
        return np.squeeze(self._t_max)

    @AbstractArrivalDistribution.check_density_input_dim
    def pdf(self, y):
        """The PDF in y at the first-passage time.

        :param y: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the y parameter of the distribution.

        :returns: A float or a np.array, the value of the PDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        p_y_t = lambda y_, t: norm.pdf(y_, loc=self._ev_t(t), scale=np.sqrt(self._var_t(t)))
        return np.squeeze(integrate.quad(lambda t: p_y_t(y, t) * self._htd.pdf(t), self.t_min, self.t_max)[0])  # TODO: Das geht wsl. schief für batchsize != 1

    @AbstractArrivalDistribution.check_density_input_dim
    def cdf(self, y):
        """The cumulative distribution function (CDF) of the distribution in y at the first-passage time.

        :param y: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the y parameter of the distribution.

        :returns: A float or a np.array, the value of the CDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size] (with sample_size > 1), then returns a np.array of shape
                    [sample_size].
            - If the distribution's batch_size is > 1
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size] (with sample_size > 1), then returns a np.array of
                    shape [batch_size, sample_size].
        """
        prob_y_t = lambda y_, t: norm.cdf(y_, loc=self._ev_t(t), scale=np.sqrt(self._var_t(t)))
        return np.squeeze(integrate.quad(lambda t: prob_y_t(y, t) * self._htd.pdf(t), self.t_min, self.t_max)[0])

    def ppf(self, q, disable_double_check=False):
        """The quantile function / percent point function (PPF) of the distribution in y at the first-passage time.

         :param q: A float, the confidence parameter of the distribution, 0 <= q <= 1.
         :param disable_double_check: Boolean, if False, the results for the ppf are inserted into the cdf and
             double-checked. Disable this to save computation time.

         :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
         """
        # TODO
        raise NotImplementedError()

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._t_min = values.t_min  # we force calculating this values since it's computationally not that demanding
        self._t_max = values.t_max

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._t_min = values.t_min[
            indices]  # we force calculating this values since it's computationally not that demanding
        self._t_max = values.t_max[indices]

        super()._left_hand_indexing(indices, values)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according t scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        super().scale_params(length_scaling_factor, time_scaling_factor)
        self._t_min *= time_scaling_factor  # TODO: Stimmt das?
        self._t_max *= time_scaling_factor

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = {}
        hit_stats['PDF'] = self.pdf
        hit_stats['CDF'] = self.cdf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['SKEW'] = self.skew
        return hit_stats


class AbstractUniformHittingLocationDistribution(AbstractUniformArrivalDistribution, AbstractHittingLocationDistribution, ABC):  # TODO: Die sehen von der intialisierung nicht aus wie die oberen, ändern!
    """Uses point predictors for the distribution in y at the first-passage time and a uniform distribution.

    This distribution corresponds to the "usual" case where we define a fixed deflection window.
    """
    def __init__(self, name="AbstractUniformHittingLocationDistribution", **kwargs):
        """Initializes the distribution.

        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name,
                         **kwargs)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._point_prediction *= length_scaling_factor    # TODO. Das gilt aber auch nicht immer (gilt nur wenn nur ort gescaled). Wird das benötigt oder sowieso überschrieben?
        self._window_length *= length_scaling_factor

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'CDF': self.cdf,
                          'PPF': self.ppf,
                          })
        return hit_stats


class AbstractMCHittingLocationDistribution(AbstractMCArrivalDistribution, AbstractHittingLocationDistribution, ABC):
    """Wraps the histogram derived by a Monte-Carlo approach to obtain the distribution in y at the first-passage time
    using scipy.stats.rv_histogram.

    """
    def __init__(self, y_range, y_samples, name="AbstractMCHittingLocationDistribution", **kwargs):
        """Initializes the distribution.

        :param y_range: A list of length 2 representing the limits for the histogram of the distribution in y at the
            first-passage time histogram (the number of bins within y_range will correspond to bins).
        :param y_samples: A np.array of shape [num_samples] containing the y-position at the first-passage
            times of the particles.
        :param name: String, the (default) name for the distribution.
        """
        y_samples = y_samples[np.isfinite(y_samples)]  # there are default values, remove them from array
        super().__init__(name=name,
                         range=y_range,
                         samples=y_samples,
                         **kwargs)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._samples *= length_scaling_factor  # TODO. Das gilt aber auch nicht immer (gilt nur wenn nur ort gescaled). Wird das benötigt oder sowieso überschrieben?
        self._range *= length_scaling_factor
        # Recalculate the distribution
        self._density = self._build_distribution_from_samples(self._samples, self._range)

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'CDF': self.cdf,
                          'PPF': self.ppf,
                          })
        return hit_stats


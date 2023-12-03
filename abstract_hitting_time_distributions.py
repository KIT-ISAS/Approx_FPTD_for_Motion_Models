"""Abstract distribution classes used by all first-passage time distribution sub-types.

"""

import os
from absl import logging

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

import matplotlib.pyplot as plt
from timeit import time

from abstract_distributions import AbstractArrivalDistribution, AbstractNormalArrivalDistribution, AbstractUniformArrivalDistribution, AbstractMCArrivalDistribution


class AbstractHittingTimeDistribution(AbstractArrivalDistribution, ABC):
    """A base class for all the hitting time distributions."""

    def __init__(self, x_predTo, t_L, name='AbstractHittingTimeDistribution'):
        """Initializes the distribution.

        :param x_predTo: A float or np.array of shape [batch_size], the position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name)

        self._x_predTo = np.broadcast_to(x_predTo, self.batch_size)  # this itself raises an error if not compatible
        self._t_L = t_L

        # # For properties  # TODO: braucht man die hier?
        # self._ev = None
        # self._var = None
        # self._ev_third = None
        # self._stddev = None

    @property
    def x_predTo(self):
        """The position of the boundary.

        :returns A float or np.array of shape [batch_size], the position of the boundary.
        """
        return np.squeeze(self._x_predTo, axis=0)

    @property
    def t_L(self):
        """The initial time.

        :returns A float, the time of the last state/measurement (initial time).
        """
        return self._t_L

    @abstractmethod
    @AbstractArrivalDistribution.batch_size_one_function
    def trans_density(self, dt, theta):
        """The transition density p(x(dt+theta)| x(theta) =x_predTo) from going fromx_predTo at time theta to
        x(dt+theta) at time dt+theta.

        Note that in terms of the used approximation, this can be seen as the first returning time tox_predTo after
        a crossing ofx_predTo at theta.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param dt: A float, the time difference. dt is zero at time = theta.
        :param theta: A float, the (assumed) time at which x(theta) = x_pred_to.

        :returns: A scipy.stats.norm object, the transition density for the given dt and theta.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def _ev_t(self, t):
        """The mean function of the motion model in x.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the mean in x at time t.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def _var_t(self, t):
        """The variance function of the motion model in x.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the variance in x at time t.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @AbstractArrivalDistribution.batch_size_one_function
    def returning_probs_inverse_transform_sampling_do_not_use(self, t, num_samples=1000, deterministic_samples=True):
        """Calculates approximate returning probabilities using a numerical integration (MC integration) based on
        samples from the approximate first-passage time distribution (using inverse transform sampling).

        DO NOT USE, APPROACH MAY BE NOT CORRECT!

        Approach:

         P(t < T_a , x(t) < a) = int_{_t_L}^t fptd(theta) P(x(t) < a | x(theta) = a) d theta

                               ≈ 1 / N sum_{theta_i} P(x(t) < a | x(theta_i) = a) ,  theta_i samples from the
                                    approximation (N samples in total) in [_t_L, t] ,

          with theta the time, when x(theta) = a.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param t: A float, the time parameter of the distribution.
        :param num_samples: An integer, the number of samples to approximate the integral.
        :param deterministic_samples: A Boolean, whether to use random samples (False) or deterministic samples (True).

        :returns: A float, an approximation for the probability P(t < T_a , x(t) < a), i.e., the probability that a
            sample path has crossed the boundary at a time theta < t, but is smaller than the boundary at time t.
        """
        q_max_to_use = self.cdf(t)

        if not deterministic_samples:
            q_samples = np.random.uniform(low=0, high=q_max_to_use, size=num_samples)
        else:
            # low=0, high=1, num_samples=5 -> [0.16, 0.33, 0.5, 0.67, 0.83]
            q_samples = np.linspace(0, q_max_to_use, num=num_samples + 1, endpoint=False)[1:]

        theta_samples = [self.ppf(q) for q in q_samples]

        return np.nanmean(
            [self.trans_density(dt=t - theta, theta=theta).cdf(self.x_predTo) for theta in theta_samples])

    @AbstractArrivalDistribution.batch_size_one_function
    def returning_probs_uniform_samples(self, t, num_samples=100, deterministic_samples=True):
        """Calculates approximate returning probabilities using a numerical integration (MC integration) based on
        samples from a uniform distribution.

        Approach:

         P(t < T_a , x(t) < a) = int_{_t_L}^t fptd(theta) P(x(t) < a | x(theta) = a) d theta

                               ≈  (t - _t_L) / N sum_{theta_i} FPTD(theta_i) * P(x(t) < a | x(theta_i) = a) ,  theta_i
                                    samples from a uniform distribution (N samples in total) in [_t_L, t] ,

          with theta the time, when x(theta) = a.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param t: A float, the time parameter of the distribution.
        :param num_samples: An integer, the number of samples to approximate the integral.
        :param deterministic_samples: A Boolean, whether to use random samples (False) or deterministic samples (True).

        :returns: A float, an approximation for the probability P(t < T_a , x(t) < a), i.e., the probability that a
            sample path has crossed the boundary at a time theta < t, but is smaller than the boundary at time t.
        """
        if not deterministic_samples:
            theta_samples = np.random.uniform(low=self._t_L, high=t, size=num_samples)
        else:
            # low=0, high=1, num_samples=5 -> [0.16, 0.33, 0.5, 0.67, 0.83]
            theta_samples = np.linspace(self._t_L, t, num=num_samples + 1, endpoint=False)[1:]

        return (t - self._t_L) * np.nanmean(
            [self.pdf(theta) * self.trans_density(dt=t - theta, theta=theta).cdf(self.x_predTo) for
             theta in theta_samples])

    @AbstractArrivalDistribution.batch_size_one_function
    def returning_probs_integrate_quad(self, t):
        """Calculates approximate returning probabilities using numerical integration.

        Approach:

         P(t < T_a , x(t) < a) = int_{_t_L}^t fptd(theta) P(x(t) < a | x(theta) = a) d theta ,

          with theta the time, when x(theta) = a.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param t: A float, the time parameter of the distribution.

        :returns: A float, an approximation for the probability P(t < T_a , x(t) < a), i.e., the probability that a
            sample path has crossed the boundary at a time theta < t, but is smaller than the boundary at time t.
        """
        fn = lambda theta: self.pdf(theta) * self.trans_density(dt=t - theta, theta=theta).cdf(self.x_predTo)
        a = np.finfo(np.float64).eps if self._t_L == 0 else self._t_L
        return integrate.quad(fn, a=a, b=t)[0]  # this is a tuple

    @AbstractArrivalDistribution.batch_size_one_function
    def plot_quantile_function(self, q_min=0.005, q_max=0.995, save_results=False, result_dir=None, for_paper=True):  # TODO
        """Plot the quantile function.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param q_min: A float, the smallest value of the confidence plot range.
        :param q_max: A float, the highest value of the confidence plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        """
        # TODO: Die plot functions müssen noch auf batch betrieb umgestellt werden, z. B. als abstract machen, docstrings
        # TODO: Welcher dieser Funktionnen verwenden?
        if not isinstance(self.ppf(0.5), float):
            raise ValueError('Plotting the PPF is only supported for distributions of batch size equal to 1.')

        plot_q = np.arange(q_min, q_max, 0.01)
        # TODO: Perhaps silent warnings
        plot_quant = [self.ppf(q) for q in plot_q]
        plt.plot(plot_q, plot_quant)
        plt.xlabel('Confidence level')
        plt.ylabel('Time t in s')

        if not for_paper:
            plt.title('Quantile Function (Inverse CDF) for ' + self.name)

        if save_results:
            plt.savefig(result_dir + self.name + '_quantile_function.pdf')
            plt.savefig(result_dir + self.name + '_quantile_function.png')
            plt.savefig(result_dir + self.name + '_quantile_function.pgf')
        plt.show()

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._x_predTo *= length_scaling_factor
        self._t_L *= time_scaling_factor

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        # sanity checks
        if self._t_L != values._t_L:
            raise ValueError(
                'When assigning values to {}, both instances must have the same parameter _t_L'.format(
                    self.__class__.__name__))

        self._x_predTo[indices] = values._x_predTo  # TODO: Call to super?

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._x_predTo = values._x_predTo[indices]
        self._t_L = values._t_L
        super()._left_hand_indexing(indices, values)

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = {}   # TODO: Warum keine PPF?
        hit_stats['PDF'] = self.pdf
        hit_stats['CDF'] = self.cdf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['SKEW'] = self.skew  # TODO: immer an? bei machen wird es auch extra hinzugefügt
        return hit_stats


class AbstractGaussTaylorHittingTimeDistribution(AbstractNormalArrivalDistribution, AbstractHittingTimeDistribution, ABC):
    """A simple Gaussian approximation for the first-passage time problem using a Taylor approximation and error
    propagation.

    """
    def __init__(self, name='AbstractGaussTaylorHittingTimeDistribution', **kwargs):
        """Initializes the distribution.

        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name,
                         **kwargs,
                         )

    @abstractmethod
    def _compute_var(self, point_prediction, x_L, C_L, t_L, S_w):
        """Computes the variance of the first-passage time problem based on error propagation.

        :param point_prediction: A np.array of shape [batch_size], a point prediction used as the distribution's
            expectation.
        :param x_L: A np.array of shape [batch_size, state_length] representing the expected value of the initial state.
        :param C_L: A np.array of shape [batch_size, state_length, state_length] representing the covariance matrix of
            the initial state.
        :param t_L: A float, the time of the last state/measurement (initial time).
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
        self._ev *= time_scaling_factor   # TODO. Das gilt aber auch nicht immer (gilt nur wenn nur zeit gescaled?). Wird das benötigt oder sowieso überschrieben?
        self._var *= time_scaling_factor**2

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'PPF': self.ppf,
                          })
        return hit_stats


class AbstractNoReturnHittingTimeDistribution(AbstractHittingTimeDistribution, ABC):
    """An approximation to the first-passage time distribution using the assumption that particles are unlikely to move
    back once they have passed the boundary.

    """
    def __init__(self, name="AbstractNoReturnHittingTimeDistribution", **kwargs):
        """Initializes the distribution.

        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name,
                         **kwargs,
                         )

        # for properties
        self._q_max = None
        self._t_max = None
        self._ev = None
        self._var = None
        self._third_central_moment = None
        self._compute_moment = None

    @property
    def q_max(self):
        """The maximum value of the CDF (may be not equal to 1).

        :returns: A float or np.array of shape [batch_size], the maximum value of the CDF.
        """
        if self._q_max is None:
            self._q_max, self._t_max = self._get_max_cdf_value_and_location()
        return np.squeeze(self._q_max, axis=0)

    @property
    def t_max(self):
        """The time, when the CDF visits its maximum (the maximum of the CDF may be not equal to 1).

        :returns: A float or np.array of shape [batch_size], the time, when the CDF visits its maximum.
        """
        if self._t_max is None:
            self._q_max, self._t_max = self._get_max_cdf_value_and_location()
        return np.squeeze(self._t_max, axis=0)

    @property
    def compute_moment(self):
        """A wrapper / helper function that builds an integrator for the computation of the moments.

         Format compute_moment:

           f(t) --> moments of f(t)

         where
            - f(t) is a callable of t implementing the function to be integrated, e.g., f(t) = t (expectation) or
                f(t) = (t - self.ev)^2 (variance).
            - moments of f(t) is tuple of length 4 of floats or np.arrays of shape [batch_size] containing the
                moments of f(t) based on lower sums, upper sums, and the absolute and relative difference between both,
                respectively.

        :returns: A callable that can be used to compute the moments.
        """
        if self._compute_moment is None:
            self._compute_moment = self._get_numerical_moment_integrator()
        return self._compute_moment

    @property
    def ev(self):
        """The expected value of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the expected value.
        """
        if self._ev is None:
            start_time = time.time()
            # Calculating moments with integrate.quad takes much time
            # TODO: For the integrate.quad method to work, the integration limits need to be chosen as for the compute_moment method
            # self._ev = integrate.quad(lambda t: t * self.pdf(t), self.ppf(0.0005), self.ppf(0.9995))[
            # 0]  # this is a tuple
            self._ev, _, abs_dev, rel_dev = self.compute_moment(lambda t: t)
            logging.info('EV integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return np.squeeze(self._ev, axis=0)

    @property
    def ev_available(self):
        """Indicator that shows if the expected value was already calculated.

        :returns: True if the expected value was already calculated, False otherwise.
        """
        return self._ev is not None

    @property
    def var(self):
        """The variance of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the variance.
        """
        if self._var is None:
            start_time = time.time()
            # Calculating moments with integrate.quad takes much time
            # self._var = integrate.quad(lambda t: (t - self.ev) ** 2 * self.pdf(t), self.ppf(0.0005), self.ppf(0.9995))[
            #     0]  # this yields much better results
            # self._var = self.compute_moment(lambda t: t**2) - self.ev ** 2 # don't calculate the variance in
            # # this way because it causes high numerical errors
            self._var, _, abs_dev, rel_dev = self.compute_moment(
                lambda t: (t - self.ev) ** 2)  # this yields much better results
            logging.info('Var integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return np.squeeze(self._var, axis=0)

    @property
    def var_available(self):
        """Indicator that shows if the variance was already calculated.

        :returns: True if the variance was already calculated, False otherwise.
        """
        return self._var is not None

    @property
    def third_central_moment(self):
        """The third central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the third central moment.
        """
        if self._third_central_moment is None:
            start_time = time.time()
            # Calculate the third central moment because calculating the third moment directly causes high numerical errors
            # Calculating moments with integrate.quad takes much time
            # self._third_central_moment = integrate.quad(lambda t: (t - self.ev)**3 * self.pdf(t), self.ppf(0.00o05), self.ppf(0.99995))[
            #              0]  # this yields much better results
            self._third_central_moment, _, abs_dev, rel_dev = self.compute_moment(
                lambda t: (t - self.ev) ** 3)  # this yields much better results
            logging.info('E3 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return np.squeeze(self._third_central_moment, axis=0)

    @property
    def third_central_moment_available(self):
        """Indicator that shows if the third central moment was already calculated.

        :returns: True if the third central moment was already calculated, False otherwise.
        """
        return self._third_central_moment is not None

    @abstractmethod
    def _pdf(self, t):
        """Time-derivative of self._cdf.

        Can be calculated from the standard Gaussian PDF N( ) with an argument (x_predTo - ev(t))/stddev(t) times the
        derivative w.r.t. of these argument (chain rule), i.e.,

           d/dt [ 1 - int( p(x(t), x= -infty ..x_predTo ) ] = d/dt [ PHI( (x_predTo - ev(t))/stddev(t) ) ]

                                                             = d/dt (x_predTo - ev(t))/stddev(t) )
                                                                        * N(x_predTo; ev(t), stddev(t)^2 )

           with PHI( ) being the standard Gaussian CDF.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the time derivative of self._cdf.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def _ppf(self, q):
        """The quantile function / percent point function (PPF) of the first-passage time distribution.

        Approach:

              1 - q = int(N(x, mu(t), var(t)), x = -inf ..x_predTo) = PHI ( (x_predTo - mu(t)) / sqrt(var(t))
              PHI^-1(1 -q) = (x_predTo - mu(t)) / sqrt(var(t)) -> solve for t...

        We solve the equation for t = t - _t_L to simplify calculations and add _t_L at the end of the function.

        :param q: A float, the confidence parameter of the distribution, 0 <= q <= 1.

        :returns:
            t: A np.array of shape [batch_size], the value of the PPF for q.
            candidate_roots: A np.array of shape [batch_size, num_possible_solutions] containing the values of all
                possible roots.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def _get_max_cdf_location_roots(self):
        """Method that finds the argmax roots of the CDF of the approximation.

        Approach:

            set self._pdf(t) = 0, solve for t.

        We solve the equation for t = t - _t_L to simplify calculations and add _t_L at the end of the function.

        :returns:
            roots: A numpy array of shape [batch_size, num_roots], candidates for the maximum value of the CDF.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    @AbstractArrivalDistribution.batch_size_one_function
    def trans_dens_ppf(self, theta, q=0.95):
        """The PPF of 1 - int ( p(x(dt+theta)| x(theta) =x_predTo), x(dt+theta) = - infty ..x_predTo),
        i.e., the inverse CDF of the event that particles are abovex_predTo once they have reached it at time theta.

        Note that in terms of the used approximation, this can be seen as PPF of the approximate first-passage
        returning time distribution w.r.t. the boundary x_pred_to.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param theta: A float, the (assumed) time at which x(theta) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.

        :returns: A np.array, the value of the PPF for q and theta, note that this a delta time w.r.t. theta.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def _cdf(self, t):
        """The probability P( x(t) > a).

        Approach: Under some additional assumptions (see self.cdf), it holds

            P( T_a > t) ≈ P( x(t) > a) = 1 - int( p(x(t), x= -infty ..x_predTo ) .

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the probability P( x(t) > a).
        """
        p_x_given_t = norm(loc=self._ev_t(t), scale=np.sqrt(self._var_t(t)))
        return 1 - p_x_given_t.cdf(self._x_predTo)  # TODO e.g. for t=t_q_max, this raises warnings

    @AbstractArrivalDistribution.check_density_input_dim
    def cdf(self, t):
        """The cumulative distribution function (CDF) of the first-passage time distribution.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A float or a np.array, the value of the CDF for t:
            - If the distribution is scalar (batch_size = 1)
                - and t is scalar, then returns a float,
                - and t is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and t is scalar, then returns a np.array of shape [batch_size],
                - and t is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        # t = np.asarray([t]) if np.isscalar(t) else np.asarray(t)
        # cdf_value = self._cdf(t)
        #
        # # The CDF is defined as a piecewise function that keeps its height once it has reaches q_max
        # if self.q_max.size == 1:  # batchsize = 1
        #     # cdf_value and t are of shape [sample_size], t_max and q_max are of shape [1]
        #     cdf_value[t > self.t_max] = self.q_max
        #     cdf_value = np.squeeze(cdf_value)
        # elif t.size == 1:  # batchsize != 1 and scalar t
        #     # cdf_value is of shape [batchsize], t_max and q_max are of shape [batchsize] and t is of shape [1]
        #     cdf_value[t > self.t_max] = self.q_max[t > self.t_max]  # piecewise function
        #     cdf_value = np.squeeze(cdf_value)
        # else: # batchsize != 1 and non-scalar t
        #     # cdf_value and t are of shape [batchsize, sample_size], t_max and q_max are of shape [batchsize]
        #     q_max_array = np.tile(self.q_max[:, np.newaxis], t.shape[1])
        #     cdf_value[t > self.t_max[:, np.newaxis]] = q_max_array[t > self.t_max[:, np.newaxis]]  # piecewise function
        # return cdf_value
        cdf_value = self._cdf(t)

        # The CDF is defined as a piecewise function that keeps its height once it has reaches q_max
        # q_max_array = np.tile(self.q_max[:, np.newaxis], t.shape[1])  # TODO: kann man das besser machen? np.broadcast?
        # cdf_value[t > self.t_max[:, np.newaxis]] = q_max_array[t > self.t_max[:, np.newaxis]]  # piecewise function
        cdf_value[t > self.t_max] = np.broadcast_to(self.q_max, shape=cdf_value.shape)[t > self.t_max]  # piecewise function
        return np.squeeze(cdf_value)

    @AbstractArrivalDistribution.check_density_input_dim
    def pdf(self, t):
        """The first-passage time distribution (FPTD).

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A float or a np.array, the value of the FPTD for t:
            - If the distribution is scalar (batch_size = 1)
                - and t is scalar, then returns a float,
                - and t is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and t is scalar, then returns a np.array of shape [batch_size],
                - and t is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        pdf_value = self._pdf(t)

        # The PDF is defined as a piecewise function that remains at 0 once it has reached it
        # pdf_value[t > self.t_max[:, np.newaxis]] = 0  # piecewise function
        pdf_value[t > self.t_max] = 0  # piecewise function
        return np.squeeze(pdf_value)

    def ppf(self, q, disable_double_check=False):
        """The quantile function / percent point function (PPF) of the first-passage time distribution.

        :param q: A float, the confidence parameter of the distribution, 0 <= q <= 1.
        :param disable_double_check: Boolean, if False, the results for the ppf are inserted into the cdf and
            double-checked. Disable this to save computation time.

        :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
        """
        # perform sanity checks on input
        # perform sanity checks on input
        if not np.isscalar(q):
            raise ValueError('Currently, only scalar q are supported.')
        if q <= 0.0 or q >= 1.0:  # TODO: Passt nicht ganz mit dem was in den docstrings steht! überall anpassen!
            raise ValueError('Confidence level q must be in interval (0, 1).')
        if np.any(q > self.q_max):
            logging.warning(
                'Approximation yields a maximum confidence of {}, '
                'which is lower than the desired confidence level of {}. Computed values may be wrong.'.format(
                    np.round(self.q_max, 4), np.round(q, 4)))

        # solve the inverse problem
        t, candidate_roots = self._ppf(q)

        # test it (but not in range [0.45, 0.55] as numerical issues arise in the vicinity of 0.5)
        if not disable_double_check and (q > 0.55 or q < 0.45):
            q_test = self._cdf(t)
            non_valids = np.logical_not(np.logical_or(np.isclose(q, q_test, atol=5e-2, rtol=3e-1), np.isnan(t)))
            if np.any(non_valids):
                raise ValueError(
                    'The PPF for q={0}, {1} was computed with high errors for the roots {2} with CDF {3}.'.format(q,
                                                                                                                  q_test[
                                                                                                                      non_valids],
                                                                                                                  candidate_roots[
                                                                                                                      non_valids],
                                                                                                                  self.cdf(
                                                                                                                      candidate_roots)[
                                                                                                                      non_valids]))
        return np.squeeze(t, axis=0)

    def _get_max_cdf_value_and_location(self):
        """Method that finds the maximum of the CDF of the approximation and its location.

        Approach:

            set self._pdf(t) = 0, solve for t.

        We solve the equation for t = t - _t_L to simplify calculations and add _t_L at the end of the function.

        :returns:
            q_max: A numpy array of shape [batch_size], the maximum value of the CDF.
            t_q_max: A numpy array of shape [batch_size], the time when the CDF visits its maximum.
        """
        roots = self._get_max_cdf_location_roots()
        q_max = self._cdf(roots)

        valid_idx = np.logical_and(np.isfinite(q_max),
                                   np.greater(q_max, 0, where=np.isfinite(q_max)),  # to silent the warnings
                                   np.less(q_max, 1, where=np.isfinite(q_max)))
        ambiguous = np.sum(valid_idx, axis=1) > 1
        no_valids = np.sum(valid_idx, axis=1) == 0

        if np.any(ambiguous):
            # import code
            # code.interact(local=dict(globals(), **locals()))
            raise RuntimeError('Ambiguous roots {} for t_q_max found.'.format(roots[ambiguous]))

        # Get the valid t and CDF values
        t_q_max = np.empty(valid_idx.shape[0])
        q_max_filtered = np.empty(valid_idx.shape[0])
        ro = roots[valid_idx]
        # print(ro.shape)
        t_q_max[np.logical_not(np.logical_or(ambiguous, no_valids))] = ro
        q_max_filtered[np.logical_not(np.logical_or(ambiguous, no_valids))] = q_max[valid_idx]

        if np.any(no_valids):
            # To overwrite all values, where no valid distribution can be found
            t_q_max[no_valids] = np.nan
            q_max_filtered[no_valids] = np.nan
            self._t_L[no_valids, :] = np.nan  # to force nan outputs for all calculations  #TODO: nicht sehr elegant
            logging.warning('No valid roots {} for t_q_max found.'.format(roots[no_valids]))
            # raise ValueError('No valid roots {} for t_q_max found.'.format(roots[no_valids]))

        return q_max_filtered, t_q_max

    def _get_numerical_moment_integrator(self, n=400, t_min=None, t_max=None):
        """Generator that builds a numerical integrator based on Riemann sums.

         Format compute_moment:

           f(t) --> moments of f(t)

         where
            - f(t) is a callable of t implementing the function to be integrated, e.g., f(t) = t (expectation) or
                f(t) = (t - self.ev)^2 (variance).
            - moments of f(t) is tuple of length 4 of floats or np.arrays of shape [batch_size] containing the
                moments of f(t) based on lower sums, upper sums, and the absolute and relative difference between both,
                respectively.

        :param n: An integer, the number of integration points.
        :param t_min: A float or a np.array of shape [batch_size], the location of the smallest integration point. If
            None, self.ppf(0.00005) will be used.
        :param t_max: A float or a np.array of shape [batch_size], the  location of the largest integration point. If
            None, and if self.q_max >=  0.99995, self.t_max will be used, otherwise self.ppf(0.99995) is used.

        :returns compute_moment: A callable that can be used to compute the moments
        """
        t_min = np.atleast_1d(self.ppf(0.00005) if t_min is None else t_min)
        if t_max is None:
            t_max = np.atleast_1d(self.t_max)
            t_max[0.99995 < self.q_max] = np.atleast_1d(self.ppf(0.99995))[0.99995 < self.q_max]
        else:
            t_max = np.atleast_1d(t_max)

        # # shared variables
        # delta_t = (t_max - t_min) / n
        # t_k = np.array([t_min + k * delta_t for k in range(n + 1)])  # shape n + 1
        # cdf_tk = self.cdf(t_k)  # shape n + 1
        # cdf_tk_plus_one = cdf_tk[1:]
        # interval_probs = cdf_tk_plus_one - cdf_tk[:-1]  # shape n

        # shared variables
        delta_t = (t_max - t_min) / n
        t_k = np.array([t_min + k * delta_t for k in range(n + 1)]).T  # shape [batch_size, n + 1]
        cdf_tk = self.cdf(t_k)  # shape [batch_size, n + 1]
        if cdf_tk.ndim == 1:
            # self.cdf squeezes, so in this case we need to expand again
            cdf_tk = np.expand_dims(cdf_tk, axis=0)
        cdf_tk_plus_one = cdf_tk[:, 1:]
        interval_probs = cdf_tk_plus_one - cdf_tk[:, :-1]  # shape [batch_size, n]

        # def compute_moment(fn, abs_tol=1.e-5, rel_tol=1.e-3):
        #     """Function that computes the moments based on Riemann sums.
        #
        #     The function computes the moments using the actual probability mass in each bin, which is calculated
        #     using the CDF of the approximation.
        #
        #     :param fn: function of which the expected value should be computed. E.g. use lambda t: t for the mean,
        #             lambda t: t**2, for the second moment, etc.
        #     :param abs_tol: A float, represents the absolute tolerance between lower and upper sums. If the error is
        #             higher than abs_tol, the function will throw a warning.
        #     :param rel_tol: A float, represents the relative tolerance between lower and upper sums. If the error is
        #             higher than rel_tol, the function will throw a warning.
        #
        #     :returns:
        #         lower_sum: A float, the moment computed based on lower sums.
        #         upper_sum: A float, the moment computed based on upper sums.
        #         abs_dev: A float, the absolute difference between lower and upper sum.
        #         rel_dev: A float, the relative difference between lower and upper sum.
        #     """
        #     fn_t_k = np.array(fn(t_k[:-1]))  # shape n
        #     fn_t_k_plus_one = np.array(fn(t_k[1:]))  # shape n
        #
        #     # return lower, upper sum and deviations
        #     lower_sum = np.dot(interval_probs, fn_t_k)
        #     upper_sum = np.dot(interval_probs, fn_t_k_plus_one)
        #
        #     abs_dev = abs(upper_sum - lower_sum)
        #     rel_dev = abs_dev / max(upper_sum, lower_sum)
        #
        #     if abs_dev > abs_tol:
        #         logging.warning(
        #             'Absolute Difference between lower and upper some is greater than {}. Try increasing integration points'.format(
        #                 abs_tol))
        #     if rel_dev > rel_tol:
        #         logging.warning(
        #             'Relative Difference between lower and upper some is greater than {}. Try increasing integration points'.format(
        #                 rel_tol))
        #
        #     return lower_sum, upper_sum, abs_dev, rel_dev

        def compute_moment(fn, abs_tol=1.e-3, rel_tol=1.e-2):
            """Function that computes the moments based on Riemann sums.
            The function computes the moments using the actual probability mass in each bin, which is calculated
            using the CDF of the approximation.

             Format compute_moment:

                f(t) --> moments of f(t)

             where
                - f(t) is a callable of t implementing the function to be integrated, e.g., f(t) = t (expectation) or
                    f(t) = (t - self.ev)^2 (variance).
                - moments of f(t) is tuple of length 4 of floats or np.arrays of shape [batch_size] containing the
                    moments of f(t) based on lower sums, upper sums, and the absolute and relative difference between
                    both, respectively.

            :param fn: function of which the expected value should be computed. E.g. use lambda t: t for the mean,
                    lambda t: t**2, for the second moment, etc.
            :param abs_tol: A float, represents the absolute tolerance between lower and upper sums. If the error is
                    higher than abs_tol, the function will throw a warning.
            :param rel_tol: A float, represents the relative tolerance between lower and upper sums. If the error is
                    higher than rel_tol, the function will throw a warning.
            :return:
                lower_sum: A float or a np.array of shape [batch_size], the moment computed based on lower sums.
                upper_sum: A float or a np.array of shape [batch_size], the moment computed based on upper sums.
                abs_dev: A float or a np.array of shape [batch_size], the absolute difference between lower and upper
                    sum.
                rel_dev: A float or a np.array of shape [batch_size], the relative difference between lower and upper
                    sum.
            """
            fn_t_k = np.array(fn(t_k[:, :-1]))  # shape [batchsize, n]
            fn_t_k_plus_one = np.array(fn(t_k[:, 1:]))  # shape [batchsize, n]

            # return lower, upper sum and deviations
            # lower_sum = np.dot(interval_probs, fn_t_k)
            # upper_sum = np.dot(interval_probs, fn_t_k_plus_one)
            lower_sum = np.sum(interval_probs * fn_t_k, axis=1)  # shape n
            upper_sum = np.sum(interval_probs * fn_t_k_plus_one, axis=1) # shape n

            abs_dev = np.absolute(upper_sum - lower_sum)
            rel_dev = abs_dev / np.maximum(upper_sum, lower_sum)

            if np.any(abs_dev > abs_tol):
                abs_dev_too_high = abs_dev[abs_dev > abs_tol]
                logging.warning(
                    'Absolute Difference {0} between lower and upper sum is greater than {1}. Try increasing the number '
                    'of integration points'.format(
                        abs_dev_too_high, abs_tol))
            if np.any(rel_dev > rel_tol):
                rel_dev_too_high = rel_dev[rel_dev > rel_tol]
                logging.warning(
                    'Relative Difference {0} between lower and upper sum is greater than {1}. Try increasing the number '
                    'of integration points'.format(
                        rel_dev_too_high, rel_tol))

            return np.squeeze(lower_sum), np.squeeze(upper_sum), np.squeeze(abs_dev), np.squeeze(rel_dev)

        return compute_moment

    @AbstractArrivalDistribution.batch_size_one_function
    def plot_valid_regions(self,
                           theta=None,
                           q=0.95,
                           plot_t_min=0.0,
                           plot_t_max=None,
                           save_results=False,
                           result_dir=None,
                           for_paper=True,
                           no_show=False):
        """Plot the (approximate) probabilities that the track doesn't intersect withx_predTo once it has reached
        it at time theta in dependency on the time difference dt (t = dt + theta) and theta.

        Note that, there are intervals in time for which is it very unlikely (with confidence q) that the track falls
        belowx_predTo again. These are the desired regions of validity.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param theta: A float, the (assumed) time at which x(theta) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.
        :param plot_t_min: A float, the lower time value of the plot range.
        :param plot_t_max: A float, the upper time value of the plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        :param no_show: Boolean, whether to show the plots (False).
        """
        t_pred = self.ppf(0.5)  # first-passage solution for the mean function
        if theta is None:
            multipliers = np.arange(start=0.4, stop=1.8, step=0.2)
            plot_theta = [t_pred * m for m in multipliers]
        else:
            # We only plot the plot for the given theta
            plot_theta = [theta]

        roots = self.trans_dens_ppf(plot_theta[0], q)  # take the first one, as it gives the largest delta t (at least
        # in the vicinity of t_pred)
        root = roots if np.isscalar(roots) else roots[0]  # take the largest one, roots are in descending order
        plot_t_max = plot_t_max if plot_t_max is not None else root * 1.4  # take the largest one, roots are in
        # descending order
        plot_t = np.linspace(plot_t_min, plot_t_max, num=10000)

        # Compute the probability that the tracks stay above x_pred_to
        # This is an approximation since it doesn't take into account tracks that fall below and climb back above
        # x_pred_to before dt.

        # Probability of the event that the transition density for x(theta + dt) is higher than x_pred_to.
        trans_fpt = lambda dt, theta: 1 - self.trans_density(dt, theta).cdf(self.x_predTo)

        fig, ax = plt.subplots()
        plt.hlines(q, xmin=plot_t_min, xmax=plot_t_max, color='black', linestyle='dashdot',
                   label='Confidence q={}'.format(q))

        # Plot the Probabilities
        for i, theta in enumerate(plot_theta):
            plot_prob = [trans_fpt(dt, theta) for dt in plot_t]

            if len(plot_theta) == 1:
                label = None
            else:
                label = r"$\theta$: {} $\cdot$ det. pred.".format(round(multipliers[i], 1))

            if np.isclose(theta, t_pred):
                # Mark the valid regions
                # TODO: The second case is not necessary
                roots = self.trans_dens_ppf(theta, q)
                if np.isscalar(roots) or roots.shape[0] == 1:
                    plt.axvspan(0, np.squeeze(roots), alpha=0.6, color='green', label='Valid region')
                elif roots.shape[0] == 2:
                    plt.axvspan(roots[1], roots[0], alpha=0.6, color='green', label='Valid region')
                else:
                    logging.WARNING('{} valid roots detected.'.format(roots.shape[0]))

            ax.plot(plot_t, plot_prob, label=label)

        ax.set_xlim(0, None)
        plt.ylabel('Confidence')
        plt.xlabel('Time difference in s')
        plt.legend(loc='upper left')

        if not for_paper:
            plt.title('Probability of Staying Above the Boundary')

        if save_results:
            plot_name = "_valid_regions" if len(plot_theta) != 1 else "_valid_region_for_" + str(round(theta, 4))
            basename = os.path.basename(os.path.normpath(result_dir))
            process_name_save = basename.lower().replace(" ", "_")
            plt.savefig(os.path.join(result_dir, process_name_save + plot_name + '.pdf'))
            plt.savefig(os.path.join(result_dir, process_name_save + plot_name + '.png'))
            plt.savefig(os.path.join(result_dir, process_name_save + plot_name + '.pgf'))
        if not no_show:
            plt.show()
        plt.close()

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._x_predTo[indices] = values._x_predTo   # TODO: hier sollte man nicht nochmal xpretdo überschreiben
        self._q_max[indices] = values.q_max  # TODO: Call to super?
        self._t_max[indices] = values.t_max
        if self.ev_available and values.ev_available:
            self._ev[indices] = values.ev
        else:
            self._ev = None
        if self.var_available and values.var_available:
            self._var[indices] = values.var
        else:
            self._var = None
        if self.third_central_moment_available and values.third_central_moment_available:
            self._third_central_moment[indices] = values.third_central_moment
        else:
            self._third_central_moment = None

        self._compute_moment = None  # numerical moment integrator must be recalculated

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._q_max = values.q_max[indices]
        self._t_max = values.t_max[indices]
        if values.ev_available:
            self._ev = values.ev[indices]
        if values.var_available:
            self._var = values.var[indices]
        if values.third_central_moment_available:
            self._third_central_moment = values.third_central_moment[indices]
        super()._left_hand_indexing(indices, values)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according t scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        super().scale_params(length_scaling_factor, time_scaling_factor)
        # Force recalculating all privates
        self._q_max = None
        self._t_max = None
        self._ev = None
        self._var = None
        self._third_central_moment = None
        self._compute_moment = None  # numerical moment integrator must be recalculated after a scaling

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'PPF': self.ppf,
                          'Median': self.ppf(0.5),
                          'FirstQuantile': self.ppf(0.25),
                          'ThirdQuantile': self.ppf(0.75),
                          'q_max': self.q_max,
                          't_max': self.t_max,
                          # 'ReturningProbs': self.returning_probs,  # do not use
                          'ReturningProbs': self.returning_probs_uniform_samples,
                          # 'ReturningProbs': self.returning_probs_integrate_quad,
                          })
        return hit_stats


class AbstractUniformHittingTimeDistribution(AbstractUniformArrivalDistribution, AbstractHittingTimeDistribution, ABC):
    """Uses point predictors for the arrival time prediction and a uniform distribution.

    This distribution corresponds to the "usual" case where we define a fixed ejection window.
    """
    def __init__(self, name='AbstractUniformHittingTimeDistribution', **kwargs):
        """Initializes the distribution.

        :param name: String, the (default) name for the distribution.
        """
        super().__init__(name=name,
                         **kwargs,
                         )

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'PPF': self.ppf,
                          })
        return hit_stats

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._point_prediction *= time_scaling_factor  # TODO. Das gilt aber auch nicht immer (gilt nur wenn nur zeit gescaled?). Wird das benötigt oder sowieso überschrieben?
        self._window_length *= time_scaling_factor


class AbstractMCHittingTimeDistribution(AbstractMCArrivalDistribution, AbstractHittingTimeDistribution, ABC):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem to a distribution
     using scipy.stats.rv_histogram.

    """
    def __init__(self, t_range, t_samples=None, name="AbstractMCHittingTimeDistribution", **kwargs):
        """Initializes the distribution.
.
        :param name: String, the (default) name for the distribution.
          :param t_range: A list of length 2 representing the limits for the first-passage time histogram (the number of
            bins within t_range will correspond to bins).
        :param t_samples: None or a np.array of shape [num_samples] containing the first-passage times of the particles.
            If None, t_samples will be created by a call to a sampling method. If given, given values will be used.
        """
        super().__init__(name=name,
                         range=t_range,
                         samples=t_samples,
                         **kwargs)

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'PPF': self.ppf,
                          # 'ReturningProbs': self.returning_probs,  # do not use
                          'ReturningProbs': self.returning_probs_uniform_samples,  # TODO: was machen wir damit?
                          # 'ReturningProbs': self.returning_probs_integrate_quad,
                          })
        return hit_stats

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._samples *= time_scaling_factor   # TODO. Das gilt aber auch nicht immer (gilt nur wenn nur zeit gescaled?). Wird das benötigt oder sowieso überschrieben?
        self._range *= time_scaling_factor
        # Recalculate the distribution
        self._density = self._build_distribution_from_samples(self._samples, self._range)


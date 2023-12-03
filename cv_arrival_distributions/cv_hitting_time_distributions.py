from absl import logging

from abc import ABC, abstractmethod

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm, multivariate_normal

from abstract_distributions import AbstractArrivalDistribution
from abstract_hitting_time_distributions import AbstractHittingTimeDistribution, \
    AbstractGaussTaylorHittingTimeDistribution, AbstractNoReturnHittingTimeDistribution, \
    AbstractUniformHittingTimeDistribution, AbstractMCHittingTimeDistribution
from cv_arrival_distributions.cv_utils import get_system_matrices_from_parameters, create_ty_cv_samples_hitting_time  # TODO: ist das file nun noch notwendig?


class AbstractCVHittingTimeDistribution(AbstractHittingTimeDistribution, ABC):
    """A base class for the CV hitting time distributions."""

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name=' AbstractCVHittingTimeDistribution', **kwargs):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, ... (e.g., pos_x, velo_x, if a CV model for both x and y is used)]

        :param x_L: A np.array of shape [state_length] or [batch_size, state_length] representing the expected value of
            the initial state. We use index L here because it usually corresponds to the last time we see a particle in
            our optical belt sorting scenario.
        :param C_L: A np.array of shape [state_size, state_length] or [batch_size, state_length, state_length]
            representing the covariance matrix of the initial state.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in x-direction.
        :param x_predTo: A float or np.array of shape [batch_size], the position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, the (default) name for the distribution.
        """
        # sanity checks
        if x_L.shape[-1] < 2:
            raise ValueError('The state length must be at least 2.')
        if not np.array_equal(np.atleast_2d(x_L).shape, np.atleast_3d(C_L).shape[:2]):
            raise ValueError('Shapes of x_L and C_L do not match.')
        if C_L.shape[-2] != C_L.shape[-1]:
            raise ValueError('C_L must be a symmetric matrix.')

        self._x_L = np.atleast_2d(x_L)
        self._C_L = np.atleast_3d(C_L)
        self._S_w = np.broadcast_to(S_w, shape=self.batch_size)  # this itself raises an error if not compatible

        super().__init__(x_predTo=x_predTo,
                         t_L=t_L,
                         name=name,
                         **kwargs)

    @property
    def x_L(self):
        """The expected value of the initial state.

        :returns: A np.array of shape [state_length] or [batch_size, state_length], the expected value of the initial
            state.
        """
        return np.squeeze(self._x_L, axis=0)

    @property
    def C_L(self):
        """The covariance matrix of the initial state.

        :returns: A np.array of shape [state_length, state_length] or [batch_size, state_length, state_length]
            representing the covariance matrix of the initial state.
        """
        return np.squeeze(self._C_L, axis=0)

    @property
    def S_w(self):
        """The power spectral density (PSD) in x-direction.

        :returns A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        return np.squeeze(self._S_w, axis=0)

    @S_w.setter
    @abstractmethod
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def batch_size(self):
        """The batch size of the distribution.

        :returns: An integer, the batch size.
        """
        return len(self._x_L)

    def _ev_t(self, t):  # TODO: Was machen wir hier mit point predictor? Braucht man die Funktion hier überhaupt oder nur in klassen die sie aufrufen
        """The mean function of the CV motion model in x.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the mean in x at time t.
        """
        # return self.x_L[:, 0] + self.x_L[:, 1] * (t - self._t_L)
        return self._x_L[:, np.newaxis, 0] + self._x_L[:, np.newaxis, 1] * (t - self._t_L)

    def _var_t(self, t):
        """The variance function of the CV motion model in x.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the variance in x at time t.
        """
        # return self.C_L[:, 0, 0] + 2 * self.C_L[:, 1, 0] * (t - self._t_L) + self.C_L[:, 1, 1] * (
        #             t - self._t_L) ** 2 + self.S_w[:, ] * pow(t - self._t_L, 3) / 3
        return self._C_L[:, np.newaxis, 0, 0] + 2 * self._C_L[:, np.newaxis, 1, 0] * (t - self._t_L) \
               + self._C_L[:, np.newaxis, 1, 1] * (t - self._t_L) ** 2 \
               + self._S_w * pow(t - self._t_L, 3) / 3

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
        cov_theta = np.array([[self.C_L[0, 0] + 2 * self.C_L[1, 0] * (theta - self._t_L)
                               + self.C_L[1, 1] * (theta - self._t_L) ** 2 + self.S_w * pow(theta - self._t_L, 3) / 3,
                               self.C_L[1, 0] + self.C_L[1, 1] * (theta - self._t_L) + self.S_w * pow(theta - self._t_L,
                                                                                                      2) / 2],
                              [self.C_L[1, 0] + self.C_L[1, 1] * (theta - self._t_L) + self.S_w * pow(theta - self._t_L,
                                                                                                      2) / 2,
                               self.C_L[1, 1] + self.S_w * (theta - self._t_L)]])
        trans_mu = self.x_predTo + dt * (self.x_L[1] + cov_theta[1, 0] / cov_theta[0, 0] * (
                    self.x_predTo - np.squeeze(self._ev_t(theta), axis=0)))
        trans_var = dt ** 2 * (
                cov_theta[1, 1] - cov_theta[1, 0] / cov_theta[0, 0] * cov_theta[0, 1]) + self.S_w * 1 / 3 * dt ** 3

        return norm(loc=trans_mu, scale=np.sqrt(trans_var))

    @AbstractArrivalDistribution.batch_size_one_function
    def trans_prob_with_positive_velocity(self, dt, theta):
        """The transition probability P(x(dt+theta) > a | x(theta) =x_predTo, v(theta)= > 0) from going fromx_predTo
        at time theta and positive velocity to x(dt+theta) < a at time dt+theta.

        P(x(t) < a | x(theta) = a, v(theta) > 0) is calculated according to

            P(x(t) < a | x(theta) = a, v(theta) > 0) = P(x(t) < a , v(theta) > 0 | x(theta) = a) /
                                                            P( v(theta) > 0 | x(theta) = a) ,

            and further

            P(x(t) < a , v(theta) > 0 | x(theta) = a) = P(x(t) < a , v(theta) < inf | x(theta) = a)
                                                            - P(x(t) < a , v(theta) < 0| x(theta) = a) .

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param dt: A float, the time difference. dt is zero at time = theta.
        :param theta: A float, the (assumed) time at which x(theta) = x_pred_to.

        :returns: A float, the probability P(x(dt+theta) > a | x(theta) =x_predTo, v(theta)= > 0).
        """
        F_theta, Q_theta = get_system_matrices_from_parameters(theta - self._t_L, self.S_w)
        F_t, Q_t = get_system_matrices_from_parameters(theta + dt - self._t_L, self.S_w)
        mean_theta = np.dot(F_theta, self.x_L)
        cov_theta = np.matmul(np.matmul(F_theta, self.C_L), F_theta.T) + Q_theta
        mean_t = np.dot(F_t, self.x_L)
        cov_t = np.matmul(np.matmul(F_t, self.C_L), F_t.T) + Q_t
        F_trans, Q_trans = get_system_matrices_from_parameters(dt, self.S_w)
        auto_cov_theta_to_t = np.matmul(F_trans, cov_theta)

        # we begin with P( v(theta) > 0 | x(theta) = a )
        p_v_theta_given_x_theta_mean = mean_theta[1] + cov_theta[1, 0]/cov_theta[0, 0]*(self.x_predTo - mean_theta[0])
        p_v_theta_given_x_theta_cov = cov_theta[1, 1] - cov_theta[1, 0] / cov_theta[0, 0] * cov_theta[0, 1]
        prob_v_theta_greater_zero_given_x_theta = 1 - norm.cdf(0,
                                                               loc=p_v_theta_given_x_theta_mean,
                                                               scale=np.sqrt(p_v_theta_given_x_theta_cov))

        # now, P(x(t) < a , v(theta) > 0 | x(theta) = a ), the joint of [x(theta), v(theta), x(t), v(t)] is given by
        joint_theta_t_mean = np.block([mean_theta[:2], mean_t[:2]])
        joint_theta_t_cov = np.block([[cov_theta[:2, :2], auto_cov_theta_to_t[:2, :2].T],
                                      [auto_cov_theta_to_t[:2, :2], cov_t[:2, :2]]])

        # thus, conditioning on x(theta) yields
        p_joint_v_theta_x_t_given_x_theta_mean = joint_theta_t_mean[[1, 2]] + joint_theta_t_cov[[1, 2], [0]] / \
                                                 joint_theta_t_cov[0, 0] * (self.x_predTo - mean_theta[0])
        p_joint_v_theta_x_t_given_x_theta_cov = joint_theta_t_cov[1:3, 1:3] - 1 / joint_theta_t_cov[
            0, 0] * np.outer(joint_theta_t_cov[[1, 2], [0]], joint_theta_t_cov[[0], [1, 2]])

        joint_v_theta_x_t_given_x_theta = multivariate_normal(mean=p_joint_v_theta_x_t_given_x_theta_mean,
                                                              cov=p_joint_v_theta_x_t_given_x_theta_cov,
                                                              allow_singular=True)  # To allow for numerical edge cases
        prob_joint_v_theta_greater_zero_x_t_smaller_a_given_x_theta = joint_v_theta_x_t_given_x_theta.cdf(
            np.array([np.inf, self.x_predTo])) - joint_v_theta_x_t_given_x_theta.cdf(np.array([0, self.x_predTo]))

        return prob_joint_v_theta_greater_zero_x_t_smaller_a_given_x_theta / prob_v_theta_greater_zero_given_x_theta

    @AbstractArrivalDistribution.batch_size_one_function
    def returning_probs_conditioned_on_positive_velocity_integrate_quad(self, t):
        """Calculates approximate returning probabilities using numerical integration thereby using the transition
        probabilities that are conditioned on a positive velocity at the time of a crossing.

        Approach:

         P(t < T_a , x(t) < a) = int_{_t_L}^t fptd(theta) P(x(t) < a | x(theta) = a, v(theta) > 0) d theta ,

          with theta the time, when x(theta) = a.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param t: A float, the time parameter of the distribution.

        :returns: A float, an approximation for the probability P(t < T_a , x(t) < a), i.e., the probability that a
            sample path  has crossed the boundary at a time theta < t, but is smaller than the boundary at time t.
        """
        fn = lambda theta: self.pdf(theta) * self.trans_prob_with_positive_velocity(dt=t - theta,
                                                                                    theta=theta)
        a = np.finfo(np.float64).eps if self._t_L == 0 else self._t_L
        return integrate.quad(fn, a=a, b=t)[0]  # this is a tuple

    @AbstractArrivalDistribution.batch_size_one_function
    def returning_probs_conditioned_on_positive_velocity_uniform_samples(self, t, num_samples=100, deterministic_samples=True):
        """Calculates approximate returning probabilities (with the transition probabilities that are conditioned on a
        positive velocity at the time of a crossing) using a numerical integration (MC integration) based on
        samples from a uniform distribution.

        Approach:

         P(t < T_a , x(t) < a) = int_{_t_L}^t fptd(theta) P(x(t) < a | x(theta) = a, v(theta) > 0) d theta

                          ≈  (t - _t_L) / N sum_{theta_i} FPTD(theta_i) * P(x(t) < a | x(theta_i) = a, v(theta_i) > 0) ,
                               theta_i samples from a uniform distribution (N samples in total) in [_t_L, t] ,

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
            [self.pdf(theta) * self.trans_prob_with_positive_velocity(dt=t - theta,
                                                                      theta=theta) for
             theta in theta_samples])

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        super().scale_params(length_scaling_factor, time_scaling_factor)

        self._x_L[:, [0, 2]] *= length_scaling_factor
        self._x_L[:, [1, 3]] *= length_scaling_factor / time_scaling_factor

        self._C_L[:, 0, 0] *= length_scaling_factor ** 2
        self._C_L[:, 2, 2] *= length_scaling_factor ** 2
        self._C_L[:, 1, 1] *= (length_scaling_factor / time_scaling_factor) ** 2
        self._C_L[:, 3, 3] *= (length_scaling_factor / time_scaling_factor) ** 2
        self._C_L[:, 0, 1] *= length_scaling_factor ** 2 / time_scaling_factor
        self._C_L[:, 1, 0] *= length_scaling_factor ** 2 / time_scaling_factor
        self._C_L[:, 2, 3] *= length_scaling_factor ** 2 / time_scaling_factor
        self._C_L[:, 3, 2] *= length_scaling_factor ** 2 / time_scaling_factor

        self._S_w *= length_scaling_factor ** 2 / time_scaling_factor ** 3

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._x_L[indices] = values.x_L  # TODO: Call to super?
        self._C_L[indices] = values.C_L
        self._S_w[indices] = values.S_w

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._x_L = values.x_L[indices]
        self._C_L = values.C_L[indices]
        self._S_w = values.S_w[indices]
        super()._left_hand_indexing(indices, values)

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'SKEW': self.skew,  # TODO. in der abstract ist es immer drin, macht das sinn?
                          # 'ReturningProbs': self.returning_probs_conditioned_on_positive_velocity_integrate_quad,
                          # 'ReturningProbs': self.returning_probs_conditioned_on_positive_velocity_uniform_samples,
                          })
        return hit_stats


class GaussTaylorCVHittingTimeDistribution(AbstractCVHittingTimeDistribution, AbstractGaussTaylorHittingTimeDistribution):
    """A simple Gaussian approximation for the first-passage time problem using a Taylor approximation and error
    propagation that can be used for CV models.

    """
    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, point_predictor, name='Gauß--Taylor approx.'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, pos_y, velo_y, ...]

         Format point_predictor:

            (pos_last, v_last)  --> dt_pred

         where
            - pos_last is a np.array of shape [batch_size, 2] and format [x, y] containing the positions at the _t_L.
            - v_last is a np.array of shape [batch_size, 2] and format [x, y] containing the velocities at the _t_L.
            - dt_pred is a np.array of shape [batch_size] with arrival time point estimates as difference times w.r.t.
                _t_L.

        :param x_L: A np.array of shape [state_length] or [batch_size, state_length] representing the expected value of
            the initial state. We use index L here because it usually corresponds to the last time we see a particle in
            our optical belt sorting scenario.
        :param C_L: A np.array of shape [state_size, state_length] or [batch_size, state_length, state_length]
            representing the covariance matrix of the initial state.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in x-direction.
        :param x_predTo: A float or np.array of shape [batch_size], the position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param point_predictor: A callable, a function that returns an estimate for the arrival time.
        :param name: String, the name for the distribution.
        """
        # sanity checks
        if not callable(point_predictor):
            raise ValueError('point_predictor must be a callable.')

        AbstractCVHittingTimeDistribution.__init__(self,
                                                   x_L=x_L,
                                                   C_L=C_L,
                                                   S_w=S_w,
                                                   x_predTo=x_predTo,
                                                   t_L=t_L,
                                                   name=name,
                                                   )

        ev = point_predictor(self._x_L[:, [0, 2]], self._x_L[:, [1, 3]])
        var = self._compute_var(ev, self._x_L, self._C_L, self._t_L, self._S_w)
        AbstractGaussTaylorHittingTimeDistribution.__init__(self,
                                                            ev=ev,
                                                            var=var,
                                                            name=name,
                                                            )

        # pos_last = np.atleast_2d(x_L)[:, [0, 2]]  # TODO: Könnte man sich das alles mit der richtigen Vererbungsreihenfolge sparen?
        # v_last = np.atleast_2d(x_L)[:, [1, 3]]
        # ev = point_predictor(pos_last, v_last)
        #
        # # Evaluate the equation at the time at the boundary
        # var = self._compute_var(ev, np.atleast_2d(x_L), np.atleast_3d(C_L), _t_L, np.broadcast_to(S_w, shape=np.atleast_2d(x_L).shape[0]))
        # super().__init__(x_L=x_L,
        #                  C_L=C_L,
        #                  S_w=S_w,
        #                 x_predTo=_x_predTo,
        #                  _t_L=_t_L,
        #                  name=name,
        #                  ev=ev,
        #                  var=var)

        # self._ev = _t_L + (x_predTo - x_L[0]) / x_L[1]
        # # Evaluate the equation at the time at the boundary
        # v_L = x_L[1]
        # dt_p = self.ev - _t_L
        # sigma_x = np.sqrt(C_L[0, 0] + 2 * C_L[1, 0] * dt_p + C_L[1, 1] * dt_p**2 + S_w * pow(dt_p, 3)/3)
        # sigma_xv = C_L[1, 0] + C_L[1, 1] * dt_p + S_w * pow(dt_p, 2)/2
        # sigma_v = np.sqrt(C_L[1, 1] + S_w * dt_p)
        # dx = 0
        # self._var = (1 / v_L) ** 2 * sigma_x ** 2 + (dx / v_L ** 2) ** 2 * sigma_v ** 2 + 2 * dx / v_L ** 3 * sigma_xv

    def _compute_var(self, point_prediction, x_L, C_L, t_L, S_w):  # TODO. Die methode ist eig. static
        """Computes the variance of the first-passage time problem based on error propagation.

        :param point_prediction: A np.array of shape [batch_size], a point_prediction used as the distribution's
            expectation.
        :param x_L: A np.array of shape [state_length] or [batch_size, state_length] representing the expected value of
            the initial state.
        :param C_L: A np.array of shape [state_size, state_length] or [batch_size, state_length, state_length]
            representing the covariance matrix of the initial state.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param S_w: A np.array of shape [batch_size], the power spectral density (PSD).

        :returns: A np.array of shape [batch_size], the variance of the approximation.
        """
        # TODO: Das kann man vereinfachen
        dt_p = point_prediction - t_L
        sigma_x = np.sqrt(
            C_L[:, 0, 0] + 2 * C_L[:, 1, 0] * dt_p + C_L[:, 1, 1] * dt_p ** 2 + S_w * pow(dt_p, 3) / 3)
        var = (1 / x_L[:, 1]) ** 2 * sigma_x ** 2
        return var

    @AbstractCVHittingTimeDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size)
        # Recalculate the variance
        self._var = self._compute_var(self._ev, self._x_L, self._C_L, self._t_L, self._S_w)


class NoReturnCVHittingTimeDistribution(AbstractCVHittingTimeDistribution, AbstractNoReturnHittingTimeDistribution):
    """An approximation to the first-passage time distribution for CV models using the assumption that particles are
    unlikely to move back once they have passed the boundary.

    """
    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name="No-return approx."):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, ... (e.g., pos_x, velo_x, if a CV model for both x and y is used)]

        :param x_L: A np.array of shape [state_length] or [batch_size, state_length] representing the expected value of
            the initial state. We use index L here because it usually corresponds to the last time we see a particle in
            our optical belt sorting scenario.
        :param C_L: A np.array of shape [state_size, state_length] or [batch_size, state_length, state_length]
            representing the covariance matrix of the initial state.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in x-direction.
        :param x_predTo: A float or np.array of shape [batch_size], the position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, the name for the distribution.
        """
        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         name=name)

        if np.any(self.q_max) < 0.95:
            q_max_too_small = self.q_max[self.q_max < 0.95]
            logging.warning(
                "{} does not seem to be applicable as max. confidence {} < 0.95.".format(self.__class__.__name__,
                                                                                         np.round(q_max_too_small, 2)))

    # @classmethod
    # def _from_private_arguments(cls, x_L, C_L, S_w,x_predTo, _t_L, q_max, t_max,
    #                             ev=None,
    #                             var=None,
    #                             third_central_moment=None,
    #                             compute_moment=None,
    #                             name="No-return approx."):
    #     """Initializes the distribution with given private arguments.
    #
    #     This may be used for creating and returning a (potentially modified) copy of the distribution.
    #
    #     :param x_L: A np.array of shape [state_length] or [batch_size, state_length] representing the expected value of
    #         the initial state. We use index L here because it usually corresponds to the last time we see a particle in
    #         our optical belt sorting scenario.
    #     :param C_L: A np.array of shape [state_size, state_length] or [batch_size, state_length, state_length]
    #         representing the covariance matrix of the initial state.
    #     :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD).
    #     :paramx_predTo: A float or np.array of shape [batch_size], the position of the boundary.
    #     :param _t_L: A float, the time of the last state/measurement (initial time).
    #     :param q_max: A float or np.array of shape [batch_size], the maximum value of the CDF.
    #     :param t_max: A float or np.array of shape [batch_size], the time, when the CDF visits its maximum.
    #     :param ev: A float or a np.array of shape [batch_size], the expected value of the first-passage time
    #         distribution.
    #     :param var: A float or a np.array of shape [batch_size], the variance of the first-passage time distribution.
    #     :param third_central_moment: A float or a np.array of shape [batch_size], the third central moment of the
    #         first-passage time distribution.
    #     :param compute_moment: A callable that can be used to compute the moments.
    #     :param name: String, the (default) name for the distribution.
    #
    #     :returns: A NoReturnCVHittingTimeDistribution object, a (potentially modified) copy of the distribution.
    #     """
    #     obj = cls.__new__(cls)  # create a new object
    #     super(cls, obj).__init__(x_L, C_L, S_w,x_predTo, _t_L, name)
    #
    #     # overwrite all privates
    #     obj._q_max = q_max
    #     obj._t_max = t_max
    #     obj._ev = ev
    #     obj._var = var
    #     obj._third_central_moment = third_central_moment
    #     obj._compute_moment = compute_moment
    #
    #     return obj

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
        # gauss = norm.pdf((self._x_predTo - self._ev_t(t))/np.sqrt(self._var_t(t)))  # Std. Gauss pdf
        # if t.ndim == 2:
        #     der_arg = self.x_L[:, np.newaxis, 1] / np.sqrt(self._var_t(t)) + (self._x_predTo - self._ev_t(t)) * (
        #             0.5 * self.S_w * t ** 2 - self.S_w * t * self._t_L + 0.5 * self.S_w * self._t_L ** 2 + t * self.C_L[:, np.newaxis,
        #         1, 1] - self._t_L * self.C_L[:, np.newaxis, 1, 1] + self.C_L[:, np.newaxis, 1, 0]) / (self._var_t(t)) ** (3.0 / 2.0)
        # else:
        #     der_arg = self.x_L[:, 1] / np.sqrt(self._var_t(t)) + (self._x_predTo - self._ev_t(t)) * (
        #                 0.5 * self.S_w * t ** 2 - self.S_w * t * self._t_L + 0.5 * self.S_w * self._t_L ** 2 + t * self.C_L[:,
        #             1, 1] - self._t_L * self.C_L[:, 1, 1] + self.C_L[:, 1, 0]) / (self._var_t(t)) ** (3.0 / 2.0)
        # return gauss * der_arg

        gauss = norm.pdf((self._x_predTo - self._ev_t(t)) / np.sqrt(self._var_t(t)))  # Std. Gauss pdf

        der_arg = self._x_L[:, np.newaxis, 1] / np.sqrt(self._var_t(t)) + (self._x_predTo - self._ev_t(t)) * (
                0.5 * self._S_w * t ** 2 - self._S_w * t * self._t_L + 0.5 * self._S_w * self._t_L ** 2 + t * self._C_L[:,
                np.newaxis, 1, 1] - self._t_L * self._C_L[:, np.newaxis, 1, 1] + self._C_L[:, np.newaxis, 1, 0]) / (
                self._var_t(t)) ** (3.0 / 2.0)

        return gauss * der_arg

    def _ppf(self, q):
        """The quantile function / percent point function (PPF) of the first-passage time distribution.

        Approach:

              1 - q = int(N(x, mu(t), var(t)), x = -inf ..x_predTo) = PHI ( (x_predTo - mu(t)) / sqrt(var(t))
              PHI^-1(1 -q) = (x_predTo - mu(t)) / sqrt(var(t)) -> solve for t...

        We solve the equation for t = t - _t_L to simplify calculations and add _t_L at the end of the function.

        :param q: A float, the confidence parameter of the distribution, 0 <= q <= 1.

        :returns:
            t: A np.array of shape [batch_size], the value of the PPF for q.
            candidate_roots: A np.array of shape [batch_size, 3] containing the values of all possible roots.
        """
        # We compute the ppf for t = t - _t_L to simplify calculations.

        # Special case q = 0.5, this corresponds to the median.
        # 0 = (x_predTo - mu(t)) / sqrt(var(t) ->x_predTo = mu(t) -> solve for t...
        if q == 0.5:
            t = (self._x_predTo - self._x_L[:, 0]) / self._x_L[:, 1]
            return np.squeeze(t + self._t_L)

        # cubic function
        # At**3 + B*t**2 + C*t + D = 0
        qf = norm.ppf(1 - q)  # Standard-Gaussian quantile function
        epsilon = 0.0  # to avoid divisions by zero, 0.01 is a good value for Cardano

        A = self._S_w / 3 * (qf ** 2 + epsilon)
        B = self._C_L[:, 1, 1] * (qf ** 2 + epsilon) - self._x_L[:, 1] ** 2
        C = 2 * self._C_L[:, 1, 0] * (qf ** 2 + epsilon) + 2 * (self._x_predTo - self._x_L[:, 0]) * self._x_L[:, 1]
        D = self._C_L[:, 0, 0] * (qf ** 2 + epsilon) - (self._x_predTo - self._x_L[:, 0]) ** 2
        # A may be a scalar or a np.array, so ensure that the shape is equal to the ones of B, C, D.
        A = np.tile(A, B.shape[0]) if np.isscalar(A) else A

        roots = self._find_cubic_roots_cardano_complex(A, B, C, D) + self._t_L
        t = roots[:, 1] if q < 0.5 else roots[:,
                                        2]  # TODO: Gibt es hier egal welche funktion man verwendet immer eine natürliche Ordnung? Ansonsten ist das nicht failsafe..

        return t, roots

    @staticmethod
    def _find_cubic_roots_cardano_complex(a, b, c, d):
        """Find the real roots of a cubic polynomial a*x^3 + b*x^2 + c*x + d = 0 using Cardano's method and
        complex numbers. This is numerically more stable than with trigonometric functions. The functions supports
        batch-wise calculations. In this case, a, b, c, and d are 1-D arrays consisting of the coefficients for the
        individual, independent polynomials.

        Based on https://stackoverflow.com/questions/39474254/cardanos-formula-not-working-with-numpy

        :param a: A np.array of shape [batch_size].
        :param b: A np.array of shape [batch_size].
        :param c: A np.array of shape [batch_size].
        :param d: A np.array of shape [batch_size].

        :returns: A np.array of shape [batch_size, 3]. Complex roots are replaced by np.nan.
        """
        if a.shape != b.shape or b.shape != c.shape or c.shape != d.shape:
            raise ValueError('Shape of all passed array must be equal.')
        if a.ndim == 0 or a.ndim > 1:
            raise ValueError('The arrays must be one-dimensional.')

        a, b, c, d = a + 0j, b + 0j, c + 0j, d + 0j
        all_ = (a != np.pi)

        Q = (3 * a * c - b ** 2) / (9 * a ** 2)
        R = (9 * a * b * c - 27 * a ** 2 * d - 2 * b ** 3) / (54 * a ** 3)
        D = Q ** 3 + R ** 2

        # S = 0
        # if np.isreal(R + np.sqrt(D)):
        #     S = cbrt(np.real(R + np.sqrt(D)))
        # else:
        #     S = (R + np.sqrt(D)) ** (1 / 3)
        S = np.zeros(R.shape, dtype=complex)
        S[np.isreal(R + np.sqrt(D))] = np.cbrt(np.real(R + np.sqrt(D)))[np.isreal(R + np.sqrt(D))]
        S[np.logical_not(np.isreal(R + np.sqrt(D)))] = np.power(R + np.sqrt(D), 1 / 3)[
            np.logical_not(np.isreal(R + np.sqrt(D)))]

        # T = 0
        # if np.isreal(R - np.sqrt(D)):
        #     T = cbrt(np.real(R - np.sqrt(D)))
        # else:
        #     T = (R - np.sqrt(D)) ** (1 / 3)
        T = np.zeros(R.shape, dtype=complex)
        T[np.isreal(R - np.sqrt(D))] = np.cbrt(np.real(R - np.sqrt(D)))[np.isreal(R - np.sqrt(D))]
        T[np.logical_not(np.isreal(R - np.sqrt(D)))] = np.power(R - np.sqrt(D), 1 / 3)[
            np.logical_not(np.isreal(R - np.sqrt(D)))]

        result = np.zeros(tuple(list(a.shape) + [3])) + 0j
        result[all_, 0] = - b / (3 * a) + (S + T)
        result[all_, 1] = - b / (3 * a) - (S + T) / 2 + 0.5j * np.sqrt(3) * (S - T)
        result[all_, 2] = - b / (3 * a) - (S + T) / 2 - 0.5j * np.sqrt(3) * (S - T)

        real_result = np.zeros(result.shape)
        real_result[np.isreal(result)] = result[np.isreal(result)].real
        real_result[np.logical_not(np.isreal(result))] = np.nan

        return real_result

    def _get_max_cdf_location_roots(self):
        """Method that finds the argmax roots of the CDF of the approximation.

        Approach:

            set self._pdf(t) = 0, solve for t.

        We solve the equation for t = t - _t_L to simplify calculations and add _t_L at the end of the function.

        :returns:
            roots: A numpy array of shape [batch_size, num_roots], candidates for the maximum value of the CDF.
        """
        A = -1 / 3 * self._S_w * self._x_L[:,
                                1]  # TODO: Die Formeln gelten nur für t0=0!! das ändern wie bei ppf, ebenfalls in dem anderen Repo machen!
        B = (self._x_predTo - self._x_L[:, 0]) * self._S_w
        C = 2 * (self._x_L[:, 1] * self._C_L[:, 0, 1] + (self._x_predTo - self._x_L[:, 0]) * self._C_L[:, 1, 1])
        D = 2 * (self._x_L[:, 1] * self._C_L[:, 0, 0] + (self._x_predTo - self._x_L[:, 0]) * self._C_L[:, 0, 1])

        roots = self._find_cubic_roots_cardano_complex(A, B, C, D) + self._t_L  # TODO: Nun ist es ergänzt, PAsst das nun?
        return roots

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
        # One could alternatively use scipy.norm's ppf function on self.trans_dens
        cov_theta = np.array([[self.C_L[0, 0] + 2 * self.C_L[1, 0] * (theta - self._t_L)
                               + self.C_L[1, 1] * (theta - self._t_L) ** 2 + self.S_w * pow(theta - self._t_L, 3) / 3,
                              self.C_L[1, 0] + self.C_L[1, 1] * (theta - self._t_L) + self.S_w * pow(theta - self._t_L,
                                                                                                     2) / 2],
                              [self.C_L[1, 0] + self.C_L[1, 1] * (theta - self._t_L) + self.S_w * pow(theta - self._t_L,
                                                                                                     2) / 2,
                              self.C_L[1, 1] + self.S_w * (theta - self._t_L)]])
        b = cov_theta[1, 1] - cov_theta[1, 0] / cov_theta[0, 0] * cov_theta[0, 1]
        c = cov_theta[1, 0] / cov_theta[0, 0] * (self.x_predTo - np.squeeze(self._ev_t(theta), axis=0))

        qf = norm.ppf(1 - q)

        A = self.S_w / 3 * qf**2
        B = b*qf**2 - (self.x_L[1] + c)**2
        C = 0
        D = 0

        positive_roots = np.array([-B/A])  # TODO. Warum np.array? Docstrings anpassen! Auch in selber funktion abstract hitting time distribution!
        return positive_roots

    @AbstractCVHittingTimeDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size)  # this itself raises an error if not compatible
        # Force recalculating all privates
        self._q_max = None
        self._t_max = None
        self._ev = None
        self._var = None
        self._third_central_moment = None
        self._compute_moment = None  # numerical moment integrator must be recalculated after a change of S_w

        # # check if the private properties contain values, transfer if so
        # q_max = self._q_max[indices] if self._q_max is not None else None
        # t_max = self._t_max[indices] if self._t_max is not None else None
        # ev = self._ev[indices] if self._ev is not None else None
        # var = self._var[indices] if self._var is not None else None
        # tcm = self._third_central_moment[indices] if self._third_central_moment is not None else None
        #
        # return type(self)._from_private_arguments(x_L=self._x_L[indices],
        #                                           C_L=self._C_L[indices],
        #                                          x_predTo=self._x_predTo[indices],
        #                                           _t_L=self._t_L,
        #                                           S_w=self._S_w[indices],
        #                                           q_max=q_max,
        #                                           t_max=t_max,
        #                                           ev=ev,
        #                                           var=var,
        #                                           third_central_moment=tcm,
        #                                           compute_moment=None,  # numerical moment integrator must be
        #                                           # recalculated
        #                                           name=self.name,
        #                                           )

    @AbstractArrivalDistribution.check_setitem
    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        super().__setitem__(indices, values)

    def scale_params(self, length_scaling_factor, time_scaling_factor):  # TODO. Die kann kann raus (steht nur hier, damit ich es in CA nicht vergesse)
        """Scales the parameters of the distribution according t scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        super().scale_params(length_scaling_factor, time_scaling_factor)
        # Force recalculating all privates

    def get_statistics(self):  # TODO: Property? Docstrings?
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'RAW_CDF': self._cdf,
                          })
        return hit_stats


class UniformCVHittingTimeDistribution(AbstractCVHittingTimeDistribution, AbstractUniformHittingTimeDistribution):
    """Uses point predictors for the CV arrival time prediction and a uniform distribution.

    This distribution corresponds to the "usual" case where we define a fixed ejection window.
    """
    def __init__(self, x_L, x_predTo, t_L, point_predictor, window_length, a=0.5, name='Uniform model'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, pos_y, velo_y, ...]

         Format point_predictor:

            (pos_last, v_last)  --> dt_pred

         where
            - pos_last is a np.array of shape [batch_size, 2] and format [x, y] containing the positions at the _t_L.
            - v_last is a np.array of shape [batch_size, 2] and format [x, y] containing the velocities at the _t_L.
            - dt_pred is a np.array of shape [batch_size] with arrival time point estimates as difference times w.r.t.
                _t_L.

        :param x_L: A np.array of shape [state_length] or [batch_size, state_length] representing the expected value of
            the initial state. We use index L here because it usually corresponds to the last time we see a particle in
            our optical belt sorting scenario.
        :param x_predTo: A float or np.array of shape [batch_size], the position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param point_predictor: A callable, a function that returns an estimate for the arrival time.
        :param window_length: A float or np.array of shape [batch_size], the window length of the distribution.
        :param a: A float or np.array of shape [batch_size], the ratio of the window length, where the point prediction
            is located.
        :param name: String, the name for the distribution.
        """
        # TODO: In welcher Einheit ist window length, auch für spatial klären!
        # sanity checks
        if not callable(point_predictor):
            raise ValueError('point_predictor must be a callable.')

        AbstractCVHittingTimeDistribution.__init__(self,
                                                   x_L=x_L,
                                                   C_L=np.zeros(
                                                       (np.atleast_2d(x_L).shape[0], x_L.shape[-1], x_L.shape[-1])),
                                                   # always zero
                                                   S_w=0,  # always zero
                                                   x_predTo=x_predTo,
                                                   t_L=t_L,
                                                   name=name,
                                                   )

        t_predicted = point_predictor(self._x_L[:, [0, 2]], self._x_L[:, [1, 3]])
        AbstractUniformHittingTimeDistribution.__init__(self,
                                                        point_prediction=t_predicted,
                                                        window_length=window_length,
                                                        a=a,
                                                        name=name,
                                                        )

        #
        # pos_last = np.atleast_2d(x_L)[:, [0, 2]]  # TODO: Könnte man sich das alles mit der richtigen Vererbungsreihenfolge sparen?
        # v_last = np.atleast_2d(x_L)[:, [1, 3]]
        # pos_last = x_L[:, [0, 2]]
        # v_last = x_L[:, [1, 3]]
        # _t_predicted = point_predictor(pos_last, v_last)
        # # _t_predicted = _t_L + (x_predTo - x_L[:, 0]) / x_L[:, 1]
        #
        # super().__init__(x_L=x_L,
        #                  C_L=np.zero((np.atleast_2d(x_L).shape[0], x_L.shape[-1], x_L.shape[-1])),  # always zero
        #                  S_w=0,
        #                 x_predTo=_x_predTo,
        #                  _t_L=_t_L,
        #                  name=name,
        #                  point_prediction=_t_predicted,
        #                  window_length=window_length,
        #                  a=a,
        #                  )

    @AbstractCVHittingTimeDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        raise NotImplementedError('S_w for {} is always zero.'.format(self.__class__.__name__))


class MCCVHittingTimeDistribution(AbstractCVHittingTimeDistribution, AbstractMCHittingTimeDistribution):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem for a CV process
    to a distribution using scipy.stats.rv_histogram.

    Note that this distribution class does not support a batch dimension.
    """
    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, t_range, bins=100, t_samples=None, name='MC simulation'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, ... (e.g., pos_x, velo_x, if a CV model for both x and y is used)]

        :param x_L: A np.array of shape [state_length] or [batch_size, state_length] representing the expected value of
            the initial state. We use index L here because it usually corresponds to the last time we see a particle in
            our optical belt sorting scenario.
        :param C_L: A np.array of shape [state_size, state_length] or [batch_size, state_length, state_length]
            representing the covariance matrix of the initial state.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in x-direction.
        :param x_predTo: A float or np.array of shape [batch_size], the position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param t_range: A list of length 2 representing the limits for the first-passage time histogram (the number of
            bins within t_range will correspond to bins).
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param t_samples: None or a np.array of shape [num_samples] containing the first-passage times of the particles.
            If None, t_samples will be created by a call to a sampling method. If given, given values will be used.
        :param name: String, the name for the distribution.
        """
        # sanity checks
        if np.squeeze(x_L).ndim != 1:
            raise ValueError(
                'Batch size must be equal to 1. Note that {} does not support a batch dimension.'.format(
                    self.__class__.__name__))

        if t_samples is None:
            t_samples, _, _ = create_ty_cv_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L)

        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         t_range=t_range,
                         t_samples=t_samples,
                         bins=bins,
                         name=name)

    @AbstractCVHittingTimeDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float, the power spectral density in x-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size)
        # Resample und recalculate the distribution
        t_samples, _, _ = create_ty_cv_samples_hitting_time(self.x_L, self.C_L, self.S_w, self.x_predTo, self._t_L)
        self._density = self._build_distribution_from_samples(self._samples, self._range)


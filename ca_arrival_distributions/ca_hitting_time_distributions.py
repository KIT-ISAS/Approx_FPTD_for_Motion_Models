from absl import logging

from abc import ABC, abstractmethod
from timeit import time

import numpy as np
from scipy.stats import norm

from abstract_distributions import AbstractArrivalDistribution
from abstract_hitting_time_distributions import AbstractHittingTimeDistribution, \
    AbstractGaussTaylorHittingTimeDistribution, AbstractNoReturnHittingTimeDistribution, \
    AbstractUniformHittingTimeDistribution, AbstractMCHittingTimeDistribution
from ca_arrival_distributions.ca_utils import create_ty_ca_samples_hitting_time, get_system_matrices_from_parameters


# Approaches to solve problem
class AbstractCAHittingTimeDistribution(AbstractHittingTimeDistribution, ABC):
    """A base class for the CA hitting time distributions."""

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name='CA hitting time model', **kwargs):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, acc_x, ... (e.g., pos_x, velo_x, acc_y, if a CA model for both x and y is used)]

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
        if C_L.shape[-2] != C_L.shape[-1]:
            raise ValueError('C_L must be a symmetric matrix.')
        if not np.array_equal(np.atleast_2d(x_L).shape, self.batch_atleast_3d(C_L).shape[:2]):
            raise ValueError('Shapes of x_L and C_L do not match.')

        self._x_L = np.atleast_2d(x_L).astype(float)
        self._C_L = self.batch_atleast_3d(C_L).astype(float)
        self._S_w = np.broadcast_to(S_w, shape=self.batch_size).copy().astype(float)  # this itself raises an error if
        # not compatible

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
        return np.squeeze(self._x_L)

    @property
    def C_L(self):
        """The covariance matrix of the initial state.

        :returns: A np.array of shape [state_length, state_length] or [batch_size, state_length, state_length]
            representing the covariance matrix of the initial state.
        """
        return np.squeeze(self._C_L)

    @property
    def S_w(self):
        """The power spectral density (PSD) in x-direction.

        :returns A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        return np.squeeze(self._S_w)

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
    @abstractmethod
    def fourth_central_moment(self):
        """The fourth central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fourth central moment.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def fifth_central_moment(self):
        """The fifth central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fifth central moment.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def fourth_moment(self):
        """The fourth moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fourth moment.
        """
        return self.fourth_central_moment + 4 * self.ev * self.third_moment \
               - 6 * self.ev ** 2 * self.second_moment + 3 * self.ev ** 4

    @property
    def fifth_moment(self):
        """The fifth moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fifth moment.
        """
        return self.fifth_central_moment + 5 * self.ev * self.fourth_moment \
               - 10 * self.ev ** 2 * self.third_moment + 10 * self.ev ** 3 * self.second_moment \
               - 4 * self.ev ** 5

    @property
    def batch_size(self):
        """The batch size of the distribution.

        :returns: An integer, the batch size.
        """
        return len(self._x_L)

    def _ev_t(self, t):  # TODO: Was machen wir hier mit point predictor? Braucht man die Funktion hier überhaupt oder nur in klassen die sie aufrufen
        """The mean function of the CA motion model in x.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the mean in x at time t.
        """
        # return self.x_L[0] + self.x_L[1] * (t - self._t_L) + self.x_L[2] / 2 * (t - self._t_L) ** 2
        return self._x_L[:, np.newaxis, 0] + self._x_L[:, np.newaxis, 1] * (t - self._t_L) + self._x_L[:, np.newaxis,
                                                                                          2] / 2 * (t - self._t_L) ** 2

    def _var_t(self, t):
        """The variance function of the CA motion model in x.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the variance in x at time t.
        """
        # return self.C_L[0, 0] + 2*self.C_L[0, 1] * (t - self._t_L) \
        #                + (self.C_L[0, 2] + self.C_L[1, 1]) * (t - self._t_L) ** 2 \
        #                + self.C_L[1, 2] * (t - self._t_L) ** 3 \
        #                + 1 / 4 * self.C_L[2, 2] * (t - self._t_L) ** 4 \
        #                + 1 / 20 * self.S_w * (t - self._t_L) ** 5
        return self._C_L[:, np.newaxis, 0, 0] + 2 * self._C_L[:, np.newaxis, 0, 1] * (t - self._t_L) \
               + (self._C_L[:, np.newaxis, 0, 2] + self._C_L[:, np.newaxis, 1, 1]) * (t - self._t_L) ** 2 \
               + self._C_L[:, np.newaxis, 1, 2] * (t - self._t_L) ** 3 \
               + 1 / 4 * self._C_L[:, np.newaxis, 2, 2] * (t - self._t_L) ** 4 \
               + 1 / 20 * self._S_w * (t - self._t_L) ** 5

    @AbstractArrivalDistribution.batch_size_one_function
    def trans_density(self, dt, theta):
        """The transition density p(x(dt+theta)| x(theta) =x_predTo) from going from x_predTo at time theta to
        x(dt+theta) at time dt+theta.

        Note that in terms of the used approximation, this can be seen as the first returning time to x_predTo after
        a crossing of x_predTo at theta.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param dt: A float, the time difference. dt is zero at time = theta.
        :param theta: A float, the (assumed) time at which x(theta) = x_pred_to.

        :returns: A scipy.stats.norm object, the transition density for the given dt and theta.
        """
        Phi = lambda dt: np.array([[1, dt, dt ** 2 / 2],
                                   [0, 1, dt],
                                   [0, 0, 1]])  # TODO: Die andere Funkion hierfür nutzen?

        Q = lambda dt: self.S_w * np.array([[pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6],
                                            [pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2],
                                            [pow(dt, 3) / 6, pow(dt, 2) / 2, dt]])

        cov_theta = np.matmul(np.matmul(Phi(theta - self.t_L), self.C_L[0:3, 0:3]),
                              np.transpose(Phi(theta - self.t_L))) + Q(theta - self.t_L)

        sel_ma = np.array([[0, 0], [1, 0], [0, 1]])
        phi_sel = np.matmul(Phi(dt), sel_ma)
        p_theta_mu = np.array([self.x_L[1] + self.x_L[2] * (theta - self.t_L), self.x_L[2]]) \
                     + cov_theta[1:, 0] / cov_theta[0, 0] * (self.x_predTo - np.squeeze(self._ev_t(theta)))
        p_theta_var = cov_theta[1:, 1:] - np.outer(cov_theta[1:, 0], cov_theta[0, 1:]) / cov_theta[0, 0]
        trans_mu = np.array([1, 0, 0]) * self.x_predTo + np.matmul(phi_sel, p_theta_mu)
        trans_var = np.matmul(np.matmul(phi_sel, p_theta_var), np.transpose(phi_sel)) + Q(dt)
        return norm(loc=trans_mu[0], scale=np.sqrt(trans_var[0, 0]))

    def scale_params(self, length_scaling_factor, time_scaling_factor):  # TODO: Bleibt das so? Privates ergänzen?
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        super().scale_params(length_scaling_factor, time_scaling_factor)

        self._x_L[:, [0, 2]] *= length_scaling_factor
        self._x_L[:, [1, 3]] *= length_scaling_factor / time_scaling_factor
        self._x_L[:, [2, 4]] *= length_scaling_factor / time_scaling_factor ** 2

        self._C_L[:, 0, 0] *= length_scaling_factor ** 2
        self._C_L[:, 2, 2] *= length_scaling_factor ** 2
        self._C_L[:, 1, 1] *= (length_scaling_factor / time_scaling_factor) ** 2
        self._C_L[:, 3, 3] *= (length_scaling_factor / time_scaling_factor) ** 2
        self._C_L[:, 0, 1] *= length_scaling_factor ** 2 / time_scaling_factor
        self._C_L[:, 1, 0] *= length_scaling_factor ** 2 / time_scaling_factor
        self._C_L[:, 2, 3] *= length_scaling_factor ** 2 / time_scaling_factor
        self._C_L[:, 3, 2] *= length_scaling_factor ** 2 / time_scaling_factor

        self._S_w *= length_scaling_factor ** 2 / time_scaling_factor ** 3

    def __setitem__(self, indices, values): # TODO: Bleibt das so? Privates ergänzen?
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
        hit_stats.update({'SKEW': self.skew,
                          })
        return hit_stats


class GaussTaylorCAHittingTimeDistribution(AbstractCAHittingTimeDistribution, AbstractGaussTaylorHittingTimeDistribution):
    """A simple Gaussian approximation for the first-passage time problem using a Taylor approximation and error
    propagation that can be used for CA models.

    """
    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, point_predictor, name='Gauß--Taylor approx.'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, acc_y, pos_y, velo_y, acc_y, ...]

         Format point_predictor:

            (pos_last, v_last, a_last, x_predTo)  --> dt_pred

         where
            - pos_last is a np.array of shape [batch_size, 2] and format [x, y] containing the positions at the t_L.
            - v_last is a np.array of shape [batch_size, 2] and format [x, y] containing the velocities at the t_L.
            - a_last is a np.array of shape [batch_size, 2] and format [x, y] containing the accelerations at the t_L.
            - dt_pred is a np.array of shape [batch_size] with arrival time point estimates as difference times w.r.t.
                t_L.

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

        ev = point_predictor(x_L[..., [0, 2]], x_L[..., [1, 3]], x_L[..., [2, 4]], x_predTo) + t_L

        # ev must be resizeable to shape [batch_size]
        if not np.atleast_1d(ev).ndim == 1 or np.atleast_1d(ev).shape[0] != np.atleast_2d(x_L).shape[0]:
            raise ValueError('point predictor must output a float or a np.array of shape [batch_size].')

        var = self._compute_var(ev, x_L, C_L, t_L, S_w)

        # AbstractCAHittingTimeDistribution.__init__(self,
        #                                            x_L=x_L,
        #                                            C_L=C_L,
        #                                            S_w=S_w,
        #                                            x_predTo=x_predTo,
        #                                            t_L=t_L,
        #                                            name=name,
        #                                            )
        #
        # ev = point_predictor(self._x_L[:, [0, 2]], self._x_L[:, [1, 3]], self.x_L[:, [2, 4]]) + t_L
        # var = self._compute_var(ev, self._x_L, self._C_L, self._t_L, self._S_w)
        #
        # AbstractGaussTaylorHittingTimeDistribution.__init__(self,
        #                                                     ev=ev,
        #                                                     var=var,
        #                                                     name=name,
        #                                                     )


        # pos_last = x_L[:, [0, 2]]
        # v_last = x_L[:, [1, 3]]
        # a_last = x_L[:, [2, 4]]
        # ev = point_predictor.predict(pos_last, v_last, a_last)

        # # Evaluate the equation at the time at the boundary
        # var = self._compute_var(ev, x_L, C_L, t_L, S_w)
        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         name=name,
                         ev=ev,
                         var=var)

        # self._ev = t_L - x_L[1] / x_L[2] + np.sign(x_L[2]) * \
        #            np.sqrt((x_L[1] / x_L[2]) ** 2 + 2 / x_L[2] * (x_predTo - x_L[0]))
        # F, _ = _get_system_matrices_from_parameters(self._ev, self.S_w)
        # x_p = np.dot(F, self.x_L)
        # self._var = (1 / x_p[1]) ** 2 * self._var_t(self._ev)

    def _compute_var(self, point_prediction, x_L, C_L, t_L, S_w):  # TODO: Das kann man vereinfachen, abstract wäre dann nicht mehr static, generell wie sieht es aus mit den input und output shapes, passen die wie beschrieben?
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
        # F, _ = get_system_matrices_from_parameters(point_prediction, S_w)
        # # x_p = np.dot(F, x_L)
        # x_p = np.matmul(self.batch_atleast_3d(F), np.atleast_2d(x_L)[:, :, np.newaxis]).squeeze(-1)   # TODO: unschön, nächste zeile auch
        # return (1 / np.atleast_2d(x_p)[:, 1]) ** 2 * self._var_t(self._ev)

        dt_p = point_prediction - t_L  # TODO: Alles unschön, dann besser machen, oder it CV besser integrieren

        def var_t(t, x_L, C_L, t_L, S_w):
            return C_L[..., 0, 0] + 2 * C_L[..., 0, 1] * (t - t_L) \
                + (C_L[..., 0, 2] + C_L[..., 1, 1]) * (t - t_L) ** 2 \
                + C_L[..., 1, 2] * (t - t_L) ** 3 \
                + 1 / 4 * C_L[..., 2, 2] * (t - t_L) ** 4 \
                + 1 / 20 * S_w * (t - t_L) ** 5

        ev_v_t = x_L[..., 1] + x_L[..., 2] * dt_p
        return var_t(point_prediction, x_L, C_L, t_L, S_w) / ev_v_t ** 2


    @property
    def fourth_central_moment(self):
        """The fourth central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fourth central moment.
        """
        return 3 * self.var ** 2  # Gaussian fourth central moment

    @property
    def fifth_central_moment(self):
        """The fifth central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fifth central moment.
        """
        return np.squeeze(np.zeros(self.batch_size))  # Gaussian fifth central moment

    @AbstractCAHittingTimeDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size).copy()
        # Recalculate the variance
        self._var = self._compute_var(self._ev, self._x_L, self._C_L, self._t_L, self._S_w)

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'PPF': self.ppf,
                          })
        return hit_stats


class NoReturnCAHittingTimeDistribution(AbstractCAHittingTimeDistribution, AbstractNoReturnHittingTimeDistribution):
    """An approximation to the first-passage time distribution for CA models using the assumption that particles
    are unlikely to move back once they have passed the boundary.

    """
    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name="No-return approx."):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, acc_x, ... (e.g., pos_x, velo_x, acc_y, if a CA model for both x and y is used)]

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
            
        # for properties
        self._fourth_central_moment = None
        self._fifth_central_moment = None
            
    # @classmethod
    # def _from_private_arguments(cls, x_L, C_L, S_w,x_predTo, t_L, q_max, t_max,
    #                             ev=None,
    #                             var=None,
    #                             third_central_moment=None,
    #                             fourth_central_moment=None,
    #                             fifth_central_moment=None,
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
    #     :param t_L: A float, the time of the last state/measurement (initial time).
    #     :param q_max: A float or np.array of shape [batch_size], the maximum value of the CDF.
    #     :param t_max: A float or np.array of shape [batch_size], the time, when the CDF visits its maximum.
    #     :param ev: A float or a np.array of shape [batch_size], the expected value of the first-passage time
    #         distribution.
    #     :param var: A float or a np.array of shape [batch_size], the variance of the first-passage time distribution.
    #     :param third_central_moment: A float or a np.array of shape [batch_size], the third central moment of the
    #         first-passage time distribution.
    #     :param fourth_central_moment: A float or a np.array of shape [batch_size], the fourth central moment of the
    #         first-passage time distribution.
    #     :param fifth_central_moment: A float or a np.array of shape [batch_size], the fifth central moment of the
    #         first-passage time distribution.
    #     :param compute_moment: A callable that can be used to compute the moments.
    #     :param name: String, the (default) name for the distribution.
    #
    #     :returns: A NoReturnCAHittingTimeDistribution object, a (potentially modified) copy of the distribution.
    #     """
    #     obj = cls.__new__(cls)  # create a new object
    #     super(cls, obj).__init__(x_L, C_L, S_w,x_predTo, t_L, name)
    #
    #     # overwrite all privates
    #     obj._q_max = q_max
    #     obj._t_max = t_max
    #     obj._ev = ev
    #     obj._var = var
    #     obj._third_central_moment = third_central_moment
    #     obj._fourth_central_moment = fourth_central_moment
    #     obj._fifth_central_moment = fifth_central_moment
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
        # return derivative(self._cdf, t, dx=1e-6)
        # gauss = norm.pdf((self._x_predTo - self._ev_t(t)) / np.sqrt(self._var_t(t)))  # Std. Gauss pdf
        # der_ev_t = self.x_L[1] + self.x_L[2] * (t - self._t_L)
        # der_var_t = 2 * self.C_L[0, 1] + 2 * (self.C_L[0, 2] + self.C_L[1, 1]) * (t - self._t_L) + \
        #             3 * self.C_L[1, 2] * (t - self._t_L) ** 2 + self.C_L[2, 2] * (t - self._t_L) ** 3 + \
        #             1 / 4 * self.S_w * (t - self._t_L) ** 4
        # neg_der_arg = der_ev_t / np.sqrt(self._var_t(t)) + (self._x_predTo - self._ev_t(t)) * der_var_t / (
        #         2 * self._var_t(t) ** (3 / 2))  # negative of the derivative
        # return gauss * neg_der_arg
        gauss = norm.pdf((self._x_predTo - self._ev_t(t)) / np.sqrt(self._var_t(t)))  # Std. Gauss pdf

        der_ev_t = self._x_L[:, np.newaxis, 1] + self._x_L[:, np.newaxis, 2] * (t - self._t_L)
        der_var_t = 2 * self._C_L[:, np.newaxis, 0, 1] + 2 * (
                    self._C_L[:, np.newaxis, 0, 2] + self._C_L[:, np.newaxis, 1, 1]) * (t - self._t_L) + \
                    3 * self._C_L[:, np.newaxis, 1, 2] * (t - self._t_L) ** 2 + self._C_L[:, np.newaxis, 2, 2] * (
                                t - self._t_L) ** 3 + \
                    1 / 4 * self.S_w * (t - self._t_L) ** 4
        neg_der_arg = der_ev_t / np.sqrt(self._var_t(t)) + (self._x_predTo - self._ev_t(t)) * der_var_t / (
                2 * self._var_t(t) ** (3 / 2))  # negative of the derivative
        return gauss * neg_der_arg

    def _ppf(self, q):
        """The quantile function / percent point function (PPF) of the first-passage time distribution.

        Approach:

              1 - q = int(N(x, mu(t), var(t)), x = -inf ..x_predTo) = PHI ( (x_predTo - mu(t)) / sqrt(var(t))
              PHI^-1(1 -q) = (x_predTo - mu(t)) / sqrt(var(t)) -> solve for t...

        We solve the equation for t = t - t_L to simplify calculations and add t_L at the end of the function.

        :param q: A float, the confidence parameter of the distribution, 0 <= q <= 1.

        :returns:
            t: A np.array of shape [batch_size], the value of the PPF for q.
            candidate_roots: A np.array of shape [batch_size, num_possible_solutions] containing the values of all
                possible roots.
        """
        # Special case q = 0.5, this corresponds to the median.
        # 0 =  (x_predTo - mu(t)) / sqrt(var(t) ->x_predTo = mu(t) -> solve for t...
        if q == 0.5:
            # solve: self._x_predTo = self.x_L[0] + self.x_L[1] * t + self.x_L[2] / 2 * t**2
            pp = self._x_L[:, 1]/self._x_L[:, 2]
            qq = 2/self._x_L[:, 2]*(self._x_L[:, 0] - self._x_predTo)
            # Sign depends on the sign of x_L[2]
            t = - pp + np.sign(self._x_L[:, 2]) * np.sqrt(pp**2 - qq) + self._t_L
            return t, t

        # Polynomial of degree 5
        # At**5 + B*t**4 + C*t**3 + D*t**2 + E*t + F = 0
        qf = norm.ppf(1 - q)  # Standard-Gaussian quantile function
        A = 1/20*self._S_w   # TODO: Stimmt das hier mit ohne qf?
        B = 1/4*(self._C_L[:, 2, 2] - self._x_L[:, 2]**2/qf**2)
        C = self._C_L[:, 1, 2] - self._x_L[:, 1]*self._x_L[:, 2]/qf**2
        D = self._C_L[:, 0, 2] + self._C_L[:, 1, 1] + (-self._x_L[:, 1]**2 + self._x_L[:, 2]*(self._x_predTo - self._x_L[:, 0]))/qf**2
        E = 2*self._C_L[:, 0, 1] + 2*self._x_L[:, 1]*(self._x_predTo - self._x_L[:, 0])/qf**2
        F = self._C_L[:, 0, 0] - (self._x_predTo - self._x_L[:, 0])**2/qf**2

        # Use numerical methods as there is no analytical solution for polynomials of degree 5.
        real_roots = np.empty((self.batch_size, 5))  # TODO: Passt das?
        real_roots[:] = np.nan
        t = np.empty(self.batch_size)
        t[:] = np.nan
        for i in range(self.batch_size):
            # unfortunately, there exists no vectorized implementation
            roots_i = np.roots([A[i], B[i], C[i], D[i], E[i], F[i]])

            # TODO: Vectorize this
            # roots are in descending order, the first root is always too large.
            real_roots_i = roots_i.real[np.logical_not(np.iscomplex(roots_i))] + self.t_L  # TODO: Stimmt das so wie es hier steht?, allgemeine Regel?
            if real_roots_i.shape[0] == 5:
                t[i] = float(real_roots_i[4] if q < 0.5 else real_roots_i[3])
            elif real_roots_i.shape[0] >= 2:
                t[i] = float(real_roots_i[2] if q < 0.5 else real_roots_i[1])
            elif real_roots_i.shape[0] == 1:
                t[i] = float(real_roots_i)
            else:
                raise ValueError('Unsupported number of roots.')
            real_roots[i, :real_roots_i.shape[0]] = real_roots_i

        return t, real_roots

    def _get_max_cdf_location_roots(self):
        """Method that finds the argmax roots of the CDF of the approximation.

        Approach:

            set self._pdf(t) = 0, solve for t.

        We solve the equation for t = t - t_L to simplify calculations and add t_L at the end of the function.

        :returns:
            roots: A numpy array of shape [batch_size, num_roots], candidates for the maximum value of the CDF.
        """
        A = -1 / 40 * self._S_w * self._x_L[:, 2]
        B = -1 / 20 * 3 * self._S_w * self._x_L[:, 1]
        C = 1 / 40 * (10 * self._S_w * self._x_predTo - 10 * self._S_w * self._x_L[:, 0] \
                      + 20 * self._C_L[:, 1, 2] * self._x_L[:, 2] - 20 * self._C_L[:, 2, 2] * self._x_L[:, 1])
        D = 1 / 40 * ((40 * self._C_L[:, 1, 1] + 40 * self._C_L[:, 0, 2]) * self._x_L[:, 2] \
                      - 40 * self._C_L[:, 1, 2] * self._x_L[:, 1] + 40 * self._C_L[:, 2, 2] * self._x_predTo \
                      - 40 * self._C_L[:, 2, 2] * self._x_L[:, 0])
        E = 1 / 40 * (120 * self._C_L[:, 0, 1] * self._x_L[:, 2] \
                      + 120 * self._C_L[:, 1, 2] * (self._x_predTo - self._x_L[:, 0]))
        F = 1 / 40 * (80 * self._C_L[:, 0, 0] * self._x_L[:, 2] \
                      + (80 * self._C_L[:, 1, 1] + 80 * self._C_L[:, 0, 2]) * self._x_predTo + (
                              -80 * self._C_L[:, 1, 1] - 80 * self._C_L[:, 0, 2]) * self._x_L[:, 0] \
                      + 80 * self._x_L[:, 1] * self._C_L[:, 0, 1])
        G = 2 * self._x_predTo * self._C_L[:, 0, 1] + 2 * self._C_L[:, 0, 0] * self._x_L[:, 1] \
            - 2 * self._C_L[:, 0, 1] * self._x_L[:, 0]

        roots = np.empty((self.batch_size, 6))  # TODO: Passt das?
        roots[:] = np.nan
        for i in range(self.batch_size):
            # unfortunately, there exists no vectorized implementation
            roots_i = np.roots([A[i], B[i], C[i], D[i], E[i], F[i], G[i]])
            roots_i = roots_i[np.isreal(roots_i)].real + self._t_L
            roots[i, :roots_i.shape[0]] = roots_i
        return roots

    @property
    def fourth_central_moment(self):
        """The fourth central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fourth central moment.
        """
        if self._fourth_central_moment is None:
            start_time = time.time()
            # Calculate the third central moment because calculating the third moment directly causes high numerical errors
            # Calculating moments with integrate.quad takes much time
            # self._third_central_moment = integrate.quad(lambda t: (t - self.ev)**4 * self.pdf(t), self.ppf(0.00o05), self.ppf(0.99995))[
            #              0]  # this yields much better results
            self._fourth_central_moment, _, abs_dev, rel_dev = self.compute_moment(
                lambda t: (t - self.ev) ** 3)  # this yields much better results
            logging.info('E4 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return np.squeeze(self._fourth_central_moment)

    @property
    def fourth_central_moment_available(self):
        """Indicator that shows if the fourth central moment was already calculated.

        :returns: True if the fourth central moment was already calculated, False otherwise.
        """
        return self._fourth_central_moment is not None

    @property
    def fifth_central_moment(self):
        """The fifth central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fifth central moment.
        """
        if self._fifth_central_moment is None:
            start_time = time.time()
            # Calculate the third central moment because calculating the third moment directly causes high numerical errors
            # Calculating moments with integrate.quad takes much time
            # self._third_central_moment = integrate.quad(lambda t: (t - self.ev)**5 * self.pdf(t), self.ppf(0.00o05), self.ppf(0.99995))[
            #              0]  # this yields much better results
            self._fifth_central_moment, _, abs_dev, rel_dev = self.compute_moment(
                lambda t: (t - self.ev) ** 5)  # this yields much better results
            logging.info('E5 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return np.squeeze(self._fifth_central_moment)

    @property
    def fifth_central_moment_available(self):
        """Indicator that shows if the fifth central moment was already calculated.

        :returns: True if the fifth central moment was already calculated, False otherwise.
        """
        return self._fifth_central_moment is not None

    @AbstractArrivalDistribution.batch_size_one_function
    def trans_dens_ppf(self, theta, q=0.9):
        """The PPF of 1 - int ( p(x(dt+theta)| x(theta) =x_predTo), x(dt+theta) = - infty ..x_predTo),
        i.e., the inverse CDF of the event that particles are abovex_predTo once they have reached it at time theta.

        Note that in terms of the used approximation, this can be seen as PPF of the approximate first-passage
        returning time distribution w.r.t. the boundary x_pred_to.

        This function does not support batch-wise processing, i.e., a batch dimension of 1 is required.

        :param theta: A float, the (assumed) time at which x(theta) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.

        :returns: A np.array, the value of the PPF for q and theta, note that this a delta time w.r.t. theta.The value of the PPF for q and theta, note that this a delta time w.r.t. theta.
        """
        # TODO: One could alternatively use scipy.norm's ppf function on self.trans_dens
        F, Q = get_system_matrices_from_parameters(theta - self.t_L, self.S_w)

        cov_theta = np.matmul(np.matmul(F, self.C_L[0:3, 0:3]),
                              np.transpose(F)) + Q

        p_theta_var = cov_theta[1:, 1:] - np.outer(cov_theta[1:, 0], cov_theta[0, 1:]) / cov_theta[0, 0]
        p_theta_mu = np.array([self.x_L[1] + self.x_L[2]*(theta - self.t_L), self.x_L[2]]) \
                     + cov_theta[1:, 0]/cov_theta[0, 0]*(self.x_predTo - np.squeeze(self._ev_t(theta)))

        qf = norm.ppf(1 - q)
        A = 1 / 20 * self.S_w * qf ** 2
        B = 1 / 4 * (p_theta_var[1, 1] * qf ** 2 - p_theta_mu[1]** 2)
        C = p_theta_var[0, 1] * qf ** 2 - p_theta_mu[0] * p_theta_mu[1]
        D = p_theta_var[0, 0] * qf ** 2 - p_theta_mu[0] ** 2
        E = 0
        F = 0

        # Use np.roots here as it is in general the most stable approach
        dt = np.roots([A, B, C, D])

        # Check all possible solutions  # TODO: Check auch in CV?
        # Probability of the event that the transition density for x(theta + dt) is higher than x_pred_to.
        trans_fpt_values = np.array([1 - self.trans_density(t, theta).cdf(self.x_predTo) for t in dt])
        valid_roots = dt[np.isclose(trans_fpt_values, q)].real
        positive_roots = valid_roots[valid_roots > 0]
        return positive_roots  # TODO. Warum np.array? Docstrings anpassen! A

    @AbstractCAHittingTimeDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size).copy()  # this itself raises an error if not
        # compatible
        # Force recalculating all privates
        self._q_max = None
        self._t_max = None
        self._ev = None
        self._var = None
        self._third_central_moment = None
        self._fourth_central_moment = None
        self._fifth_central_moment = None
        self._compute_moment = None  # numerical moment integrator must be recalculated after a change of S_w

    # def __getitem__(self, indices):
    #     """Takes elements along the batch shape. Use this for fancy indexing (e.g., new_distr = distr[:2]).
    #
    #     :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to extract.
    #
    #     :returns: A copy of the distribution with the extracted values.
    #     """
    #     if not isinstance(indices, slice):
    #         indices = np.array(indices)  # e.g. if it is a list or boolean
    #
    #     # check if the private properties contain values, transfer if so
    #     q_max = self._q_max[indices] if self._q_max is not None else None
    #     t_max = self._t_max[indices] if self._t_max is not None else None
    #     ev = self._ev[indices] if self._ev is not None else None
    #     var = self._var[indices] if self._var is not None else None
    #     tcm = self._third_central_moment[indices] if self._third_central_moment is not None else None
    #     focm = self._fourth_central_moment[indices] if self._fourth_central_moment is not None else None
    #     ficm = self._fifth_central_moment[indices] if self._fifth_central_moment is not None else None
    #
    #     return type(self)._from_private_arguments(x_L=self._x_L[indices],
    #                                               C_L=self._C_L[indices],
    #                                              x_predTo=self._x_predTo[indices],
    #                                               t_L=self._t_L,
    #                                               S_w=self._S_w[indices],
    #                                               q_max=q_max,
    #                                               t_max=t_max,
    #                                               ev=ev,
    #                                               var=var,
    #                                               third_central_moment=tcm,
    #                                               fourth_central_moment=focm,
    #                                               fifth_central_moment=ficm,
    #                                               compute_moment=None,  # numerical moment integrator must be
    #                                               # recalculated
    #                                               name=self.name,
    #                                               )

    @AbstractArrivalDistribution.check_setitem
    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        super().__setitem__(indices, values)

        if self.fourth_central_moment_available and values.fourth_central_moment_available:
            self._fourth_central_moment[indices] = values.fourth_central_moment
        else:
            self._fourth_central_moment = None
        if self.fifth_central_moment_available and values.fifth_central_moment_available:
            self._fifth_central_moment[indices] = values.fifth_central_moment
        else:
            self._fifth_central_moment = None

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        if values.fourth_central_moment_available:
            self._fourth_central_moment = values.fourth_central_moment[indices]
        if values.fifth_central_moment_available:
            self._fifth_central_moment = values.fifth_central_moment[indices]
        super()._left_hand_indexing(indices, values)

    def scale_params(self, length_scaling_factor, time_scaling_factor):  # TODO. Die kann kann raus (steht nur hier, damit ich es in CA nicht vergesse)
        """Scales the parameters of the distribution according t scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        super().scale_params(length_scaling_factor, time_scaling_factor)
        # Force recalculating all privates
        self._fourth_central_moment = None
        self._fifth_central_moment = None

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'RAW_CDF': lambda t: np.squeeze(self._cdf(t)),
                          })
        return hit_stats


class UniformCAHittingTimeDistribution(AbstractUniformHittingTimeDistribution, AbstractCAHittingTimeDistribution):
    """Uses point predictors for the CA arrival time prediction and a uniform distribution.

    This distribution corresponds to the "usual" case where we define a fixed deflection window.
    """
    def __init__(self, x_L, x_predTo, t_L, point_predictor, window_length, a=0.5, name='Uniform model'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, acc_y, pos_y, velo_y, acc_y, ...]

         Format point_predictor:

            (pos_last, v_last, a_last, x_predTo)  --> dt_pred

         where
            - pos_last is a np.array of shape [batch_size, 2] and format [x, y] containing the positions at the t_L.
            - v_last is a np.array of shape [batch_size, 2] and format [x, y] containing the velocities at the t_L.
            - a_last is a np.array of shape [batch_size, 2] and format [x, y] containing the accelerations at the t_L.
            - dt_pred is a np.array of shape [batch_size] with arrival time point estimates as difference times w.r.t.
                t_L.

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

        t_predicted = point_predictor(x_L[..., [0, 3]], x_L[..., [1, 4]], x_L[..., [2, 5]], x_predTo) + t_L

        # t_predicted must be resizeable to shape [batch_size]
        if not np.atleast_1d(t_predicted).ndim == 1 or np.atleast_1d(t_predicted).shape[0] != np.atleast_2d(x_L).shape[
            0]:
            raise ValueError('point predictor must output a float or a np.array of shape [batch_size].')

        # AbstractCAHittingTimeDistribution.__init__(self,
        #                                            x_L=x_L,
        #                                            C_L=np.zeros((np.atleast_2d(x_L).shape[0], 4, 4)),  # always zero
        #                                            S_w=0,  # always zero
        #                                            x_predTo=x_predTo,
        #                                            t_L=t_L,
        #                                            name=name,
        #                                            )
        #
        # t_predicted = point_predictor(self._x_L[:, [0, 2]], self._x_L[:, [1, 3]], self._x_L[:, [2, 4]]) + t_L
        # AbstractUniformHittingTimeDistribution.__init__(self,
        #                                                 point_prediction=t_predicted,
        #                                                 window_length=window_length,
        #                                                 a=a,
        #                                                 name=name,
        #                                                 )



        # pos_last = x_L[:, [0, 2]]
        # v_last = x_L[:, [1, 3]]
        # a_last = x_L[:, [2, 4]]
        # _t_predicted = point_predictor.predict(pos_last, v_last, a_last)
        # # _t_predicted = t_L + (x_predTo - x_L[:, 0]) / x_L[:, 1]
        #
        super().__init__(x_L=x_L,
                         C_L=np.zeros((np.atleast_2d(x_L).shape[0], x_L.shape[-1], x_L.shape[-1])),  # always zero
                         S_w=0,  # always zero
                         x_predTo=x_predTo,
                         t_L=t_L,
                         name=name,
                         point_prediction=t_predicted,
                         window_length=window_length,
                         a=a,
                         )

    @property
    def fourth_central_moment(self):
        """The fourth central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fourth central moment.
        """
        return np.squeeze(1 / 80 * (self.x_max - self.x_min) ** 4)  # Uniform fourth central moment

    @property
    def fifth_central_moment(self):
        """The fifth central moment of the first-passage time distribution.

        :returns: A float or a np.array of shape [batch_size], the fifth central moment.
        """
        return np.squeeze(np.zeros(self.batch_size))  # Uniform fifth central moment

    @AbstractCAHittingTimeDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in x-direction.
        """
        raise NotImplementedError('S_w for {} is always zero.'.format(self.__class__.__name__))


class MCCAHittingTimeDistribution(AbstractMCHittingTimeDistribution, AbstractCAHittingTimeDistribution):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem for a CA process
    to a distribution using scipy.stats.rv_histogram.

    Note that this distribution class does not support a batch dimension.
    """
    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, t_range, bins=100, t_samples=None, name='MC simulation'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, acc_x, ... (e.g., pos_x, velo_x, acc_y, if a CA model for both x and y is used)]

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
            dt = (np.max(t_range) - t_L) / 200  # we want to use approx. 200 time steps in the MC simulation
            # round dt to the first significant digit
            dt = np.round(dt, -np.floor(np.log10(np.abs(dt))).astype(int))
            (t_samples, _, _), _ = create_ty_ca_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L, dt=dt)

        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         t_range=t_range,
                         t_samples=t_samples,
                         bins=bins,
                         name=name)

    @property
    def fourth_central_moment(self):
        """The fourth central moment of the first-passage time distribution.

        :returns: A float, the fourth central moment.
        """
        return self.fourth_moment - 4 * self.ev * self.third_moment + 6 * self.ev ** 2 * self.second_moment - 3 * self.ev ** 4

    @property
    def fourth_moment(self):
        """The fourth moment of the first-passage time distribution.

        :returns: A float, the fourth moment.
        """
        return self._density.moment(4)

    @property
    def fifth_central_moment(self):
        """The fifth central moment of the first-passage time distribution.

        :returns: A float, the fifth central moment.
        """
        return self.fifth_moment - 5 * self.ev * self.fourth_moment + 10 * self.ev ** 2 * self.third_moment - 10 * self.ev ** 3 * self.second_moment + 4 * self.ev ** 5

    @property
    def fifth_moment(self):
        """The fifth moment of the first-passage time distribution.

        :returns: A float, the fifth moment.
        """
        return self._density.moment(5)

    @AbstractCAHittingTimeDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in x-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float, the power spectral density in x-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size).copy()
        # Resample und recalculate the distribution
        dt = (np.max(self._range) - self._t_L) / 200  # we want to use approx. 200 time steps in the MC simulation
        # round dt to the first significant digit
        dt = np.round(dt, -np.floor(np.log10(np.abs(dt))).astype(int))
        (self._samples, _, _), _ = create_ty_ca_samples_hitting_time(self.x_L, self.C_L, self.S_w, self.x_predTo,
                                                                     self._t_L, dt=dt)
        self._density = self._build_distribution_from_samples(self._samples,
                                                              self._range)  # TODO: Self.range dann auch anpassen?

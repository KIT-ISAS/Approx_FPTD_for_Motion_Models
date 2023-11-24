from absl import logging

from abc import ABC, abstractmethod
from timeit import time

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm, multivariate_normal
from scipy.special import cbrt

from abstract_hitting_time_models import AbstractHittingTimeModel, AbstractTaylorHittingTimeModel, \
    AbstractEngineeringApproxHittingTimeModel, AbstractMCHittingTimeModel
from cv_utils import _get_system_matrices_from_parameters, create_ty_cv_samples_hitting_time


class AbstractCVHittingTimeModel(AbstractHittingTimeModel, ABC):
    """A base class for the CV hitting time models."""

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name='CV hitting time model', **kwargs):
        """Initialize the model.

        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, pos_y, vel_y].
        :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param x_predTo: A float, position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, (default) name for the model.
        """
        self._x_L = x_L
        self._C_L = C_L
        self._S_w = S_w
        super().__init__(x_predTo=x_predTo,
                         t_L=t_L,
                         name=name,
                         **kwargs)

        # For properties
        self._third_central_moment = None
        self._second_moment = None
        self._third_moment = None
        self._skew = None

    @property
    def x_L(self):
        """The expected value of the initial state."""
        return self._x_L

    @property
    def C_L(self):
        """The covariance matrix of the initial state."""
        return self._C_L

    @property
    def S_w(self):
        """The power spectral density of the model."""
        return self._S_w

    @property
    @abstractmethod
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def second_moment(self):
        """The second moment of the first passage time distribution."""
        if self._second_moment is None:
            self._second_moment = self.var + self.ev**2
        return self._second_moment

    @property
    def third_moment(self):
        """The third moment of the first passage time distribution."""
        if self._third_moment is None:
            self._third_moment = self.third_central_moment + 3*self.ev*self.var + self.ev**3
        return self._third_moment

    @property
    def skew(self):
        """The skew of the first passage time distribution."""
        if self._skew is None:
            # third standardized moment / stddev**3
            self._skew = self.third_central_moment / self.stddev**3
        return self._skew

    def ev_t(self, t):
        """The mean function of the CV motion model in x.

        :param t: A float or np.array, the time parameter of the mean function.
        """
        return self.x_L[0] + self.x_L[1] * (t - self.t_L)

    def var_t(self, t):
        """The variance function of the CV motion model in x.

        :param t: A float or np.array, the time parameter of the variance function.
        """
        return self.C_L[0, 0] + 2 * self.C_L[1, 0] * (t - self.t_L) + self.C_L[1, 1] * (
                    t - self.t_L) ** 2 + self.S_w * pow(t - self.t_L, 3) / 3

    def trans_density(self, dt, theta):
        """The transition density p(x(dt+theta)| x(theta) = x_predTo) from going from x_predTo at time theta to
        x(dt+theta) at time dt+theta.

        Note that in terms of the used approximation, this can be seen as the first returning time to x_predTo after
        a crossing of x_predTo at theta.

        :param dt: A float or np.array, the time difference. dt is zero at time = theta.
        :param theta: A float or np.array, the (assumed) time at which x(theta) = x_pred_to.

        :returns: A scipy.stats.norm object, the transition density for the given dt and theta.
        """
        cov_theta = np.array([[self.C_L[0, 0] + 2 * self.C_L[1, 0] * (theta - self.t_L)
                               + self.C_L[1, 1] * (theta - self.t_L) ** 2 + self.S_w * pow(theta - self.t_L, 3) / 3,
                               self.C_L[1, 0] + self.C_L[1, 1] * (theta - self.t_L) + self.S_w * pow(theta - self.t_L,
                                                                                                     2) / 2],
                              [self.C_L[1, 0] + self.C_L[1, 1] * (theta - self.t_L) + self.S_w * pow(theta - self.t_L,
                                                                                                     2) / 2,
                               self.C_L[1, 1] + self.S_w * (theta - self.t_L)]])
        trans_mu = self.x_predTo + dt * (
                    self.x_L[1] + cov_theta[1, 0] / cov_theta[0, 0] * (self.x_predTo - self.ev_t(theta)))
        trans_var = dt ** 2 * (
                    cov_theta[1, 1] - cov_theta[1, 0] / cov_theta[0, 0] * cov_theta[0, 1]) + self.S_w * 1 / 3 * dt ** 3

        return norm(loc=trans_mu, scale=np.sqrt(trans_var))

    def trans_prob_with_positive_velocity(self, dt, theta):
        """The transition probability P(x(dt+theta) > a | x(theta) = x_predTo, v(theta)= > 0) from going from x_predTo
        at time theta and positive velocity to x(dt+theta) < a at time dt+theta.

        P(x(t) < a | x(theta) = a, v(theta) > 0) is calculated according to

            P(x(t) < a | x(theta) = a, v(theta) > 0) = P(x(t) < a , v(theta) > 0 | x(theta) = a) /
                                                            P( v(theta) > 0 | x(theta) = a) ,

            and further

            P(x(t) < a , v(theta) > 0 | x(theta) = a) = P(x(t) < a , v(theta) < inf | x(theta) = a)
                                                            - P(x(t) < a , v(theta) < 0| x(theta) = a) .

        :param dt: A float or np.array, the time difference. dt is zero at time = theta.
        :param theta: A float or np.array, the (assumed) time at which x(theta) = x_pred_to.

        :returns: A float, the probability P(x(dt+theta) > a | x(theta) = x_predTo, v(theta)= > 0).
        """
        F_theta, Q_theta = _get_system_matrices_from_parameters(theta - self.t_L, self.S_w)
        F_t, Q_t = _get_system_matrices_from_parameters(theta + dt - self.t_L, self.S_w)
        mean_theta = np.dot(F_theta, self.x_L)
        cov_theta = np.matmul(np.matmul(F_theta, self.C_L), F_theta.T) + Q_theta
        mean_t = np.dot(F_t, self.x_L)
        cov_t = np.matmul(np.matmul(F_t, self.C_L), F_t.T) + Q_t
        F_trans, Q_trans = _get_system_matrices_from_parameters(dt, self.S_w)
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
        # print('DT',dt)
        # print('theta', theta)
        # print( p_joint_v_theta_x_t_given_x_theta_cov )

        joint_v_theta_x_t_given_x_theta = multivariate_normal(mean=p_joint_v_theta_x_t_given_x_theta_mean,
                                                              cov=p_joint_v_theta_x_t_given_x_theta_cov,
                                                              allow_singular=True)  # To allow for numerical edge cases
        prob_joint_v_theta_greater_zero_x_t_smaller_a_given_x_theta = joint_v_theta_x_t_given_x_theta.cdf(
            np.array([np.inf, self.x_predTo])) - joint_v_theta_x_t_given_x_theta.cdf(np.array([0, self.x_predTo]))

        return prob_joint_v_theta_greater_zero_x_t_smaller_a_given_x_theta / prob_v_theta_greater_zero_given_x_theta

    def returning_probs_conditioned_on_positive_velocity_integrate_quad(self, t):
        """Calculates approximate returning probabilities using numerical integration thereby using the transition
        probabilities that are conditioned on a positive velocity at the time of a crossing.

        Approach:

         P(t < T_a , x(t) < a) = int_{t_L}^t fptd(theta) P(x(t) < a | x(theta) = a, v(theta) > 0) d theta ,

          with theta the time, when x(theta) = a.

        :param t: A float or np.array, the time parameter of the distribution.

        :returns: An approximation for the probability P(t < T_a , x(t) < a), i.e., the probability that a sample path
            has crossed the boundary at a time theta < t, but is smaller than the boundary at time t.
        """
        fn = lambda theta: self.pdf(theta) * self.trans_prob_with_positive_velocity(dt=t - theta,
                                                                                    theta=theta)
        a = np.finfo(np.float64).eps if self.t_L == 0 else self.t_L
        return integrate.quad(fn, a=a, b=t)[0]  # this is a tuple

    def returning_probs_conditioned_on_positive_velocity_uniform_samples(self, t, num_samples=100, deterministic_samples=True):
        """Calculates approximate returning probabilities (with the transition probabilities that are conditioned on a
        positive velocity at the time of a crossing) using a numerical integration (MC integration) based on
        samples from a uniform distribution.

        Approach:

         P(t < T_a , x(t) < a) = int_{t_L}^t fptd(theta) P(x(t) < a | x(theta) = a, v(theta) > 0) d theta

                          ≈  (t - t_L) / N sum_{theta_i} FPTD(theta_i) * P(x(t) < a | x(theta_i) = a, v(theta_i) > 0) ,
                               theta_i samples from a uniform distribution (N samples in total) in [t_L, t] ,

          with theta the time, when x(theta) = a.

        :param t: A float or np.array, the time parameter of the distribution.
        :param num_samples: An integer, the number of samples to approximate the integral.
        :param deterministic_samples: A Boolean, whether to use random samples (False) or deterministic samples (True).

        :returns: An approximation for the probability P(t < T_a , x(t) < a), i.e., the probability that a sample path
            has crossed the boundary at a time theta < t, but is smaller than the boundary at time t.
        """
        if not deterministic_samples:
            theta_samples = np.random.uniform(low=self.t_L, high=t, size=num_samples)
        else:
            # low=0, high=1, num_samples=5 -> [0.16, 0.33, 0.5, 0.67, 0.83]
            theta_samples = np.linspace(self.t_L, t, num=num_samples + 1, endpoint=False)[1:]

        return (t - self.t_L) * np.nanmean(
            [self.pdf(theta) * self.trans_prob_with_positive_velocity(dt=t - theta,
                                                                      theta=theta) for
             theta in theta_samples])

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'SKEW': self.skew,
                          # 'ReturningProbs': self.returning_probs_conditioned_on_positive_velocity_integrate_quad,
                          # 'ReturningProbs': self.returning_probs_conditioned_on_positive_velocity_uniform_samples,
                          })
        return hit_stats


class TaylorHittingTimeModel(AbstractCVHittingTimeModel, AbstractTaylorHittingTimeModel):
    """A simple Gaussian approximation for the first hitting time problem using a Taylor approximation and error
    propagation.
    """

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name='Gauß--Taylor approx.'):
        """Initialize the model.

        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, pos_y, vel_y].
        :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param x_predTo: A float, position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, name for the model.
        """
        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         name=name)

        self._ev = t_L + (x_predTo - x_L[0]) / x_L[1]
        # Evaluate the equation at the time at the boundary
        v_L = x_L[1]
        dt_p = self.ev - t_L
        sigma_x = np.sqrt(C_L[0, 0] + 2 * C_L[1, 0] * dt_p + C_L[1, 1] * dt_p**2 + S_w * pow(dt_p, 3)/3)
        sigma_xv = C_L[1, 0] + C_L[1, 1] * dt_p + S_w * pow(dt_p, 2)/2
        sigma_v = np.sqrt(C_L[1, 1] + S_w * dt_p)
        dx = 0
        self._var = (1 / v_L) ** 2 * sigma_x ** 2 + (dx / v_L ** 2) ** 2 * sigma_v ** 2 + 2 * dx / v_L ** 3 * sigma_xv

    @property
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        return 0  # Gaussian third central moment


class EngineeringApproxHittingTimeModel(AbstractCVHittingTimeModel, AbstractEngineeringApproxHittingTimeModel):
    """An approximation to the first passage time distribution using the (engineering) assumption that particles
    are unlikely to move back once they have passed the boundary.
    """

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name="No-return approx."):
        """Initialize the model.

        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, pos_y, vel_y].
        :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param x_predTo: A float, position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, name for the model.
        """
        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         name=name)

        if self.q_max < 0.95:
            logging.warning(self.name + " does not seem to be applicable as max. confidence {} < 0.95.".format(
                round(self.q_max, 2)))

    def _cdf(self, t):
        """Approach: 1 - int( p(x(t), x= -infty .. x_predTo )

        :param t: A float or np.array, the time parameter of the distribution.
        """
        p_x_given_t = norm(loc=self.ev_t(t), scale=np.sqrt(self.var_t(t)))
        return 1 - p_x_given_t.cdf(self.x_predTo)

    def cdf(self, t):
        """The CDF of the first passage time distribution.

        :param t: A float or np.array, the time parameter of the distribution.
        """
        t = np.asarray([t]) if np.isscalar(t) else np.asarray(t)
        cdf_value = self._cdf(t)
        cdf_value[t > self.t_max] = self.q_max  # piecewise function
        return np.squeeze(cdf_value)

    def _pdf(self, t):
        """Derivative of self._cdf. Can be calculated from the standard Gauss pdf with an argument
        (x_predTo - ev(t))/stddev(t) times the derivative with respect to t of these argument (chain rule).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        gauss = norm.pdf((self.x_predTo - self.ev_t(t))/np.sqrt(self.var_t(t)))  # Std. Gauss pdf
        der_arg = self.x_L[1] / np.sqrt(self.var_t(t)) + (self.x_predTo - self.ev_t(t)) * (
                    0.5 * self.S_w * t ** 2 - self.S_w * t * self.t_L + 0.5 * self.S_w * self.t_L ** 2 + t * self.C_L[
                1, 1] - self.t_L * self.C_L[1, 1] + self.C_L[1, 0]) / (self.var_t(t)) ** (3.0 / 2.0)
        return gauss * der_arg

    def pdf(self, t):
        """The first passage time distribution (FPTD).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        t = np.asarray([t]) if np.isscalar(t) else np.asarray(t)
        pdf_value = self._pdf(t)
        pdf_value[t > self.t_max] = 0  # piecewise function
        return np.squeeze(pdf_value)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        # perform sanity checks on input
        if q > self.q_max:
            logging.warning(
                'Approximation yields a maximum confidence of {}, '
                'which is lower than the desired confidence level of {}. Computed values may be wrong.'.format(
                    np.round(self.q_max, 4), np.round(q, 4)))

        return self._ppf(q)

    def _ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        Approach:

              1 - q = int(N(x, mu(t), var(t)), x = -inf .. x_predTo) = PHI ( (x_predTo - mu(t)) / sqrt(var(t))
              PHI^-1(1 -q) = (x_predTo - mu(t)) / sqrt(var(t)) -> solve for t...

        We solve the equation for t = t - t_L to simplify calculations and add t_L at the end of the function.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        if q < 0.0 or q > 1.0:
            raise ValueError('Confidence level q must be between 0 and 1.')

        # Special case q = 0.5, this corresponds to the median.
        # 0 = (x_predTo - mu(t)) / sqrt(var(t) -> x_predTo = mu(t) -> solve for t...
        if q == 0.5:
            t = (self.x_predTo - self.x_L[0])/self.x_L[1]
            return t + self.t_L

        # cubic function
        # At**3 + B*t**2 + C*t + D = 0
        qf = norm.ppf(1 - q)  # Standard-Gaussian quantile function
        epsilon = 0.0  # to avoid divisions by zero, 0.01 is a good value for Cardano

        # A = self.S_w / 3
        # B = self.C_L[1, 1] - self.x_L[1]**2/(qf**2 + epsilon)
        # C = 2*self.C_L[1, 0] + 2*(self.x_predTo - self.x_L[0])*self.x_L[1]/(qf**2 + epsilon)
        # D = self.C_L[0, 0] - (self.x_predTo - self.x_L[0])**2/(qf**2 + epsilon)

        A = self.S_w / 3 * (qf**2 + epsilon)
        B = self.C_L[1, 1]*(qf**2 + epsilon) - self.x_L[1]**2
        C = 2*self.C_L[1, 0]*(qf**2 + epsilon) + 2*(self.x_predTo - self.x_L[0])*self.x_L[1]
        D = self.C_L[0, 0]*(qf**2 + epsilon) - (self.x_predTo - self.x_L[0])**2

        def find_cubic_roots_numpy(A, B, C, D):
            """Find the relevant root using numpy's np.roots function."""
            roots = np.roots([A, B, C, D])
            # roots are in descending order, the first root is always too large.

            # real_roots = roots.real[np.logical_not(np.iscomplex(roots))]
            # if real_roots.shape[0] == 1:
            #     t = float(real_roots)
            # if real_roots.shape[0] == 3:
            #     t = float(real_roots[2] if q < 0.5 else real_roots[1])
            return float(roots[2].real if q < 0.5 else roots[1].real)

        def find_cubic_roots_cardano_trigonometry(A, B, C, D):
            """Find the relevant root using Cardano's method and trigonometric functions.

            This function has some numerical problems in the vicinity of q=0.5.
            """
            # t**3 + a*t**2 + b*t + c = 0
            a = B/A
            b = C/A
            c = D/A

            # substitute t = z - a/3 -> z**3 + p*z + q = 0
            p = b - a**2/3
            qq = 2*a**3/27 - a*b/3 + c

            # Cardano formula
            Delta = (qq/2)**2 + (p/3)**3

            if np.isnan(Delta):
                logging.warning('Nan-value detected for confidence level {}.'.format(q))
                return np.nan
            else:
                eps = 1E-12
                # Handle the cases separately, ignore complex solutions
                if Delta > eps:
                    # Case: Delta > 0
                    # u = (-qq / 2 + np.sqrt(Delta)) ** (1.0 / 3.0)
                    arg_u = -qq / 2 + np.sqrt(Delta)
                    u = np.sign(arg_u)*(np.abs(arg_u)) ** (1.0 / 3.0)
                    # v = (-qq / 2 - np.sqrt(Delta)) ** (1.0 / 3.0)
                    arg_v = -qq / 2 - np.sqrt(Delta)
                    v = np.sign(arg_v)*(np.abs(arg_v)) ** (1.0 / 3.0)
                    t = u + v - B/(3*A)


                elif np.isclose(Delta, 0, atol=eps):
                    # Case: Delta = 0
                    if np.isclose(p, 0, atol=eps):
                        t = -B/(3*A)
                    else:
                        t_1 = 3*qq/p - B/(3*A)
                        t_2 = - 3 * qq / (2*p) - B / (3 * A)
                        t = t_1 if q < 0.5 else t_2

                else:
                    # Case: Delta < 0:
                    t_1 = - np.sqrt(-4/3*p) * np.cos(1/3*np.arccos(-qq/2*np.sqrt(-27/p**3)) + np.pi/3) - B/(3*A)
                    t_2 = np.sqrt(-4/3 * p) * np.cos(1 / 3 * np.arccos(-qq / 2 * np.sqrt(-27 / p ** 3))) - B / (3 * A)
                    t_3 = - np.sqrt(-4/3*p) * np.cos(1/3*np.arccos(-qq/2*np.sqrt(-27/p**3)) - np.pi/3) - B/(3*A)

                    # Function must be positive for all confidence levels (because t is starting at 0),
                    # but we have a sign shift at q=0.5. Thus:
                    t = t_3 if q < 0.5 else t_1

                return t

        def find_cubic_roots_cardano_complex(a, b, c, d):
            """Find the relevant root using Cardano's method and complex numbers.

            This is numerically more stable then with trigonometric functions.
            From https://stackoverflow.com/questions/39474254/cardanos-formula-not-working-with-numpy
            """
            a, b, c, d = a + 0j, b + 0j, c + 0j, d + 0j
            all_ = (a != np.pi)

            Q = (3 * a * c - b ** 2) / (9 * a ** 2)
            R = (9 * a * b * c - 27 * a ** 2 * d - 2 * b ** 3) / (54 * a ** 3)
            D = Q ** 3 + R ** 2
            S = 0
            if np.isreal(R + np.sqrt(D)):
                S = cbrt(np.real(R + np.sqrt(D)))
            else:
                S = (R + np.sqrt(D)) ** (1 / 3)
            T = 0
            if np.isreal(R - np.sqrt(D)):
                T = cbrt(np.real(R - np.sqrt(D)))
            else:
                T = (R - np.sqrt(D)) ** (1 / 3)

            result = np.zeros(tuple(list(a.shape) + [3])) + 0j
            result[all_, 0] = - b / (3 * a) + (S + T)
            result[all_, 1] = - b / (3 * a) - (S + T) / 2 + 0.5j * np.sqrt(3) * (S - T)
            result[all_, 2] = - b / (3 * a) - (S + T) / 2 - 0.5j * np.sqrt(3) * (S - T)

            # Function must be positive for all confidence levels (because t is starting at 0),
            # but we have a sign shift at q=0.5. Thus:

            # t = float(result[1].real if q < 0.5 else result[2].real)
            real_result = np.zeros(result.shape)
            real_result[np.isreal(result)] = result[np.isreal(result)].real
            real_result[np.logical_not(np.isreal(result))] = np.nan
            real_result[real_result < 0] = np.nan
            t = real_result[1] if q < 0.5 else real_result[2]

            # Do not use only real solution since in the vicinity of 0.5, due to numerical problems, there
            # are some solutions with small imaginary parts
            #
            # # Check all possible solutions
            # probs = 1 - norm.cdf(self.x_predTo, loc=self.ev_t(result.real),
            #                      scale=np.sqrt(self.var_t(result.real)))
            #
            # valid_roots = result.real[np.isclose(probs, q, atol=1E-02)]
            # # if min_dt_q_max.shape[0] != 1:
            # #
            # if valid_roots.shape[0] == 0:
            #     t = np.nan
            # elif valid_roots.shape[0] == 1:
            #     t = float(valid_roots)
            # else:
            #     closest_root = valid_roots[np.abs(probs - q).argmin()]
            #     t = closest_root
            #     #t = float(valid_roots[0] if q < 0.5 else valid_roots[1])
            # # else:
            # #     raise ValueError(
            # #         'Calculation of PPF not successful, there are {} possible roots.'.format(valid_roots.shape[0]))

            return t

        # t = find_cubic_roots_numpy(A, B, C, D) # works fine for all q
        # t = find_cubic_roots_cardano_trigonometry(A, B, C, D)  # works fine for all q except for q approx. 0.5
        # t = find_cubic_roots_cardano_complex(A, B, C, D)  # works fine for all q except for small q
        if q > 0.4 and q < 0.6:
            # t = find_cubic_roots_cardano_complex(A, B, C, D)
            t = find_cubic_roots_numpy(A, B, C, D)
        else:
            t = find_cubic_roots_cardano_trigonometry(A, B, C, D)

        return t + self.t_L

    def _get_max_cdf_value_and_location(self):
        """Method that finds the maximum of the CDF of the approximation and its location.

        Approach:

            set self.pdf(t) = 0, solve for t.

        We solve the equation for t = t - t_L to simplify calculations and add t_L at the end of the function.

        :returns,:
            q_max: A float, the maximum value of the CDF.
            t_q_max: A float, the time, when the CDF visits its maximum.
        """
        A = -1/3*self.S_w*self.x_L[1]
        B = (self.x_predTo - self.x_L[0])*self.S_w
        C = 2*(self.x_L[1]*self.C_L[0, 1] + (self.x_predTo - self.x_L[0])*self.C_L[1, 1])
        D = 2*(self.x_L[1]*self.C_L[0, 0] + (self.x_predTo - self.x_L[0])*self.C_L[0, 1])

        roots = np.roots([A, B, C, D])
        t_q_max = roots[np.isreal(roots)].real[-1]  # get the highest one (it should be only one)
        t_q_max += self.t_L

        # Get the CDF Value
        q_max = self._cdf(t_q_max)
        return q_max, t_q_max

    @property
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
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
        return self._third_central_moment

    def trans_dens_ppf(self, theta, q=0.95):
        """The PPF of 1 - int ( p(x(dt+theta)| x(theta) = x_predTo), x(dt+theta) = - infty .. x_predTo),
        i.e., the inverse CDF of the event that particles are above x_predTo once they have reached it at time theta.

        Note that in terms of the used approximation, this can be seen as PPF of the approximate first passage
        returning time distribution w.r.t. the boundary x_pred_to.

        :param theta: A float or np.array, the (assumed) time at which x(theta) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.

        :returns: A np.array, the value of the PPF for q and theta, note that this a delta time w.r.t. theta.
        """
        # TODO: One could alternatively use scipy.norm's ppf function on self.trans_dens
        cov_theta = np.array([[self.C_L[0, 0] + 2 * self.C_L[1, 0] * (theta - self.t_L)
                               + self.C_L[1, 1] * (theta - self.t_L) ** 2 + self.S_w * pow(theta - self.t_L, 3) / 3,
                              self.C_L[1, 0] + self.C_L[1, 1] * (theta - self.t_L) + self.S_w * pow(theta - self.t_L,
                                                                                                     2) / 2],
                              [self.C_L[1, 0] + self.C_L[1, 1] * (theta - self.t_L) + self.S_w * pow(theta - self.t_L,
                                                                                                     2) / 2,
                              self.C_L[1, 1] + self.S_w * (theta - self.t_L)]])
        b = cov_theta[1, 1] - cov_theta[1, 0] / cov_theta[0, 0] * cov_theta[0, 1]
        c = cov_theta[1, 0] / cov_theta[0, 0] * (self.x_predTo - self.ev_t(theta))

        qf = norm.ppf(1 - q)

        A = self.S_w / 3 * qf**2
        B = b*qf**2 - (self.x_L[1] + c)**2
        C = 0
        D = 0

        positive_roots = np.array([-B/A])
        return positive_roots

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'RAW_CDF': self._cdf,
                          })
        return hit_stats


class MCHittingTimeModel(AbstractCVHittingTimeModel, AbstractMCHittingTimeModel):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem to a distribution
     using scipy.stats.rv_histogram.
    """

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, t_range, bins=100, t_samples=None, name='MC simulation'):
        """Initialize the model.

        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, pos_y, vel_y].
        :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param x_predTo: A float, position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param t_range: A list of length 2 representing the limits for the first passage time histogram (the number of
            bins within t_range will correspond to bins).
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param t_samples: None or a np.array of shape [N] containing the first passage times of the particles. If None,
            t_samples will be created by a call to a sampling method. If given, given values will be used.
        :param name: String, name for the model.
        """
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

    @property
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        return self.third_moment - 3 * self.ev * self.var - self.ev ** 3

    @property
    def third_moment(self):
        """The third moment of the first passage time distribution."""
        return self._density.moment(3)

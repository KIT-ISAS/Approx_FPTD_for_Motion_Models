from absl import logging

from abc import ABC, abstractmethod

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

from abstract_hitting_location_models import AbstractHittingLocationModel, AbstractTaylorHittingLocationModel, AbstractSimpleGaussHittingLocationModel, AbstractUniformHittingLocationModel, AbstractMCHittingLocationModel
from cv_utils import create_ty_cv_samples_hitting_time


class AbstractCVHittingLocationModel(AbstractHittingLocationModel, ABC):
    """A base class for the CV hitting location models.

    These models calculate the distribution in y at the first passage time.
    """

    def __init__(self, hitting_time_model, S_w, name='CV hitting location model', **kwargs):
        """Initialize the model.

        :param hitting_time_model, a HittingTimeModel object.
        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, pos_y, vel_y].
        :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.

        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, (default) name for the model.
        """
        super().__init__(hitting_time_model=hitting_time_model,
                         name=name,
                         S_w=S_w,
                         **kwargs,
                         )

    @property
    def ev(self):
        """The expected value of the distribution in y at the first passage time."""
        if self._ev is None:
            self._ev = self._ht.x_L[2] + self._ht.x_L[3] * (self._ht.ev - self._ht.t_L)
        return self._ev

    @property
    def var(self):
        """The variance of the distribution in y at the first passage time."""
        if self._var is None:
            self._var = self._ht.C_L[2, 2] + 2 * self._ht.C_L[2, 3] * (self._ht.ev - self._ht.t_L) \
                        + self._ht.C_L[3, 3] * (self._ht.second_moment - 2 * self._ht.ev * self._ht.t_L + self._ht.t_L ** 2) \
                        + self._S_w / 3 * (
                                self._ht.third_moment - 3 * self._ht.second_moment * self._ht.t_L + 3 * self._ht.ev * self._ht.t_L ** 2
                                - self._ht.t_L ** 3)
        return self._var
    
    @staticmethod
    def _compute_ev(hitting_time_model):
        return hitting_time_model.x_L[2] + hitting_time_model.x_L[3] * (hitting_time_model.ev - hitting_time_model.t_L)

    @staticmethod
    def _compute_var(hitting_time_model, S_w):  # TODO: Mit oberer verbinden
        var = hitting_time_model.C_L[2, 2] + 2 * hitting_time_model.C_L[2, 3] * (
                    hitting_time_model.ev - hitting_time_model.t_L) \
              + hitting_time_model.C_L[3, 3] * (
                          hitting_time_model.second_moment - 2 * hitting_time_model.ev * hitting_time_model.t_L + hitting_time_model.t_L ** 2) \
              + S_w / 3 * (
                      hitting_time_model.third_moment - 3 * hitting_time_model.second_moment * hitting_time_model.t_L + 3 * hitting_time_model.ev * hitting_time_model.t_L ** 2
                      - hitting_time_model.t_L ** 3)
        return var

    def ev_t(self, t):
        """The mean function of the CV motion model in y.

        :param t: A float or np.array, the time parameter of the mean function.
        """
        return self._ht.x_L[2] + self._ht.x_L[3] * (t - self._ht.t_L)

    def var_t(self, t):
        """The variance function of the CV motion model in y.

        :param t: A float or np.array, the time parameter of the variance function.
        """
        return self._ht.C_L[2, 2] + 2 * self._ht.C_L[2, 3] * (t - self._ht.t_L) + self._ht.C_L[3, 3] * (
                    t - self._ht.t_L) ** 2 + self._S_w * pow(t - self._ht.t_L, 3) / 3


class CVTaylorHittingLocationModel(AbstractTaylorHittingLocationModel, AbstractCVHittingLocationModel):  # TODO: Benennung einheitlich machen, wo steht CV etc.
    """A simple Gaussian approximation for the distribution in y at the first passage time problem using a
    Taylor approximation and error propagation.

    Note that this method, although it may capture the shape of the distribution very well, does not have the exact
    moments as calculated in the AbstractCVHittingLocationModel base class.
    """

    def __init__(self, hitting_time_model, S_w, name='Gauß--Taylor approx.'):
        """Initialize the model.

        :param hitting_time_model, a HittingTimeModel object.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param name: String, name for the model.
        """

        # Uncertainty prediction in spatial dimension
        dt_p = hitting_time_model.ev - hitting_time_model.t_L
        sigma_y = np.sqrt(hitting_time_model.C_L[2, 2] + 2 * hitting_time_model.C_L[3, 2] * dt_p + hitting_time_model.C_L[3, 3] * dt_p ** 2 + S_w * pow(dt_p, 3) / 3)
        # TODO: Replace sigma y with var_t(dt_p). Should be the same.  Es ist sogar dasselbe für CA und CV -> Formel (nicht ergebnis) kann auch in die abstract
        # y-velocity at boundary
        vy = hitting_time_model.x_L[3]
        # y-position at boundary
        # overwrite the moments of the base class
        ev = hitting_time_model.x_L[2] + (hitting_time_model.ev - hitting_time_model.t_L) * hitting_time_model.x_L[3]
        var = sigma_y ** 2 + vy ** 2 * hitting_time_model.var

        super().__init__(hitting_time_model=hitting_time_model,
                         S_w=S_w,
                         name=name,
                         ev=ev,
                         var=var,
                         )


class SimpleGaussCVHittingLocationModel(AbstractSimpleGaussHittingLocationModel, AbstractCVHittingLocationModel):
    """A purely Gaussian approximation for the distribution in y at the first passage time problem by simply using the
    mean and variance of the distribution in y.

    Note that the mean and variance can be calculated directly (an independently of the used model)
    with the given FPTD as done in the AbstractCVHittingLocationModel parent class.

    Compared with the TaylorHittingLocationModel, this model has the exact moments, but its shape may capture
    the underlying distribution less well.
    """

    def __init__(self, hitting_time_model, S_w, name='Gauß approx.'):
        """Initialize the model.

        :param hitting_time_model, a HittingTimeModel object.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param name: String, name for the model.
        """

        ev = self._compute_ev(hitting_time_model)
        var = self._compute_var(hitting_time_model, S_w)

        super().__init__(hitting_time_model=hitting_time_model,
                         S_w=S_w,
                         name=name,
                         ev=ev,
                         var=var,
                         )


class ProjectionCVHittingLocationModel(AbstractCVHittingLocationModel):
    """Jakob Thumm's method. Marginalize (project) the distribution at theta with theta the solution of

    # TODO: Passt das
    E{x(t)} = x_predTo w.r.t to t to y(t) using that
        p( y(t) | y(theta), vy(theta)) ~= delta( y(t) - y(theta) - vy(theta)(t - theta) ).

    This uses a Bayesian model

        p(y(t)) = int ( p(y(t) | y(theta), vy(theta), t)
                            p (y(theta), vy(theta) ) p(t) d y(theta), d vy(theta) d t

    and to get the distribution p(y(t)), where p(t) ist the FPTD.
    Note that a numerical method for the (double) integration is needed,
    resulting in high computational times ( > 2 min).
    Furthermore, the moments are not the same as the calculated for the base class.
    """

    def __init__(self, hitting_time_model, S_w, name='Projection method'):
        """Initialize the model.

        :param hitting_time_model, a HittingTimeModel object.
        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, pos_y, vel_y].
        :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, name for the model.
        """
        super().__init__(hitting_time_model=hitting_time_model,
                         S_w=S_w,
                         name=name,
                         )

        # Shared variables (for temporal and spacial)
        self._t_predicted = 1.0 * (self._ht.t_L + (self._ht.x_predTo - self._ht.x_L[0]) / self._ht.x_L[1])
        self._v_L = self._ht.x_L[1]

    # Uncertainty prediction in spatial dimension
    def pdf(self, y):
        """The PDF in y at the first passage time.

        :param y: A float or np.array, the y parameter of the distribution.
        """
        C_L = self._ht.C_L
        dt_p = self._ht.dt_p  # TODO: Das ist nicht mehr definiert...
        x_L = self._ht.x_L

        sigma_y = np.sqrt(C_L[2, 2] + 2 * C_L[3, 2] * dt_p + C_L[3, 3] * dt_p ** 2 + self._S_w * pow(dt_p, 3) / 3)
        # w := velocity in y direction
        sigma_yw = C_L[3, 2] + C_L[3, 3] * dt_p + self._S_w * pow(dt_p, 2) / 2
        sigma_w = np.sqrt(C_L[3, 3] + self._S_w * dt_p)
        # y-velocity at boundary
        vy = x_L[3]
        # y-position at boundary
        y_L = x_L[2] + self._t_predicted * x_L[3]
        det_C = sigma_y ** 2 * sigma_w ** 2 - sigma_yw ** 2

        return 1 / (2 * np.pi * np.sqrt(det_C)) * integrate.dblquad(lambda t_i, vy_i:
                                                                    np.exp(-1 / (2 * det_C) * ((
                                                                                                       y - vy_i * t_i - y_L) ** 2 * sigma_w ** 2 -
                                                                                               2 * (
                                                                                                       y - vy_i * t_i - y_L) * (
                                                                                                       vy_i - vy) * sigma_yw +
                                                                                               (
                                                                                                       vy_i - vy) ** 2 * sigma_y ** 2))
                                                                    * self._ht.f_t_predicted(t_i - 0)
                                                                    , -2.3, 2.7, lambda vy_i: -0.01,
                                                                    lambda vy_i: 0.01)[0]

    def cdf(self, y):
        """The CDF in y at the first passage time.

        :param y: A float or np.array, the y parameter of the distribution.
        """
        raise NotImplementedError()

    @property
    def ev(self):
        """The expected value of the distribution in y at the first passage time."""
        if self._ev is None:
            self._ev = integrate.quad(lambda y: y * self.pdf(y), 0, np.inf)[0]
        return self._ev

    @property
    def var(self):
        """The variance of the distribution in y at the first passage time."""
        if self._var is None:
            self._var = integrate.quad(lambda y: y ** 2 * self.pdf(y), 0, np.inf)[
                            0] - self.ev ** 2
        return self._var

    def ppf(self, q):
        raise NotImplementedError()


class UniformCVHittingPlaceDistribution(AbstractUniformHittingLocationModel, AbstractCVHittingLocationModel):
    """A simple Gaussian approximation for the for the distribution in y at the first passage time using a Taylor
    approximation and error
    propagation. # TODO
    """

    def __init__(self, hitting_time_model, point_predictor, window_length, a=0.5, name='Uniform model'):
        """Initialize the model.
        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, pos_y, vel_y].
        :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param x_predTo: A float, position of the boundary.
        :param name: String, name for the model.
        """
        pos_last = hitting_time_model.x_L[:, [0, 2]]
        v_last = hitting_time_model.x_L[:, [1, 3]]
        y_predicted = point_predictor.predict(pos_last, v_last, dt_pred=hitting_time_model.ev - hitting_time_model.t_L) + hitting_time_model.t_L  # TODO. Das stimmt nicht, wo stimmt es noch nicht?

        super().__init__(hitting_time_model=hitting_time_model,
                         S_w=0,
                         name=name,
                         point_prediction=y_predicted,
                         window_length=window_length,
                         a=a)


class MCCVHittingLocationModel(AbstractMCHittingLocationModel, AbstractCVHittingLocationModel):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem to a distribution
     using scipy.stats.rv_histogram.
    """

    def __init__(self, hitting_time_model, S_w, y_range, bins=100, y_samples=None, name='MC simulation'):
        """Initialize the model.

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
        if y_samples is None:
            _, y_samples, _ = create_ty_cv_samples_hitting_time(self._ht.x_L, self._ht.C_L, S_w, self._ht.x_predTo,
                                                                self._ht.t_L)

        super().__init__(hitting_time_model=hitting_time_model,
                         S_w=S_w,
                         name=name,
                         y_samples=y_samples,
                         y_range=y_range,
                         bins=bins,
                         )

from abstract_hitting_location_distributions import *
from cv_arrival_distributions.cv_utils import create_ty_cv_samples_hitting_time


class AbstractCVHittingLocationDistribution(AbstractHittingLocationDistribution, ABC):
    """A base class for the CV hitting location distributions.

    These models calculate the distribution in y at the first-passage time.
    """

    def __init__(self, htd, S_w, name="AbstractCVHittingLocationDistribution", **kwargs):
        """Initializes the distribution.

        State format:

            [..., pos_y, velo_y]

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in y-direction.
        :param name: String, the (default) name for the distribution.
        """
        super().__init__(htd=htd,
                         name=name,
                         **kwargs,
                         )

        self._S_w = np.broadcast_to(S_w, shape=self.batch_size)  # this itself raises an error if not compatible

    @property
    def S_w(self):
        """The power spectral density (PSD) in y-direction.

        :returns A float or np.array of shape [batch_size], the power spectral density.
        """
        return np.squeeze(self._S_w, axis=0)

    @S_w.setter
    @abstractmethod
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in y-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in y-direction.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def ev(self):
        """The expected value of the distribution in y at the first-passage time.

        :returns: A float or a np.array of shape [batch_size], the expected value.
        """
        return self._htd.x_L[..., -2] + self._htd.x_L[..., -1] * (self._htd.ev - self._htd._t_L)

    @property
    def var(self):
        """The variance of the distribution in y at the first-passage time.

        :returns: A float or a np.array of shape [batch_size], the variance.
        """
        return self._compute_var(self._htd, self._S_w)

    @staticmethod
    def _compute_var(htd, S_w):  # TODO: In die abstract? Die haben wir schon für die Taylor? Brauchen wir die wirklich im Allgemeinen?
        """Computes the variance of the distribution in y at the first-passage time.

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A np.array of shape [batch_size], the power spectral density (PSD).

        :returns: A np.array of shape [batch_size], the variance of the approximation.  # TODO: Das ist nun nicht unbedingt shape = batchsize, schlimm?
        """
        var = htd.C_L[..., -2, -2] + 2 * htd.C_L[..., -2, -1] * (
                htd.ev - htd._t_L) \
              + htd.C_L[..., -1, -1] * (
                      htd.second_moment - 2 * htd.ev * htd._t_L + htd._t_L ** 2) \
              + S_w / 3 * (
                      htd.third_moment - 3 * htd.second_moment * htd._t_L + 3 * htd.ev * htd._t_L ** 2
                      - htd._t_L ** 3)
        return var

    # TODO: Eine Plot funktion um den Verlauf der Tracks zu sehen wäre noch sehr hilfreicg

    def _ev_t(self, t):
        """The mean function of the motion model in y.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the mean in y at time t.
        """
        return self._htd.x_L[..., -2] + self._htd.x_L[..., -1] * (t - self._htd._t_L)

    def _var_t(self, t):
        """The variance function of the motion model in y.

        :param t: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the time parameter of the distribution.

        :returns: A np.array of shape [batch_size, sample_size], the variance in y at time t.
        """
        return self._htd.C_L[..., -2, -2] + 2 * self._htd.C_L[..., -2, -1] * (t - self._htd._t_L) + self._htd.C_L[
            ..., -1, -1] * (t - self._htd._t_L) ** 2 + self._S_w * pow(t - self._htd._t_L, 3) / 3

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        super().scale_params(length_scaling_factor, time_scaling_factor)
        self._S_w *= length_scaling_factor ** 2 / time_scaling_factor ** 3

    def __setitem__(self, indices, values):
        """Assigns elements along the batch shape at the given indices. Use this for fancy indexing
        (e.g., distr[:2] = old_distr).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._S_w[indices] = values.S_w
        super().__setitem__(indices, values)

    def _left_hand_indexing(self, indices, values):
        """Takes elements of values and assigns elements along the batch shape at the given indices. This is a helper
        function for __getitem__, which is used for fany indexing (e.g., new_distr = distr[:2]).

        :param indices: Slices, or list, or np.array of integers or Booleans. The indices of the values to assign.
        :param values: An object of the same type as self, the object from which to take the elements.
        """
        self._S_w = values.S_w[indices]
        super()._left_hand_indexing(indices, values)


class GaussTaylorCVHittingLocationDistribution(AbstractGaussTaylorHittingLocationDistribution, AbstractCVHittingLocationDistribution):
    """A simple Gaussian approximation for the distribution in y at the first-passage time problem using a
    Taylor approximation and error propagation that can be used for CV models.

    Note that this method, although it may capture the shape of the distribution very well, does not have the exact
    moments as calculated in the AbstractCVHittingLocationModel base class.
    """
    def __init__(self, htd, S_w, point_predictor, name='Gauß--Taylor approx.'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, ..., pos_y, velo_y]

         Format point_predictor:

            (pos_last, v_last, dt_pred)  --> dy_pred

         where
            - pos_last is a np.array of shape [batch_size, 2] and format [x, y] containing the positions at the _t_L.
            - v_last is a np.array of shape [batch_size, 2] and format [x, y] containing the velocities at the _t_L.
            - dt_pred is a np.array of shape [batch_size] with arrival time point estimates as difference times w.r.t.
                _t_L.
            - dy_pred is a np.array of shape [batch_size] with point estimates for the arrival location along the
                actuator array as difference w.r.t. x_L.

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in y-direction.
        :param point_predictor: A callable, a function that returns an estimate for the arrival location.
        :param name: String, the name for the distribution.
        """
        # sanity checks
        if not callable(point_predictor):
            raise ValueError('point_predictor must be a callable.')

        ev = point_predictor(htd.x_L[..., [0, -2]], htd.x_L[..., [1, -1]], dt_pred=htd.ev - htd._t_L) + htd.x_L[..., -2]
        var = self._compute_var(htd, S_w)

        # # Uncertainty prediction in spatial dimension
        # dt_p = htd.ev - htd._t_L
        # sigma_y = np.sqrt(htd.C_L[2, 2] + 2 * htd.C_L[3, 2] * dt_p + htd.C_L[3, 3] * dt_p ** 2 + S_w * pow(dt_p, 3) / 3)
        # # TODO: Replace sigma y with _var_t(dt_p). Should be the same.  Es ist sogar dasselbe für CA und CV -> Formel (nicht ergebnis) kann auch in die abstract
        # # y-velocity at boundary
        # vy = htd.x_L[3]
        # # y-position at boundary
        # # overwrite the moments of the base class
        # ev = htd.x_L[2] + (htd.ev - htd._t_L) * htd.x_L[3]
        # var = sigma_y ** 2 + vy ** 2 * htd.var

        super().__init__(htd=htd,
                         S_w=S_w,
                         point_predictor=point_predictor,
                         name=name,
                         ev=ev,
                         var=var,
                         )
    @staticmethod
    def _compute_var(htd, S_w):  # TODO: Das noch vereinfachen! Entspricht nicht der signatur der base-methode
        """Computes the variance of the distribution in y at the first-passage time based on error propagation.

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A np.array of shape [batch_size], the power spectral density (PSD).

        :returns: A np.array of shape [batch_size], the variance of the approximation.
        """
        dt_p = htd.ev - htd._t_L
        sigma_y = np.sqrt(
            htd.C_L[..., -2, -2] + 2 * htd.C_L[..., -1, -2] * dt_p + htd.C_L[..., -1, -1] * dt_p ** 2 + S_w * pow(dt_p,
                                                                                                                  3) / 3)
        var = sigma_y ** 2 + htd.x_L[..., -1] ** 2 * htd.var
        return var
    
    @AbstractCVHittingLocationDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in y-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in y-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size)
        # Recalculate the variance
        self._var = self._compute_var(self._htd, self._S_w)
    
    
class SimpleGaussCVHittingLocationDistribution(AbstractSimpleGaussHittingLocationDistribution, AbstractCVHittingLocationDistribution):
    """A purely Gaussian approximation for the distribution in y at the first-passage time problem by simply using the
    (theoretic) mean and variance of the distribution in y given the hitting time model that can be used for CV models.

    Note that the mean and variance can be calculated directly (and independently of the used approximation for the
    distribution of y at the first-passage time) with the given FPTD as done by the parent class.

    Compared with the GaussTaylorHittingLocationDistribution, this distribution uses the exact first and second moments,
    but its shape may capture the underlying distribution less well.
    """
    def __init__(self, htd, S_w, point_predictor, name='Gauß approx.'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, ..., pos_y, velo_y]

         Format point_predictor:

            (pos_last, v_last, dt_pred)  --> dy_pred

         where
            - pos_last is a np.array of shape [batch_size, 2] and format [x, y] containing the positions at the _t_L.
            - v_last is a np.array of shape [batch_size, 2] and format [x, y] containing the velocities at the _t_L.
            - dt_pred is a np.array of shape [batch_size] with arrival time point estimates as difference times w.r.t.
                _t_L.
            - dy_pred is a np.array of shape [batch_size] with point estimates for the arrival location along the
                actuator array as difference w.r.t. x_L.

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in y-direction.
        :param point_predictor: A callable, a function that returns an estimate for the arrival location.
        :param name: String, the name for the distribution.
        """
        # sanity checks
        if not callable(point_predictor):
            raise ValueError('point_predictor must be a callable.')

        ev = point_predictor(htd.x_L[..., [0, -2]], htd.x_L[..., [1, -1]], dt_pred=htd.ev - htd._t_L) + htd.x_L[..., -2]
        var = self._compute_var(htd, S_w)

        super().__init__(htd=htd,
                         S_w=S_w,
                         point_predictor=point_predictor,
                         name=name,
                         ev=ev,
                         var=var,
                         )
    
    @AbstractCVHittingLocationDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in y-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in y-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size)
        # Recalculate the variance
        self._var = self._compute_var(self._htd, self._S_w)


class UniformCVHittingLocationDistribution(AbstractUniformHittingLocationDistribution, AbstractCVHittingLocationDistribution):
    """Uses point predictors for the distribution in y at the first-passage time and a uniform distribution assuming a
    CV model.

    This distribution corresponds to the "usual" case where we define a fixed ejection window.
    """
    def __init__(self, htd, point_predictor, window_length, a=0.5, name='Uniform model'):
        """Initializes the distribution.

         State format:

            [pos_x, velo_x, ..., pos_y, velo_y]

         Format point_predictor:

            (pos_last, v_last, dt_pred)  --> dy_pred

         where
            - pos_last is a np.array of shape [batch_size, 2] and format [x, y] containing the positions at the _t_L.
            - v_last is a np.array of shape [batch_size, 2] and format [x, y] containing the velocities at the _t_L.
            - dt_pred is a np.array of shape [batch_size] with arrival time point estimates as difference times w.r.t.
                _t_L.
            - dy_pred is a np.array of shape [batch_size] with point estimates for the arrival location along the
                actuator array as difference w.r.t. x_L.

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param point_predictor: A callable, a function that returns an estimate for the arrival location.
        :param window_length: A float or np.array of shape [batch_size], the window length of the distribution.
        :param a: A float or np.array of shape [batch_size], the ratio of the window length, where the point prediction
            is located.
        :param name: String, the name for the distribution.
        """
        # sanity checks
        if not callable(point_predictor):
            raise ValueError('point_predictor must be a callable.')

        y_predicted = point_predictor(htd.x_L[..., [0, -2]],
                                      htd.x_L[..., [1, -1]],
                                      dt_pred=htd.ev - htd._t_L) + htd.x_L[..., -2]
        super().__init__(htd=htd,
                         S_w=0,  # always zero
                         point_prediction=y_predicted,
                         name=name,
                         window_length=window_length,
                         a=a)

    @AbstractCVHittingLocationDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in y-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in y-direction.
        """
        raise NotImplementedError('S_w for {} is always zero.'.format(self.__class__.__name__))


class BayesMixtureCVHittingLocationDistribution(AbstractBayesMixtureHittingLocationDistribution, AbstractCVHittingLocationDistribution):
    """Mathematically exact way to solve the problem of finding the distribution in y at the first-passage time. Sets up
    the joint distribution of the CV process in y and the approximation for the given first-passage time distribution
    and performs a marginalization over the latter.

    The integration is done using a Riemann-sum-like approach by summing Gaussian random variables that represent
    the densities in y at different times weighted by the first-passage time probability in a (small) range around
    these times.
    """
    def __init__(self, htd, S_w, t_min=None, t_max=None, n=100, name='Mixture Method'):
        """Initializes the distribution.

        State format:

            [..., pos_y, velo_y]

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in y-direction.
        :param t_min: A float or a np.array of shape [batch_size], the lower integration limit.
        :param t_max: A float or a np.array of shape [batch_size], the upper integration limit.
        :param n: An integer, the number of integration points to use.
        :param name: String, the name for the distribution.
        """
        super().__init__(htd=htd,
                         S_w=S_w,
                         t_min=t_min,
                         t_max=t_max,
                         n=n,
                         name=name,
                         )

    @AbstractCVHittingLocationDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in y-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in y-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size)
        # Force recalculating all privates, except of t_min and t_max
        self._weights = None
        self._locations = None
        self._scales = None


class BayesianCVHittingLocationDistribution(AbstractBayesianHittingLocationDistribution, AbstractCVHittingLocationDistribution):
    """ Mathematically exact way to solve the problem of finding the distribution in y at the first-passage time. Sets up
    the joint distribution of the CV process in y and the approximation for the given first-passage time distribution
    and performs a marginalization over the latter.

    The integration is done using scipy's integrate quad function.
    """
    def __init__(self, htd, S_w, t_min=None, t_max=None, name='Bayesian method'):
        """Initializes the distribution.

        State format:

            [..., pos_y, velo_y]

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A float or np.array of shape [batch_size], the power spectral density (PSD) in y-direction.
        :param t_min: A float or a np.array of shape [batch_size], the lower integration limit.
        :param t_max: A float or a np.array of shape [batch_size], the upper integration limit.
        :param name: String, the name for the distribution.
        """
        super().__init__(htd=htd,
                         S_w=S_w,
                         name=name,
                         t_min=t_min,
                         t_max=t_max,
                         )

    @AbstractCVHittingLocationDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in y-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float or np.array of shape [batch_size], the power spectral density in y-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size)


class MCCVHittingLocationDistribution(AbstractMCHittingLocationDistribution, AbstractCVHittingLocationDistribution):
    """Wraps the histogram derived by a Monte-Carlo approach to obtain the distribution in y at the first-passage time
    assuming a CV model using scipy.stats.rv_histogram.

    """
    def __init__(self, htd, S_w, y_range, bins=100, y_samples=None, name='MC simulation'):
        """Initializes the distribution.

        State format:

            [..., pos_y, velo_y]

        :param htd: An AbstractHittingTimeDistribution object, the used hitting time distribution.
        :param S_w: A float, the power spectral density (PSD) in y-direction.
        :param y_range: A list of length 2 representing the limits for the histogram of the distribution in y at the
            first-passage time histogram (the number of bins within y_range will correspond to bins).
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param y_samples: None or a np.array of shape [num_samples] containing the y-position at the first-passage
            times of the particles. If None, y_samples will be created by a call to a sampling method. If given, given
            values will be used.
        :param name: String, the name for the distribution.
        """
        # sanity checks
        if htd.batch_size != 1 or len(np.atleast_1d(S_w)) != 0:
            raise ValueError(
                'Batch size must be equal to 1. Note that {} does not support a batch dimension.'.format(
                    self.__class__.__name__))

        if y_samples is None:
            _, y_samples, _ = create_ty_cv_samples_hitting_time(htd.x_L,
                                                                htd.C_L,
                                                                S_w,
                                                                htd._x_predTo,
                                                                htd._t_L)

        super().__init__(htd=htd,
                         S_w=S_w,
                         name=name,
                         y_samples=y_samples,
                         y_range=y_range,
                         bins=bins,
                         )

    @AbstractCVHittingLocationDistribution.S_w.setter
    def S_w(self, value):
        """The setter of the power spectral density (PSD) S_w in y-direction. Depending on the distribution, S_w might
        be its hyperparameter and therefore we may want to adjust it after initializing.

        :param value: S_w: A float, the power spectral density in y-direction.
        """
        self._S_w = np.broadcast_to(value, shape=self.batch_size)
        # Resample und recalculate the distribution
        t_samples, _, _ = create_ty_cv_samples_hitting_time(self._htd.x_L,
                                                            self._htd.C_L,
                                                            self._S_w,
                                                            self._htd._x_predTo,
                                                            self._htd._t_L)
        self._density = self._build_distribution_from_samples(self._samples, self._range)

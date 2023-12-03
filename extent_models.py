from abc import ABC, abstractmethod
import numpy as np

from abstract_distributions import AbstractArrivalDistribution
from abstract_hitting_time_distributions import AbstractHittingTimeDistribution
from abstract_hitting_location_distributions import AbstractHittingLocationDistribution
from cv_arrival_distributions.cv_hitting_time_distributions import AbstractCVHittingTimeDistribution
from ca_arrival_distributions.ca_hitting_time_distributions import AbstractCAHittingTimeDistribution


class AbstractHittingTimeWithExtentsModel(ABC):
    """A base class for all models that calculate the temporal deflection windows based on particle extents.

    """
    def __init__(self, length, name='AbstractHittingTimeWithExtentsModel'):
        """Initializes the model.

        :param length: A float or np.array of shape [batch_size], the length (in transport direction) of the particles.
        :param name: String, the (default) name for the model.
        """
        self._length = np.atleast_1d(length)
        self.name = name

    @abstractmethod
    def calculate_confidence_bounds(self, q):
        """Calculates confidence bounds used as temporal deflection windows.

        :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the involved
            distributions.

        :returns:
            t_start: A float or np.array of shape [batch_size], the start time of the deflection window.
            t_end: A float or np.array of shape [batch_size], the terminal time of the deflection window.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')


class HittingTimeWithExtentsModel(AbstractHittingTimeWithExtentsModel):
    """A model that calculates the temporal deflection windows based on the marginal distributions of particle front
     and particle back arrival.

     """
    def __init__(self, length, htd_class, htd_kwargs, name="Extended hitting time model"):
        """Initializes the model.

        :param length: A float or np.array of shape [batch_size], the length (in transport direction) of the particles.
        :param htd_class: A subclass of AbstractHittingTimeDistribution.
        :param htd_kwargs: A dict of kwargs for the htd_class.
        :param name: String, the name for the model.
        """
        # sanity check
        if not issubclass(htd_class, AbstractHittingTimeDistribution):
            raise ValueError('htd_class must be a subclass of AbstractHittingTimeDistribution.')

        super().__init__(length=length,
                         name=name,
                         )

        # remove x_preTo from the kwargs (note thatx_predTo must be in the kwargs)
        htd_kwargs = htd_kwargs.copy()
        x_predTo = htd_kwargs.pop('x_predTo')

        self._front_arrival_distribution = htd_class(x_predTo=x_predTo - self._length / 2,
                                                     **htd_kwargs)
        self._back_arrival_distribution = htd_class(x_predTo=x_predTo + self._length / 2,
                                                    **htd_kwargs)

    @property
    def front_arrival_distribution(self):
        """The particle front arrival time distribution at the actuator array.

        :returns: A child instance of AbstractHittingTimeDistribution, the particle front arrival time distribution.
        """
        return self._front_arrival_distribution

    @property
    def back_arrival_distribution(self):
        """The particle back arrival time distribution at the actuator array.

         :returns: A child instance of AbstractHittingTimeDistribution, the particle back arrival time distribution.
         """
        return self._back_arrival_distribution

    def calculate_confidence_bounds(self, q):
        """Calculates confidence bounds used as temporal deflection windows.

         :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the marginal
             distributions.

         :returns:
             t_start: A float or np.array of shape [batch_size], the start time of the deflection window.
             t_end: A float or np.array of shape [batch_size], the terminal time of the deflection window.
         """
        q_front = (1 - q) / 2
        q_back = (1 + q) / 2

        t_start = self._front_arrival_distribution.ppf(q_front)
        t_end = self._back_arrival_distribution.ppf(q_back)
        return t_start, t_end


class HittingTimeWithExtentsSimplifiedModel(AbstractHittingTimeWithExtentsModel):
    """A model that calculates the temporal deflection windows based on the simplified assumption of constant velocity
    as the particle passes by.

     """
    def __init__(self, length, htd_class, htd_kwargs, name="Simplified Extended hitting time model"):
        """Initializes the model.

        :param length: A float or np.array of shape [batch_size], the length (in transport direction) of the particles.
        :param htd_class: A subclass of AbstractHittingTimeDistribution.
        :param htd_kwargs: A dict of kwargs for the htd_class.
        :param name: String, the name for the model.
        """
        # sanity check
        if not issubclass(htd_class, (AbstractCVHittingTimeDistribution, AbstractCAHittingTimeDistribution)):
            raise ValueError('htd_class must be a subclass of AbstractCVHittingTimeDistribution or '
                             'AbstractCAHittingTimeDistribution.')

        super().__init__(length=length,
                         name=name,
                         )

        self._arrival_model = htd_class(**htd_kwargs)

    def calculate_confidence_bounds(self, q):
        """Calculates confidence bounds used as temporal deflection windows.

        :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the involved
            distributions.

        :returns:
            t_start: A float or np.array of shape [batch_size], the start time of the deflection window.
            t_end: A float or np.array of shape [batch_size], the terminal time of the deflection window.
        """
        q_front = (1 - q) / 2
        q_back = (1 + q) / 2

        t_start = self._arrival_model.ppf(q_front) - self._length/(2 * self._arrival_model.x_L[..., 1])   # TODO: Ist das überhaupt so wie gedacht?
        t_end = self._arrival_model.ppf(q_back) + self._length/(2 * self._arrival_model.x_L[..., 1])
        return t_start, t_end


class HittingLocationWithExtentsModel:
    """A model that calculates the spatial deflection windows based on the marginal distributions of the uppermost and
     lowermost particle edge.

     """
    def __init__(self,
                 width,
                 htwe_model,
                 hld_class,
                 hld_kwargs,
                 name,
                 ):
        """Initializes the model.

        :param width: A float or np.array of shape [batch_size], the width (orthogonal to the transport direction) of
            the particles.
        :param htwe_model: A HittingTimeWithExtentsModel instance, the hitting time with extents model to use.
        :param hld_class: A subclass of AbstractHittingLocationDistribution.
        :param hld_kwargs: A dict of kwargs for the hld_class.
        :param name: String, the name for the model.
        """
        # sanity check
        if not isinstance(htwe_model, HittingTimeWithExtentsModel):
            raise ValueError('htwe_model must be a HittingTimeWithExtentsModel instance.')
        if not issubclass(hld_class, AbstractHittingLocationDistribution):
            raise ValueError('htd_class must be a subclass of AbstractHittingLocationDistribution.')

        self.name = name
        self._width = np.atleast_1d(width)

        # remove htd from the kwargs
        if 'htd' in hld_kwargs.keys():
            hld_kwargs = hld_kwargs.copy()
            del hld_kwargs['htd']

        self._front_location_model = hld_class(htd=htwe_model.front_arrival_distribution,
                                               **hld_kwargs)
        self._back_location_model = hld_class(htd=htwe_model.back_arrival_distribution,
                                              **hld_kwargs)

        self._min_y_model = MinYMonotonouslyMotionDDistribution(self._front_location_model, self._back_location_model, self._width)

        self._max_y_model = MaxYMonotonouslyMotionDistribution(self._front_location_model, self._back_location_model, self._width)

    @property
    def front_location_model(self):
        """The particle front arrival location distribution at the actuator array.

        :returns: A child instance of AbstractHittingLocationDistribution, the particle front arrival location
            distribution.
        """
        return self._front_location_model

    @property
    def back_location_model(self):
        """The particle back arrival location distribution at the actuator array.

        :returns: A child instance of AbstractHittingLocationDistribution, the particle back arrival location
            distribution.
        """
        return self._back_location_model

    @property
    def max_y_model(self):
        """The location distribution of the particles' uppermost edge at the actuator array.

        :returns: A MaxYMonotonouslyMotionDistribution object, the distribution of the particles' uppermost edge at the actuator array.
        """
        return self._max_y_model

    @property
    def min_y_model(self):
        """The location distribution of the particles' lowermost edge at the actuator array.

        :returns: A MaxYMonotonouslyMotionDistribution object, the distribution of the particles' lowermost edge at the actuator array.
        """
        return self._min_y_model

    def calculate_ejection_windows(self, q):
        """Calculates confidence bounds used as spatial deflection windows.

         :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the marginal
             distributions.

         :returns:
             y_start: A float or np.array of shape [batch_size], the lower boundary of the deflection window.
             y_end: A float or np.array of shape [batch_size], the upper boundary of the deflection window.
         """
        q_low = (1 - q) / 2
        q_up = (1 + q) / 2

        y_start = self._min_y_model.ppf(q_low)
        y_end = self._max_y_model.ppf(q_up)
        return y_start, y_end


class AbstractMinMaxYDistribution(AbstractArrivalDistribution):
    """A base class for distributions that model the maximum or minimum location value of a particle passing the
    actuator array.

    """
    def __init__(self, front_location_distribution, back_location_distribution, width,
                 name='AbstractMinMaxYDistribution'):
        """Initializes the distribution
        
        :param front_location_distribution: A child instance of AbstractHittingLocationDistribution, the distribution of
            the particle front arrival location.
        :param back_location_distribution: A child instance of AbstractHittingLocationDistribution, the distribution of
            the particle back arrival location.
        :param width: A float or np.array of shape [batch_size], the width (orthogonal to the transport direction) of
            the particles.
        :param name: String, the (default) name for the distribution.
        """
        # sanity checks
        if not isinstance(front_location_distribution, AbstractHittingLocationDistribution):
            raise ValueError('front_location_distribution must be a child of AbstractHittingLocationDistribution.')
        if not isinstance(back_location_distribution, AbstractHittingLocationDistribution):
            raise ValueError('back_location_distribution must be a child of AbstractHittingLocationDistribution.')

        self._front_location_model = front_location_distribution
        self._back_location_model = back_location_distribution
        self._width = np.atleast_1d(width)

        super().__init__(name=name)

    def batch_size(self):
        return self._front_location_model.batch_size

    @property
    def ev(self):
        raise NotImplementedError('Not supported for this model.')

    @property
    def var(self):
        raise NotImplementedError('Not supported for this model.')

    @property
    def third_central_moment(self):
        raise NotImplementedError('Not supported for this model.')

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        raise NotImplementedError('Not supported for this model.')

    def __setitem__(self, key, value):
        raise NotImplementedError('Not supported for this model.')

    @abstractmethod
    def front_location_cdf(self, y):
        """The cumulative distribution function (CDF) of particle front arrival location distribution at the actuator
        array.

        :param y: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the CDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def front_location_pdf(self, y):
        """The particle front arrival location distribution at the actuator array.

        :param y: A float, a np.array of shape [sample_size], a np.array of shape [batch_size], or a np.array of
            [batch_size, sample_size], the parameter of the distribution.

        :returns: A float or a np.array, the value of the PDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def back_location_cdf(self, y):
        """The cumulative distribution function (CDF) of particle back arrival location distribution at the actuator
        array.

        :returns: A float or a np.array, the value of the CDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def back_location_pdf(self, y):
        """The particle back arrival location distribution at the actuator array.

        :returns: A float or a np.array, the value of the PDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')


class MaxYMonotonouslyMotionDistribution(AbstractMinMaxYDistribution):
    """An approximation for the distributions of the maximum y-value of a particle passing the actuator array using the
    assumption of monotonously motion (either monotonously increasing or decreasing, but which is the case is a priori
    not known).

    Note that this model only yields an accurate estimation for low and high probability (CDF) values, but not for
    intermediate values.
    """
    def __init__(self, front_location_distribution, back_location_distribution, width,
                 name='MaxYMonotonouslyMotionDistribution'):
        """Initializes the distribution

        :param front_location_distribution: A child instance of AbstractHittingLocationDistribution, the distribution of
            the particle front arrival location.
        :param back_location_distribution: A child instance of AbstractHittingLocationDistribution, the distribution of
            the particle back arrival location.
        :param width: A float or np.array of shape [batch_size], the width (orthogonal to the transport direction) of
            the particles.
        :param name: String, the name for the distribution.
        """
        super().__init__(front_location_distribution=front_location_distribution,
                         back_location_distribution=back_location_distribution,
                         width=width,
                         name=name,
                         )

    def cdf(self, y):
        """The cumulative distribution function (CDF) of the location distribution of the particles' uppermost edge at
        the actuator array.

        :returns: A float or a np.array, the value of the CDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        return np.min(np.stack([self.back_location_cdf(y), self.front_location_cdf(y)]), axis=0)  # np.stack stacks
        # along new axis 0

    def pdf(self, y):
        """The location distribution of the particles' uppermost edge at the actuator array.

        :returns: A float or a np.array, the value of the DDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        return np.stack([self.back_location_pdf(y), self.front_location_pdf(y)])[
            np.argmin(np.stack([self.back_location_cdf(y), self.front_location_cdf(y)]), axis=0)]  # np.stack stacks
        # along new axis 0

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the location distribution of the particles' uppermost
        edge at the actuator array.

        :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the distribution.

        :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
        """
        back_location_value = self._back_location_model.ppf(q) + self._width / 2
        front_location_value = self._front_location_model.ppf(q) + self._width / 2
        return np.max(np.stack([back_location_value, front_location_value]), axis=0)  # np.stack stacks along new axis 0

    def back_location_cdf(self, y):
        return self._back_location_model.cdf(y - self._width / 2)

    def front_location_cdf(self, y):
        return self._front_location_model.cdf(y - self._width / 2)

    def back_location_pdf(self, y):
        return self._back_location_model.pdf(y - self._width / 2)

    def front_location_pdf(self, y):
        return self._front_location_model.pdf(y - self._width / 2)


class MinYMonotonouslyMotionDDistribution(AbstractMinMaxYDistribution):
    """An approximation for the distributions of the maximum y-value of a particle passing the actuator array using the
    assumption of monotonously motion (either monotonously increasing or decreasing, but which is the case is a priori
    not known).

    Note that this model only yields an accurate estimation for low and high probability (CDF) values, but not for
    intermediate values.
    """
    def __init__(self, front_location_distribution, back_location_distribution, width, name="MinYMonotonouslyMotionDDistribution"):
        """Initializes the distribution

        :param front_location_distribution: A child instance of AbstractHittingLocationDistribution, the distribution of
            the particle front arrival location.
        :param back_location_distribution: A child instance of AbstractHittingLocationDistribution, the distribution of
            the particle back arrival location.
        :param width: A float or np.array of shape [batch_size], the width (orthogonal to the transport direction) of
            the particles.
        :param name: String, the name for the distribution.
        """
        super().__init__(front_location_distribution=front_location_distribution,
                         back_location_distribution=back_location_distribution,
                         width=width,
                         name=name,
                         )

    def cdf(self, y):
        """The cumulative distribution function (CDF) of the location distribution of the particles' lowermost edge at
        the actuator array.

        :returns: A float or a np.array, the value of the CDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        return np.max(np.stack([self.back_location_cdf(y), self.front_location_cdf(y)]), axis=0)  # np.stack stacks
        # along new axis 0

    def pdf(self, y):
        """The location distribution of the particles' lowermost edge at the actuator array.

        :returns: A float or a np.array, the value of the DDF for y:
            - If the distribution is scalar (batch_size = 1)
                - and y is scalar, then returns a float,
                - and y is np.array of shape [sample_size], then returns a np.array of shape [sample_size].
            - If the distribution's batch_size is > 1 )
                - and y is scalar, then returns a np.array of shape [batch_size],
                - and y is a np.array of [batch_size, sample_size], then returns a np.array of shape
                    [batch_size, sample_size].
        """
        return np.stack([self.back_location_pdf(y), self.front_location_pdf(y)])[
            np.argmax(np.stack([self.back_location_cdf(y), self.front_location_cdf(y)]), axis=0)]  # np.stack stacks
        # along new axis 0

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the location distribution of the particles' lowermost
        edge at the actuator array.

        :param q: A float or np.array of shape [batch_size] in [0, 1], the confidence parameter of the distribution.

        :returns: A float or a np.array of shape [batch_size], the value of the PPF for q.
        """
        back_location_value = self._back_location_model.ppf(q) - self._width / 2
        front_location_value = self._front_location_model.ppf(q) - self._width / 2
        return np.min(np.stack([back_location_value, front_location_value]), axis=0)  # np.stack stacks along new axis 0

    def back_location_cdf(self, y):
        return self._back_location_model.cdf(y + self._width / 2)

    def front_location_cdf(self, y):
        return self._front_location_model.cdf(y + self._width / 2)

    def back_location_pdf(self, y):
        return self._back_location_model.pdf(y + self._width / 2)

    def front_location_pdf(self, y):
        return self._front_location_model.pdf(y + self._width / 2)

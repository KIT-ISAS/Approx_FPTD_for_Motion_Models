from abstract_distributions import AbstractUniformArrivalDistribution, AbstractNormalArrivalDistribution, AbstractGaussianMixtureArrivalDistribution


class UniformGenericArrivalTimeDistribution(AbstractUniformArrivalDistribution):
    """A simple approximation using a uniform distribution for the arrival time distribution. The distribution can be
    directly initialized with a given, predicted value for the time of arrival. Use e.g. in combination with neural
    network predictors.

    """
    def __init__(self, t_predicted, window_length, a=0.5, name='Uniform generic'):
        """Initializes the distribution.

        :param t_predicted: A float or a np.array of shape [batch_size], the predicted time of arrival at the actuator
            array.
        :param window_length: A float or np.array of shape [batch_size], the window length of the distribution.
        :param a: A float or np.array of shape [batch_size], the ratio of the window length, where the point prediction
            is located.
        :param name: String, the name for the distribution.
        """
        super().__init__(t_predicted, window_length, a, name)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._point_prediction *= time_scaling_factor
        self._window_length *= time_scaling_factor


class NormalGenericArrivalTimeDistribution(AbstractNormalArrivalDistribution):
    """A simple approximation using a Gaussian distribution for the arrival time distribution. The distribution can be
    directly initialized with a given, predicted value for the time of arrival and its variance. Use e.g. in combination
    with neural networks.

    """
    def __init__(self,  t_predicted, var_t_predicted, name='Normal generic'):
        """Initializes the distribution.

        :param t_predicted: A float or a np.array of shape [batch_size], the predicted time of arrival at the actuator
            array.
        :param var_t_predicted: A float or a np.array of shape [batch_size], the predicted variance of the time of
            arrival.
        :param name: String, the name for the distribution.
        """
        super().__init__(t_predicted, var_t_predicted, name)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._ev *= time_scaling_factor
        self._var *= time_scaling_factor**2


class GaussianMixtureGenericArrivalTimeDistribution(AbstractGaussianMixtureArrivalDistribution):
    """A more sophisticated approximation using a Gaussian mixture distribution for the arrival time distribution.
    The distribution can be directly initialized with given, predicted values for the time of arrival and their variance,
    as well as the mixture weights. Use e.g. in combination with a multiple-model approach.

    """
    def __init__(self, mus, sigmas, weights, name='GMM generic'):
        """Initializes the distribution.

        :param mus: A np.array of shape [component_size] or [batch_size, component_size], the means of the component
            Gaussian distributions.
        :param sigmas: A np.array of shape [component_size] or [batch_size, component_size], the standard deviations of
            the component Gaussian distributions.
        :param weights: A np.array of shape [component_size] or [batch_size, component_size], the weights
            (probabilities) for the component Gaussian distributions. Weights must be in [0, 1] and sum to 1.
        :param name: String, the name for the distribution.
        """
        super().__init__(mus, sigmas, weights, name)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._mus *= time_scaling_factor
        self._sigmas *= time_scaling_factor

        # reinitialize tfd
        self._distr = self._build_dist()

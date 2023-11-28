from abstract_distributions import AbstractUniformArrivalDistribution, AbstractNormalArrivalDistribution, AbstractGaussianMixtureArrivalDistribution


class UniformGenericArrivalLocationDistribution(AbstractUniformArrivalDistribution):
    """A simple model using a uniform distribution for the arrival location distribution. The distribution can be
    directly initialized with a given, predicted value for the y-position at the time of arrival. Use e.g. in
    combination with neural network predictors.

    """
    def __init__(self, y_predicted, window_length, a=0.5, name='Uniform generic'):
        """Initializes the distribution.

        :param y_predicted: A float or np.array of shape [batch_size], the predicted time of arrival at the actuator
            array, i.e., the predicted y-position at the first-passage time.
        :param name: String, the name for the distribution.
        """
        super().__init__(y_predicted, window_length, a, name)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._point_prediction *= length_scaling_factor
        self._window_length *= length_scaling_factor


class NormalGenericArrivalLocationDistribution(AbstractNormalArrivalDistribution):
    """A simple approximation using a Gaussian distribution for the arrival location distribution. The distribution can
    be directly initialized with a given, predicted value for the y-position at the time of arrival and its variance.
   Use in combination with neural networks.

    """
    def __init__(self,  y_predicted, var_y_predicted, name='Normal NN Spatial Ejection Model'):
        """Initializes the distribution.

        :param y_predicted: A float or np.array of shape [batch_size], the predicted time of arrival at the actuator
            array, i.e., the predicted y-position at the first-passage time.
        :param var_y_predicted: A float or a np.array of shape [batch_size], the predicted variance of the time of
            arrival, i.e., the variance of the predicted y-position at the first-passage time.
        :param name: String, the name for the distribution.
        """
        super().__init__(y_predicted, var_y_predicted, name)

    def scale_params(self, length_scaling_factor, time_scaling_factor):
        """Scales the parameters of the distribution according to the scaling factor.

        :param length_scaling_factor: Float, the scaling factor for lengths.
        :param time_scaling_factor: Float, the scaling factor for times.
        """
        self._ev *= length_scaling_factor
        self._var *= length_scaling_factor ** 2


class GaussianMixtureGenericArrivalLocationDistribution(AbstractGaussianMixtureArrivalDistribution):
    """A more sophisticated approximation using a Gaussian mixture distribution for the arrival location distribution.
      The distribution can be directly initialized with given, predicted values for the y-position at the time of
      arrival and their variance as well as the mixture weights. Use e.g. in combination with a multiple-model approach.

      """
    def __init__(self, mus, sigmas, weights, name='GaussianMixtureTemporalEjectionDistribution'):
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
        self._mus *= length_scaling_factor
        self._sigmas *= length_scaling_factor

        # reinitialize tfd
        self._distr = self._build_dist()

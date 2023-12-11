import numpy as np


class KalmanFilter:
    """The Kalman filter.

    This class can be used in two ways:

        i)  by initializing a new object with a system model, and an initial mean and covariance and subsequent state
                management by this class.
        ii) by calling the static update-method on states managed by another algorithm (a tracker, a controller, ...).

    Note that the second possibility allows for flexible operations on the states whereas the first one provides a
    simple interface for standard problems.
    """

    def __init__(self, system_model, init_state_mean, init_state_cov, init_k=0):
        """Initializes a Kalman filter with states managed by this class.

        :param system_model: A callable mapping (motion_state_mean, motion_state_cov, time_step) to (motion_state_mean, motion_state_cov), where
            motion_state_mean, motion_state_cov, time_step are of same type and format as init_state_mean, init_state_cov, init_k.
            The callable represents the system model to be used for filtering.
        :param init_state_mean: A np.array with shape [batch_size, state_length], the mean of the initial state.
        :param init_state_cov: A np.array with shape [batch_size, state_length, state_length], the covariance of the
            initial state.
        :param init_k: An integer, the initial time step.
        """
        if init_state_mean.shape[-1] != init_state_cov.shape[-1]:
            raise ValueError('The dimension of init_state_mean and init_state_cov must match.')
        if init_state_cov.shape[-1] != init_state_cov.shape[-2]:
            raise ValueError('init_state_cov must match must be a quadratic matrix.')
        if not isinstance(init_k, int):
            raise ValueError('The initial time step k must be an integer.')

        self._system_model = system_model
        self._state_mean = init_state_mean
        self._state_cov = init_state_cov
        self._k = init_k

    @property
    def time_step(self):
        """The current time step.

        :returns: An integer, the current time step.
        """
        return self._k

    @property
    def state_mean(self):
        """The state's mean.

        :returns: A np.array with shape [batch_size, state_length], the states mean.
        """
        return self._state_mean

    @property
    def state_cov(self):
        """The state's covariance.

        :returns: A np.array with shape [batch_size, state_length, state_length], the states covariance.
        """
        return self._state_cov

    def predict_own_state(self):
        """Calculates the transition p(x_{k+1} | x_k), where x_k represents the state at time step k.

        This function predicts the state managed by the KalmanFilter class.
        """
        self._state_mean, self._state_cov = self._system_model(self._state_mean, self._state_cov, self._k)
        self._k += 1

    def update_own_state(self, measurement, H, C_v):
        """Calculates the measurement update p(x_k | z_{init_k}, ..., x_k), where x_k represents the state at time step
        k and z_{init_k}, ..., x_k the sequence of measurements up to time step k.

        The Kalman filter assumes a measurement equation of the form

            z_k = H x_k + eps_k , with eps a zero-mean measurement noise with covariance C_v.

        This function updates the state managed by the KalmanFilter class.

        :param measurement: A np.array of shape [batch_size, measurement_length], the current measurement z_k.
        :param H: A np.array of shape [measurement_length, state_length], the measurement matrix of the
            linear (or linearized) measurement equation.
        :param: C_v: A np.array of shape [measurement_length, measurement_length], the measurement noise
            covariance.
        """
        self._state_cov, self._state_cov = self.update(measurement, self._state_mean, self._state_cov, H, C_v,
                                                       copy=False)

    @staticmethod
    def update(measurement, state_mean, state_cov, H, C_v, copy=True):
        """The Kalman filter update step, calculates the measurement update p(x_k | z_{init_k}, ..., z_k), where x_k
        represents the state at time step k and z_{init_k}, ..., x_k the sequence of measurements up to time step k.

        The Kalman filter assumes a measurement equation of the form

            z_k = H_k x_k + eps_k , with eps a zero-mean measurement noise with covariance C_v.

        :param measurement: A np.array of shape [batch_size, measurement_length], the current measurement z_k.
        :param state_mean: A np.array of shape [batch_size, state_length], the states mean.
        :param state_cov: A np.array of shape [batch_size, state_length, state_length], the state covariance.
        :param H: A np.array of shape [measurement_length, state_length], the measurement matrix of the
            linear (or linearized) measurement equation.
        :param: C_v: A np.array of shape [measurement_length, measurement_length], the measurement noise
            covariance.
        :param copy: A Boolean, whether to copy the input array (True) or use inplace replacement (False). Inplace
            replacement should be used with caution.

        :returns:
            motion_state_mean: A np.array of shape [batch_size, state_length], the states mean after the update step
            motion_state_cov: A np.array of shape [batch_size, state_length, state_length], the states covariance after
                the update step.
        """
        if copy:
            state_mean = state_mean.copy()
            state_cov = state_cov.copy()

        for i, (pos_measurement_p, state_mean_i, state_cov_i) in enumerate(
                zip(measurement, state_mean, state_cov)):  # TODO: Vectorize/parallelize the implementation
            # Calculate gain N = C_p * H^T * inv(C_v + H * C_p * H^T)
            M_inv = C_v + np.matmul(np.matmul(H, state_cov_i), H.T)
            K = np.matmul(np.matmul(state_cov_i, H.T), np.linalg.inv(M_inv))
            # Update state x_e = x_p + N * (y - H * x_p)
            state_mean[i, :] = state_mean_i + np.matmul(K, pos_measurement_p - np.matmul(H, state_mean_i))
            # Update covariance matrix C_e = C_p - N * H * C_p
            state_cov[i, :, :] = state_cov_i - np.matmul(np.matmul(K, H), state_cov_i)
        return state_mean, state_cov
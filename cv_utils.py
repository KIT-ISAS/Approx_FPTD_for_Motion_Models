import numpy as np

from sampler import create_lgssm_hitting_time_samples


def _get_system_matrices_from_parameters(dt, S_w):
    """Returns the transition matrix (F) and the noise covariance of the transition (Q) of the model.

    Both matrices can be used, e.g., to simulate the discrete-time counterpart of the model.

     Assumed CV state format:

        [pos_x, velo_x, pos_y, velo_y]

    :param dt: A float, time increment.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.

    :returns:
        F: A np.array of shape [4, 4], the transition matrix.
        Q: A np.array of shape [4, 4], the transition noise covariance matrix.
    """
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    Q = S_w * np.array([[pow(dt, 3) / 3, pow(dt, 2) / 2, 0, 0],
                        [pow(dt, 2) / 2, dt, 0, 0],
                        [0, 0, pow(dt, 3) / 3, pow(dt, 2) / 2],
                        [0, 0, pow(dt, 2) / 2, dt]])
    return F, Q


def create_ty_cv_samples_hitting_time(x_L,
                                      C_L,
                                      S_w,
                                      x_predTo,
                                      t_L=0.0,
                                      N=100000,
                                      dt=1 / 1000,
                                      break_after_n_time_steps=1000,
                                      break_min_time=None):
    """Monte Carlo approach to solve the first passage time problem. Propagates particles through the 2D discrete-time
    CV motion model and determines their first passage at x_predTo as well as the location in y at the first passage by
    interpolating the positions between the last time before and the first time after the boundary.

    Note that particles that do not reach the boundary after break_after_n_time_steps time_steps are handled with a
    fallback value of max(t_samples) + 1 in the t_samples and np.nan in the y_samples.

    :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, vel_x, pos_y, vel_y].
    :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, position of the boundary.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_after_n_time_steps: Integer, maximum number of time steps for the simulation.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        t_samples: A np.array of shape [N] containing the first passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first passage times of the particles.
        fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
    """
    F, Q = _get_system_matrices_from_parameters(dt, S_w)

    time_before_arrival, x_before_arrival, x_after_arrival, x_term, fraction_of_returns = create_lgssm_hitting_time_samples(
        F,
        Q,
        x_L,
        C_L,
        x_predTo,
        t_L=t_L,
        N=N,
        dt=dt,
        break_after_n_time_steps=break_after_n_time_steps,
        break_min_time=break_min_time)

    # Linear interpolation to get time
    v_interpolated = (x_after_arrival[x_term, 1] - x_before_arrival[x_term, 1]) / (
            x_after_arrival[x_term, 0] - x_before_arrival[x_term, 0]) * (x_predTo - x_before_arrival[x_term, 0]) + \
                     x_before_arrival[x_term, 1]
    last_t = (x_predTo - x_before_arrival[x_term, 0]) / v_interpolated
    t_samples = time_before_arrival
    t_samples[x_term] = time_before_arrival[x_term] + last_t
    t_samples[np.logical_not(x_term)] = int(
        max(t_samples)) + 1  # default value for particles that do not arrive

    y_samples = x_before_arrival[:, 2]
    y_samples[x_term] = x_before_arrival[x_term, 2] + last_t * x_before_arrival[x_term, 3]
    y_samples[np.logical_not(x_term)] = np.nan  # default value for particles that do not arrive

    return t_samples, y_samples, fraction_of_returns
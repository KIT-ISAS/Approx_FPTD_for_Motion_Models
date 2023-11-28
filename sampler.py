"""Sampling functions used by the MC simulation and sample plot methods.

"""

from absl import logging

from timeit import time

import numpy as np
import tensorflow as tf


def create_hitting_time_samples(initial_samples,
                                compute_x_next_func,
                                x_predTo,
                                t_L=0.0,
                                N=100000,
                                dt=1 / 1000,
                                break_after_n_time_steps=1000,
                                break_min_time=None):
    """Monte Carlo approach to solve the first-passage time problem. Propagates particles through the motion model and
    determines time before arrival, and positions before and after the arrival as well as more statistics.

    :param initial_samples: A np.array of shape [num_samples, state_size], the initial particles samples at time step
        t_L.
    :param compute_x_next_func: A function propagating the samples to the next time step, i.e., the one time step
        transition function.
    :param x_predTo: A float, position of the boundary.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_after_n_time_steps: Integer, maximum number of time steps for the simulation.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least.
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        t_samples: A np.array of shape [N] containing the first-passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first-passage times of the particles.
        fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
    """
    start_time = time.time()
    # Let the samples move to the boundary
    
    x_curr = initial_samples
    x_term = np.zeros(initial_samples.shape[0], dtype=bool)
    t = t_L
    ind = 0
    time_before_arrival = np.full(N, t_L, dtype=np.float64)
    x_before_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_before_arrival[:] = np.nan
    x_after_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_after_arrival[:] = np.nan
    fraction_of_returns = []
    while True:
        if ind % 100 == 0:
            logging.info('Timestep {0}, x equals approx. {1}'.format(ind, x_curr[0, 0]))
        fraction_of_returns.append(np.sum(np.logical_and(x_curr[:, 0] < x_predTo, x_term)) / N)
        
        x_next = compute_x_next_func(x_curr)

        first_passage = np.logical_and(np.logical_not(x_term), x_next[:, 0] >= x_predTo)
        x_term[first_passage] = True
        x_before_arrival[first_passage] = x_curr[first_passage]
        x_after_arrival[first_passage] = x_next[first_passage]
        if break_min_time is None:
            if np.all(x_term):
                break
        else:
            if t >= break_min_time and np.all(x_term):
                break
        if break_after_n_time_steps is not None and ind >= break_after_n_time_steps:
            logging.info(
                'Warning: Sampling interrupted because {}. reached. Please adjust break_after_n_time_steps if you want to move the particles more timesteps.'.format(
                    break_after_n_time_steps))
            break
        x_curr = x_next
        t += dt   # TODO: Stimmt die Reihenfolge hier? müsste man nicht erst nach der nächsten Zeile +dt rechnen?
        time_before_arrival[np.logical_not(x_term)] = t
        ind += 1

    logging.info('MC time: {0}ms'.format(round(1000 * (time.time() - start_time))))
    return time_before_arrival, x_before_arrival, x_after_arrival, x_term, np.array(fraction_of_returns)


def create_lgssm_hitting_time_samples(F,
                                      Q,
                                      x_L,
                                      C_L,
                                      x_predTo,
                                      t_L=0.0,
                                      N=100000,
                                      dt=1 / 1000,
                                      break_after_n_time_steps=1000,
                                      break_min_time=None):
    """Monte Carlo approach to solve the first-passage time problem. Propagates particles through the discrete-time
    LGSSM motion model and determines their first-passage at x_predTo as well as the location in y at the first-passage 
    by interpolating the positions between the last time before and the first time after the boundary.

    :param F: A np.array of shape [4, 4], the transition matrix of the LGSSM.
    :param Q: A np.array of shape [4, 4], the transition noise covariance matrix of the LGSSM.
    :param x_L: A np.array of shape [length_state] representing the expected value of the initial state. We use index L 
        here because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, ..., pos_y, ...].
    :param C_L: A np.array of shape [length_state, length_state] representing the covariance matrix of the initial state.
    :param x_predTo: A float, position of the boundary.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least.
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        t_samples: A np.array of shape [N] containing the first-passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first-passage times of the particles.
        fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
    """
    initial_samples = np.random.multivariate_normal(mean=x_L, cov=C_L, size=N)
    
    mean_w = np.zeros(x_L.shape[0])

    def compute_x_next_func(x_curr):
        x_curr_tf = tf.convert_to_tensor(x_curr)
        x_next = tf.linalg.matvec(F, x_curr_tf).numpy()
        w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=N)
        x_next = x_next + w_k
        return x_next

    time_before_arrival, x_before_arrival, x_after_arrival, x_term, fraction_of_returns = create_hitting_time_samples(
        initial_samples,
        compute_x_next_func,
        x_predTo,
        t_L=t_L,
        N=N,
        dt=dt,
        break_after_n_time_steps=break_after_n_time_steps,
        break_min_time=break_min_time)

    return time_before_arrival, x_before_arrival, x_after_arrival, x_term, fraction_of_returns


def get_example_tracks_lgssm(x_L, C_L, S_w, get_system_matrices_from_parameters_func):
    """Generator that creates a function for simulation of example tracks of LGSSMs. Used for plotting purpose only.

    :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, velo_x, pos_y, velo_y].
    :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param get_system_matrices_from_parameters_func: A function that returns the system matrices of the LGSSM.
        Signature: f(dt, S_w), with dt a float being the time increment.

    :returns:
        get_example_tracks: A function that can be used for simulation of example tracks.
    """

    def get_example_tracks(plot_t, N=5):
        """Create data (only x-positions) of some tracks.

        :param plot_t: A np.array of shape [n_plot_points], point in time, when a point in the plot should be displayed.
            Consecutive points must have the same distance.
        :param N: Integer, number of tracks to create.

        :returns:
            x_tracks: A np.array of shape [num_time_steps, N] containing the x-positions of the tracks.  # TODO: Anpassen, sodass es auch mit den y-Positionen geht!
        """
        dt = plot_t[1] - plot_t[0]
        F, Q = get_system_matrices_from_parameters_func(dt, S_w)

        initial_samples = np.random.multivariate_normal(mean=x_L, cov=C_L, size=N)  # [length_state, N]
        mean_w = np.zeros(initial_samples.shape[1])

        # Let the samples move to the boundary
        tracks = np.expand_dims(initial_samples, axis=2)
        for _ in range(plot_t.size - 1):
            x_curr_tf = tf.convert_to_tensor(tracks[:, :, -1])
            x_next = tf.linalg.matvec(F, x_curr_tf).numpy()
            w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=N)
            x_next = np.expand_dims(x_next + w_k, axis=-1)
            tracks = np.concatenate((tracks, x_next), axis=-1)

        x_tracks = tracks[:, 0, :].T  # [N, length_state, num_time_steps] -> [num_time_steps, N]

        return x_tracks

    return get_example_tracks

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
                                length=None,
                                y_pos_ind=None,
                                N=100000,
                                dt=1 / 1000,
                                break_after_n_time_steps=1000,
                                break_min_time=None,
                                ):
    """Monte Carlo approach to solve the first-passage time problem. Propagates particles through the motion model and
    determines time before arrival, and positions before and after the arrival as well as more statistics.

     State format:

        [x_pos, ..., y_pos at index y_pos_ind, ... ]

    :param initial_samples: A np.array of shape [num_samples, state_size], the initial particles samples at time step
        t_L.
    :param compute_x_next_func: A function propagating the samples to the next time step, i.e., the one time step
        transition function.
    :param x_predTo: A float, the position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
    :param length: None or a float, the length (in transport direction) of the particle. If None, no
        extent_passage_statistics will be calculated.
    :param y_pos_ind: An integer, the index where the position component in y-direction is located in the state-vector.
        Must be provided if length is not None.
    :param N: Integer, number of samples to use.
    :param dt: A float, the time increment.
    :param break_after_n_time_steps: Integer, maximum number of time steps for the simulation.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least.
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        first_passage_statistics: A tuple containing
            - time_before_arrival: A np.array of shape [num_samples] containing the time of the last time step before
                the first-passage.
            - x_before_arrival: A np.array of shape [num_samples] containing the state of the last time step before
                the first-passage.
            - x_after_arrival: A np.array of shape [num_samples] containing the state of the first time step after
                the first-passage.
            - x_term: A Boolean np.array of shape [num_samples] indicating whether the particle has crossed the boundary
                or not.
            - fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
                tracks that have previously reached the boundary, but then fall below the boundary until the respective
                time step.
        extent_passage_statistics: None or a tuple containing
            - time_before_first_front_arrival: A np.array of shape [num_samples] containing the time of the last time
                step before the particle fronts pass the boundary for the first time.
            - time_before_first_back_not_in: A np.array of shape [num_samples] containing the time of the last time
                step before the particle backs pass the boundary for the first time.
            - x_before_front_arrival: A np.array of shape [num_samples] containing the state of the last time step before
                the particle fronts pass the boundary for the first time.
            - x_after_front_arrival: A np.array of shape [num_samples] containing the state of the first time step after
                the particle fronts pass the boundary for the first time.
            - x_before_first_back_not_in: A np.array of shape [num_samples] containing the state of the last time step
                before the particle backs pass the boundary for the first time.
            - x_after_first_back_not_in: A np.array of shape [num_samples] containing the state of the first time step
                after the particle backs pass the boundary for the first time.
            - x_term_first_front_arrival: A Boolean np.array of shape [num_samples] indicating whether the particle
                front has crossed the boundary or not.
            - x_term_first_back_not_in: A Boolean np.array of shape [num_samples] indicating whether the particle
                back has crossed the boundary or not.
            - y_min: A np.array of shape [num_samples] containing the minimum y-position of all time steps when the
                particle passes the boundary.
            - y_max: A np.array of shape [num_samples] containing the maximum y-position of all time steps when the
                particle passes the boundary.
    """
    # sanity check
    if length is not None and y_pos_ind is None:
        raise ValueError("If length is given, also y_pos_ind must be provided.")

    start_time = time.time()
    # Let the samples move to the boundary

    t = t_L
    ind = 0
    x_curr = initial_samples

    fraction_of_returns = []

    x_term = np.zeros(initial_samples.shape[0], dtype=bool)
    time_before_arrival = np.full(N, t_L, dtype=np.float64)
    x_before_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_before_arrival[:] = np.nan
    x_after_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_after_arrival[:] = np.nan

    if length is not None:

        x_term_first_front_arrival = np.zeros(initial_samples.shape[0], dtype=bool)
        x_term_first_back_not_in = np.zeros(initial_samples.shape[0], dtype=bool)
        time_before_first_front_arrival = np.full(N, t_L, dtype=np.float64)
        time_before_first_back_not_in = np.full(N, t_L, dtype=np.float64)

        x_before_front_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
        x_before_front_arrival[:] = np.nan
        x_before_first_back_not_in = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
        x_before_first_back_not_in[:] = np.nan

        x_after_front_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
        x_after_front_arrival[:] = np.nan
        x_after_first_back_not_in = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
        x_after_first_back_not_in[:] = np.nan

        y_min = np.empty((initial_samples.shape[0]))
        y_min[:] = np.inf
        y_max = np.empty((initial_samples.shape[0]))
        y_max[:] = - np.inf

    while True:

        if ind % 100 == 0:
            logging.info('Timestep {0}, x equals approx. {1}'.format(ind, x_curr[0, 0]))

        fraction_of_returns.append(np.sum(np.logical_and(x_curr[:, 0] < x_predTo, x_term)) / N)

        x_next = compute_x_next_func(x_curr)

        first_passage = np.logical_and(np.logical_not(x_term), x_next[:, 0] >= x_predTo)
        x_term[first_passage] = True
        x_before_arrival[first_passage] = x_curr[first_passage]
        x_after_arrival[first_passage] = x_next[first_passage]

        if length is not None:
            first_front_arrival = np.logical_and(np.logical_not(x_term_first_front_arrival),
                                                 x_next[:, 0] + length / 2 >= x_predTo)
            first_back_not_in = np.logical_and(np.logical_not(x_term_first_back_not_in),
                                               x_next[:, 0] - length / 2 > x_predTo)
            x_term_first_front_arrival[first_front_arrival] = True
            x_term_first_back_not_in[first_back_not_in] = True

            x_before_front_arrival[first_front_arrival] = x_curr[first_front_arrival]
            x_after_front_arrival[first_front_arrival] = x_next[first_front_arrival]
            x_before_first_back_not_in[first_back_not_in] = x_curr[first_back_not_in]
            x_after_first_back_not_in[first_back_not_in] = x_next[first_back_not_in]

            particle_passes_mask = np.logical_and(x_term_first_front_arrival, np.logical_not(x_term_first_back_not_in))
            y_min[np.logical_and(particle_passes_mask, x_next[:, y_pos_ind] < y_min)] = x_next[
                np.logical_and(particle_passes_mask, x_next[:, y_pos_ind] < y_min), y_pos_ind]
            y_max[np.logical_and(particle_passes_mask, x_next[:, y_pos_ind] > y_max)] = x_next[
                np.logical_and(particle_passes_mask, x_next[:, y_pos_ind] > y_max), y_pos_ind]

        if (length is None and np.all(x_term)) or (length is not None and np.all(x_term_first_back_not_in)):
            if break_min_time is None:
                break
            elif t >= break_min_time:
                break
        if break_after_n_time_steps is not None and ind >= break_after_n_time_steps:
            logging.info(
                'Warning: Sampling interrupted because {}. reached. Please adjust break_after_n_time_steps if you want to move the particles more timesteps.'.format(
                    break_after_n_time_steps))
            break

        x_curr = x_next
        t += dt

        time_before_arrival[np.logical_not(x_term)] = t  # it's the updated t (i.e., t + dt) since x_term is based on
        # x_next
        if length is not None:
            time_before_first_front_arrival[np.logical_not(x_term_first_front_arrival)] = t
            time_before_first_back_not_in[np.logical_not(x_term_first_back_not_in)] = t

        ind += 1

    logging.info('MC time: {0}ms'.format(round(1000 * (time.time() - start_time))))

    first_passage_statistics = (
        time_before_arrival, x_before_arrival, x_after_arrival, x_term, np.array(fraction_of_returns))

    if length is not None:
        first_arrival_interval_statistics = (
            time_before_first_front_arrival, time_before_first_back_not_in, x_before_front_arrival, x_after_front_arrival,
            x_before_first_back_not_in, x_after_first_back_not_in, x_term_first_front_arrival, x_term_first_back_not_in,
            y_min, y_max)
    else:
        first_arrival_interval_statistics = None

    return first_passage_statistics, first_arrival_interval_statistics


def create_lgssm_hitting_time_samples(F,
                                      Q,
                                      x_L,
                                      C_L,
                                      t_L,
                                      x_predTo,
                                      calculate_intersection_delta_time_fn,
                                      calculate_delta_y,
                                      y_pos_ind,
                                      u=None,
                                      length=None,
                                      N=100000,
                                      dt=1 / 1000,
                                      break_after_n_time_steps=1000,
                                      break_min_time=None):
    """Monte Carlo approach to solve the first-passage time problem. Propagates particles through the discrete-time
    LGSSM motion model and determines their first-passage at x_predTo as well as the location in y at the first-passage
    by interpolating the positions between the last time before and the first time after the boundary.

     Format calculate_intersection_delta_time_fn:

        (x_before_arrival, x_after_arrival, x_predTo, x_term)  --> delta_t

         where
            - x_before_arrival is a np.array of shape [num_samples] containing the state of the last time step before
                the first-passage,
            -  x_after_arrival is a np.array of shape [num_samples] containing the state of the first time step after
                the first-passage,
            - x_term is A Boolean np.array of shape [num_samples] indicating whether the particle has crossed the
                boundary or not.

     Format calculate_delta_y:

        (x_before_arrival, x_term, delta_t)  --> delta_y

        where
            - delta_y is a float, the position in y as delta w.r.t. the position of x_before_arrival.

    Note that particles that do not reach the boundary after break_after_n_time_steps time_steps are handled with a
    fallback value of max(t_samples) + 1 in the t_samples and np.nan in the y_samples and all other samples.

    :param F: A np.array of shape [4, 4], the transition matrix of the LGSSM.
    :param Q: A np.array of shape [4, 4], the transition noise covariance matrix of the LGSSM.
    :param x_L: A np.array of shape [length_state] representing the expected value of the initial state. We use index L 
        here because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, ..., pos_y, ...].
    :param C_L: A np.array of shape [length_state, length_state] representing the covariance matrix of the initial state.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param x_predTo: A float, the position of the boundary.
    :param calculate_intersection_delta_time_fn: A callable, a function that returns the intersection time with x_predTo
        as delta w.r.t. the time of the last time step.
    :param calculate_delta_y: A callable, a function that returns the position in y as delta w.r.t. the position of
        the last state.
    :param y_pos_ind: An integer, the index where the position component in y-direction is located in the state-vector.
    :param u: None or a np.array of shape [length_state], the input.
    :param length: None or a float, the length (in transport direction) of the particle. If None, no
        extent_passage_statistics will be calculated.
    :param N: Integer, number of samples to use.
    :param dt: A float, the time increment.
    :param break_after_n_time_steps: Integer, maximum number of time steps for the simulation.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least.
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        first_passage_statistics: A tuple containing
            - t_samples: A np.array of shape [num_samples] containing the first-passage times of the particles.
            - y_samples: A np.array of shape [num_samples] containing the y-position at the first-passage times of the
                particles.
            - fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
                tracks that have previously reached the boundary, but then fall below the boundary until the respective
                time step.
        extent_passage_statistics: None or a tuple containing
            - t_samples_first_front_arrival: A np.array of shape [num_samples] containing the first-passage times of the
                particle fronts.
            - t_samples_first_back_arrival: A np.array of shape [num_samples] containing the first-passage times of the
                particle backs.
            - y_min_samples: A np.array of shape [num_samples] containing the minimum y-position in the time interval
                when the particle passed the boundary.
            - y_max_samples: A np.array of shape [num_samples] containing the maximum y-position in the time interval
                when the particle passed the boundary.
            - y_samples_first_front_arrival: A np.array of shape [num_samples] containing the y-position at the
                first-passage times of the particle fronts.
            - y_samples_first_front_arrival: A np.array of shape [num_samples] containing the y-position at the
                first-passage times of the particle backs.
    """
    # sanity check
    if not callable(calculate_intersection_delta_time_fn):
        raise ValueError("calculate_intersection_delta_time_fn must be callable.")
    if not callable(calculate_delta_y):
        raise ValueError("calculate_delta_y must be callable.")

    initial_samples = np.random.multivariate_normal(mean=x_L, cov=C_L, size=N)

    u = np.zeros_like(x_L) if u is None else u
    mean_w = np.zeros(x_L.shape[0])

    def compute_x_next_func(x_curr):
        x_curr_tf = tf.convert_to_tensor(x_curr)
        x_next = tf.linalg.matvec(F, x_curr_tf).numpy() + u
        w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=N)
        x_next = x_next + w_k
        return x_next

    first_passage_statistics, first_arrival_interval_statistics = create_hitting_time_samples(
        initial_samples,
        compute_x_next_func,
        x_predTo,
        t_L=t_L,
        length=length,
        y_pos_ind=y_pos_ind,
        N=N,
        dt=dt,
        break_after_n_time_steps=break_after_n_time_steps,
        break_min_time=break_min_time)

    # first for the first-passage statistics
    time_before_arrival, x_before_arrival, x_after_arrival, x_term, fraction_of_returns = first_passage_statistics

    # first arrival time
    t_samples = time_before_arrival
    delta_t_first_arrival = calculate_intersection_delta_time_fn(x_before_arrival=x_before_arrival,
                                                                 x_after_arrival=x_after_arrival,
                                                                 x_predTo=x_predTo,
                                                                 x_term=x_term)
    t_samples[x_term] += delta_t_first_arrival
    t_samples[np.logical_not(x_term)] = int(
        max(t_samples)) + 1  # default value for particles that do not arrive

    # first arrival location
    y_samples = x_before_arrival[:, y_pos_ind]
    y_samples[x_term] += calculate_delta_y(x_before_arrival=x_before_arrival,
                                           x_term=x_term,
                                           delta_t=delta_t_first_arrival)
    y_samples[np.logical_not(x_term)] = np.nan  # default value for particles that do not arrive

    first_passage_statistics = t_samples, y_samples, fraction_of_returns

    # then for the first-interval statistics
    if length is not None:

        time_before_first_front_arrival, time_before_first_back_not_in, x_before_front_arrival, x_after_front_arrival, x_before_first_back_not_in, x_after_first_back_not_in, x_term_first_front_arrival, x_term_first_back_not_in, y_min_samples, y_max_samples = first_arrival_interval_statistics

        # front arrival time
        t_samples_first_front_arrival = time_before_first_front_arrival
        delta_t_first_front_arrival = calculate_intersection_delta_time_fn(x_before_arrival=x_before_front_arrival,
                                                                           x_after_arrival=x_after_front_arrival,
                                                                           x_predTo=x_predTo - length / 2,
                                                                           x_term=x_term_first_front_arrival)
        t_samples_first_front_arrival[x_term_first_front_arrival] += delta_t_first_front_arrival
        # default value for particles that do not arrive
        t_samples_first_front_arrival[np.logical_not(x_term_first_front_arrival)] = int(
            max(t_samples_first_front_arrival)) + 1

        # back arrival time
        t_samples_first_back_arrival = time_before_first_back_not_in
        delta_t_first_back_arrival = calculate_intersection_delta_time_fn(x_before_arrival=x_before_first_back_not_in,
                                                                          x_after_arrival=x_after_first_back_not_in,
                                                                          x_predTo=x_predTo + length / 2,
                                                                          x_term=x_term_first_back_not_in)
        t_samples_first_back_arrival[x_term_first_back_not_in] += delta_t_first_back_arrival
        # default value for particles that do not arrive
        t_samples_first_back_arrival[np.logical_not(x_term_first_back_not_in)] = int(
            max(t_samples_first_back_arrival)) + 1

        # front arrival location
        y_samples_first_front_arrival = x_before_front_arrival[:, y_pos_ind]
        y_samples_first_front_arrival[x_term_first_front_arrival] += calculate_delta_y(
            x_before_arrival=x_before_front_arrival,
            x_term=x_term_first_front_arrival,
            delta_t=delta_t_first_front_arrival)
        # default value for particles that do not arrive
        y_samples_first_front_arrival[np.logical_not(x_term_first_front_arrival)] = np.nan

        # back arrival location
        y_samples_first_back_arrival = x_before_first_back_not_in[:, y_pos_ind]
        y_samples_first_back_arrival[x_term_first_back_not_in] += calculate_delta_y(
            x_before_arrival=x_before_first_back_not_in,
            x_term=x_term_first_back_not_in,
            delta_t=delta_t_first_back_arrival)
        # default value for particles that do not arrive
        y_samples_first_back_arrival[np.logical_not(x_term_first_back_not_in)] = np.nan

        # min and max
        # default value for particles that do not arrive
        y_min_samples[np.logical_not(x_term_first_front_arrival)] = np.nan
        y_max_samples[np.logical_not(x_term_first_front_arrival)] = np.nan

        y_min_samples[y_samples_first_back_arrival < y_min_samples] = y_samples_first_back_arrival[
            y_samples_first_back_arrival < y_min_samples]
        y_max_samples[y_samples_first_back_arrival > y_min_samples] = y_samples_first_back_arrival[
            y_samples_first_back_arrival > y_min_samples]

        y_min_samples[y_samples_first_front_arrival < y_min_samples] = y_samples_first_front_arrival[
            y_samples_first_front_arrival < y_min_samples]
        y_max_samples[y_samples_first_front_arrival > y_min_samples] = y_samples_first_front_arrival[
            y_samples_first_front_arrival > y_min_samples]

        # build the tuple
        first_arrival_interval_statistics = (
            t_samples_first_front_arrival, t_samples_first_back_arrival, y_min_samples, y_max_samples,
            y_samples_first_front_arrival, y_samples_first_back_arrival)
    else:
        first_arrival_interval_statistics = None

    return first_passage_statistics, first_arrival_interval_statistics


def get_example_tracks_lgssm(x_L, C_L, S_w, get_system_matrices_from_parameters_func, return_component_ind=0, u=None):
    """Generator that creates a function for simulation of example tracks of LGSSMs. Used for plotting purpose only.

    :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, velo_x, pos_y, velo_y].
    :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param get_system_matrices_from_parameters_func: A function that returns the system matrices of the LGSSM.
        Signature: f(dt, S_w), with dt a float being the time increment.
    :param return_component_ind: An integer, the index where the state component to be returned is located in the
        state-vector.
    :param u: None or a np.array of shape [length_state], the input.

    :returns:
        _get_example_tracks: A function that can be used for simulation of example tracks.
    """
    u = np.zeros_like(x_L) if u is None else u

    def get_example_tracks(plot_t, N=5):
        """Create data (only x-positions) of some tracks.

        :param plot_t: A np.array of shape [n_plot_points], point in time, when a point in the plot should be displayed.
            Consecutive points must have the same distance.
        :param N: Integer, the number of tracks to create.

        :returns:
            x_tracks: A np.array of shape [num_time_steps, N] containing the x-positions of the tracks.
        """
        dt = plot_t[1] - plot_t[0]
        F, Q = get_system_matrices_from_parameters_func(dt, S_w)
        F = np.block([[F, np.zeros_like(F)], [np.zeros_like(F), F]])
        Q = np.block([[Q, np.zeros_like(Q)], [np.zeros_like(Q), Q]])

        initial_samples = np.random.multivariate_normal(mean=x_L, cov=C_L, size=N)  # [length_state, N]
        mean_w = np.zeros(initial_samples.shape[1])

        # Let the samples move to the boundary
        tracks = np.expand_dims(initial_samples, axis=2)
        for _ in range(plot_t.size - 1):
            x_curr_tf = tf.convert_to_tensor(tracks[:, :, -1])
            x_next = tf.linalg.matvec(F, x_curr_tf).numpy() + u
            w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=N)
            x_next = np.expand_dims(x_next + w_k, axis=-1)
            tracks = np.concatenate((tracks, x_next), axis=-1)

        component_tracks = tracks[:, return_component_ind,
                           :].T  # [N, length_state, num_time_steps] -> [num_time_steps, N]

        return component_tracks

    return get_example_tracks

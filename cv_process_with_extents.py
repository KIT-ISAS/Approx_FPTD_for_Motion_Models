from absl import logging
from absl import app
from absl import flags

from abc import ABC, abstractmethod
from timeit import time

import numpy as np
import tensorflow as tf


from cv_process import TaylorHittingTimeModel, EngineeringApproxHittingTimeModel
from sampler import get_example_tracks_lgssm as get_example_tracks
from cv_utils import _get_system_matrices_from_parameters, create_ty_cv_samples_hitting_time
from cv_hitting_location_model import CVTaylorHittingLocationModel, SimpleGaussCVHittingLocationModel, ProjectionCVHittingLocationModel, MCCVHittingLocationModel

from evaluators.hitting_time_evaluator_with_extents import HittingTimeEvaluatorWithExtents, HittingLocationEvaluatorWithExtents


# Delete all FLAGS defined by CV process as we here not want them to be overwritten by the following flags.
for name in list(flags.FLAGS):
    if name in ['load_samples', 'save_samples', 'save_path', 'save_results', 'result_dir', 'no_show', 'for_paper',
                'measure_computational_times', 'verbosity_level']:
        delattr(flags.FLAGS, name)


flags.DEFINE_bool('load_samples', default=False,
                    help='Whether the samples should be loaded from a .npz  file.')
flags.DEFINE_bool('save_samples', default=False,
                    help='Whether the samples should be saved to a .npz  file.')
flags.DEFINE_string('save_path', default='/mnt/cv_with_system_noise_samples.npz',
                    help='The path to save the .npz  file.')
flags.DEFINE_bool('save_results', default=False,
                    help='Whether to save the results.')
flags.DEFINE_string('result_dir', default='/mnt/results/',
                    help='The directory where to save the results.')
flags.DEFINE_bool('no_show', default=False,
                  help='Set this to True if you do not want to show evaluation graphics and only save them.')
flags.DEFINE_bool('for_paper', default=False,
                  help='Boolean, whether to use the plots for publication (omit headers, etc.)..')
flags.DEFINE_bool('measure_computational_times', default=False,
                    help='Whether to measure the computational times.')

flags.DEFINE_string('verbosity_level', default='INFO', help='Verbosity options.')
flags.register_validator('verbosity_level',
                         lambda value: value in ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                         message='dataset_type must one of ' + str(['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']))

FLAGS = flags.FLAGS


def main(args):
    del args

    # Setup logging
    logging.set_verbosity(logging.FLAGS.verbosity_level)

    # Define system parameters
    # System noise
    S_w = 10
    # Covariance matrix at last timestep
    C_L = np.array([[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]])
    # Mean position at last timestep
    x_L = np.array([0.3, 6.2, 0.5, 0.0])
    # length and width of the particle
    particle_size = [0.1, 0.1]  # TODO: Die sind halt recht klein! Ist das nicht in m?

    # Boundary position
    x_predTo = 0.6458623971412047
    # Last time step
    t_L = 0  # In principle, we could assume w.l.o.g. that t_L = 0 (t_L is just a location argument).

    # Run the experiment
    run_experiment(x_L, C_L, t_L, S_w, x_predTo,
                   particle_size=particle_size,
                   measure_computational_times=FLAGS.measure_computational_times,
                   load_samples=FLAGS.load_samples,
                   save_samples=FLAGS.save_samples,
                   save_path=FLAGS.save_path,
                   save_results=FLAGS.save_results,
                   result_dir=FLAGS.result_dir,
                   for_paper=FLAGS.for_paper,
                   no_show=FLAGS.no_show,
                   )


def run_experiment(x_L, C_L, t_L, S_w, x_predTo,
                   particle_size,
                   t_range=None,
                   y_range=None,
                   measure_computational_times=False,
                   load_samples=False,
                   save_samples=False,
                   save_path=None,
                   save_results=False,
                   result_dir=None,
                   no_show=False,
                   for_paper=False):
    """Runs an experiment including a comparison with Monte Carlo simulation with the given settings.

    The underlying process is a 2D (x, y) constant velocity (CV) model with independent components in x, y.
    Therefore, the state is [pos_x, vel_x, pos_y, vel_y].

    :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, vel_x, pos_y, vel_y].
    :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, position of the boundary.
    :param length: TODO
    :param t_range: A list of length 2 representing the plot limits for the first passage time.
    :param y_range: A list of length 2 representing the plot limits for the y component at the first passage time.
    :param measure_computational_times: A Boolean, whether to measure the computational times.
    :param load_samples: Boolean, whether to load the samples for the Monte Carlo simulation from file.
    :param save_samples: Boolean, whether to save the samples for the Monte Carlo simulation from file.
    :param save_path: String, path where to save the .npz file with the samples (suffix .npz).
    :param save_results: Boolean, whether to save the plots.
    :param result_dir: String, directory where to save the plots.
    :param no_show: Boolean, whether to show the plots (False).
    :param for_paper: Boolean, whether to use the plots for a publication (omit headers, etc.).
    """
    hitting_time_model_kwargs = {"x_L": x_L,
                                 "C_L": C_L,
                                 "S_w": S_w,
                                 "x_predTo": x_predTo,
                                 "t_L": t_L}

    # Deterministic predictions
    t_predicted = t_L + (x_predTo - x_L[0]) / x_L[1]
    y_predicted = x_L[2] + (t_predicted - t_L) * x_L[3]

    # Plot settings
    if t_range is None:
        t_range = [t_predicted - 0.3 * (t_predicted - t_L) - 1 / 2 * particle_size[0] / x_L[1],
                   t_predicted + 0.3 * (t_predicted - t_L) + particle_size[0] / x_L[1]]
    if y_range is None:
        y_range = [0.7 * y_predicted - 1 / 2 * particle_size[1],
                   1.3 * y_predicted + 1 / 2 * particle_size[1]]
    plot_t = np.arange(t_range[0], t_range[1], 0.00001)
    plot_y = np.arange(y_range[0], y_range[1], 0.001)

    # Create base class
    hte = HittingTimeEvaluatorWithExtents('CV Process', x_predTo, plot_t, t_predicted, t_L,
                                          get_example_tracks_fn=get_example_tracks(x_L,
                                                                                   C_L,
                                                                                   S_w,
                                                                                   _get_system_matrices_from_parameters),
                                          # TODO: Nicht mehr privat
                                          save_results=save_results,
                                          result_dir=result_dir,
                                          no_show=no_show,
                                          for_paper=for_paper)

    # Create samples
    dt = 1 / 1000
    t_samples_first_front_arrival, t_samples_first_back_arrival, y_min_samples, y_max_samples = create_ty_cv_samples_hitting_time(
        x_L=x_L,
        C_L=C_L,
        S_w=S_w,
        x_predTo=x_predTo,
        t_L=t_L,
        length=particle_size[0],
        dt=dt,
        N=100000)

    # Set up the hitting time approaches
    taylor_model_with_extents = HittingTimeWithExtentsModel(particle_size[0], TaylorHittingTimeModel,
                                                            hitting_time_model_kwargs,
                                                            name="Gauß-Taylor with extent")
    approx_model_with_extents = HittingTimeWithExtentsModel(particle_size[0], EngineeringApproxHittingTimeModel,
                                                            hitting_time_model_kwargs,
                                                            name="No-return approx. with extent")
    simplified_taylor_model_with_extents = HittingTimeWithExtentsSimplifiedModel(particle_size[0], TaylorHittingTimeModel,
                                                                                 hitting_time_model_kwargs,
                                                                                 name="Gauß-Taylor with extent (simplified)")
    simplified_approx_model_with_extents = HittingTimeWithExtentsSimplifiedModel(particle_size[0],
                                                                                 EngineeringApproxHittingTimeModel,
                                                                                 hitting_time_model_kwargs,
                                                                                 name="No-return approx. with extent (simplified)")
    approaches_temp_ls = [taylor_model_with_extents,
        approx_model_with_extents,
        simplified_taylor_model_with_extents,
        simplified_approx_model_with_extents,
    ]

    # plot the distribution of the particle front and back arrival time at one axis (the time axis)
    # hte.plot_first_arrival_interval_distribution_on_time_axis(t_samples_first_front_arrival,
    #                                                           t_samples_first_back_arrival,
    #                                                           approaches_temp_ls,
    #                                                           plot_hist_for_all_particles=True,
    #                                                           )
    #
    # # plot the joint distribution of the particle front and back arrival time (2-dimensional distribution, heatmap)
    # hte.plot_joint_first_arrival_interval_distribution(t_samples_first_front_arrival,
    #                                                    t_samples_first_back_arrival,
    #                                                    approaches_temp_ls,
    #                                                    plot_hist_for_all_particles=True,
    #                                                    plot_cdfs=False,
    #                                                    )

    # # plot a simplifies joint distribution (based on the marginals and independence assumption) of the particle front
    # # and back arrival time (2-dimensional distribution, heatmap)
    # hte.plot_joint_first_arrival_interval_distribution(t_samples_first_front_arrival,
    #                                                    t_samples_first_back_arrival,
    #                                                    approaches_temp_ls,
    #                                                    plot_hist_for_all_particles=True,
    #                                                    use_independent_joint=True,
    #                                                    )

    # plot the calibration
    # hte.plot_calibration(t_samples_first_front_arrival,
    #                      t_samples_first_back_arrival,
    #                      approaches_temp_ls,
    #                      )

    # Set up the hitting location approaches
    hitting_location_model_kwargs = {'S_w': hitting_time_model_kwargs['S_w']}
    taylor_model_with_extents = HittingLocationWithExtentsModel(*particle_size,
                                                                TaylorHittingTimeModel,
                                                                CVTaylorHittingLocationModel,
                                                                hitting_time_model_kwargs,
                                                                hitting_location_model_kwargs,
                                                                name="Gauß-Taylor with extent",
                                                                )


    # Results for spatial uncertainties
    hte_spatial = HittingLocationEvaluatorWithExtents('CV Process', x_predTo, t_predicted, y_predicted, plot_y, t_L,
                                                      get_example_tracks_fn=get_example_tracks(x_L,
                                                                                               C_L,
                                                                                               S_w,
                                                                                               _get_system_matrices_from_parameters),
                                                      save_results=save_results,
                                                      result_dir=result_dir,
                                                      no_show=no_show,
                                                      for_paper=for_paper)

    approaches_spatial_ls = [taylor_model_with_extents]

    # plot the distribution of the particle front and back arrival time at one axis (the time axis)
    hte_spatial.plot_y_at_first_arrival_interval_distribution_on_y_axis(y_min_samples - particle_size[1] / 2,
                                                                        y_max_samples + particle_size[1] / 2,
                                                                        approaches_spatial_ls,
                                                                        plot_hist_for_all_particles=True,
                                                                        )

    # plot the joint distribution of the particle front and back arrival time (2-dimensional distribution, heatmap)
    hte_spatial.plot_joint_y_at_first_arrival_interval_distribution(y_min_samples - particle_size[1] / 2,
                                                                    y_max_samples + particle_size[1] / 2,
                                                                    approaches_spatial_ls,
                                                                    plot_hist_for_all_particles=True,
                                                                    plot_marginals=False,
                                                                    )
    # plot the calibration
    hte_spatial.plot_calibration(y_min_samples - particle_size[1] / 2,
                                 y_max_samples + particle_size[1] / 2,
                                 approaches_spatial_ls,
                                 )


class AbstractHittingTimeWithExtentsModel(ABC):

    def __init__(self, length, name):

        self._length = length
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def calculate_ejection_windows(self, q):
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')


class HittingTimeWithExtentsModel(AbstractHittingTimeWithExtentsModel):

    def __init__(self, length, hitting_time_model_class, hitting_time_model_kwargs, name):

        super().__init__(length=length,
                         name=name,
                         )

        htm_kwargs = hitting_time_model_kwargs.copy()
        x_predTo = htm_kwargs.pop('x_predTo')

        self._front_arrival_model = hitting_time_model_class(x_predTo=x_predTo - length / 2,
                                                             **htm_kwargs)
        self._back_arrival_model = hitting_time_model_class(x_predTo=x_predTo + length / 2,
                                                            **htm_kwargs)

    @property
    def front_arrival_model(self):
        return self._front_arrival_model

    @property
    def back_arrival_model(self):
        return self._back_arrival_model

    def calculate_ejection_windows(self, q):
        q_front = (1 - q) / 2
        q_back = (1 + q) / 2

        t_start = self._front_arrival_model.ppf(q_front)
        t_end = self._back_arrival_model.ppf(q_back)
        return t_start, t_end


class HittingTimeWithExtentsSimplifiedModel(AbstractHittingTimeWithExtentsModel):

    def __init__(self, length, hitting_time_model_class, hitting_time_model_kwargs, name):

        super().__init__(length=length,
                         name=name,
                         )

        self._arrival_model = hitting_time_model_class(**hitting_time_model_kwargs)

    def calculate_ejection_windows(self, q):
        q_front = (1 - q) / 2
        q_back = (1 + q) / 2

        t_start = self._arrival_model.ppf(q_front) - self._length/(2 * self._arrival_model._x_L[1])   # TODO: Ist das überhaupt so wie gedacht?
        t_end = self._arrival_model.ppf(q_back) + self._length/(2 * self._arrival_model._x_L[1])
        return t_start, t_end


class AbstractHittingLocationWithExtentsModel(ABC):

    def __init__(self, length, width, name):

        self._length = length
        self._width = width
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def calculate_ejection_windows(self, q):
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')


class HittingLocationWithExtentsModel(AbstractHittingLocationWithExtentsModel):

    def __init__(self,
                 length,
                 width,
                 hitting_time_model_class,
                 hitting_location_model_class,
                 hitting_time_model_kwargs,
                 hitting_location_model_kwargs,
                 name,
                 ):
        super().__init__(length=length,
                         width=width,
                         name=name,
                         )

        htm_kwargs = hitting_time_model_kwargs.copy()
        x_predTo = htm_kwargs.pop('x_predTo')

        self._front_arrival_model = hitting_time_model_class(x_predTo=x_predTo - length / 2,
                                                             **htm_kwargs)
        self._back_arrival_model = hitting_time_model_class(x_predTo=x_predTo + length / 2,
                                                            **htm_kwargs)
        self._front_location_model = hitting_location_model_class(hitting_time_model=self._front_arrival_model,
                                                                  **hitting_location_model_kwargs)  # TODO: Das geht so allgemein nicht
        self._back_location_model = hitting_location_model_class(hitting_time_model=self._back_arrival_model,
                                                                 **hitting_location_model_kwargs)  # TODO: Das geht so allgemein nicht

    @property
    def front_location_model(self):  # TODO: Brauchen wir die zwei überhaupt?
        return self._front_location_model

    @property
    def back_location_model(self):
        return self._back_location_model

    @property
    def max_y_model(self):
        return MaxYModel(self._front_location_model, self._back_location_model, self._width)

    @property
    def min_y_model(self):
        return MinYModel(self._front_location_model, self._back_location_model, self._width)

    def calculate_ejection_windows(self, q):
        q_low = (1 - q) / 2
        q_up = (1 + q) / 2

        y_start = self.min_y_model.ppf(q_low)
        y_end = self.max_y_model.ppf(q_up)
        return y_start, y_end


class MaxYModel(object):

    def __init__(self, front_location_model, back_location_model, width):
        self._front_location_model = front_location_model
        self._back_location_model = back_location_model

        self._width = width

    def cdf(self, y):
        # by independence assumption
        # return self._front_location_model.cdf(y - self._width / 2) * self._back_location_model.cdf(y - self._width / 2)
        # by no return assumption
        # return self._back_location_model.cdf(y - self._width / 2)
        # return self._front_location_model.cdf(y - self._width / 2)
        back_location_value = self._back_location_model.cdf(y - self._width / 2)
        front_location_value = self._front_location_model.cdf(y - self._width / 2)
        return np.min(np.array([back_location_value, front_location_value]))

    def pdf(self, y):
        # by no return assumption
        return self._back_location_model.pdf(y - self._width / 2)
        # return self._front_location_model.pdf(y - self._width / 2)
        back_location_value = self._back_location_model.pdf(y - self._width / 2)
        front_location_value = self._front_location_model.pdf(y - self._width / 2)
        back_location_cdf_value = self._back_location_model.cdf(y - self._width / 2)
        front_location_cdf_value = self._front_location_model.cdf(y - self._width / 2)
        return np.array([back_location_value, front_location_value])[
            np.argmin(np.array([back_location_cdf_value, front_location_cdf_value]))]

    def ppf(self, q):
        # by no return assumption
        # return self._back_location_model.ppf(q) + self._width / 2
        # return self._front_location_model.ppf(q) + self._width / 2
        back_location_value = self._back_location_model.ppf(q) + self._width / 2
        front_location_value = self._front_location_model.ppf(q) + self._width / 2
        return np.max(np.array([back_location_value, front_location_value]))

    def cdf_values(self, range):
        cdf_values = [self.cdf(ys) for ys in range]
        return cdf_values

    def pdf_values(self, range):
        pdf_values = np.gradient(self.cdf_values(range), range)
        return pdf_values


class MinYModel(object):

    def __init__(self, front_location_model, back_location_model, width):
        self._front_location_model = front_location_model
        self._back_location_model = back_location_model
        self._width = width

    def cdf(self, y):
        # # by independence assumption
        # return 1 - (1 - self._front_location_model.cdf(y + self._width / 2)) * (
        #         1 - self._back_location_model.cdf(y + self._width / 2))
        # by no return assumption
        # return self._back_location_model.cdf(y + self._width / 2)
        back_location_value = self._back_location_model.cdf(y + self._width / 2)
        front_location_value = self._front_location_model.cdf(y + self._width / 2)
        return np.max(np.array([back_location_value, front_location_value]))

    def pdf(self, y):
        # by no return assumption
        # return self._back_location_model.pdf(y + self._width / 2)
        back_location_value = self._back_location_model.pdf(y + self._width / 2)
        front_location_value = self._front_location_model.pdf(y + self._width / 2)
        back_location_cdf_value = self._back_location_model.cdf(y + self._width / 2)
        front_location_cdf_value = self._front_location_model.cdf(y + self._width / 2)
        return np.array([back_location_value, front_location_value])[
            np.argmax(np.array([back_location_cdf_value, front_location_cdf_value]))]

    def ppf(self, q):
        # by no return assumption
        # return self._back_location_model.ppf(q) - self._width / 2
        back_location_value = self._back_location_model.ppf(q) - self._width / 2
        front_location_value = self._front_location_model.ppf(q) - self._width / 2
        return np.min(np.array([back_location_value, front_location_value]))

    def cdf_values(self, range):
        cdf_values = [self.cdf(ys) for ys in range]
        return cdf_values

    def pdf_values(self, range):
        pdf_values = np.gradient(self.cdf_values(range), range)
        return pdf_values


def create_hitting_time_samples(initial_samples,
                                compute_x_next_func,
                                x_predTo,
                                length,
                                t_L=0.0,
                                N=100000,
                                dt=1 / 1000,
                                break_after_n_time_steps=1000,
                                break_min_time=None):
    """Monte Carlo approach to solve the first passage time problem. Propagates particles through the motion model and
    determines time before arrival, and positions before and after the arrival as well as more statistics.

    :param initial_samples: A np.array of shape [num_samples, state_size], the initial particles samples at time step
        t_L.
    :param compute_x_next_func: A function propagating the samples to the next time step, i.e., the one time step
        transition function.
    :param x_predTo: A float, position of the boundary.
    :param length: TODO
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_after_n_time_steps: Integer, maximum number of time steps for the simulation.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least.
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        t_samples: A np.array of shape [N] containing the first passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first passage times of the particles.
        fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
        TODO
    """
    start_time = time.time()
    # Let the samples move to the boundary

    x_curr = initial_samples
    x_term_first_front_arrival = np.zeros(initial_samples.shape[0], dtype=bool)
    x_term_first_back_not_in = np.zeros(initial_samples.shape[0], dtype=bool)
    t = t_L
    ind = 0
    time_before_first_front_arrival = np.full(N, t_L, dtype=np.float64)
    time_before_first_back_not_in = np.full(N, t_L, dtype=np.float64)
    x_before_front_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_before_front_arrival[:] = np.nan
    x_after_front_arrival = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_after_front_arrival[:] = np.nan
    x_before_first_back_not_in = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_before_first_back_not_in[:] = np.nan
    x_after_first_back_not_in = np.empty((initial_samples.shape[0], initial_samples.shape[1]))
    x_after_first_back_not_in[:] = np.nan
    y_min = np.empty((initial_samples.shape[0]))
    y_min[:] = np.inf
    y_max = np.empty((initial_samples.shape[0]))
    y_max[:] = - np.inf
    while True:
        if ind % 100 == 0:
            logging.info('Timestep {0}, x equals approx. {1}'.format(ind, x_curr[0, 0]))

        x_next = compute_x_next_func(x_curr)

        first_front_arrival = np.logical_and(np.logical_not(x_term_first_front_arrival), x_next[:, 0] + length/2 >= x_predTo)
        first_back_not_in = np.logical_and(np.logical_not(x_term_first_back_not_in), x_next[:, 0] - length/2 > x_predTo)

        x_term_first_front_arrival[first_front_arrival] = True
        x_term_first_back_not_in[first_back_not_in] = True
        
        x_before_front_arrival[first_front_arrival] = x_curr[first_front_arrival]
        x_after_front_arrival[first_front_arrival] = x_next[first_front_arrival]
        x_before_first_back_not_in[first_back_not_in] = x_curr[first_back_not_in]
        x_after_first_back_not_in[first_back_not_in] = x_next[first_back_not_in]

        particle_passes_mask = np.logical_and(x_term_first_front_arrival, np.logical_not(x_term_first_back_not_in))
        y_min[np.logical_and(particle_passes_mask, x_next[:, 2] < y_min)] = x_next[
            np.logical_and(particle_passes_mask, x_next[:, 2] < y_min), 2]
        y_max[np.logical_and(particle_passes_mask, x_next[:, 2] > y_max)] = x_next[
            np.logical_and(particle_passes_mask, x_next[:, 2] > y_max), 2]

        if break_min_time is None:
            if np.all(x_term_first_front_arrival) and np.all(x_term_first_back_not_in):
                break
        else:
            if t >= break_min_time and np.all(x_term_first_front_arrival) and np.all(x_term_first_back_not_in):
                break
        if break_after_n_time_steps is not None and ind >= break_after_n_time_steps:
            logging.info(
                'Warning: Sampling interrupted because {}. reached. Please adjust break_after_n_time_steps if you want to move the particles more timesteps.'.format(
                    break_after_n_time_steps))
            break
        x_curr = x_next
        t += dt
        time_before_first_front_arrival[np.logical_not(x_term_first_front_arrival)] = t
        time_before_first_back_not_in[np.logical_not(x_term_first_back_not_in)] = t
        ind += 1

    logging.info('MC time: {0}ms'.format(round(1000 * (time.time() - start_time))))
    return time_before_first_front_arrival, time_before_first_back_not_in, x_before_front_arrival, x_after_front_arrival, x_before_first_back_not_in, x_after_first_back_not_in, x_term_first_front_arrival, x_term_first_back_not_in, y_min, y_max


def create_lgssm_hitting_time_samples(F,
                                      Q,
                                      x_L,
                                      C_L,
                                      x_predTo,
                                      length,
                                      t_L=0.0,
                                      N=100000,
                                      dt=1 / 1000,
                                      break_after_n_time_steps=1000,
                                      break_min_time=None):
    """Monte Carlo approach to solve the first passage time problem. Propagates particles through the discrete-time
    LGSSM motion model and determines their first passage at x_predTo as well as the location in y at the first passage 
    by interpolating the positions between the last time before and the first time after the boundary.

    :param F: A np.array of shape [4, 4], the transition matrix of the LGSSM.
    :param Q: A np.array of shape [4, 4], the transition noise covariance matrix of the LGSSM.
    :param x_L: A np.array of shape [length_state] representing the expected value of the initial state. We use index L 
        here because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, ..., pos_y, ...].
    :param C_L: A np.array of shape [length_state, length_state] representing the covariance matrix of the initial state.
    :param x_predTo: A float, position of the boundary.
    :param length: TODO
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_min_time: A float, the time (not the time step) up to which is simulated at least.
        (break_after_n_time_steps dominates break_min_time).

    :returns:
        t_samples: A np.array of shape [N] containing the first passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first passage times of the particles.
        fraction_of_returns: A np.array of shape[num_simulated_time_steps], the fraction in each time steps of
            tracks that have previously reached the boundary, but then fall below the boundary until the respective
            time step.
        TODO
    """
    initial_samples = np.random.multivariate_normal(mean=x_L, cov=C_L, size=N)

    mean_w = np.zeros(x_L.shape[0])

    def compute_x_next_func(x_curr):
        x_curr_tf = tf.convert_to_tensor(x_curr)
        x_next = tf.linalg.matvec(F, x_curr_tf).numpy()
        w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=N)
        x_next = x_next + w_k
        return x_next

    time_before_first_front_arrival, time_before_first_back_not_in, x_before_front_arrival, x_after_front_arrival, x_before_first_back_not_in, x_after_first_back_not_in, x_term_first_front_arrival, x_term_first_back_not_in, y_min, y_max = create_hitting_time_samples(
        initial_samples,
        compute_x_next_func,
        x_predTo,
        length,
        t_L=t_L,
        N=N,
        dt=dt,
        break_after_n_time_steps=break_after_n_time_steps,
        break_min_time=break_min_time)

    return time_before_first_front_arrival, time_before_first_back_not_in, x_before_front_arrival, x_after_front_arrival, x_before_first_back_not_in, x_after_first_back_not_in, x_term_first_front_arrival, x_term_first_back_not_in, y_min, y_max


def create_ty_cv_samples_hitting_time(x_L,
                                      C_L,
                                      S_w,
                                      x_predTo,
                                      length,
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
    :param length: TODO
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
        TODO
    """
    F, Q = _get_system_matrices_from_parameters(dt, S_w)

    time_before_first_front_arrival, time_before_first_back_not_in, x_before_front_arrival, x_after_front_arrival, x_before_first_back_not_in, x_after_first_back_not_in, x_term_first_front_arrival, x_term_first_back_not_in, y_min_samples, y_max_samples = create_lgssm_hitting_time_samples(
        F,
        Q,
        x_L,
        C_L,
        x_predTo,
        length,
        t_L=t_L,
        N=N,
        dt=dt,
        break_after_n_time_steps=break_after_n_time_steps,
        break_min_time=break_min_time)

    # Linear interpolation to get time
    v_interpolated_arrival = (x_after_front_arrival[x_term_first_front_arrival, 1] - x_before_front_arrival[
        x_term_first_front_arrival, 1]) / (
                                     x_after_front_arrival[x_term_first_front_arrival, 0] - x_before_front_arrival[
                                 x_term_first_front_arrival, 0]) * (
                                     x_predTo - x_before_front_arrival[x_term_first_front_arrival, 0]) + \
                             x_before_front_arrival[x_term_first_front_arrival, 1]
    delta_t_first_arrival = ((x_predTo - length / 2) - x_before_front_arrival[
        x_term_first_front_arrival, 0]) / v_interpolated_arrival
    t_samples_first_front_arrival = time_before_first_front_arrival  # TODO
    t_samples_first_front_arrival[x_term_first_front_arrival] = time_before_first_front_arrival[
                                                                    x_term_first_front_arrival] + delta_t_first_arrival
    t_samples_first_front_arrival[np.logical_not(x_term_first_front_arrival)] = int(
        max(t_samples_first_front_arrival)) + 1  # default value for particles that do not arrive

    v_interpolated_last_in = (x_after_first_back_not_in[x_term_first_back_not_in, 1] - x_before_first_back_not_in[
        x_term_first_back_not_in, 1]) / (
                                     x_after_first_back_not_in[x_term_first_back_not_in, 0] -
                                     x_before_first_back_not_in[x_term_first_back_not_in, 0]) * (
                                     (x_predTo + length / 2) - x_before_first_back_not_in[
                                 x_term_first_back_not_in, 0]) + \
                             x_before_first_back_not_in[x_term_first_back_not_in, 1]
    delta_t_last_in = ((x_predTo + length / 2) - x_before_first_back_not_in[
        x_term_first_back_not_in, 0]) / v_interpolated_last_in
    t_samples_first_back_arrival = time_before_first_back_not_in  # TODO
    t_samples_first_back_arrival[x_term_first_back_not_in] = time_before_first_back_not_in[
                                                                 x_term_first_back_not_in] + delta_t_last_in
    t_samples_first_back_arrival[np.logical_not(x_term_first_back_not_in)] = int(
        max(t_samples_first_back_arrival)) + 1  # default value for particles that do not arrive

    y_samples_first_front_arrival = x_before_front_arrival[:, 2]
    y_samples_first_front_arrival[x_term_first_front_arrival] = x_before_front_arrival[
                                                                    x_term_first_front_arrival, 2] + delta_t_first_arrival * \
                                                                x_before_front_arrival[x_term_first_front_arrival, 3]
    y_samples_first_front_arrival[
        np.logical_not(x_term_first_front_arrival)] = np.nan  # default value for particles that do not arrive

    y_samples_first_back_arrival = x_before_first_back_not_in[:, 2]
    y_samples_first_back_arrival[x_term_first_back_not_in] = x_before_first_back_not_in[
                                                                 x_term_first_back_not_in, 2] + delta_t_last_in * \
                                                             x_before_first_back_not_in[x_term_first_back_not_in, 3]
    y_samples_first_back_arrival[
        np.logical_not(x_term_first_back_not_in)] = np.nan  # default value for particles that do not arrive

    y_min_samples[y_samples_first_back_arrival < y_min_samples] = y_samples_first_back_arrival[
        y_samples_first_back_arrival < y_min_samples]
    y_max_samples[y_samples_first_back_arrival > y_min_samples] = y_samples_first_back_arrival[
        y_samples_first_back_arrival > y_min_samples]

    y_min_samples[y_samples_first_front_arrival < y_min_samples] = y_samples_first_front_arrival[
        y_samples_first_front_arrival < y_min_samples]
    y_max_samples[y_samples_first_front_arrival > y_min_samples] = y_samples_first_front_arrival[
        y_samples_first_front_arrival > y_min_samples]

    return t_samples_first_front_arrival, t_samples_first_back_arrival, y_min_samples, y_max_samples


if __name__ == "__main__":
    app.run(main)

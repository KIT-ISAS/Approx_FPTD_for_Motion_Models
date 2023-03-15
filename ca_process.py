"""
############################################ ca_process.py  ###########################################
Authors: Marcel Reith-Braun (ISAS, marcel.reith-braun@kit.edu), Jakob Thumm
#######################################################################################################
Calculates approximate first passage time distributions for a constant acceleration model using different
approaches.

usage:
 - run docker container - tested with tensorflow/approx_fptd:2.8.0-gpu image:
    $ docker run -u $(id -u):$(id -g) \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
            tensorflow/approx_fptd:2.8.0-gpu
 - within container:
     $   python3 /mnt/ca_process.py \\
requirements:
  - Required tensorflow/approx_fptd:2.8.0-gpu image: See corresponding dockerfile.
  - Volume mounts: Specify a path </path/to/repo/> that points to the repo.
"""


from absl import logging
from absl import app
from absl import flags

from abc import ABC, abstractmethod
from timeit import time

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm

from hitting_time_uncertainty_utils import HittingTimeEvaluator
from abstract_distributions import AbstractHittingTimeModel, AbstractTaylorHittingTimeModel, \
    AbstractEngineeringApproxHittingTimeModel, AbstractMCHittingTimeModel
from sampler import create_lgssm_hitting_time_samples, get_example_tracks_lgssm
from timer import measure_computation_times


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
    S_w = 1000
    # Covariance matrix at last timestep
    C_L = np.array([[2E-7, 2E-5, 8E-4, 0, 0, 0], [2E-5, 3E-3, 1.5E-1, 0, 0, 0], [8E-4, 1.5E-1, 1.3E1, 0, 0, 0],
                     [0, 0, 0, 2E-7, 2E-5, 8E-4], [0, 0, 0, 2E-5, 3E-3, 1.5E-1], [0, 0, 0, 8E-4, 1.5E-1, 1.3E1]])
    # Mean position at last timestep
    x_L = np.array([0.3, 6.2, 4.4, 0.5, 0.2, 2.8])

    # Boundary position
    x_predTo = 0.6458623971412047
    # Last time step
    t_L = 0

    # Run the experiment
    run_experiment(x_L, C_L, t_L, S_w, x_predTo,
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

    The underlying process is a 2D (x, y) constant acceleration (CA) model with independent components in x, y.
    Therefore, the state is [pos_x, vel_x, acc_x, pos_y, vel_y, acc_y].

    :param x_L: A np.array of shape [6] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, vel_x, acc_x, pos_y, vel_y, acc_y].
    :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, position of the boundary.
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
    # Deterministic predictions
    t_predicted = t_L - x_L[1] / x_L[2] + np.sign(x_L[2]) * \
                  np.sqrt((x_L[1] / x_L[2]) ** 2 + 2 / x_L[2] * (x_predTo - x_L[0]))
    y_predicted = x_L[3] + (t_predicted - t_L) * x_L[4] + 1 / 2 * (t_predicted - t_L) ** 2 * x_L[5]

    # Plot settings
    if t_range is None:
        t_range = [t_predicted - 0.2*(t_predicted - t_L), t_predicted + 0.2*(t_predicted - t_L)]
    if y_range is None:
        y_range = [0.9 * y_predicted, 1.1 * y_predicted]
    plot_t = np.arange(t_range[0], t_range[1], 0.00001)
    plot_y = np.arange(y_range[0], y_range[1], 0.001)

    # Create base class
    hte = HittingTimeEvaluator('CA Process', x_predTo, plot_t, t_predicted, t_L,
                               plot_y=plot_y,
                               y_predicted=y_predicted,
                               get_example_tracks_fn=get_example_tracks_lgssm(x_L,
                                                                              C_L,
                                                                              S_w,
                                                                              _get_system_matrices_from_parameters),
                               save_results=save_results,
                               result_dir=result_dir,
                               no_show=no_show,
                               for_paper=for_paper)

    # Create samples
    dt = 1 / 1000
    if not load_samples:
        t_samples, y_samples, fraction_of_returns = create_ty_ca_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L, dt=dt)
        if save_samples:
            np.savez(save_path, name1=t_samples, name2=y_samples)
            logging.info("Saved samples.")
    else:
        data = np.load(save_path)
        t_samples = data['name1']
        y_samples = data['name2']
        fraction_of_returns = data['name3']
    hte.plot_sample_histogram(t_samples)
    # hte.plot_sample_histogram(y_samples, x_label='y-Coordinate')

    # Show example tracks and visualize uncertainties over time
    # hte.plot_example_tracks(N=5)
    ev_fn = lambda t: x_L[0] + x_L[1] * (t - t_L) + x_L[2]/ 2 *(t - t_L)**2
    var_fn = lambda t: C_L[0, 0] + (C_L[0, 1] + C_L[1, 0]) * (t - t_L) \
                       + (1 / 2 * C_L[0, 2] + 1 / 2 * C_L[2, 0] + C_L[1, 1]) * (t - t_L) ** 2 \
                       + (1 / 2 * C_L[1, 2] + 1 / 2 * C_L[2, 1]) * (t - t_L) ** 3 \
                       + 1 / 4 * C_L[2, 2] * (t - t_L) ** 4 + 1 / 20 * S_w * (t - t_L) ** 5
    hte.plot_mean_and_stddev_over_time(ev_fn, var_fn, show_example_tracks=True)

    # Set up the approaches
    taylor_model = TaylorHittingTimeModel(x_L, C_L, S_w, x_predTo, t_L)
    approx_model = EngineeringApproxHittingTimeModel(x_L, C_L, S_w, x_predTo, t_L)
    mc_model = MCHittingTimeModel(x_L, C_L, S_w, x_predTo, t_L, t_range, t_samples=t_samples)

    # Results for temporal uncertainties
    logging.info('MAX CDF: {} at {}'.format(approx_model.q_max, approx_model.t_max))
    approx_model.plot_valid_regions(theta=t_predicted, save_results=save_results, result_dir=result_dir, for_paper=True,
                                    no_show=no_show)
    # approx_model.plot_valid_regions(save_results=save_results, result_dir=result_dir, for_paper=True, no_show=no_show)
    logging.info('tau_max: {}'.format(approx_model.trans_dens_ppf(t_predicted)[0]))
    logging.info('Mass inside invalid region: {}'.format(
        1 - approx_model.cdf(t_predicted + approx_model.trans_dens_ppf(t_predicted)[0])))
    logging.info('Approximate returning probs after a crossing until time t_max: {}'.format(
        approx_model.get_statistics()['ReturningProbs'](approx_model.t_max)))

    approaches_temp_ls = [taylor_model, approx_model]

    # Plot the quantile functions
    hte.plot_quantile_functions(approaches_temp_ls)
    # Calculate moments and compare the results
    hte.compare_moments_temporal(approaches_temp_ls)
    # Calculate the skewness and compare the results
    hte.compare_skewness_temporal(approaches_temp_ls)

    # Calculate wasserstein distance and compare results
    hte.compare_wasserstein_distances_temporal(t_samples, approaches_temp_ls)
    # Calculate the Hellinger distance
    hte.compare_hellinger_distances_temporal(t_samples, approaches_temp_ls)
    # Calculate the first wasserstein distance
    hte.compare_first_wasserstein_distances_temporal(t_samples, approaches_temp_ls)
    # Calculate the kolmogorov distance
    hte.compare_kolmogorv_distances_temporal(t_samples, approaches_temp_ls)

    # Plot histogram of samples and hitting time distributions
    hte.plot_first_hitting_time_distributions(t_samples, approaches_temp_ls, plot_hist_for_all_particles=True)
    hte.plot_fptd_and_paths_in_one(ev_fn, var_fn, t_samples, approaches_temp_ls, plot_hist_for_all_particles=True)
    # Plot histogram of samples for returning distribution and estimated returning distribution
    # hte.plot_returning_probs_from_fptd_histogram(ev_fn, var_fn, t_samples, approaches_temp_ls)   # this is too noisy
    hte.plot_returning_probs_from_sample_paths(fraction_of_returns, dt, approaches_temp_ls)

    if measure_computational_times:
        logging.info('Measuring computational time for ca process.')
        model_class_ls = [MCHittingTimeModel, TaylorHittingTimeModel, EngineeringApproxHittingTimeModel]
        model_attributes_ls = [[x_L, C_L, S_w, x_predTo, t_L, t_range]] + 2 * [[x_L, C_L, S_w, x_predTo, t_L]]
        measure_computation_times(model_class_ls, model_attributes_ls, t_range=t_range)


# Approaches to solve problem
class CAHittingTimeModel(AbstractHittingTimeModel, ABC):
    """A base class for the CA hitting time models."""

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name='CA hitting time model', **kwargs):
        """Initialize the model.

        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, acc_y, pos_y, vel_y, acc_y].
        :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param x_predTo: A float, position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, (default) name for the model.
        """
        self._x_L = x_L
        self._C_L = C_L
        self._S_w = S_w
        super().__init__(x_predTo=x_predTo,
                         t_L=t_L,
                         name=name,
                         **kwargs)

        # For properties
        self._third_central_moment = None
        self._fourth_central_moment = None
        self._fifth_central_moment = None
        self._second_moment = None
        self._third_moment = None
        self._fourth_moment = None
        self._fifth_moment = None
        self._skew = None

    @property
    def x_L(self):
        """The expected value of the initial state."""
        return self._x_L

    @property
    def C_L(self):
        """The covariance matrix of the initial state."""
        return self._C_L

    @property
    def S_w(self):
        """The power spectral density of the model."""
        return self._S_w

    @property
    @abstractmethod
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def fourth_central_moment(self):
        """The fourth central moment of the first passage time distribution."""
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def fifth_central_moment(self):
        """The fifth central moment of the first passage time distribution."""
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def second_moment(self):
        """The second moment of the first passage time distribution."""
        if self._second_moment is None:
            self._second_moment = self.var + self.ev**2
        return self._second_moment

    @property
    def third_moment(self):
        """The third moment of the first passage time distribution."""
        if self._third_moment is None:
            self._third_moment = self.third_central_moment + 3*self.ev*self.var + self.ev**3
        return self._third_moment

    @property
    def fourth_moment(self):
        """The fourth moment of the first passage time distribution."""
        if self._fourth_moment is None:
            self._fourth_moment = self.fourth_central_moment + 4*self.ev*self.third_moment \
                                  - 6*self.ev**2*self.second_moment + 3*self.ev**4
        return self._fourth_moment

    @property
    def fifth_moment(self):
        """The fifth moment of the first passage time distribution."""
        if self._fifth_moment is None:
            self._fifth_moment = self.fifth_central_moment + 5*self.ev*self.fourth_moment \
                                  - 10*self.ev**2*self.third_moment + 10*self.ev**3*self.second_moment \
                                  - 4*self.ev**5
        return self._fifth_moment

    @property
    def skew(self):
        """The skew of the first passage time distribution."""
        if self._skew is None:
            # third standardized moment / stddev**3
            self._skew = self.third_central_moment / self.stddev**3
        return self._skew

    def ev_t(self, t):
        """The mean function of the CA motion model in x.

        :param t: A float or np.array, the time parameter of the mean function.
        """
        return self.x_L[0] + self.x_L[1] * (t - self.t_L) + self.x_L[2] / 2 * (t - self.t_L) ** 2

    def var_t(self, t):
        """The variance function of the CA motion model in x.

        :param t: A float or np.array, the time parameter of the variance function.
        """
        return self.C_L[0, 0] + 2*self.C_L[0, 1] * (t - self.t_L) \
                       + (self.C_L[0, 2] + self.C_L[1, 1]) * (t - self.t_L) ** 2 \
                       + self.C_L[1, 2] * (t - self.t_L) ** 3 \
                       + 1 / 4 * self.C_L[2, 2] * (t - self.t_L) ** 4 \
                       + 1 / 20 * self.S_w * (t - self.t_L) ** 5

    def trans_density(self, dt, theta):
        """The transition density p(x(dt+theta)| x(theta) = x_predTo) from going from x_predTo at time theta to
        x(dt+theta) at time dt+theta.

        Note that in terms of the used approximation, this can be seen as the first returning time to x_predTo after
        a crossing of x_predTo at theta.

        :param dt: A float or np.array, the time difference. dt is zero at time = theta.
        :param theta: A float or np.array, the (assumed) time at which x(theta) = x_pred_to.

        :returns: The value of the transition density for the given dt and theta.
        """
        Phi = lambda dt: np.array([[1, dt, dt ** 2 / 2],
                                   [0, 1, dt],
                                   [0, 0, 1]])

        Q = lambda dt: self.S_w * np.array([[pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6],
                                            [pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2],
                                            [pow(dt, 3) / 6, pow(dt, 2) / 2, dt]])

        cov_theta = np.matmul(np.matmul(Phi(theta - self.t_L), self.C_L[0:3, 0:3]),
                              np.transpose(Phi(theta - self.t_L))) + Q(theta - self.t_L)

        sel_ma = np.array([[0, 0], [1, 0], [0, 1]])
        phi_sel = np.matmul(Phi(dt), sel_ma)
        p_theta_mu = np.array([self.x_L[1] + self.x_L[2]*(theta - self.t_L), self.x_L[2]]) \
                     + cov_theta[1:, 0]/cov_theta[0, 0]*(self.x_predTo - self.ev_t(theta))
        p_theta_var = cov_theta[1:, 1:] - np.outer(cov_theta[1:, 0], cov_theta[0, 1:]) / cov_theta[0, 0]
        trans_mu = np.array([1, 0, 0]) * self.x_predTo + np.matmul(phi_sel, p_theta_mu)
        trans_var = np.matmul(np.matmul(phi_sel, p_theta_var), np.transpose(phi_sel)) + Q(dt)
        return norm(loc=trans_mu[0], scale=np.sqrt(trans_var[0, 0]))

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'SKEW': self.skew,
                          })
        return hit_stats


class TaylorHittingTimeModel(CAHittingTimeModel, AbstractTaylorHittingTimeModel):
    """A simple Gaussian approximation for the first hitting time problem using a Taylor approximation and error
    propagation.
    """

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name='Gauß--Taylor approx.'):
        """Initialize the model.

        :param x_L: A np.array of shape [6] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, acc_x pos_y, vel_y, acc_y].
        :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param x_predTo: A float, position of the boundary.
        :param name: String, name for the model.
        """
        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         name=name)

        self._ev = t_L - x_L[1] / x_L[2] + np.sign(x_L[2]) * \
                   np.sqrt((x_L[1] / x_L[2]) ** 2 + 2 / x_L[2] * (x_predTo - x_L[0]))
        F, _ = _get_system_matrices_from_parameters(self._ev, self.S_w)
        x_p = np.dot(F, self.x_L)
        self._var = (1 / x_p[1]) ** 2 * self.var_t(self._ev)

    @property
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        return 0  # Gaussian third central moment

    @property
    def fourth_central_moment(self):
        """The fourth central moment of the first passage time distribution."""
        return 3*self.var**2  # Gaussian fourth central moment

    @property
    def fifth_central_moment(self):
        """The fifth central moment of the first passage time distribution."""
        return 0  # Gaussian fifth central moment


class EngineeringApproxHittingTimeModel(CAHittingTimeModel, AbstractEngineeringApproxHittingTimeModel):
    """An approximation to the first passage time distribution using the (engineering) assumption that particles
    are unlikely to move back once they have passed the boundary.
    """

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name="No-return approx."):
        """Initialize the model.

        :param x_L: A np.array of shape [6] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, acc_x, pos_y, vel_y, acc_y].
        :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
        :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
        :param x_predTo: A float, position of the boundary.
        :param t_L: A float, the time of the last state/measurement (initial time).
        :param name: String, name for the model.
        """
        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         name=name)

        if self.q_max < 0.95:
            logging.warning(self.name + " does not seem to be applicable as max. confidence {} < 0.95.".format(
                round(self.q_max, 2)))

    def _cdf(self, t):
        """Approach: 1 - int( p(x(t), x= -infty .. x_predTo )

        :param t: A float or np.array, the time parameter of the distribution.
        """
        p_x_given_t = norm(loc=self.ev_t(t), scale=np.sqrt(self.var_t(t)))
        return 1 - p_x_given_t.cdf(self.x_predTo)

    def cdf(self, t):
        """The CDF of the first passage time distribution.

        :param t: A float or np.array, the time parameter of the distribution.
        """
        t = np.asarray([t]) if np.isscalar(t) else np.asarray(t)
        cdf_value = self._cdf(t)
        cdf_value[t > self.t_max] = self.q_max  # piecewise function
        return np.squeeze(cdf_value)

    def _pdf(self, t):
        """Derivative of self._cdf. Can be calculated from the standard Gauss pdf with an argument
        (x_predTo - ev(t))/stddev(t) times the derivative with respect to t of these argument (chain rule).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        # return derivative(self._cdf, t, dx=1e-6)
        gauss = norm.pdf((self.x_predTo - self.ev_t(t)) / np.sqrt(self.var_t(t)))  # Std. Gauss pdf
        der_ev_t = self.x_L[1] + self.x_L[2] * (t - self.t_L)
        der_var_t = 2 * self.C_L[0, 1] + 2 * (self.C_L[0, 2] + self.C_L[1, 1]) * (t - self.t_L) + \
                    3 * self.C_L[1, 2] * (t - self.t_L) ** 2 + self.C_L[2, 2] * (t - self.t_L) ** 3 + \
                    1 / 4 * self.S_w * (t - self.t_L) ** 4
        neg_der_arg = der_ev_t / np.sqrt(self.var_t(t)) + (self.x_predTo - self.ev_t(t)) * der_var_t / (
                    2 * self.var_t(t) ** (3 / 2))  # negative of the derivative
        return gauss * neg_der_arg

    def pdf(self, t):
        """The first passage time distribution (FPTD).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        t = np.asarray([t]) if np.isscalar(t) else np.asarray(t)
        pdf_value = self._pdf(t)
        pdf_value[t > self.t_max] = 0  # piecewise function
        return np.squeeze(pdf_value)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        # perform sanity checks on input
        if q > self.q_max:
            logging.warning(
                'Approximation yields a maximum confidence of {}, '
                'which is lower than the desired confidence level of {}. Computed values may be wrong.'.format(
                    np.round(self.q_max, 4), np.round(q, 4)))
        return self._ppf(q)

    def _ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        Approach:

              1 - q = int(N(x, mu(t), var(t)), x = -inf .. x_predTo) = PHI ( (x_predTo - mu(t)) / sqrt(var(t))
              PHI^-1(1 -q) = (x_predTo - mu(t)) / sqrt(var(t)) -> solve for t...

        We solve the equation for t = t - t_L to simplify calculations and add t_L at the end of the function.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        if q < 0.0 or q > 1.0:
            raise ValueError('Confidence level q must be between 0 and 1.')

        # Special case q = 0.5, this corresponds to the median.
        # 0 =  (x_predTo - mu(t)) / sqrt(var(t) -> x_predTo = mu(t) -> solve for t...
        if q == 0.5:
            # solve: self.x_predTo = self.x_L[0] + self.x_L[1] * t + self.x_L[2] / 2 * t**2
            pp = self.x_L[1]/self.x_L[2]
            qq = 2/self.x_L[2]*(self.x_L[0] - self.x_predTo)
            # Sign depends on the sign of x_L[2]
            t = - pp + np.sign(self.x_L[2]) * np.sqrt(pp**2 - qq)
            return t + self.t_L

        # Polynomial of degree 5
        # At**5 + B*t**4 + C*t**3 + D*t**2 + E*t + F = 0
        qf = norm.ppf(1 - q)  # Standard-Gaussian quantile function
        A = 1/20*self.S_w
        B = 1/4*(self.C_L[2, 2] - self.x_L[2]**2/qf**2)
        C = self.C_L[1, 2] - self.x_L[1]*self.x_L[2]/qf**2
        D = self.C_L[0, 2] + self.C_L[1, 1] + (-self.x_L[1]**2 + self.x_L[2]*(self.x_predTo - self.x_L[0]))/qf**2
        E = 2*self.C_L[0, 1] + 2*self.x_L[1]*(self.x_predTo - self.x_L[0])/qf**2
        F = self.C_L[0, 0] - (self.x_predTo - self.x_L[0])**2/qf**2

        # Use numerical methods as there is no analytical solution for polynomials of degree 5.
        roots = np.roots([A, B, C, D, E, F])

        # roots are in descending order, the first root is always to large.
        real_roots = roots.real[np.logical_not(np.iscomplex(roots))]
        if real_roots.shape[0] == 5:
            t = float(real_roots[4] if q < 0.5 else real_roots[3])
        elif real_roots.shape[0] >= 2:
            t = float(real_roots[2] if q < 0.5 else real_roots[1])
        elif real_roots.shape[0] == 1:
            t = float(real_roots)
        else:
            raise ValueError('Unsupported number of roots.')

        return t + self.t_L

    def _get_max_cdf_value_and_location(self):
        """Method that finds the maximum of the CDF of the approximation and its location.

        Approach:

            set self.pdf(t) = 0, solve for t.

        We solve the equation for t = t - t_L to simplify calculations and add t_L at the end of the function.

        :returns:
            q_max: A float, the maximum value of the CDF.
            t_q_max: A float, the time, where the CDF visits its maximum.
        """
        A = -1 / 40 * self.S_w * self.x_L[2]
        B = -1 / 20 * 3 * self.S_w * self.x_L[1]
        C = 1 / 40 * (10 * self.S_w * self.x_predTo - 10 * self.S_w * self.x_L[0] + 20 * self.C_L[1, 2] * self.x_L[
            2] - 20 * self.C_L[2, 2] * self.x_L[1])
        D = 1 / 40 * (
                    (40 * self.C_L[1, 1] + 40 * self.C_L[0, 2]) * self.x_L[2] - 40 * self.C_L[1, 2] * self.x_L[1] + 40 *
                    self.C_L[2, 2] * self.x_predTo - 40 * self.C_L[2, 2] * self.x_L[0])
        E = 1 / 40 * (120 * self.C_L[0, 1] * self.x_L[2] + 120 * self.C_L[1, 2] * (self.x_predTo - self.x_L[0]))
        F = 1 / 40 * (
                    80 * self.C_L[0, 0] * self.x_L[2] + (80 * self.C_L[1, 1] + 80 * self.C_L[0, 2]) * self.x_predTo + (
                        -80 * self.C_L[1, 1] - 80 * self.C_L[0, 2]) * self.x_L[0] + 80 * self.x_L[1] * self.C_L[0, 1])
        G = 2 * self.x_predTo * self.C_L[0, 1] + 2 * self.C_L[0, 0] * self.x_L[1] - 2 * self.C_L[0, 1] * self.x_L[0]

        roots = np.roots([A, B, C, D, E, F, G])
        real_roots = roots[np.isreal(roots)].real
        t_q_max = max(real_roots)  # get the highest one
        t_q_max += self.t_L

        # Get the CDF Value
        q_max = self._cdf(t_q_max)
        return q_max, t_q_max

    @property
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        if self._third_central_moment is None:
            start_time = time.time()
            # Calculate the third central moment because calculating the third moment directly causes high numerical errors
            # Calculating moments with integrate.quad takes much time
            # self._third_central_moment = integrate.quad(lambda t: (t - self.ev)**3 * self.pdf(t), self.ppf(0.00o05), self.ppf(0.99995))[
            #              0]  # this yields much better results
            self._third_central_moment, _, abs_dev, rel_dev = self.compute_moment(
                lambda t: (t - self.ev) ** 3)  # this yields much better results
            logging.info('E3 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return self._third_central_moment

    @property
    def fourth_central_moment(self):
        """The fourth central moment of the first passage time distribution."""
        if self._fourth_central_moment is None:
            start_time = time.time()
            # Calculate the third central moment because calculating the third moment directly causes high numerical errors
            # Calculating moments with integrate.quad takes much time
            # self._third_central_moment = integrate.quad(lambda t: (t - self.ev)**4 * self.pdf(t), self.ppf(0.00o05), self.ppf(0.99995))[
            #              0]  # this yields much better results
            self._fourth_central_moment, _, abs_dev, rel_dev = self.compute_moment(
                lambda t: (t - self.ev) ** 3)  # this yields much better results
            logging.info('E4 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return self._fourth_central_moment

    @property
    def fifth_central_moment(self):
        """The fifth central moment of the first passage time distribution."""
        if self._fifth_central_moment is None:
            start_time = time.time()
            # Calculate the third central moment because calculating the third moment directly causes high numerical errors
            # Calculating moments with integrate.quad takes much time
            # self._third_central_moment = integrate.quad(lambda t: (t - self.ev)**5 * self.pdf(t), self.ppf(0.00o05), self.ppf(0.99995))[
            #              0]  # this yields much better results
            self._fifth_central_moment, _, abs_dev, rel_dev = self.compute_moment(
                lambda t: (t - self.ev) ** 5)  # this yields much better results
            logging.info('E5 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return self._fifth_central_moment

    def trans_dens_ppf(self, theta, q=0.9):
        """The PPF of 1 - int ( p(x(dt+theta)| x(theta) = x_predTo), x(dt+theta) = - infty .. x_predTo),
        i.e., the inverse CDF of the event that particles are above x_predTo once they have reached it at time theta.

        Note that in terms of the used approximation, this can be seen as PPF of the approximate first passage
        returning time distribution w.r.t. the boundary x_pred_to.

        :param theta: A float or np.array, the (assumed) time at which x(theta) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.

        :returns: A np.array, the value of the PPF for q and theta, note that this a delta time w.r.t. theta.The value of the PPF for q and theta, note that this a delta time w.r.t. theta.
        """
        # TODO: One could alternatively use scipy.norm's ppf function on self.trans_dens
        F, Q = _get_system_matrices_from_parameters(theta - self.t_L, self.S_w)
        F = F[:3, :3]
        Q = Q[:3, :3]

        cov_theta = np.matmul(np.matmul(F, self.C_L[0:3, 0:3]),
                              np.transpose(F)) + Q

        p_theta_var = cov_theta[1:, 1:] - np.outer(cov_theta[1:, 0], cov_theta[0, 1:]) / cov_theta[0, 0]
        p_theta_mu = np.array([self.x_L[1] + self.x_L[2]*(theta - self.t_L), self.x_L[2]]) \
                     + cov_theta[1:, 0]/cov_theta[0, 0]*(self.x_predTo - self.ev_t(theta))

        qf = norm.ppf(1 - q)
        A = 1 / 20 * self.S_w * qf ** 2
        B = 1 / 4 * (p_theta_var[1, 1] * qf ** 2 - p_theta_mu[1]** 2)
        C = p_theta_var[0, 1] * qf ** 2 - p_theta_mu[0] * p_theta_mu[1]
        D = p_theta_var[0, 0] * qf ** 2 - p_theta_mu[0] ** 2
        E = 0
        F = 0

        # Use np.roots here as it is in general the most stable approach
        dt = np.roots([A, B, C, D])

        # Check all possible solutions
        # Probability of the event that the transition density for x(theta + dt) is higher than x_pred_to.
        trans_fpt_values = np.array([1 - self.trans_density(t, theta).cdf(self.x_predTo) for t in dt])
        valid_roots = dt[np.isclose(trans_fpt_values, q)].real
        positive_roots = valid_roots[valid_roots > 0]
        return positive_roots

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = super().get_statistics()
        hit_stats.update({'RAW_CDF': self._cdf,
                          })
        return hit_stats


class MCHittingTimeModel(CAHittingTimeModel, AbstractMCHittingTimeModel):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem to a distribution
     using scipy.stats.rv_histogram.
    """

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, t_range, bins=100, t_samples=None, name='MC simulation'):
        """Initialize the model.

        :param x_L: A np.array of shape [4] representing the expected value of the initial state. We use index L here
            because it corresponds to the last time we see a particle in our optical belt sorting scenario.
            Format: [pos_x, vel_x, pos_y, vel_y].
        :param C_L: A np.array of shape [4, 4] representing the covariance matrix of the initial state.
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
        if t_samples is None:
            t_samples, _, _ = create_ty_ca_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L)

        super().__init__(x_L=x_L,
                         C_L=C_L,
                         S_w=S_w,
                         x_predTo=x_predTo,
                         t_L=t_L,
                         t_range=t_range,
                         t_samples=t_samples,
                         bins=bins,
                         name=name)

    @property
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        return self.third_moment - 3 * self.ev * self.var - self.ev ** 3

    @property
    def third_moment(self):
        """The third moment of the first passage time distribution."""
        return self._density.moment(3)

    @property
    def fourth_central_moment(self):
        """The fourth central moment of the first passage time distribution."""
        return self.fourth_moment - 4 * self.ev * self.third_moment + 6 * self.ev ** 2 * self.second_moment - 3 * self.ev ** 4

    @property
    def fourth_moment(self):
        """The fourth moment of the first passage time distribution."""
        return self._density.moment(4)

    @property
    def fifth_central_moment(self):
        """The fifth central moment of the first passage time distribution."""
        return self.fifth_moment - 5 * self.ev * self.fourth_moment + 10 * self.ev ** 2 * self.third_moment - 10 * self.ev ** 3 * self.second_moment + 4 * self.ev ** 5

    @property
    def fifth_moment(self):
        """The fifth moment of the first passage time distribution."""
        return self._density.moment(5)


def _get_system_matrices_from_parameters(dt, S_w):
    """Returns the transition matrix (F) and the noise covariance of the transition (Q) of the model.

    Both matrices can be used, e.g., to simulate the discrete-time counterpart of the model.

     Assumed CA state format:

        [pos_x, velo_x, acc_x, pos_y, velo_y, acc_y]

    :param dt: A float, time increment.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.

    :returns:
        F: A np.array of shape [6, 6], the transition matrix.
        Q: A np.array of shape [6, 6], the transition noise covariance matrix.
    """
    F = np.array([[1, dt, dt ** 2 / 2, 0, 0, 0], [0, 1, dt, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, dt ** 2 / 2], [0, 0, 0, 0, 1, dt], [0, 0, 0, 0, 0, 1]])
    Q = S_w * np.array([[pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6, 0, 0, 0],
                        [pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2, 0, 0, 0],
                        [pow(dt, 3) / 6, pow(dt, 2) / 2, dt, 0, 0, 0],
                        [0, 0, 0, pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6],
                        [0, 0, 0, pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2],
                        [0, 0, 0, pow(dt, 3) / 6, pow(dt, 2) / 2, dt]])
    return F, Q


def create_ty_ca_samples_hitting_time(x_L,
                                      C_L,
                                      S_w,
                                      x_predTo,
                                      t_L=0.0,
                                      N=100000,
                                      dt=1 / 1000,
                                      break_after_n_time_steps=1000,
                                      break_min_time=None):
    """Monte Carlo approach to solve the first passage time problem. Propagates particles through the 2D discrete-time
    CA motion model and determines their first passage at x_predTo as well as the location in y at the first passage by
    interpolating the positions between the last time before and the first time after the boundary.

    Note that particles that do not reach the boundary after break_after_n_time_steps time steps are handled with a
    fallback value of max(t_samples) + 1 in the t_samples and np.nan in the y_samples.

    :param x_L: A np.array of shape [6] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, vel_x, acc_x, pos_y, vel_y, acc_y].
    :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, position of the boundary.
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
    t_samples = time_before_arrival
    last_t = - x_before_arrival[x_term, 1] / x_before_arrival[x_term, 2] + np.sign(x_before_arrival[x_term, 2]) * \
             np.sqrt(
                 (x_before_arrival[x_term, 1] / x_before_arrival[x_term, 2]) ** 2 + 2 / x_before_arrival[x_term, 2] * (
                             x_predTo - x_before_arrival[x_term, 0]))
    t_samples[x_term] = time_before_arrival[x_term] + last_t
    t_samples[np.logical_not(x_term)] = int(
        max(t_samples)) + 1  # default value for particles that do not arrive

    y_samples = x_before_arrival[:, 3]
    y_samples[x_term] = x_before_arrival[x_term, 3] + last_t * x_before_arrival[x_term, 4] + 1 / 2 * last_t ** 2 * \
                        x_before_arrival[x_term, 5]
    y_samples[np.logical_not(x_term)] = np.nan  # default value for particles that do not arrive

    return t_samples, y_samples, fraction_of_returns


if __name__ == "__main__":
    app.run(main)

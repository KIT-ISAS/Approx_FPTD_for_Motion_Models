'''
############################################ ca_process.py  ###########################################
Authors: Marcel Reith-Braun (ISAS, marcel.reith-braun@kit.edu), Jakob Thumm
#######################################################################################################
Calculates approximate first passage time distributions for a constant velocity model using different
approaches. Furthermore it provides some methods for calculating the distributions of the y-component of
the process at the first passage, where y is orthogonal to x and indepent of the movement in x.

usage:
 - run docker container - tested with tracksort_neural:2.1.0-gpu-py3 image:
    $ docker run -u $(id -u):$(id -g) \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
            tensorflow/tracksort_neural:2.1.0-gpu-py3
 - within container:
     $   python3 /mnt/ca_process.py \\
requirements:
  - Required packages/tracksort_neural:2.1.0-gpu-py3 image: See corresponding dockerfile.
  - Volume mounts: Specify a path </path/to/repo/> that points to the repo.
'''


from absl import logging
from absl import app
from absl import flags

from abc import ABC, abstractmethod
from timeit import time
import os

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import scipy.integrate as integrate
from scipy.stats import norm
from scipy.stats import uniform
from scipy.misc import derivative

from hitting_time_uncertainty_utils import HittingTimeEvaluator


flags.DEFINE_bool('load_samples', default=False,
                    help='Whether the samples should be loaded from a .npz  file.')
flags.DEFINE_bool('save_samples', default=False,
                    help='Whether the samples should be saved to a .npz  file.')
flags.DEFINE_string('save_path', default='/mnt/ca_with_system_noise_samples.npz',
                    help='The path to save the .npz  file.')
flags.DEFINE_bool('save_results', default=False,
                    help='Whether to save the results.')
flags.DEFINE_string('result_dir', default='/mnt/results/',
                    help='The directory where to save the results.')

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
                   load_samples=FLAGS.load_samples,
                   save_samples=FLAGS.save_samples,
                   save_path=FLAGS.save_path,
                   save_results=FLAGS.save_results,
                   result_dir=FLAGS.result_dir,
                   )


def run_experiment(x_L, C_L, t_L, S_w, x_predTo,
                   t_range=None,
                   y_range=None,
                   load_samples=False,
                   save_samples=False,
                   save_path=None,
                   save_results=False,
                   result_dir=None,
                   no_show=False):
    """Runs an experiment including a comparision with Monte Carlo simulation with the given settings.

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
    :param load_samples: Boolean, whether to load the samples for the Monte Carlo simulation from file.
    :param save_samples: Boolean, whether to save the samples for the Monte Carlo simulation from file.
    :param save_path: String, path where to save the .npz file with the samples (suffix .npz).
    :param save_results: Boolean, whether to save the plots.
    :param result_dir: String, directory where to save the plots.
    :param no_show: Boolean, whether to show the plots (False).
    """
    # Deterministic predictions
    t_predicted = t_L + - x_L[1] / x_L[2] + np.sign(x_L[2]) * \
                  np.sqrt((x_L[1] / x_L[2]) ** 2 + 2 / x_L[2] * (x_predTo - x_L[0]))
    y_predicted = x_L[3] + (t_predicted - t_L)* x_L[4] + 1 / 2 * (t_predicted - t_L) ** 2 * x_L[5]

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
                               get_example_tracks_fn=get_example_tracks(x_L, C_L, S_w),
                               save_results=save_results,
                               result_dir=result_dir,
                               no_show=no_show)

    # Create samples
    if not load_samples:
        t_samples, y_samples = create_ty_ca_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L)
        if save_samples:
            np.savez(save_path, name1=t_samples, name2=y_samples)
            logging.info("Saved samples.")
    else:
        data = np.load(save_path)
        t_samples = data['name1']
        y_samples = data['name2']
    hte.plot_sample_histogram(t_samples)
    hte.plot_sample_histogram(y_samples, x_label='y-Coordinate')

    # Show example tracks and visualize uncertainties over time
    #hte.plot_example_tracks(N=5)
    ev_fn = lambda t: x_L[0] + x_L[1] * (t - t_L) + x_L[2]/ 2 *(t - t_L)**2
    var_fn = lambda t: C_L[0, 0] + (C_L[0, 1] + C_L[1, 0]) * (t - t_L) \
                       + (1 / 2 * C_L[0, 2] + 1 / 2 * C_L[2, 0] + C_L[1, 1]) * (t - t_L) ** 2 \
                       + (1 / 2 * C_L[1, 2] + 1 / 2 * C_L[2, 1]) * (t - t_L) ** 3 \
                       + 1 / 4 * C_L[2, 2] * (t - t_L) ** 4 + 1 / 20 * S_w * (t - t_L) ** 5
    hte.plot_mean_and_stddev_over_time(ev_fn, var_fn, show_example_tracks=True)

    # Set up the approaches
    taylor_model = TaylorHittingTimeModel(x_L, C_L, S_w, x_predTo, t_L)
    approx_model = EngineeringApproxHittingTimeModel(x_L, C_L, S_w, x_predTo, t_L)

    # Results for temporal uncertainties
    print('MAX CDF', approx_model.q_max, approx_model.t_max)
    approx_model.plot_valid_regions(theta=t_predicted, save_results=save_results, result_dir=result_dir, for_paper=True, no_show=no_show)
    #approx_model.plot_valid_regions(save_results=save_results, result_dir=result_dir, for_paper=True, no_show=no_show)
    print('tau_max', approx_model.trans_dens_ppf(t_predicted)[0])
    print('Mass inside invalid region:',
          1 - approx_model.cdf(t_predicted + approx_model.trans_dens_ppf(t_predicted)[0]))

    #approaches_temp_ls = [jakob_model]
    approaches_temp_ls = [taylor_model, approx_model]

    # Plot the quantile functions
    hte.plot_quantile_functions(approaches_temp_ls)
    # Calculate moments and compare the results
    hte.compare_moments_temporal(approaches_temp_ls)
    # Calculate the skewness and compare the results
    hte.compare_skewness_temporal(approaches_temp_ls)
    # Plot histogram of samples and hitting time distributions
    hte.plot_first_hitting_time_distributions(t_samples, approaches_temp_ls, plot_hist_for_all_particles=True)
    hte.plot_fptd_and_paths_in_one(ev_fn, var_fn, t_samples, approaches_temp_ls, plot_hist_for_all_particles=True)


# Approaches to solve problem
class HittingTimeModel(ABC):
    """A base class for the CA hitting time models."""

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name='DefaultHittingTimeName'):
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
        self.name = name
        self.x_L = x_L
        self.C_L = C_L
        self.S_w = S_w
        self.x_predTo = x_predTo
        self.t_L = t_L

        # For properties
        self._ev = None
        self._var = None
        self._third_central_moment = None
        self._fourth_central_moment = None
        self._fifth_central_moment = None
        self._stddev = None
        self._second_moment = None
        self._third_moment = None
        self._fourth_moment = None
        self._fifth_moment = None
        self._skew = None

    @property
    @abstractmethod
    def ev(self):
        """The expected value of the first passage time distribution."""
        # To be overwritten by subclass
        pass

    @property
    @abstractmethod
    def var(self):
        """The variance of the first passage time distribution."""
        # To be overwritten by subclass
        pass

    @property
    @abstractmethod
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        # To be overwritten by subclass
        pass

    @property
    @abstractmethod
    def fourth_central_moment(self):
        """The fourth central moment of the first passage time distribution."""
        # To be overwritten by subclass
        pass

    @property
    @abstractmethod
    def fifth_central_moment(self):
        """The fifth central moment of the first passage time distribution."""
        # To be overwritten by subclass
        pass

    @property
    def stddev(self):
        """The standard deviation of the first passage time distribution."""
        if self._stddev is None:
            self._stddev = np.sqrt(self.var)
        return self._stddev

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

    @abstractmethod
    def pdf(self, t):
        """The first passage time distribution (FPTD).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        # To be overwritten by subclass
        pass

    @abstractmethod
    def cdf(self, t):
        """The CDF of the first passage time distribution.

        :param t: A float or np.array, the time parameter of the distribution.
        """
        # To be overwritten by subclass
        pass

    @abstractmethod
    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        # To be overwritten by subclass
        pass

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

    def plot_quantile_function(self, q_min=0.005, q_max=0.995, save_results=False, result_dir=None, for_paper=True):
        """Plot the quantile function.

        :param q_min: A float, the smallest value of the confidence plot range.
        :param q_max: A float, the highest value of the confidence plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        """
        plot_q = np.arange(q_min, q_max, 0.01)
        plot_quant = [self.ppf(q) for q in plot_q]
        plt.plot(plot_q, plot_quant)
        plt.xlabel('Confidence level')
        plt.ylabel('Time t in s')
        if not for_paper:
            plt.title('Quantile Function (Inverse CDF) for ' + self.name)

        if save_results:
            plt.savefig(result_dir + self.name + '_quantile_function.pdf')
            plt.savefig(result_dir + self.name + '_quantile_function.png')
        plt.show()

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = {}
        hit_stats['PDF'] = self.pdf
        hit_stats['CDF'] = self.cdf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['SKEW'] = self.skew
        return hit_stats


class TaylorHittingTimeModel(HittingTimeModel):
    """A simple Gaussian approximation for the first hitting time problem using a Taylor approximation and error
    propagation.
    """

    __metaclass__ = HittingTimeModel

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name='Taylor approximation'):
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
        super().__init__(x_L, C_L, S_w, x_predTo, t_L, name)

        self._ev = t_L + - x_L[1] / x_L[2] + np.sign(x_L[2]) * \
                        np.sqrt((x_L[1] / x_L[2])**2 + 2/x_L[2]*(x_predTo-x_L[0]))
        dt_p = self._ev - t_L
        x_p = mu_t(dt_p, x_L)

        self._var = (1/x_p[1])**2 * self.var_t(dt_p)

    @property
    def ev(self):
        """The expected value of the first passage time distribution."""
        return self._ev

    @property
    def var(self):
        """The variance of the first passage time distribution."""
        return self._var

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
        """The fiftth central moment of the first passage time distribution."""
        return 0  # Gaussian fifth central moment

    def pdf(self, t):
        """The first passage time distribution (FPTD).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        return norm.pdf(t, loc=self.ev, scale=self.stddev)

    def cdf(self, t):
        """The CDF of the first passage time distribution.

        :param t: A float or np.array, the time parameter of the distribution.
        """
        return norm.cdf(t, loc=self.ev, scale=self.stddev)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        # percent point function / quantile function
        return norm.ppf(q, loc=self.ev, scale=self.stddev)

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = {}
        hit_stats['PDF'] = self.pdf
        hit_stats['CDF'] = self.cdf
        hit_stats['PPF'] = self.ppf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['SKEW'] = self.skew
        return hit_stats


class EngineeringApproxHittingTimeModel(HittingTimeModel):
    """An approximation to the first passage time distribution using the (engineering) assumption that particles
    are unlikely to move back once they have passed the boundary.
    """

    __metaclass__ = HittingTimeModel

    def __init__(self, x_L, C_L, S_w, x_predTo, t_L, name="Engineering approx."):
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
        super().__init__(x_L, C_L, S_w, x_predTo, t_L, name)

        self.q_max, self.t_max = self._get_max_cdf_value_and_location()
        if self.q_max < 0.95:
            logging.warning(self.name + " does not seem to be applicable as max. confidence {} < 0.95.".format(
                round(self.q_max, 2)))

        self.compute_moment = self.get_numerical_moment_integrator()

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
        """Derivative of self._cdf. Can be calculate from the standard Gauss pdf with an argument (x_predTo - ev(t))/sttdev(t) times
        the derivative with respect to t of these argument (chain rule).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        # return derivative(self._cdf, t, dx=1e-6)
        gauss = norm.pdf((self.x_predTo - self.ev_t(t)) / np.sqrt(self.var_t(t)))  # Std. Gauss pdf
        der_ev_t = self.x_L[1] + self.x_L[2]* (t - self.t_L)
        der_var_t = 2*self.C_L[0, 1] + 2*(self.C_L[0, 2] + self.C_L[1, 1])*(t - self.t_L) + \
                    3*self.C_L[1, 2]*(t - self.t_L)**2 + self.C_L[2, 2]*(t - self.t_L)**3 + \
                    1/4*self.S_w*(t - self.t_L)**4
        neg_der_arg = der_ev_t / np.sqrt(self.var_t(t)) + (self.x_predTo - self.ev_t(t))*der_var_t/(2*self.var_t(t)**(3/2)) # negative of the derivative
        return gauss * neg_der_arg

    def pdf(self, t):
        """The first passage time distribution (FPTD).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        t = np.asarray([t]) if np.isscalar(t) else np.asarray(t)
        pdf_value = self._pdf(t)
        pdf_value[t > self.t_max] = 0  # piecewise function
        return pdf_value

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
        """Approach:

              1 - q = int(N(x, mu(t), var(t)), x = -inf .. x_predTo) = PHI ( (x_predTo - mu(t)) / sqrt(var(t))
              PHI^-1(1 -q) = (x_predTo - mu(t)) / sqrt(var(t)) -> solve for t...

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        if q < 0.0 or q > 1.0:
            raise ValueError('Confidence level q must be between 0 and 1.')

        # We compute the ppf for t = t - t_L to simplify calculations.

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

        return t + self.t_L

    def _get_max_cdf_value_and_location(self):
        """Method that finds the maximum of the CDF of the approximation and its location.

        :return:
            q_max: A float, the maximum value of the CDF.
            t_q_max: A float, the time where the CDF visits its maximum.
        """
        # # Search for large t with pdf(t) = 0, start with 2*median
        # t_iter = 2*self._ppf(0.5)
        # while not np.isclose(self.pdf(t_iter), 0, atol=1E-3):
        #     t_iter_new = 1.5*t_iter
        #     if self.pdf(t_iter_new) > 0:
        #         t_iter = t_iter_new
        #     else:
        #         break
        # # Get the CDF Value and the corresponding time
        # q_max = self.cdf(t_iter)
        # # t_q_max = self._ppf(q_max)

        A = -1/40*self.S_w*self.x_L[2]
        B = -1/20*3*self.S_w*self.x_L[1]
        C = 1/40*(10*self.S_w*self.x_predTo - 10*self.S_w*self.x_L[0] + 20*self.C_L[1, 2]*self.x_L[2] - 20*self.C_L[2, 2]*self.x_L[1])
        D = 1/40*((40*self.C_L[1, 1] + 40*self.C_L[0, 2])*self.x_L[2] - 40*self.C_L[1, 2]*self.x_L[1] + 40*self.C_L[2, 2]*self.x_predTo - 40*self.C_L[2, 2]*self.x_L[0])
        E = 1/40*(120*self.C_L[0, 1]*self.x_L[2] + 120*self.C_L[1, 2]*(self.x_predTo - self.x_L[0]))
        F = 1/40*(80*self.C_L[0, 0]*self.x_L[2] + (80*self.C_L[1, 1] + 80*self.C_L[0, 2])*self.x_predTo + (-80*self.C_L[1, 1] - 80*self.C_L[0, 2])*self.x_L[0] + 80*self.x_L[1]*self.C_L[0, 1])
        G = 2*self.x_predTo*self.C_L[0, 1] + 2*self.C_L[0, 0]*self.x_L[1] - 2*self.C_L[0, 1]*self.x_L[0]

        roots = np.roots([A, B, C, D, E, F, G])
        real_roots = roots[np.isreal(roots)].real
        t_q_max = max(real_roots)  # get the highest one

        # Get the CDF Value
        q_max = self._cdf(t_q_max)

        return q_max, t_q_max

    def get_numerical_moment_integrator(self, n=400, t_min=None, t_max=None):
        """Generator that builds a numerical integrator based on Riemann sums.

        :param n: Integer, number of integration points.
        :param t_min: Integer, location of smallest integration point.
        :param t_max:  Integer, location of tallest integration point.

        :return:
            compute_moment: Function that can be used to compute the moments.
        """
        t_min = self.ppf(0.00005) if t_min is None else t_min
        if t_max is None:
            t_max = self.ppf(0.99995) if 0.99995 < self.q_max else self.t_max

        # shared variables
        delta_t = (t_max - t_min) / n
        t_k = np.array([t_min + k * delta_t for k in range(n + 1)])  # shape n + 1
        cdf_tk = self.cdf(t_k)  # shape n + 1
        cdf_tk_plus_one = cdf_tk[1:]
        interval_probs = cdf_tk_plus_one - cdf_tk[:-1]  # shape n

        def compute_moment(fn, abs_tol=1.e-5, rel_tol=1.e-3):
            """Function that computes the moments based on Riemann sums.

            The function computes the moments using the actual probability mass in each bin, which is calculated
            using the CDF of the approximation.

            :param fn: function of which the expected value should be computed. E.g. use lambda t: t for the mean,
                    lambda t: t**2, for the second moment, etc.
            :param abs_tol: A float, represents the absolute tolerance between lower and upper sums. If the error is
                    higher than abs_tol, the function will throw a warning.
            :param rel_tol: A float, represents the relative tolerance between lower and upper sums. If the error is
                    higher than rel_tol, the function will throw a warning.

            :return:
                lower_sum: A float, the moment computed based on lower sums.
                upper_sum: A float, the moment computed based on upper sums.
                abs_dev: A float, the absolute difference between lower and upper sum.
                rel_dev: A float, the relative difference between lower and upper sum.
            """
            fn_t_k = np.array(fn(t_k[:-1]))  # shape n
            fn_t_k_plus_one = np.array(fn(t_k[1:]))  # shape n

            # return lower, upper sum and deviations
            lower_sum = np.dot(interval_probs, fn_t_k)
            upper_sum = np.dot(interval_probs, fn_t_k_plus_one)

            abs_dev = abs(upper_sum - lower_sum)
            rel_dev = abs_dev / max(upper_sum, lower_sum)

            if abs_dev > abs_tol:
                logging.warning(
                    'Absolute Difference between lower and upper some is greater than {}. Try increasing integration points'.format(
                        abs_tol))
            if rel_dev > rel_tol:
                logging.warning(
                    'Relative Difference between lower and upper some is greater than {}. Try increasing integration points'.format(
                        rel_tol))

            return lower_sum, upper_sum, abs_dev, rel_dev

        return compute_moment

    @property
    def ev(self):
        """The expected value of the first passage time distribution."""
        if self._ev is None:
            start_time = time.time()
            # Calculating moments with integrate.quad takes much time
            # TODO: For the integrate.quad method to work, the integration limits need to be chosen as for the compute_moment method
            # self._ev = integrate.quad(lambda t: t * self.pdf(t), self.ppf(0.0005), self.ppf(0.9995))[
            # 0]  # this is a tuple
            self._ev, _, abs_dev, rel_dev = self.compute_moment(lambda t: t)
            print('EV', self._ev)
            print('EV integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return self._ev

    @property
    def var(self):
        """The variance of the first passage time distribution."""
        if self._var is None:
            start_time = time.time()
            # Calculating moments with integrate.quad takes much time
            # self._var = integrate.quad(lambda t: (t - self.ev) ** 2 * self.pdf(t), self.ppf(0.0005), self.ppf(0.9995))[
            #     0]  # this yields much better results
            # self._var = self.compute_moment(lambda t: t**2) - self.ev ** 2 # don't calculate the variance in
            # # this way because it causes high numerical errors
            self._var, _, abs_dev, rel_dev = self.compute_moment(
                lambda t: (t - self.ev) ** 2)  # this yields much better results
            print('Var integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return self._var

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
            print('E3 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
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
            print('E4 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
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
            self._fifth_central_moment,  _, abs_dev, rel_dev = self.compute_moment(lambda t: (t - self.ev)**5)  # this yields much better results
            print('E5 integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
                round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))
        return self._fifth_central_moment

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = {}
        hit_stats['PDF'] = self.pdf
        hit_stats['CDF'] = self.cdf
        hit_stats['PPF'] = self.ppf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['SKEW'] = self.skew
        hit_stats['Median'] = self.ppf(0.5)
        hit_stats['FirstQuantile'] = self.ppf(0.25)
        hit_stats['ThirdQuantile'] = self.ppf(0.75)
        return hit_stats

    def trans_density(self, dt, theta):
        """The transition density p(x(dt+theta)| x(thetha) = x_predTo) from going from x_predTo at time theta to
         x(dt+theta) at time dt+theta.

         Note that in terms of the used approximation, this can be seen as the first returning time to x_predTo after
         a crossing of x_predTo at theta.

         :param dt: A float or np.array, the time difference. dt is zero at time = theta.
         :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to.

         :return: The value of the transition density for the given dt and theta.
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

    def trans_dens_ppf(self, theta, q=0.9):
        """The PPF of 1 - int ( p(x(dt+theta)| x(thetha) = x_predTo), x(dt+theta) = - infty .. x_predTo),
        i.e., the inverse CDF of the event that particles are above x_predTo once they have reached it at time theta.

        Note that in terms of the used approximation, this can be seen as PPF of the approximate first passage
        returning time distribution w.r.t. the boundary x_pred_to.

        :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.

        :return: The value of the PPF for q and theta.
        """
        Phi = lambda dt: np.array([[1, dt, dt ** 2 / 2],
                                   [0, 1, dt],
                                   [0, 0, 1]])

        Q = lambda dt: self.S_w * np.array([[pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6],
                                            [pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2],
                                            [pow(dt, 3) / 6, pow(dt, 2) / 2, dt]])

        cov_theta = np.matmul(np.matmul(Phi(theta - self.t_L), self.C_L[0:3, 0:3]),
                              np.transpose(Phi(theta - self.t_L))) + Q(theta - self.t_L)

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

    def plot_valid_regions(self, theta=None, q=0.95, plot_t_min=0.0, plot_t_max=None, save_results=False, result_dir=None, for_paper=True, no_show=False):
        """Plot the (approximate) probabilities that the track doesn't intersect with x_predTo once it has reached
        it at time theta in dependency on the time difference dt (t = dt + theta) and theta.

        Note that, because of the linear behavior of the mean and the cubic behavior of the variance, there
        are intervals in time for which is it very unlikely (with confidence q) that the track falls below x_pred_to,
        again. These are the desired regions of validity.

        :param theta: A float, the (assumed) time at which x(thetha) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.
        :param plot_t_min: A float, the lower time value of the plot range.
        :param plot_t_max: A float, the upper time value of the plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        :param no_show: Boolean, whether to show the plots (False).
        """
        t_pred = self.t_L - self.x_L[1] / self.x_L[2] + np.sign(self.x_L[2]) * \
                        np.sqrt((self.x_L[1] / self.x_L[2])**2 + 2/self.x_L[2]*(self.x_predTo-self.x_L[0]))
        if theta is None:
            multipliers = np.arange(start=0.85, stop=1.2, step=0.05)
            plot_theta = [t_pred * m for m in multipliers]
        else:
            # We only plot the plot for the given theta
            plot_theta = [theta]

        roots = self.trans_dens_ppf(plot_theta[0], q)  # take the first one, as it gives the largest delta t (at least in the vicinity of t_pred)
        plot_t_max = plot_t_max if plot_t_max is not None else roots[0] * 1.4  # take the largest one, roots are in descending order
        plot_t = np.linspace(plot_t_min, plot_t_max, num=10000)

        # Compute the probability that the tracks stay above x_pred_to
        # This is an approximation since it doesn't take into account tracks that fall below and climb back above
        # x_pred_to before dt.

        # Probability of the event that the transition density for x(theta + dt) is higher than x_pred_to.
        trans_fpt = lambda dt, theta: 1 - self.trans_density(dt, theta).cdf(self.x_predTo)

        fig, ax = plt.subplots()
        plt.hlines(q, xmin=plot_t_min, xmax=plot_t_max, color='black', linestyle='dashdot',
                   label='Confidence q={}'.format(q))

        # Plot the Probabilities
        for i, theta in enumerate(plot_theta):
            plot_prob = [trans_fpt(dt, theta) for dt in plot_t]

            if len(plot_theta) == 1:
                label = None
            else:
                label = r"$\theta$: {} $\cdot$ det. pred.".format(round(multipliers[i], 2))

            if np.isclose(theta, t_pred):
                # Mark the valid regions
                # TODO: The second case is not necessary
                roots = self.trans_dens_ppf(theta, q)
                if roots.shape[0] == 1:
                    plt.axvspan(0, roots[0], alpha=0.6, color='green', label='Valid region')
                if roots.shape[0] == 2:
                    plt.axvspan(roots[1], roots[0], alpha=0.6, color='green', label='Valid region')
                if roots.shape[0] == 3:
                    logging.WARNING('Three valid roots detected.')

            ax.plot(plot_t, plot_prob, label=label)

        plt.ylabel('Confidence')
        plt.xlabel('Time difference in s')
        plt.legend(loc='upper right')

        if not for_paper:
            plt.title('Probability of Staying Above the Boundary')

        if save_results:
            plot_name = "_valid_regions" if len(plot_theta) != 1 else "_valid_region_for_" + str(round(theta, 4))
            basename = os.path.basename(os.path.normpath(result_dir))
            process_name_save = basename.lower().replace(" ", "_")
            plt.savefig(os.path.join(result_dir, process_name_save + plot_name + '.pdf'))
            plt.savefig(os.path.join(result_dir, process_name_save + plot_name + '.png'))
        if not no_show:
            plt.show()
        plt.close()


# Mean and variance of the six-dimensional process
# TODO: Combine the functions so that the matrices are defined only once, do the same with the CV model!
def mu_t(delta_t, mu_0):
    """Mean function of the 2D CA process.
    
    :param delta_t: A float, time difference.
    :param mu_0: A np.array of shape [6] reprenting the mean of the initial state. 
        Format: [pos_x, vel_x, acc_x, pos_y, vel_y, acc_y].
    """
    F = np.array([[1, delta_t, delta_t ** 2 / 2, 0, 0, 0],
                   [0, 1, delta_t, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, delta_t, delta_t ** 2 / 2],
                   [0, 0, 0, 0, 1, delta_t],
                   [0, 0, 0, 0, 0, 1]])
    mu_t = F.dot(mu_0)
    return mu_t

def cov_t(delta_t, cov_0, S_w):
    """Covariance (in depency on t) of the 2D CA process.

    :param delta_t: A float, time difference.
    :param cov_0: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    """
    F = np.array([[1, delta_t, delta_t ** 2 / 2, 0, 0, 0],
                   [0, 1, delta_t, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, delta_t, delta_t ** 2 / 2],
                   [0, 0, 0, 0, 1, delta_t],
                   [0, 0, 0, 0, 0, 1]])
    cov_h_t = np.dot(np.matmul(F, cov_0), F.T)  # homogeneous solution
    cov_p_t = S_w *np.array([[pow(delta_t, 5) / 20, pow(delta_t, 4) / 8, pow(delta_t, 3) / 6, 0, 0, 0],
                              [pow(delta_t, 4) / 8, pow(delta_t, 3) / 3, pow(delta_t, 2) / 2, 0, 0, 0],
                              [pow(delta_t, 3) / 6, pow(delta_t, 2) / 2, delta_t, 0, 0, 0],
                              [0, 0, 0, pow(delta_t, 5) / 20, pow(delta_t, 4) / 8, pow(delta_t, 3) / 6],
                              [0, 0, 0, pow(delta_t, 4) / 8, pow(delta_t, 3) / 3, pow(delta_t, 2) / 2],
                              [0, 0, 0, pow(delta_t, 3) / 6, pow(delta_t, 2) / 2, delta_t]])  # particular solution
    cov_t = cov_h_t + cov_p_t
    return cov_t


def create_ty_ca_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L=0.0, N=100000, dt=1 / 1000, break_after_n_timesteps=1000):
    """Monte Carlo approach to solve the first passage time problem. Propagates particles through the 2D discrete-time
    CA motion model and determines their first passage at x_predTo as well as the location in y at the first passage by
    interpolating the positions between the last time before and the first time after the boundary.

    :param x_L: A np.array of shape [6] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, vel_x, acc_x, pos_y, vel_y, acc_y].
    :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.
    :param x_predTo: A float, position of the boundary.
    :param t_L: A float, the time of the last state/measurement (initial time).
    :param N: Integer, number of samples to use.
    :param dt: A float, time increment.
    :param break_after_n_timesteps: Integer, maximum number of timesteps for the simulation.

    :return:
        t_samples: A np.array of shape [N] containing the first passage times of the particles.
        y_samples: A np.array of shape [N] containing the y-position at the first passage times of the particles.

    Note that particles that do not reach the boundary after break_after_n_timesteps timesteps are handled with a
    fallback value of max(t_samples) + 1 in the t_samples and np.nan in the y_samples.
    """
    samples = np.random.multivariate_normal(mean=x_L, cov=C_L, size=N)
    # Let the samples move to the boundary
    F = np.array([[1, dt, dt ** 2 / 2, 0, 0, 0], [0, 1, dt, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, dt, dt ** 2 / 2], [0, 0, 0, 0, 1, dt], [0, 0, 0, 0, 0, 1]])
    mean_w = np.array([0, 0, 0, 0, 0, 0])
    Q = S_w * np.array([[pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6, 0, 0, 0],
                         [pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2, 0, 0, 0],
                         [pow(dt, 3) / 6, pow(dt, 2) / 2, dt, 0, 0, 0],
                         [0, 0, 0, pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6],
                         [0, 0, 0, pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2],
                         [0, 0, 0, pow(dt, 3) / 6, pow(dt, 2) / 2, dt]])

    # t_samples = np.zeros(N)
    # y_samples = np.zeros(N)
    # for i in range(N):
    #     if i % 1000 == 0: print(i)
    #     x_curr = samples[i]
    #     t = t_L
    #     while True:
    #         x_next = np.array(F.dot(x_curr)[0])[0]
    #         w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=1)[0]
    #         x_next = x_next + w_k
    #         if x_next[0] >= x_predTo:
    #             # Linear interpolation to get time
    #             last_t = - x_curr[1] / x_curr[2] + np.sign(x_curr[2]) * \
    #                      np.sqrt((x_curr[1] / x_curr[2]) ** 2 + 2 / x_curr[2] * (x_predTo - x_curr[0]))
    #             t += last_t
    #             y = x_curr[3] + last_t * x_curr[4] + 1 / 2 * last_t ** 2 * x_curr[5]
    #             break
    #         t += dt
    #         x_curr = x_next
    #     t_samples[i] = t
    #     y_samples[i] = y

    x_curr = samples
    x_term = np.zeros(samples.shape[0], dtype=bool)
    t = t_L
    ind = 0
    time_before_arrival = np.full(N, t_L, dtype=np.float64)
    while True:
        if ind % 100 == 0:
            logging.info('Timestep {0}, x equals approx. {1}'.format(ind, x_curr[0, 0]))
        x_curr_tf = tf.convert_to_tensor(x_curr)
        x_next = tf.linalg.matvec(F, x_curr_tf).numpy()
        w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=N)
        x_next = x_next + w_k
        x_term[x_next[:, 0] >= x_predTo] = True
        if np.all(x_term):
            break
        if ind == break_after_n_timesteps:
            logging.warning('Sampling interrupted because {}. reached. Please adjust break_after_n_timesteps if you want to move the particles more timesteps.'.format(break_after_n_timesteps))
            break
        x_curr[np.logical_not(x_term)] = x_next[np.logical_not(x_term)]
        t += dt
        time_before_arrival[np.logical_not(x_term)] = t
        ind += 1

    # Linear interpolation to get time
    time_of_arrival = time_before_arrival
    last_t = - x_curr[x_term, 1] / x_curr[x_term, 2] + np.sign(x_curr[x_term, 2]) * \
             np.sqrt((x_curr[x_term, 1] / x_curr[x_term, 2]) ** 2 + 2 / x_curr[x_term, 2] * (x_predTo - x_curr[x_term, 0]))
    time_of_arrival[x_term] = time_before_arrival[x_term] + last_t
    y = x_curr[:, 3]
    y[x_term] = x_curr[x_term, 3] + last_t * x_curr[x_term, 4] + 1 / 2 * last_t ** 2 * x_curr[x_term, 5]

    time_of_arrival[np.logical_not(x_term)] = int(
        max(time_of_arrival)) + 1  # default value for particles that do not arrive
    y[np.logical_not(x_term)] = np.nan  # default value for particles that do not arrive

    # last_t = - x_curr[:, 1] / x_curr[:, 2] + np.sign(x_curr[:, 2]) * \
    #          np.sqrt((x_curr[:, 1] / x_curr[:, 2]) ** 2 + 2 / x_curr[:, 2] * (x_predTo - x_curr[:, 0]))
    #
    # time_of_arrival = time_before_arrival + last_t
    # y = x_curr[:, 3] + last_t * x_curr[:, 4] + 1 / 2 * last_t ** 2 * x_curr[:, 5]

    t_samples = time_of_arrival
    y_samples = y

    return t_samples, y_samples


def get_example_tracks(x_L, C_L, S_w):
    """Generator that creates a function for simulation of example tracks. Used for plotting purpose only.

    :param x_L: A np.array of shape [6] representing the expected value of the initial state. We use index L here
        because it corresponds to the last time we see a particle in our optical belt sorting scenario.
        Format: [pos_x, vel_x, acc_x pos_y, vel_y, acc_y].
    :param C_L: A np.array of shape [6, 6] representing the covariance matrix of the initial state.
    :param S_w: A float, power spectral density (psd) of the model. Note that we assume the same psd in x and y.

    :return:
        _get_example_tracks: A function that can be used for simulation of example tracks.
    """

    def _get_example_tracks(plot_t, N=5):
        """Create data (only x-positions) of some tracks.

        :param plot_t: A np.array of shape [n_plot_points], point in time where a point in the plot should be displayed.
            Consecutive points must have the same distance.
        :param N: Integer, number of tracks to create.

        :return:
            x_tracks: A np.array of shape [num_timesteps, N] containing the x-positions of the tracks.
        """
        dt = plot_t[1] - plot_t[0]

        samples = np.random.multivariate_normal(mean=x_L, cov=C_L, size=N)  # [6, N]
        # Let the samples move to the boundary
        F = np.array([[1, dt, dt ** 2 / 2, 0, 0, 0],
                       [0, 1, dt, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, dt, dt ** 2 / 2],
                       [0, 0, 0, 0, 1, dt],
                       [0, 0, 0, 0, 0, 1]])
        mean_w = np.array([0, 0, 0, 0, 0, 0])
        Q = S_w * np.array([[pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6, 0, 0, 0],
                             [pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2, 0, 0, 0],
                             [pow(dt, 3) / 6, pow(dt, 2) / 2, dt, 0, 0, 0],
                             [0, 0, 0, pow(dt, 5) / 20, pow(dt, 4) / 8, pow(dt, 3) / 6],
                             [0, 0, 0, pow(dt, 4) / 8, pow(dt, 3) / 3, pow(dt, 2) / 2],
                             [0, 0, 0, pow(dt, 3) / 6, pow(dt, 2) / 2, dt]])

        tracks = np.expand_dims(samples, axis=2)
        for _ in range(plot_t.size - 1):
            x_curr_tf = tf.convert_to_tensor(tracks[:, :, -1])
            x_next = tf.linalg.matvec(F, x_curr_tf).numpy()
            w_k = np.random.multivariate_normal(mean=mean_w, cov=Q, size=N)
            x_next = np.expand_dims(x_next + w_k, axis=-1)
            tracks = np.concatenate((tracks, x_next), axis=-1)

        x_tracks = tracks[:, 0, :].T  # [N, 4, num_timesteps] -> [num_timesteps, N]

        return x_tracks

    return _get_example_tracks


if __name__ == "__main__":
    app.run(main)

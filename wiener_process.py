'''
#########################################  wiener_process.py  #########################################
Authors: Marcel Reith-Braun (ISAS, marcel.reith-braun@kit.edu), Jakob Thumm
#######################################################################################################
Calculates approximate first passage time distributions for a constant velocity model using different
approaches.

usage:
 - run docker container - tested with tensorflow/approx_fptd:2.8.0-gpu: image:
    $ docker run -u $(id -u):$(id -g) \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
            tensorflow/approx_fptd:2.8.0-gpu:
 - within container:
     $   python3 /mnt/wiener_process.py \\
requirements:
  - Required packages/tensorflow/approx_fptd:2.8.0-gpu image: See corresponding dockerfile.
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

import scipy.integrate as integrate
from scipy.stats import norm
from scipy.misc import derivative

from hitting_time_uncertainty_utils import HittingTimeEvaluator

flags.DEFINE_bool('load_samples', default=False,
                    help='Whether the samples should be loaded from a .npz  file.')
flags.DEFINE_bool('save_samples', default=False,
                    help='Whether the samples should be saved to a .npz  file.')
flags.DEFINE_string('save_path', default='/mnt/wiener_samples.npz',
                    help='The path to save the .npz  file.')
flags.DEFINE_bool('save_results', default=False,
                    help='Whether to save the results.')
flags.DEFINE_string('result_dir', default='/mnt/results/',
                    help='The directory where to save the results.')
flags.DEFINE_bool('sw_fifty', default=False,
                    help='Whether to use Sw=50.')
flags.DEFINE_bool('for_paper', default=False,
                  help='Boolean, whether to use the plots for publication (omit headers, etc.)..')
flags.DEFINE_bool('no_show', default=False,
                  help='Set this to True if you do not want to show evaluation graphics and only save them.')


flags.DEFINE_string('verbosity_level', default='INFO', help='Verbosity options.')
flags.register_validator('verbosity_level',
                         lambda value: value in ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                         message='dataset_type must one of ' + str(['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']))

FLAGS = flags.FLAGS

b = 0  # offset for the initial variance, leave it at zero.

def main(args):
    del args

    ## Setup logging
    logging.set_verbosity(logging.FLAGS.verbosity_level)

    ## Define system parameters
    # System noise
    mu = 600
    sigma = 50 if FLAGS.sw_fifty else 10
    x0 = 29.0304

    ## Nozzle array position
    x_predTo = 62.5

    ## Plot settings
    t_predicted = (x_predTo - x0) / mu
    if FLAGS.sw_fifty:
        t_range = [0, 0.15]  # for sigma = 50
    else:
        t_range = [0.7 * t_predicted, 1.3* t_predicted]
    plot_t = np.arange(t_range[0], t_range[1], 0.00001)

    # Create base class
    hte = HittingTimeEvaluator('Wiener Process with Drift', x_predTo, plot_t, t_predicted,
                               get_example_tracks_fn=get_example_tracks(mu, sigma, x0),
                               save_results=FLAGS.save_results,
                               result_dir=FLAGS.result_dir,
                               for_paper=FLAGS.for_paper,
                               no_show=FLAGS.no_show)

    ## Create samples
    if not FLAGS.load_samples:
        t_samples = create_wiener_samples_hitting_time(mu, sigma, x0, x_predTo, N=500000, dt=1/20000, break_after_n_timesteps=3000)
        #t_samples = create_wiener_samples_hitting_time(mu, sigma, x0, x_predTo)
        if FLAGS.save_samples:
            np.savez(FLAGS.save_path, name1=t_samples)
            logging.info("Saved samples.")
    else:
        # data = np.load('/mnt/ty_samples_with_system_noise_new.npz')
        data = np.load(FLAGS.save_path)
        t_samples = data['name1']
    hte.plot_sample_histogram(t_samples)

    # Show example tracks and visualize uncertainties over time
    # hte.plot_example_tracks(N=5)
    ev_fn = lambda t: x0 + mu*t
    var_fn = lambda t: sigma**2*t + b
    hte.plot_mean_and_stddev_over_time(ev_fn, var_fn, show_example_tracks=True)

    # Setup the approaches
    analytic_model = AnalyticHittingTimeModel(mu, sigma, x0, x_predTo)
    approx_model = EngineeringApproxHittingTimeModel(mu, sigma, x0, x_predTo)

    # Plot valid regions
    approx_model.plot_valid_regions(theta=t_predicted, save_results=FLAGS.save_results, result_dir=FLAGS.result_dir, for_paper=True, no_show=FLAGS.no_show)
    # no difference to the one before as the process is Markov
    # approx_model.plot_valid_regions(save_results=FLAGS.save_results, result_dir=FLAGS.result_dir, for_paper=True, no_show=FLAGS.no_show)
    print('Mass inside invalid region:', approx_model.cdf(t_predicted + approx_model.trans_dens_ppf()) - approx_model.cdf(t_predicted))

    # Results for temporal uncertainties
    approaches_temp_ls = [analytic_model, approx_model]
    # Calculate moments and compare the results
    hte.compare_moments_temporal(approaches_temp_ls)
    # Plot quantile functions
    hte.plot_quantile_functions(approaches_temp_ls)
    # Plot histogram of samples and hitting time distributions
    hte.plot_first_hitting_time_distributions(t_samples, approaches_temp_ls, plot_hist_for_all_particles=True)


class HittingTimeModel(ABC):
    """A base class for the Wiener process with drift hitting time models."""

    def __init__(self, mu, sigma, x0, x_predTo, name='DefaultName'):
        """Initialize the model.

        :param mu: A float, the "velocity" (drift) of the Wiener process with drift
        :param sigma: A float, the diffusion constant of the Wiener process with drift
        :param x0: A float, the starting position x(t=0) of the process.
        :param x_predTo: A float, position of the boundary.
        :param name: String, (default) name for the model.
        """
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.x_predTo = x_predTo

       # For properties
        self._ev = None
        self._var = None
        self._ev_third = None
        self._stddev = None

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
    def stddev(self):
        """The standard deviation of the first passage time distribution."""
        if self._stddev is None:
            self._stddev = np.sqrt(self.var)
        return self._stddev

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
        """The mean function of the CV motion model in x.

        :param t: A float or np.array, the time parameter of the mean function.
        """
        return self.mu * t + self.x0

    def var_t(self, t):
        """The variance function of the CV motion model in x.

        :param t: A float or np.array, the time parameter of the variance function.
        """
        return self.sigma**2*t + b

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
        return hit_stats


class AnalyticHittingTimeModel(HittingTimeModel):
    """The analytic to the first passage time problem.

    Inverse Gaussian distribution, see e.g. Whitmore and Seshadri (1987).
    """
    __metaclass__ = HittingTimeModel

    def __init__(self, mu, sigma, x0, x_predTo, name='Analytic solution'):
        """Initialize the model.

        :param mu: A float, the "velocity" (drift) of the Wiener process with drift
        :param sigma: A float, the diffusion constant of the Wiener process with drift
        :param x0: A float, the starting position x(t=0) of the process.
        :param x_predTo: A float, position of the boundary.
        :param name: String, name for the model.
        """
        super().__init__(mu, sigma, x0, x_predTo, name)
        self._ev = (self.x_predTo - self.x0)/ self.mu
        self.lambdaa = (self.x_predTo - self.x0)**2/self.sigma**2
        self._var = self.ev**3 / self.lambdaa

    def pdf(self, t):
        """The first passage time distribution (FPTD).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        return np.sqrt(self.lambdaa/(2*np.pi*t**3)) * np.exp(-self.lambdaa*(t - self.ev)**2/(2*self.ev**2*t))

    def cdf(self, t):
        """The CDF of the first passage time distribution.

        :param t: A float or np.array, the time parameter of the distribution.
        """
        std_gauss = norm(loc=0, scale=1)
        z1 = np.sqrt(self.lambdaa/t)*(t/self.ev - 1)
        z2 = - np.sqrt(self.lambdaa / t) * (t / self.ev + 1)
        return std_gauss.cdf(z1) + np.exp(2*self.lambdaa/self.ev) * std_gauss.cdf(z2)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

         :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
         """
        raise NotImplementedError()

    @property
    def ev(self):
        """The expected value of the first passage time distribution."""
        return self._ev

    @property
    def var(self):
        """The variance of the first passage time distribution."""
        return self._var


class EngineeringApproxHittingTimeModel(HittingTimeModel):
    """An approximation to the first passage time distribution using the (engineering) assumption that particles
    are unlikely to move back once they have passed the boundary.
    """

    __metaclass__ = HittingTimeModel

    def __init__(self, mu, sigma, x0, x_predTo, name='Engineering approx.'):
        """Initialize the model.

        :param mu: A float, the "velocity" (drift) of the Wiener process with drift
        :param sigma: A float, the diffusion constant of the Wiener process with drift
        :param x0: A float, the starting position x(t=0) of the process.
        :param x_predTo: A float, position of the boundary.
        :param name: String, name for the model.
        """
        super().__init__(mu, sigma, x0, x_predTo, name)

        self.compute_moment = self.get_numerical_moment_integrator()
        self.compute_moment_riemann = self.get_numerical_moment_integrator(use_cdf=False)

    def cdf(self, t):
        """The CDF of the first passage time distribution.

        Approach:
            1 - int( p(x(t), x= -infty .. x_predTo )

        :param t: A float or np.array, the time parameter of the distribution.
        """
        return 1 - norm.cdf(self.x_predTo, loc=self.ev_t(t), scale=np.sqrt(self.var_t(t)))

    def pdf(self, t):
        """The first passage time distribution (FPTD).

        Derivative of self._cdf. Can be calculate from the standard Gauss pdf with an argument (x_predTo - ev(t))/sttdev(t) times
        the derivative with respect to t of these argument (chain rule).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        # return derivative(self.cdf, t, dx=1e-6)
        der_ev = self.mu
        der_var = self.sigma**2
        mulipl = der_ev/np.sqrt(self.var_t(t)) + (self.x_predTo - self.ev_t(t))*der_var/(2*self.var_t(t)**(3.0/2.0))
        return mulipl*1/np.sqrt(2*np.pi)*np.exp(-(self.x_predTo - self.ev_t(t))**2/(2*self.var_t(t)))

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

          Approach:
                1 - q = int(N(x, mu(t), var(t)), x = -inf .. x_predTo) = PHI ( (x_predTo - mu(t)) / sqrt(var(t))
                PHI^-1(1 -q) = (x_predTo - mu(t)) / sqrt(var(t)) -> solve for t...

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        if q < 0.0 or q > 1.0:
            raise ValueError('Confidence level q must be between 0 and 1.')

        # quadratic function
        # t**2 + p*t + qq = 0
        qf = norm.ppf(1 - q)  # Standard-Gaussian quantile function

        p = - (2*(self.x_predTo - self.x0)/self.mu + qf**2*self.sigma**2/self.mu**2)
        qq = (self.x_predTo - self.x0)**2/self.mu**2  # p**2 > qq everywhere!

        t_1 = - p/2 + np.sqrt((p/2)**2  - qq)
        t_2 = - p / 2 - np.sqrt((p / 2) ** 2 - qq)

        # Function must be positive for all confidence levels (because t is starting at 0),
        # but we have a sign shift at cl=0.5. Thus:
        t = t_1 if q > 0.5 else t_2

        return t

    def trans_density(self, dt, theta):
        """"The transition density p(x(dt+theta)| x(thetha) = x_predTo) from going from x_predTo at time theta to
        x(dt+theta) at time dt+theta.

        Note that in terms of the used approximation, this can be seen as the first returning time to x_predTo after
        a crossing of x_predTo at theta.

        Depends on the time difference dt, not on theta since the Wiener process with drift is a 1D Markov process.

        :param dt: A float or np.array, the time difference. dt is zero at time = theta.
        :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to.

        :return: The value of the transition density for the given dt and theta.
        """

        trans_mu = self.x_predTo + self.mu*dt
        trans_var = self.sigma ** 2 * dt
        return norm(loc=trans_mu, scale=np.sqrt(trans_var))

    def trans_dens_ppf(self, theta=None, q=0.95):
        """The PPF of 1 - int ( p(x(dt+theta)| x(thetha) = x_predTo), x(dt+theta) = - infty .. x_predTo),
        i.e., the inverse CDF of the event that particles are above x_predTo once they have reached it at time theta.

        Note that in terms of the used approximation, this can be seen as PPF of the approximate first passage
        returning time distribution w.r.t. the boundary x_pred_to.

        Depends on the time difference dt, not on theta since the Wiener process with drift is a 1D Markov process.

        :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.

        :return: The value of the PPF for q and theta.
        """

        # t**2 + p*t + qq = 0
        qf = norm.ppf(1 - q)
        p = - qf ** 2 * self.sigma ** 2 / self.mu ** 2
        # qq = 0

        # t = 0 is not a valid solution
        t = -p

        return t

    def plot_valid_regions(self, theta=None, q=0.95,
                           plot_t_min=0.0, plot_t_max=None,
                           save_results=False, result_dir=None,
                           for_paper=True, no_show=False):
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
        t_pred = (self.x_predTo - self.x0) / self.mu
        if theta is None:
            multipliers = np.arange(start=0.4, stop=1.8, step=0.2)
            plot_theta = [t_pred * m for m in multipliers]
        else:
            # We only plot the plot for the given theta
            plot_theta = [theta]

        root = self.trans_dens_ppf(plot_theta[0], q)  # take the first one, as it gives the largest delta t (at least in the vicinity of t_pred)
        plot_t_max = plot_t_max if plot_t_max is not None else root * 1.4  # take the largest one, roots are in descending order
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
                label = r"$\theta$: {} $\cdot$ det. pred.".format(round(multipliers[i], 1))

            if np.isclose(theta, t_pred):
                # Mark the valid regions
                root = self.trans_dens_ppf(theta, q)
                plt.axvspan(root, plot_t_max, alpha=0.6, color='green', label='Valid region')

            ax.plot(plot_t, plot_prob, label=label)

        ax.set_xlim(0, None)
        plt.ylabel('Confidence')
        plt.xlabel('Time difference in s')
        plt.legend(loc='upper left')

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

    def get_numerical_moment_integrator(self, n=400, t_min=None, t_max=None, use_cdf=True):
        """Generator that builds a numerical integrator based on Riemann sums.

        :param n: Integer, number of integration points.
        :param t_min: Integer, location of smallest integration point.
        :param t_max:  Integer, location of tallest integration point.
        :param use_cdf: Boolean, whether to use left/right Riemann sums to the PDF(False) or the CDF (recommended).
            In symmetric cases, using left/right sums yields the same results. This can be solved by using the
            infimum / supremum in each bin, but this makes things more complicated and thus is not implemented yet.

        :return:
            compute_moment: Function that can be used to compute the moments.
        """
        # fast numerical integrator
        t_min = self.ppf(0.00005) if t_min is None else t_min   # 0,001
        t_max = self.ppf(0.99995) if t_max is None else t_max  # 0,999

        # shared variables
        delta_t = (t_max - t_min) / n
        t_k = np.array([t_min + k * delta_t for k in range(n + 1)])  # shape n + 1

        if use_cdf:
            cdf_tk = self.cdf(t_k)  # shape n + 1
            cdf_tk_plus_one = cdf_tk[1:]
            interval_probs = cdf_tk_plus_one - cdf_tk[:-1]  # shape n
        else:
            # use pdf, Riemann sums
            pdf_tk = self.pdf(t_k)  # shape n + 1

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
            if use_cdf:
                lower_sum = np.dot(interval_probs, fn_t_k)
                upper_sum = np.dot(interval_probs, fn_t_k_plus_one)
            else:
                # in many scenariaos there is no difference between lower and upper sum (because the pdf at both ends is close to 0)
                lower_sum = delta_t * np.dot(pdf_tk[:-1], fn_t_k)
                upper_sum = delta_t * np.dot(pdf_tk[1:], fn_t_k_plus_one)

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
            # self._ev = integrate.quad(lambda t: t * self.pdf(t), self.ppf(0.0005), self.ppf(0.9995))[
            # 0]  # this is a tuple
            self._ev, _, abs_dev, rel_dev = self.compute_moment(lambda t: t)
            print('EV', self._ev)
            print('EV integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}, '.format(round(1000*(time.time() - start_time), 4), abs_dev, rel_dev))

            # Just to compare the two approaches..
            start_time = time.time()
            # Calculating moments with integrate.quad takes much time
            # self._ev = integrate.quad(lambda t: t * self.pdf(t), self.ppf(0.0005), self.ppf(0.9995))[
            # 0]  # this is a tuple
            _ev, _, abs_dev, rel_dev = self.compute_moment_riemann(lambda t: t)
            print('EV (Riemann)', _ev)
            print('EV integration time (Riemann): {0}ms. Abs dev: {1}, Rel. dev: {2}, '.format(
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
            self._var, _, abs_dev, rel_dev = self.compute_moment(lambda t: (t - self.ev) ** 2)  # this yields much better results
            print('Stddev', np.sqrt(self._var))
            print('Var integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(round(1000*(time.time() - start_time), 4), abs_dev, rel_dev))

            # Just to compare the two approaches..
            start_time = time.time()
            # Calculating moments with integrate.quad takes much time
            # self._ev = integrate.quad(lambda t: t * self.pdf(t), self.ppf(0.0005), self.ppf(0.9995))[
            # 0]  # this is a tuple
            _var, _, abs_dev, rel_dev = self.compute_moment_riemann(lambda t: t)
            print('Stddev (Riemann)', np.sqrt(_var))
            print('VAR integration time (Riemann): {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(
            round(1000 * (time.time() - start_time), 4), abs_dev, rel_dev))

        return self._var

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = {}
        hit_stats['PDF'] = self.pdf
        hit_stats['CDF'] = self.cdf
        hit_stats['PPF'] = self.ppf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        hit_stats['Median'] = self.ppf(0.5)
        hit_stats['FirstQuantile'] = self.ppf(0.25)
        hit_stats['ThirdQuantile'] = self.ppf(0.75)
        return hit_stats


def create_wiener_samples_hitting_time(mu, sigma, x0, x_predTo, N=100000, dt=1 / 1000, break_after_n_timesteps=1000):
    """Monte Carlo approach to solve the first passage time problem. Propagates particles through the process model
    and determines their first passage at x_predTo by interpolating the positions between the last time before and
    the first time after the boundary.

    :param mu: A float, the "velocity" (drift) of the Wiener process with drift
    :param sigma: A float, the diffusion constant of the Wiener process with drift
    :param x0: A float, the starting position x(t=0) of the process.
    :param x_predTo: A float, position of the boundary.
    :param dt: A float, time increment.
    :param N: Integer, number of samples to use.
    :param break_after_n_timesteps: Integer, maximum number of timesteps for the simulation.

    :return:
        t_samples: A np.array of shape [N] containing the first passage times of the particles.

    Note that particles that do not reach the boundary after break_after_n_timesteps timesteps are handled with a
    fallback value of max(t_samples) + 1.
    """
    samples = x0 + np.random.normal(loc=0, scale=np.sqrt(b), size=N)
    # Let the samples move to the nozzle array

    mu_discrete = mu * dt
    sigma_discrete = sigma *np.sqrt(dt)

    x_curr = samples
    x_term = np.zeros(samples.shape[0], dtype=bool)
    t = 0
    ind = 0
    time_before_arrival = np.full(N, 0, dtype=np.float64)
    while True:
        if ind % 100 == 0:
            logging.info('Timestep {0}, x equals approx. {1}'.format(ind, x_curr[0]))
        w_k = np.random.normal(loc=mu_discrete, scale=sigma_discrete, size=N)
        #w_k = np.random.normal(loc=mu, scale=sigma, size=N)
        x_next = x_curr + w_k
        x_term[x_next >= x_predTo] = True
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
    v_interpolated = (x_next[x_term] - x_curr[x_term]) / dt
    last_t = (x_predTo - x_curr[x_term]) / v_interpolated
    time_of_arrival = time_before_arrival
    time_of_arrival[x_term] = time_before_arrival[x_term] + last_t

    time_of_arrival[np.logical_not(x_term)] = int(
        max(time_of_arrival)) + 1  # default value for particles that do not arrive

    t_samples = time_of_arrival

    return t_samples


def get_example_tracks(mu, sigma, x0):
    """Generator that creates a function for simulation of example tracks. Used for plotting purpose only.

    :param mu: A float, the "velocity" (drift) of the Wiener process with drift
    :param sigma: A float, the diffusion constant of the Wiener process with drift
    :param x0: A float, the starting position x(t=0) of the process.

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
        mu_discrete = mu * dt
        sigma_discrete = sigma * np.sqrt(dt)

        samples = x0 + np.random.normal(loc=0, scale=np.sqrt(b), size=N)
        # Let the samples move to the nozzle array

        tracks = np.expand_dims(samples, axis=1)
        for _ in range(plot_t.size - 1):
            w_k = np.random.normal(loc=mu_discrete, scale=sigma_discrete, size=N)
            x_next = np.expand_dims(tracks[:, -1] + w_k, axis=-1)
            tracks = np.concatenate((tracks,  x_next), axis=-1)

        x_tracks = tracks.T

        return x_tracks

    return _get_example_tracks


if __name__ == "__main__":
    app.run(main)

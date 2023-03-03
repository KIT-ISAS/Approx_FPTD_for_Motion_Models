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
from scipy.stats import norm, invgauss
from scipy.misc import derivative

from hitting_time_uncertainty_utils import HittingTimeEvaluator
from abstract_distributions import AbstractHittingTimeModel, AbstractEngineeringApproxHittingTimeModel, \
    AbstractMCHittingTimeModel
from samplers import create_hitting_time_samples


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
    dt = 1 / 20000
    if not FLAGS.load_samples:
        t_samples, _, fraction_of_returns = create_wiener_samples_hitting_time(mu,
                                                                               sigma,
                                                                               x0,
                                                                               x_predTo,
                                                                               N=500000,
                                                                               dt=dt,
                                                                               break_after_n_time_steps=3000)
        #t_samples = create_wiener_samples_hitting_time(mu, sigma, x0, x_predTo)
        if FLAGS.save_samples:
            np.savez(FLAGS.save_path, name1=t_samples)
            logging.info("Saved samples.")
    else:
        # data = np.load('/mnt/ty_samples_with_system_noise_new.npz')
        data = np.load(FLAGS.save_path)
        t_samples = data['name1']
        fraction_of_returns = None  # TODO
    # hte.plot_sample_histogram(t_samples)

    # Show example tracks and visualize uncertainties over time
    hte.plot_example_tracks(N=5)
    ev_fn = lambda t: x0 + mu*t
    var_fn = lambda t: sigma**2*t + b
    hte.plot_mean_and_stddev_over_time(ev_fn, var_fn, show_example_tracks=True)

    # Setup the approaches
    analytic_model = AnalyticHittingTimeModel(mu, sigma, x0, x_predTo)
    approx_model = EngineeringApproxHittingTimeModel(mu, sigma, x0, x_predTo)

    # Plot valid regions
    approx_model.plot_valid_regions(theta=t_predicted, save_results=FLAGS.save_results, result_dir=FLAGS.result_dir, for_paper=True, no_show=FLAGS.no_show)
    # no difference to the one before as the process is Markov
    approx_model.plot_valid_regions(save_results=FLAGS.save_results, result_dir=FLAGS.result_dir, for_paper=True, no_show=FLAGS.no_show)
    print('Mass inside invalid region:', approx_model.cdf(t_predicted + approx_model.trans_dens_ppf()) - approx_model.cdf(t_predicted))  # TODO: Das ist anders wie bei en anderen, warum?
    # TODO: DIe ganzen vergleiche einfügen


    # Results for temporal uncertainties
    approaches_temp_ls = [analytic_model, approx_model]
    # Calculate moments and compare the results
    hte.compare_moments_temporal(approaches_temp_ls)
    # Plot quantile functions
    hte.plot_quantile_functions(approaches_temp_ls)
    # Plot histogram of samples and hitting time distributions
    hte.plot_first_hitting_time_distributions(t_samples, approaches_temp_ls, plot_hist_for_all_particles=True)
    # TODO: in one ergänzen
    # hte.plot_returning_probs_from_fptd_histogram(ev_fn, var_fn, t_samples, approaches_temp_ls,
    #                                              t_range=[t_range[0], 3 * t_range[1]])  # this is too noisy
    # for this, we require longer samples paths, but we can use a reduced time resolution
    hte.plot_returning_probs_from_sample_paths(fraction_of_returns, dt, approaches_temp_ls,
                                               t_range=[t_range[0], 3 * t_range[1]])


class WienerHittingTimeModel(AbstractHittingTimeModel, ABC):
    """A base class for the Wiener process with drift hitting time models."""

    def __init__(self, mu, sigma, x0, x_predTo, name='Wiener hitting time model', **kwargs):
        """Initialize the model.

        :param mu: A float, the "velocity" (drift) of the Wiener process with drift
        :param sigma: A float, the diffusion constant of the Wiener process with drift
        :param x0: A float, the starting position x(t=0) of the process.
        :param x_predTo: A float, position of the boundary.
        :param name: String, (default) name for the model.
        """
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0

        super().__init__(x_predTo=x_predTo,
                         t_L=0,  # methods only support t_L = 0
                         name=name,
                         **kwargs)

    def ev_t(self, t):
        """The mean function of the Wiener motion model in x.

        :param t: A float or np.array, the time parameter of the mean function.
        """
        return self.mu * t + self.x0

    def var_t(self, t):
        """The variance function of the Wiener motion model in x.

        :param t: A float or np.array, the time parameter of the variance function.
        """
        return self.sigma**2*t + b


class AnalyticHittingTimeModel(WienerHittingTimeModel):
    """The analytic to the first passage time problem.

    Inverse Gaussian distribution, see, e.g., Whitmore and Seshadri (1987).
    """

    def __init__(self, mu, sigma, x0, x_predTo, name='Analytic solution'):
        """Initialize the model.

        :param mu: A float, the "velocity" (drift) of the Wiener process with drift
        :param sigma: A float, the diffusion constant of the Wiener process with drift
        :param x0: A float, the starting position x(t=0) of the process.
        :param x_predTo: A float, position of the boundary.
        :param name: String, name for the model.
        """
        super().__init__(mu=mu,
                         sigma=sigma,
                         x0=x0,
                         x_predTo=x_predTo,
                         name=name)

        self._ev = (self.x_predTo - self.x0) / self.mu
        self.lambdaa = (self.x_predTo - self.x0) ** 2 / self.sigma ** 2
        self._var = self.ev ** 3 / self.lambdaa
        # Note that alternatively, we could use scipy.stats.invgauss(self_.ev/lambdaa, scale=self.lambdaa), see
        # https://github.com/scipy/scipy/issues/4654

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
        return invgauss(self._ev/self.lambdaa, scale=self.lambdaa).ppf(q)

    @property
    def ev(self):
        """The expected value of the first passage time distribution."""
        return self._ev

    @property
    def var(self):
        """The variance of the first passage time distribution."""
        return self._var

    def trans_density(self, dt, theta=None):   # TODO: Ab hier die nächsten Methoden löschen, wenn nicht mehr gebraucht
        """"The transition density p(x(dt+theta)| x(thetha) = x_predTo) from going from x_predTo at time theta to
        x(dt+theta) at time dt+theta.

        Note that in terms of the used approximation, this can be seen as the first returning time to x_predTo after
        a crossing of x_predTo at theta.

        Depends on the time difference dt, not on theta since the Wiener process with drift is a 1D Markov process.

        :param dt: A float or np.array, the time difference. dt is zero at time = theta.
        :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to. Not required for this
            model.

        :return: The value of the transition density for the given dt and theta.
        """
        trans_mu = self.x_predTo + self.mu*dt
        trans_var = self.sigma ** 2 * dt
        return norm(loc=trans_mu, scale=np.sqrt(trans_var))

    def returning_probs(self, t, num_samples=1000, deterministic_samples=True, mc_hitting_time_model=None):  # TODO: Die Funktion funktioniert nicht richtig, hier ist (mathematisch, implementierung) was falsch
        """Calculates approximate returning probabilities based on a numerical integration (MC integration) based on
        samples from the approximate first passage time distribution (using inverse transform sampling).

        Approach:

         P(t < T_a , x(t) < a) = int_{t_L}^t fptd(theta) P(x(t) < a | x(theta) = a) d theta

                               ≈ 1 / N sum_{theta_i} P(x(t) < a | x(theta_i) = a) ,  theta_i samples from the
                                    approximation (N samples in total) in [t_L, t].

          with theta the time where x(theta) = a.

        :param t:  # TODO
        :param num_samples: An integer, the number of samples to approximate the integral.
        :param deterministic_samples: A Boolean, whether to use random samples (False) or deterministic samples (True).

        :returns: An approximation for the probability P(t < T_a , x(t) < a), i.e., the probability that a sample path
            has crossed the boundary at a time theta < t, but is smaller than the boundary at time t.
        """
        q_max_to_use = self.cdf(t)

        if not deterministic_samples:
            q_samples = np.random.uniform(low=0, high=q_max_to_use, size=num_samples)
        else:
            # low=0, high=1, num_samples=5 -> [0.16, 0.33, 0.5, 0.67, 0.83]
            q_samples = np.linspace(0, q_max_to_use, num=num_samples + 1, endpoint=False)[1:]  # TODO: passt das?

        theta_samples = [self.ppf(q) for q in q_samples]
        # theta_samples = [mc_hitting_time_model.ppf(q) for q in q_samples]

        return np.nanmean(
            [self.trans_density(dt=t - theta, theta=theta).cdf(self.x_predTo) for theta in theta_samples])
        # return np.nanmean(self.trans_density(dt=t - theta_samples, theta=theta_samples).cdf(self.x_predTo))  # TODO: Parallelisierung ermöglichen (scipy funktion)

    def returning_probs_uniform_samples(self, t, num_samples=1000, deterministic_samples=True, mc_hitting_time_model=None):
        """Calculates approximate returning probabilities based on a numerical integration (MC integration) based on
        samples from a uniform distribution.

        Approach:

         P(t < T_a , x(t) < a) = int_{t_L}^t fptd(theta) P(x(t) < a | x(theta) = a) d theta

                               ≈  (t - t_L) / N sum_{theta_i} FPTD(theta_i) * P(x(t) < a | x(theta_i) = a) ,  theta_i
                                    samples from a uniform distribution (N samples in total) in [t_L, t].

          with theta the time where x(theta) = a.

        :param t: TODO
        :param num_samples: An integer, the number of samples to approximate the integral.
        :param deterministic_samples: A Boolean, whether to use random samples (False) or deterministic samples (True).

        :returns: An approximation for the probability P(t < T_a , x(t) < a), i.e., the probability that a sample path
            has crossed the boundary at a time theta < t, but is smaller than the boundary at time t.
        """
        if not deterministic_samples:
            theta_samples = np.random.uniform(low=self.t_L, high=t, size=num_samples)
        else:
            # low=0, high=1, num_samples=5 -> [0.16, 0.33, 0.5, 0.67, 0.83]
            theta_samples = np.linspace(self.t_L, t, num=num_samples + 1, endpoint=False)[1:]  # TODO: passt das?

        # return (t - self.t_L) * np.nanmean(
        #     [mc_hitting_time_model.pdf(theta) * self.trans_density(dt=t - theta, theta=theta).cdf(self.x_predTo) for
        #      theta in theta_samples])

        return (t - self.t_L) * np.nanmean(
            [self.pdf(theta) * self.trans_density(dt=t - theta, theta=theta).cdf(self.x_predTo) for
             theta in theta_samples])
        # return (t - self.t_L) * np.nanmean(
        #     self.pdf(theta_samples) * self.trans_density(dt=t - theta_samples, theta=theta_samples).cdf(self.x_predTo)) # TODO: Parallelisierung ermöglichen (scipy funktion)

    def returning_probs_integrate_quad(self, t):
        # TODO
        fn = lambda theta: self.pdf(theta) * self.trans_density(dt=t - theta, theta=theta).cdf(self.x_predTo)
        a = np.finfo(np.float64).eps if self.t_L == 0 else self.t_L  # TODO: Braucht man das generell?
        return integrate.quad(fn, a=a, b=t)[0]  # this is a tuple

    def true_returning_probs(self, t):
        return self.cdf(t) + norm.cdf(self.x_predTo, loc=self.ev_t(t), scale=np.sqrt(self.var_t(t))) - 1.0

    def get_statistics(self):
        hit_stats = super().get_statistics()
        hit_stats.update({# 'ReturningProbs': self.returning_probs,  # do not use
                          # 'ReturningProbs': self.returning_probs_uniform_samples,
                          'ReturningProbs': self.returning_probs_integrate_quad,
                          'TrueReturningProbs': self.true_returning_probs,
                          })
        return hit_stats


class EngineeringApproxHittingTimeModel(WienerHittingTimeModel, AbstractEngineeringApproxHittingTimeModel):
    """An approximation to the first passage time distribution using the (engineering) assumption that particles
    are unlikely to move back once they have passed the boundary.
    """

    def __init__(self, mu, sigma, x0, x_predTo, name='No-return approx.'):
        """Initialize the model.

        :param mu: A float, the "velocity" (drift) of the Wiener process with drift
        :param sigma: A float, the diffusion constant of the Wiener process with drift
        :param x0: A float, the starting position x(t=0) of the process.
        :param x_predTo: A float, position of the boundary.
        :param name: String, name for the model.
        """
        super().__init__(mu=mu,
                         sigma=sigma,
                         x0=x0,
                         x_predTo=x_predTo,
                         name=name)

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

    def trans_density(self, dt, theta=None):
        """"The transition density p(x(dt+theta)| x(thetha) = x_predTo) from going from x_predTo at time theta to
        x(dt+theta) at time dt+theta.

        Note that in terms of the used approximation, this can be seen as the first returning time to x_predTo after
        a crossing of x_predTo at theta.

        Depends on the time difference dt, not on theta since the Wiener process with drift is a 1D Markov process.

        :param dt: A float or np.array, the time difference. dt is zero at time = theta.
        :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to. Not required for this
            model.

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

        :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to. ot required for this
            model.
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

    def _get_max_cdf_value_and_location(self):
        q_max = 1  # for the Wiener model, there is no maximum of the approximation
        t_max = np.infty  # for the Wiener model, there is no maximum of the approximation
        return q_max, t_max


class MCHittingTimeModel(WienerHittingTimeModel, AbstractMCHittingTimeModel):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem to a distribution
     using scipy.stats.rv_histogram.

    """

    def __init__(self, mu, sigma, x0, x_predTo, bins=100, name='MC simulation'):
        """Initialize the model.

        :param mu: A float, the "velocity" (drift) of the Wiener process with drift
        :param sigma: A float, the diffusion constant of the Wiener process with drift
        :param x0: A float, the starting position x(t=0) of the process.
        :param x_predTo: A float, position of the boundary.
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param name: String, name for the model.
        """
        t_samples, _, _ = create_wiener_samples_hitting_time(mu,
                                                             sigma,
                                                             x0,
                                                             x_predTo,
                                                             N=500000,
                                                             dt=1 / 20000,
                                                             break_after_n_time_steps=3000)  # TODO: Diese Werte in die Default übernehmen?
        super().__init__(mu=mu,
                         sigma=sigma,
                         x0=x0,
                         x_predTo=x_predTo,
                         t_samples=t_samples,
                         bins=bins,
                         name=name)


def create_wiener_samples_hitting_time(mu,
                                       sigma,
                                       x0,
                                       x_predTo,
                                       t_L=0.0,
                                       N=100000,
                                       dt=1 / 1000,
                                       break_after_n_time_steps=1000,
                                       break_min_time=None):
    """Monte Carlo approach to solve the first passage time problem. Propagates particles through the process model
    and determines their first passage at x_predTo by interpolating the positions between the last time before and
    the first time after the boundary.

    :param mu: A float, the "velocity" (drift) of the Wiener process with drift
    :param sigma: A float, the diffusion constant of the Wiener process with drift
    :param x0: A float, the starting position x(t=0) of the process.
    :param x_predTo: A float, position of the boundary.
    :param t_L: A float, the initial time.
    :param dt: A float, time increment.
    :param N: Integer, number of samples to use.
    :param break_after_n_time_steps: Integer, maximum number of time steps for the simulation.
    # TODO

    :return:
        t_samples: A np.array of shape [N] containing the first passage times of the particles.

    Note that particles that do not reach the boundary after break_after_n_time_steps time steps are handled with a
    fallback value of max(t_samples) + 1.
    """
    initial_samples = x0 + np.random.normal(loc=0, scale=np.sqrt(b), size=N)


    mu_discrete = mu * dt
    sigma_discrete = sigma *np.sqrt(dt)

    def compute_x_next_func(x_curr):
        w_k = np.random.normal(loc=mu_discrete, scale=sigma_discrete, size=N)
        x_next = x_curr + w_k[:, None]  # expects a 2D array
        return x_next

    # Let the samples move to the nozzle array
    time_before_arrival, x_before_arrival, x_after_arrival, x_term, fraction_of_returns = create_hitting_time_samples(
        initial_samples[:, None],  # expects a 2D array
        compute_x_next_func,
        x_predTo,
        t_L=t_L,
        N=N,
        dt=dt,
        break_after_n_time_steps=break_after_n_time_steps,
        break_min_time=break_min_time)

    # Linear interpolation to get time
    x_before_arrival = np.squeeze(x_before_arrival)  # go back to 1D arrays
    x_after_arrival = np.squeeze(x_after_arrival)
    v_interpolated = (x_after_arrival[x_term] - x_before_arrival[x_term]) / dt
    last_t = (x_predTo - x_before_arrival[x_term]) / v_interpolated
    time_of_arrival = time_before_arrival
    time_of_arrival[x_term] = time_before_arrival[x_term] + last_t

    time_of_arrival[np.logical_not(x_term)] = int(
        max(time_of_arrival)) + 1  # default value for particles that do not arrive

    t_samples = time_of_arrival
    return t_samples, None, fraction_of_returns


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

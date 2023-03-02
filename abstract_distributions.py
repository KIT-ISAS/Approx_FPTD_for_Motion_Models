import os
from absl import logging

from abc import ABC, abstractmethod

import numpy as np

import matplotlib.pyplot as plt
from timeit import time


class AbstractHittingTimeModel(ABC):
    """A base class for the hitting time models."""

    def __init__(self, x_predTo, name='DefaultName'):
        """Initialize the model.

        :param x_predTo: A float, position of the boundary.
        :param name: String, (default) name for the model.
        """
        self.x_predTo = x_predTo
        self.name = name

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
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def var(self):
        """The variance of the first passage time distribution."""
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

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
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def cdf(self, t):
        """The CDF of the first passage time distribution.

        :param t: A float or np.array, the time parameter of the distribution.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def ev_t(self, t):
        """The mean function of the motion model in x.

        :param t: A float or np.array, the time parameter of the mean function.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def var_t(self, t):
        """The variance function of the motion model in x.

        :param t: A float or np.array, the time parameter of the variance function.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def plot_quantile_function(self, q_min=0.005, q_max=0.995, save_results=False, result_dir=None, for_paper=True):
        """Plot the quantile function.

        :param q_min: A float, the smallest value of the confidence plot range.
        :param q_max: A float, the highest value of the confidence plot range.
        :param save_results: Boolean, whether to save the plots.
        :param result_dir: String, directory where to save the plots.
        :param for_paper: Boolean, whether to use a publication (omit headers, etc.).
        """
        plot_q = np.arange(q_min, q_max, 0.01)
        # TODO: Perhaps silent warnings
        plot_quant = [self.ppf(q) for q in plot_q]
        plt.plot(plot_q, plot_quant)
        plt.xlabel('Confidence level')
        plt.ylabel('Time t in s')

        if not for_paper:
            plt.title('Quantile Function (Inverse CDF) for ' + self.name)

        if save_results:
            plt.savefig(result_dir + self.name + '_quantile_function.pdf')
            plt.savefig(result_dir + self.name + '_quantile_function.png')
            plt.savefig(result_dir + self.name + '_quantile_function.pgf')
        plt.show()

    def get_statistics(self):
        """Get some statistics from the model as a dict."""
        hit_stats = {}
        hit_stats['PDF'] = self.pdf
        hit_stats['CDF'] = self.cdf
        hit_stats['EV'] = self.ev
        hit_stats['STDDEV'] = self.stddev
        return hit_stats


class AbstractEngineeringApproxHittingTimeModel(AbstractHittingTimeModel, ABC):

    def __init__(self, x_predTo, name="No-return approx."):
        """Initialize the model.

        :param x_predTo: A float, position of the boundary.
        :param name: String, name for the model.
        """
        super().__init__(x_predTo, name)

        self.compute_moment = self.get_numerical_moment_integrator()
        self.compute_moment_riemann = self.get_numerical_moment_integrator(use_cdf=False) # TODO: Bleibt das?

    @abstractmethod
    def trans_density(self, dt, theta):
        """"The transition density p(x(dt+theta)| x(thetha) = x_predTo) from going from x_predTo at time theta to
        x(dt+theta) at time dt+theta.

        Note that in terms of the used approximation, this can be seen as the first returning time to x_predTo after
        a crossing of x_predTo at theta.

        :param dt: A float or np.array, the time difference. dt is zero at time = theta.
        :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to.

        :return: A scipy.stats.norm object, the transition density for the given dt and theta.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def trans_dens_ppf(self, theta=None, q=0.95):
        """The PPF of 1 - int ( p(x(dt+theta)| x(thetha) = x_predTo), x(dt+theta) = - infty .. x_predTo),
        i.e., the inverse CDF of the event that particles are above x_predTo once they have reached it at time theta.

        Note that in terms of the used approximation, this can be seen as PPF of the approximate first passage
        returning time distribution w.r.t. the boundary x_pred_to.

        :param theta: A float or np.array, the (assumed) time at which x(thetha) = x_pred_to.
        :param q: A float, the desired confidence level, 0 <= q <= 1.

        :return: The value of the PPF for q and theta.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

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
            print('EV integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(round(1000*(time.time() - start_time), 4), abs_dev, rel_dev))
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
            print('Var integration time: {0}ms. Abs dev: {1}, Rel. dev: {2}'.format(round(1000*(time.time() - start_time), 4), abs_dev, rel_dev))
        return self._var

    def get_numerical_moment_integrator(self, n=400, t_min=None, t_max=None):
        """Generator that builds a numerical integrator based on Riemann sums.

        :param n: Integer, number of integration points.
        :param t_min: Integer, location of smallest integration point.
        :param t_max:  Integer, location of tallest integration point.

        :return:
            compute_moment: Function that can be used to compute the moments.
        """
        t_min = self.ppf(0.00005) if t_min is None else t_min   # TODO Das muss man anpassen, ist bei Wiener mit drift nicht so...
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

    def returning_probs(self, t, num_samples=101, deterministic_samples=True, mc_hitting_time_model=None):
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

    def returning_probs_uniform_samples(self, t, num_samples=101, deterministic_samples=True, mc_hitting_time_model=None):
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

    def plot_valid_regions(self,
                           theta=None,
                           q=0.95,
                           plot_t_min=0.0,
                           plot_t_max=None,
                           save_results=False,
                           result_dir=None,
                           for_paper=True,
                           no_show=False):
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
        t_pred = (self.x_predTo - self.x0) / self.mu  # todo: dAS SOLLTE HOCH
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
            plt.savefig(os.path.join(result_dir, process_name_save + plot_name + '.pgf'))
        if not no_show:
            plt.show()
        plt.close()


class AbstractMCHittingTimeModel(AbstractHittingTimeModel, ABC):
    """Wraps the histogram derived by a Monte-Carlo approach to solve the first-passage time problem to a distribution
     using scipy.stats.rv_histogram.

    """

    def __init__(self, x_predTo, bins=100, name='MCHittingTimeModel'):  # TODO: Name ist hier anders
        """Initialize the model.

        :param x_predTo: A float, position of the boundary.
        :param bins: An integer, the number of bins to use to represent the histogram.
        :param name: String, name for the model.
        """
        super().__init__(x_predTo=x_predTo,
                         name=name)

        self.t_samples, _, _ = create_ty_cv_samples_hitting_time(x_L, C_L, S_w, x_predTo, t_L)
        hist = np.histogram(self.t_samples, bins=bins, density=False)
        self._density = rv_histogram(hist, density=True)

    @property
    def ev(self):
        """The expected value of the first passage time distribution."""
        return self._density.mean

    @property
    def var(self):
        """The variance of the first passage time distribution."""
        return self._density.var

    @property
    def third_central_moment(self):
        """The third central moment of the first passage time distribution."""
        return self._third_moment - 3 * self.ev * self.var - self.ev ** 3

    def third_moment(self):
        """The third moment of the first passage time distribution."""
        return self._density.moment(3)

    def pdf(self, t):
        """The first passage time distribution (FPTD).

        :param t: A float or np.array, the time parameter of the distribution.
        """
        return self._density.pdf(t)

    def cdf(self, t):
        """The CDF of the first passage time distribution.

        :param t: A float or np.array, the time parameter of the distribution.
        """
        return self._density.pdf(t)

    def ppf(self, q):
        """The quantile function / percent point function (PPF) of the first passage time distribution.

        :param q: A float or np.array, the confidence parameter of the distribution, 0 <= q <= 1.
        """
        return self._density.ppf(q)


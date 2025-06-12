import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import tensorflow as tf

from scipy.stats import norm

from cv_arrival_distributions.cv_hitting_time_distributions import GaussTaylorCVHittingTimeDistribution


class GaussTaylorCVHittingTimeDistributionTest(tf.test.TestCase):
    """Test cases for the GaussTaylorCVHittingTimeDistribution class."""

    # Define system parameters
    # System noise
    S_w = 10
    # Covariance matrix at last timestep
    C_L = np.array([[2E-7, 2E-5, 0, 0], [2E-5, 6E-3, 0, 0], [0, 0, 2E-7, 2E-5], [0, 0, 2E-5, 6E-3]])
    # Mean at last timestep
    x_L = np.array([0.3, 6.2, 0.5, 0.2])
    # Boundary position
    x_predTo = 0.6458623971412047
    # Last time step
    t_L = 0  # In principle, we could assume w.l.o.g. that t_L = 0 (t_L is just a location argument).

    comb_S_w = np.stack([S_w, 4 * S_w])
    comb_x_L = np.stack([x_L, 0.002 + x_L])
    comb_C_L = np.stack([C_L, 0.5 * C_L])
    comb_x_predTo = np.stack([x_predTo, 1.5 * x_predTo])

    def setUp(self):
        self.cv_temporal_point_predictor = lambda pos_l, v_l, x_predTo: (x_predTo - pos_l[..., 0]) / v_l[..., 0]

    def test_cdf(self):
        with self.subTest(name='Test at the mean, and mean + std, all 1D'):
            dist = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                        point_predictor=self.cv_temporal_point_predictor)

            m = (self.x_predTo - self.x_L[0]) / self.x_L[1] + self.t_L
            ev, std = dist.ev.copy(), dist.stddev.copy()
            self.assertEqual(ev, m)
            self.assertEqual(0.5, dist.cdf(ev))
            self.assertAllClose(norm.cdf(1.0), dist.cdf(ev + std))

        with self.subTest(name='Test without vs. with batch_size and scalar input'):
            dist2 = GaussTaylorCVHittingTimeDistribution(0.002 + self.x_L,
                                                         0.5 * self.C_L,
                                                         4 * self.S_w,
                                                         1.5 * self.x_predTo,
                                                         self.t_L,
                                                         point_predictor=self.cv_temporal_point_predictor)
            comb_dist = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                             self.comb_C_L,
                                                             self.comb_S_w,
                                                             self.comb_x_predTo,
                                                             self.t_L,
                                                             point_predictor=self.cv_temporal_point_predictor)

            cdf_value_1 = dist.cdf(ev + 1.2 * std)
            cdf_value_2 = dist2.cdf(ev + 1.2 * std)

            cdf_value_comb = comb_dist.cdf(ev + 1.2 * std)
            self.assertAllClose(np.stack([cdf_value_1, cdf_value_2]), cdf_value_comb)

        with self.subTest(name='Test without vs. with batch_size and matrix input'):
            dist2 = GaussTaylorCVHittingTimeDistribution(0.002 + self.x_L,
                                                         0.5 * self.C_L,
                                                         4 * self.S_w,
                                                         1.5 * self.x_predTo,
                                                         self.t_L,
                                                         point_predictor=self.cv_temporal_point_predictor)
            comb_dist = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                             self.comb_C_L,
                                                             self.comb_S_w,
                                                             self.comb_x_predTo,
                                                             self.t_L,
                                                             point_predictor=self.cv_temporal_point_predictor)

            inp_1 = ev + 1.2 * std + np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            inp_2 = ev + 1.2 * std + np.array([-0.1, -0.2, -0.3, -0.4, -0.5])

            cdf_value_1 = dist.cdf(inp_1)
            cdf_value_2 = dist2.cdf(inp_2)

            cdf_value_comb = comb_dist.cdf(np.stack([inp_1, inp_2], axis=1))
            self.assertAllClose(np.stack([cdf_value_1, cdf_value_2], axis=1), cdf_value_comb)

        with self.subTest(name='Test with batch_size and matrix input, but only one sample'):
            self.assertAllEqual((2,), comb_dist.cdf(np.array([[ev + 1.2 * std, ev + 1.2 * std]])).shape)

    def test_S_w_setter(self):
        with self.subTest(name='Test for no batch dimension'):
            dist = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                        point_predictor=self.cv_temporal_point_predictor)
            dist2 = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, 10 ** 2 * self.S_w, self.x_predTo,
                                                         self.t_L,
                                                         point_predictor=self.cv_temporal_point_predictor)
            ev, std = dist.ev.copy(), dist.stddev.copy()
            dist.S_w *= 10 ** 2
            cdf_value_1 = dist.cdf(ev + 1.2 * std)
            cdf_value_2 = dist2.cdf(ev + 1.2 * std)
            self.assertAllClose(cdf_value_1, cdf_value_2)

        with self.subTest(name='Test with batch dimension'):
            comb_dist = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                             self.comb_C_L,
                                                             self.comb_S_w,
                                                             self.comb_x_predTo,
                                                             self.t_L,
                                                             point_predictor=self.cv_temporal_point_predictor)
            comb_dist_2 = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                               self.comb_C_L,
                                                               10 ** 2 * self.comb_S_w,
                                                               self.comb_x_predTo,
                                                               self.t_L,
                                                               point_predictor=self.cv_temporal_point_predictor)
            ev, std = dist.ev.copy(), dist.stddev.copy()
            comb_dist.S_w *= 10 ** 2
            cdf_value_1 = comb_dist.cdf(ev + 1.2 * std)
            cdf_value_2 = comb_dist_2.cdf(ev + 1.2 * std)
            self.assertAllClose(cdf_value_1, cdf_value_2)

    def test_scale_params(self):
        with self.subTest(name='Test for no batch dimension'):
            dist = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                        point_predictor=self.cv_temporal_point_predictor)
            lsf = 2.0
            tsf = 3.0
            scaled_x_L = self.x_L.copy()
            scaled_x_L[[0, 2]] *= lsf
            scaled_x_L[[1, 3]] *= lsf / tsf
            scaled_C_L = self.C_L.copy()
            scaled_C_L[0, 0] *= lsf ** 2
            scaled_C_L[2, 2] *= lsf ** 2
            scaled_C_L[1, 1] *= (lsf / tsf) ** 2
            scaled_C_L[3, 3] *= (lsf / tsf) ** 2
            scaled_C_L[0, 1] *= lsf ** 2 / tsf
            scaled_C_L[1, 0] *= lsf ** 2 / tsf
            scaled_C_L[2, 3] *= lsf ** 2 / tsf
            scaled_C_L[3, 2] *= lsf ** 2 / tsf
            scaled_S_w = self.S_w * lsf ** 2 / tsf ** 3
            scaled_x_predTo = self.x_predTo * lsf
            scaled_t_L = self.t_L * lsf

            dist2 = GaussTaylorCVHittingTimeDistribution(scaled_x_L, scaled_C_L, scaled_S_w, scaled_x_predTo,
                                                         scaled_t_L,
                                                         point_predictor=self.cv_temporal_point_predictor)
            ev, std = dist.ev.copy(), dist.stddev.copy()
            dist.scale_params(length_scaling_factor=lsf, time_scaling_factor=tsf)
            cdf_value_1 = dist.cdf(ev + 1.2 * std)
            cdf_value_2 = dist2.cdf(ev + 1.2 * std)
            self.assertAllClose(cdf_value_1, cdf_value_2)

        with self.subTest(name='Test with batch dimension'):
            comb_dist = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                             self.comb_C_L,
                                                             self.comb_S_w,
                                                             self.comb_x_predTo,
                                                             self.t_L,
                                                             point_predictor=self.cv_temporal_point_predictor)
            lsf = 2.0
            tsf = 3.0
            scaled_x_L = self.comb_x_L.copy()
            scaled_x_L[:, [0, 2]] *= lsf
            scaled_x_L[:, [1, 3]] *= lsf / tsf
            scaled_C_L = self.comb_C_L.copy()
            scaled_C_L[:, 0, 0] *= lsf ** 2
            scaled_C_L[:, 2, 2] *= lsf ** 2
            scaled_C_L[:, 1, 1] *= (lsf / tsf) ** 2
            scaled_C_L[:, 3, 3] *= (lsf / tsf) ** 2
            scaled_C_L[:, 0, 1] *= lsf ** 2 / tsf
            scaled_C_L[:, 1, 0] *= lsf ** 2 / tsf
            scaled_C_L[:, 2, 3] *= lsf ** 2 / tsf
            scaled_C_L[:, 3, 2] *= lsf ** 2 / tsf
            scaled_S_w = self.comb_S_w * lsf ** 2 / tsf ** 3
            scaled_x_predTo = self.comb_x_predTo * lsf
            scaled_t_L = self.t_L * lsf

            dist2 = GaussTaylorCVHittingTimeDistribution(scaled_x_L, scaled_C_L, scaled_S_w, scaled_x_predTo,
                                                         scaled_t_L,
                                                         point_predictor=self.cv_temporal_point_predictor)
            ev, std = comb_dist.ev.copy(), comb_dist.stddev.copy()
            comb_dist.scale_params(length_scaling_factor=lsf, time_scaling_factor=tsf)
            cdf_value_1 = comb_dist.cdf(np.expand_dims(ev + 1.2 * std, axis=0))
            cdf_value_2 = dist2.cdf(np.expand_dims(ev + 1.2 * std, axis=0))
            self.assertAllClose(cdf_value_1, cdf_value_2)

    def test_getitem(self):
        comb_dist = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                         self.comb_C_L,
                                                         self.comb_S_w,
                                                         self.comb_x_predTo,
                                                         self.t_L,
                                                         point_predictor=self.cv_temporal_point_predictor)
        ev, std = comb_dist.ev.copy(), comb_dist.stddev.copy()
        cdf_value = comb_dist.cdf(np.expand_dims(ev + 1.2 * std, axis=0))
        dist = comb_dist[0]
        self.assertEqual(cdf_value[0], dist.cdf(ev[0] + 1.2 * std[0]))

    def test_setitem(self):
        comb_dist = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                         self.comb_C_L,
                                                         self.comb_S_w,
                                                         self.comb_x_predTo,
                                                         self.t_L,
                                                         point_predictor=self.cv_temporal_point_predictor)
        dist = GaussTaylorCVHittingTimeDistribution(self.x_L + 0.1, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                    point_predictor=self.cv_temporal_point_predictor)
        ev, std = comb_dist.ev.copy(), comb_dist.stddev.copy()
        cdf_value = dist.cdf(ev[0] - 4.0 * std[0])
        comb_dist[0] = dist
        self.assertEqual(cdf_value, comb_dist.cdf(np.expand_dims(ev - 4.0 * std, axis=0))[0])

    def test_get_statistics(self):
        exp_keys = {'PDF', 'CDF', 'PPF', 'EV', 'STDDEV', 'SKEW'}
        dist = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                    point_predictor=self.cv_temporal_point_predictor)
        self.assertSetEqual(exp_keys, set(dist.get_statistics().keys()))


if __name__ == "__main__":
    tf.test.main()

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
from cv_arrival_distributions.cv_hitting_location_distributions import GaussTaylorCVHittingLocationDistribution


class GaussTaylorCVHittingLocationDistributionTest(tf.test.TestCase):
    """Test cases for the GaussTaylorCVHittingLocationDistributionclass."""

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
        self.cv_spatial_point_predictor = lambda pos_l, v_l, dt_pred: v_l[..., 1] * dt_pred

    def test_cdf(self):
        with self.subTest(name='Test at the mean, and mean + std, all 1D'):
            htd = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                       point_predictor=self.cv_temporal_point_predictor)
            hld = GaussTaylorCVHittingLocationDistribution(htd, self.S_w, self.cv_spatial_point_predictor)

            t_pred = (self.x_predTo - self.x_L[0]) / self.x_L[1] + self.t_L
            m = (t_pred - self.t_L) * self.x_L[3] + self.x_L[2]
            ev, std = hld.ev.copy(), hld.stddev.copy()
            self.assertEqual(ev, m)
            self.assertEqual(0.5, hld.cdf(ev))
            self.assertAllClose(norm.cdf(1.0), hld.cdf(ev + std))

        with self.subTest(name='Test without vs. with batch_size and scalar input'):
            htd2 = GaussTaylorCVHittingTimeDistribution(0.002 + self.x_L,
                                                        0.5 * self.C_L,
                                                        4 * self.S_w,
                                                        1.5 * self.x_predTo,
                                                        self.t_L,
                                                        point_predictor=self.cv_temporal_point_predictor)
            hld2 = GaussTaylorCVHittingLocationDistribution(htd2, 4 * self.S_w, self.cv_spatial_point_predictor)
            comb_htd = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                            self.comb_C_L,
                                                            self.comb_S_w,
                                                            self.comb_x_predTo,
                                                            self.t_L,
                                                            point_predictor=self.cv_temporal_point_predictor)
            comb_hld = GaussTaylorCVHittingLocationDistribution(comb_htd, self.comb_S_w,
                                                                self.cv_spatial_point_predictor)

            cdf_value_1 = hld.cdf(ev + 1.2 * std)
            cdf_value_2 = hld2.cdf(ev + 1.2 * std)

            cdf_value_comb = comb_hld.cdf(ev + 1.2 * std)
            self.assertAllClose(np.stack([cdf_value_1, cdf_value_2]), cdf_value_comb)

        with self.subTest(name='Test without vs. with batch_size and matrix input'):
            htd2 = GaussTaylorCVHittingTimeDistribution(0.002 + self.x_L,
                                                        0.5 * self.C_L,
                                                        4 * self.S_w,
                                                        1.5 * self.x_predTo,
                                                        self.t_L,
                                                        point_predictor=self.cv_temporal_point_predictor)
            hld2 = GaussTaylorCVHittingLocationDistribution(htd2, 4 * self.S_w, self.cv_spatial_point_predictor)
            comb_htd = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                            self.comb_C_L,
                                                            self.comb_S_w,
                                                            self.comb_x_predTo,
                                                            self.t_L,
                                                            point_predictor=self.cv_temporal_point_predictor)
            comb_hld = GaussTaylorCVHittingLocationDistribution(comb_htd, self.comb_S_w,
                                                                self.cv_spatial_point_predictor)

            inp_1 = ev + 1.2 * std + np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            inp_2 = ev + 1.2 * std + np.array([-0.1, -0.2, -0.3, -0.4, -0.5])

            cdf_value_1 = hld.cdf(inp_1)
            cdf_value_2 = hld2.cdf(inp_2)

            cdf_value_comb = comb_hld.cdf(np.stack([inp_1, inp_2], axis=1))
            self.assertAllClose(np.stack([cdf_value_1, cdf_value_2], axis=1), cdf_value_comb)

        with self.subTest(name='Test with batch_size and matrix input, but only one sample'):
            self.assertAllEqual((2,), comb_hld.cdf(np.array([[ev + 1.2 * std, ev + 1.2 * std]])).shape)

    def test_S_w_setter(self):
        with self.subTest(name='Test for no batch dimension'):
            htd = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                       point_predictor=self.cv_temporal_point_predictor)
            hld = GaussTaylorCVHittingLocationDistribution(htd, self.S_w, self.cv_spatial_point_predictor)
            hld2 = GaussTaylorCVHittingLocationDistribution(htd, 10 ** 2 * self.S_w, self.cv_spatial_point_predictor)

            ev, std = hld.ev.copy(), hld.stddev.copy()
            hld.S_w *= 10 ** 2
            cdf_value_1 = hld.cdf(ev + 1.2 * std)
            cdf_value_2 = hld2.cdf(ev + 1.2 * std)
            self.assertAllClose(cdf_value_1, cdf_value_2)

        with self.subTest(name='Test with batch dimension'):
            comb_htd = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                            self.comb_C_L,
                                                            self.comb_S_w,
                                                            self.comb_x_predTo,
                                                            self.t_L,
                                                            point_predictor=self.cv_temporal_point_predictor)
            comb_hld = GaussTaylorCVHittingLocationDistribution(comb_htd, self.comb_S_w,
                                                                self.cv_spatial_point_predictor)
            comb_hld_2 = GaussTaylorCVHittingLocationDistribution(comb_htd, 10 ** 2 * self.comb_S_w,
                                                                  self.cv_spatial_point_predictor)
            ev, std = hld.ev.copy(), hld.stddev.copy()
            comb_hld.S_w *= 10 ** 2
            cdf_value_1 = comb_hld.cdf(ev + 1.2 * std)
            cdf_value_2 = comb_hld_2.cdf(ev + 1.2 * std)
            self.assertAllClose(cdf_value_1, cdf_value_2)

    def test_scale_params(self):
        with self.subTest(name='Test for no batch dimension'):
            htd = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                       point_predictor=self.cv_temporal_point_predictor)
            hld = GaussTaylorCVHittingLocationDistribution(htd, self.S_w, self.cv_spatial_point_predictor)

            lsf = 3.0
            tsf = 4.0

            scaled_S_w = self.S_w * lsf ** 2 / tsf ** 3

            hld2 = GaussTaylorCVHittingLocationDistribution(htd, scaled_S_w, self.cv_spatial_point_predictor)
            ev, std = hld.ev.copy(), hld.stddev.copy()
            hld.scale_params(length_scaling_factor=lsf, time_scaling_factor=tsf)
            cdf_value_1 = hld.cdf(ev + 1.2 * std)
            cdf_value_2 = hld2.cdf(ev + 1.2 * std)
            self.assertAllClose(cdf_value_1, cdf_value_2)

        with self.subTest(name='Test with batch dimension'):
            comb_htd = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                            self.comb_C_L,
                                                            self.comb_S_w,
                                                            self.comb_x_predTo,
                                                            self.t_L,
                                                            point_predictor=self.cv_temporal_point_predictor)
            comb_hld = GaussTaylorCVHittingLocationDistribution(comb_htd, self.comb_S_w,
                                                                self.cv_spatial_point_predictor)
            lsf = 3.0
            tsf = 4.0

            scaled_S_w = self.comb_S_w * lsf ** 2 / tsf ** 3

            hld2 = GaussTaylorCVHittingLocationDistribution(comb_htd, scaled_S_w, self.cv_spatial_point_predictor)
            ev, std = comb_hld.ev.copy(), comb_hld.stddev.copy()
            comb_hld.scale_params(length_scaling_factor=lsf, time_scaling_factor=tsf)
            cdf_value_1 = comb_hld.cdf(np.expand_dims(ev + 1.2 * std, axis=0))
            cdf_value_2 = hld2.cdf(np.expand_dims(ev + 1.2 * std, axis=0))
            self.assertAllClose(cdf_value_1, cdf_value_2)

    def test_getitem(self):
        comb_htd = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                        self.comb_C_L,
                                                        self.comb_S_w,
                                                        self.comb_x_predTo,
                                                        self.t_L,
                                                        point_predictor=self.cv_temporal_point_predictor)
        comb_hld = GaussTaylorCVHittingLocationDistribution(comb_htd, self.comb_S_w,
                                                            self.cv_spatial_point_predictor)
        ev, std = comb_hld.ev.copy(), comb_hld.stddev.copy()
        cdf_value = comb_hld.cdf(np.expand_dims(ev + 1.2 * std, axis=0))
        hld = comb_hld[0]
        self.assertEqual(cdf_value[0], hld.cdf(ev[0] + 1.2 * std[0]))

    def test_setitem(self):
        comb_htd = GaussTaylorCVHittingTimeDistribution(self.comb_x_L,
                                                        self.comb_C_L,
                                                        self.comb_S_w,
                                                        self.comb_x_predTo,
                                                        self.t_L,
                                                        point_predictor=self.cv_temporal_point_predictor)
        comb_hld = GaussTaylorCVHittingLocationDistribution(comb_htd, self.comb_S_w,
                                                            self.cv_spatial_point_predictor)
        htd = GaussTaylorCVHittingTimeDistribution(self.x_L + 0.1, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                   point_predictor=self.cv_temporal_point_predictor)
        hld = GaussTaylorCVHittingLocationDistribution(htd, self.S_w, self.cv_spatial_point_predictor)
        ev, std = comb_hld.ev.copy(), comb_hld.stddev.copy()
        cdf_value = hld.cdf(ev[0] - 4.0 * std[0])
        comb_hld[0] = hld
        self.assertEqual(cdf_value, comb_hld.cdf(np.expand_dims(ev - 4.0 * std, axis=0))[0])

    def test_get_statistics(self):
        exp_keys = {'PDF', 'CDF', 'PPF', 'EV', 'STDDEV', 'SKEW'}
        htd = GaussTaylorCVHittingTimeDistribution(self.x_L, self.C_L, self.S_w, self.x_predTo, self.t_L,
                                                   point_predictor=self.cv_temporal_point_predictor)
        hld = GaussTaylorCVHittingLocationDistribution(htd, self.S_w, self.cv_spatial_point_predictor)
        self.assertSetEqual(exp_keys, set(hld.get_statistics().keys()))


if __name__ == "__main__":
    tf.test.main()

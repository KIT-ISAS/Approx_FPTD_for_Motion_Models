import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import tensorflow as tf

from cv_arrival_distributions.cv_hitting_time_distributions import GaussTaylorCVHittingTimeDistribution


class AbstractNormalArrivalDistributionTest(tf.test.TestCase):
    """Test cases for the AbstractNormalArrivalDistribution class."""

    def setUp(self):
        self.cv_temporal_point_predictor = lambda pos_l, v_l, x_predTo: (x_predTo - pos_l[..., 0]) / v_l[..., 0]

    def test_cdf(self):
        # Create a GaussTaylorCVHittingTimeDistribution object with easy-to-understand parameters
        cov = np.diag([1, 1, 1, 1])
        na_distr = GaussTaylorCVHittingTimeDistribution(x_L=np.zeros((8, 4)),
                                                        C_L=np.stack([cov for _ in range(8)], axis=0),
                                                        S_w=0.1,
                                                        x_predTo=5.0,
                                                        t_L=0,
                                                        point_predictor=self.cv_temporal_point_predictor)
        na_distr._ev = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0])
        na_distr._var = np.ones(8)

        with self.subTest(name='Test with gating, for scalar input'):
            exp_outp_cdf = na_distr.cdf(8)
            self.assertTrue(np.allclose(exp_outp_cdf[:4], 1))
            outp_cdf = na_distr.cdf(8, use_gating=True, gating_stddev_factor=6.0)
            self.assertAllClose(exp_outp_cdf, outp_cdf)

        with self.subTest(name='Test with gating, for one-dimensional input'):
            a = np.array([[-10, -3.0, 3, 10.0, -10, 7.0, 13, 20.0]])

            exp_outp_cdf = na_distr.cdf(a)
            self.assertAllEqual((8,), exp_outp_cdf.shape)
            self.assertTrue(np.allclose(exp_outp_cdf[[0, 4]], 0))
            self.assertTrue(np.allclose(exp_outp_cdf[[3, 7]], 1))
            outp_cdf = na_distr.cdf(a, use_gating=True, gating_stddev_factor=6.0)
            self.assertAllClose(exp_outp_cdf, outp_cdf)

        with self.subTest(name='Test with gating, for two-dimensional input'):
            a_0 = np.array([-10.0, -5.0, -3.0])  # some left inside for ev[:4], all left outside for ev[4:]
            a_1 = np.array([-20.0, -15.0, -10.0])  # all left outside for ev
            a_2 = np.array([10.0, 15.0, 30.0])  # all right outside for ev[:4], some right inside for ev[4:]
            a_3 = np.array([20.0, 25.0, 40.0])  # all right outside for ev

            a = np.stack([a_0, a_1, a_2, a_3, a_0, a_1, a_2, a_3], axis=1)  # shape (3, 8)

            exp_outp_cdf = na_distr.cdf(a)
            self.assertAllEqual((3, 8), exp_outp_cdf.shape)
            self.assertTrue(np.allclose(exp_outp_cdf[0, 0], 0))
            self.assertTrue(np.allclose(exp_outp_cdf[:, 4], 0))
            self.assertTrue(np.allclose(exp_outp_cdf[:, [1, 5]], 0))
            self.assertTrue(np.allclose(exp_outp_cdf[-1, 6], 1))
            self.assertTrue(np.allclose(exp_outp_cdf[:, 2], 1))
            self.assertTrue(np.allclose(exp_outp_cdf[:, [3, 7]], 1))
            outp_cdf = na_distr.cdf(a, use_gating=True, gating_stddev_factor=6.0)
            self.assertAllClose(exp_outp_cdf, outp_cdf)

        with self.subTest(name='Test with gating, for scalar ev, stddev'):
            na_distr._ev = np.array([0.0])
            na_distr._var = np.array([1.0])

            exp_outp_cdf = na_distr.cdf(8)
            self.assertTrue(np.allclose(exp_outp_cdf, 1))
            outp_cdf = na_distr.cdf(8, use_gating=True, gating_stddev_factor=6.0)
            self.assertAllClose(exp_outp_cdf, outp_cdf)

        with self.subTest(name='Test with gating, check that probabilities sum to [0, 1]'):
            na_distr._ev = np.array([0.0])
            na_distr._var = np.array([1.0])

            a = np.array([[-15.0], [-6.00001]])  # this one is gated  --> results in zeros
            b = np.array([[-6.00001], [10.0]])  # this one is not  --> results in [0, 1]
            # probs are [0, 1]

            pb = na_distr.cdf(b)
            pa = na_distr.cdf(a)
            probs = pb - pa
            self.assertTrue(np.sum(probs, axis=0) >= 0)
            self.assertTrue(np.sum(probs, axis=0) <= 1)

            pb = na_distr.cdf(b, use_gating=True, gating_stddev_factor=6.0)
            pa = na_distr.cdf(a, use_gating=True, gating_stddev_factor=6.0)
            probs = pb - pa
            self.assertTrue(np.sum(probs, axis=0) >= 0)
            self.assertTrue(np.sum(probs, axis=0) <= 1)

            a = np.array([[-10], [6.00001]])  # this one is not gated --> results in [0, 1]
            b = np.array([[6.00001], [10.0]])  # this one is gated  --> results in ones
            # probs are [1, 0]

            pb = na_distr.cdf(b)
            pa = na_distr.cdf(a)
            probs = pb - pa
            self.assertTrue(np.sum(probs, axis=0) >= 0)
            self.assertTrue(np.sum(probs, axis=0) <= 1)

            pb = na_distr.cdf(b, use_gating=True, gating_stddev_factor=6.0)
            pa = na_distr.cdf(a, use_gating=True, gating_stddev_factor=6.0)
            probs = pb - pa
            self.assertTrue(np.sum(probs, axis=0) >= 0)
            self.assertTrue(np.sum(probs, axis=0) <= 1)


if __name__ == "__main__":
    tf.test.main()

import unittest
import numpy as np
from itertools import product
from nonrad.nonrad import fact, overlap_NM, analytic_overlap_NM, get_C


class OverlapTest(unittest.TestCase):
    def test_fact(self):
        for i, j in enumerate([1, 1, 2, 6, 24, 120]):
            self.assertEqual(fact(i), j)

    def test_overlap_NM(self):
        DQ, w1, w2 = (0.00, 0.03, 0.03)
        for m, n in product(range(10), range(10)):
            if m == n:
                self.assertAlmostEqual(overlap_NM(DQ, w1, w2, m, n), 1.)
            else:
                self.assertAlmostEqual(overlap_NM(DQ, w1, w2, m, n), 0.)
        DQ, w1, w2 = (1.00, 0.03, 0.03)
        for m, n in product(range(10), range(10)):
            if m == n:
                self.assertNotAlmostEqual(overlap_NM(DQ, w1, w2, m, n), 1.)
            else:
                self.assertNotAlmostEqual(overlap_NM(DQ, w1, w2, m, n), 0.)
        DQ, w1, w2 = (1.00, 0.15, 0.03)
        for m, n in product(range(10), range(10)):
            if m == n:
                self.assertNotAlmostEqual(overlap_NM(DQ, w1, w2, m, n), 1.)
            else:
                self.assertNotAlmostEqual(overlap_NM(DQ, w1, w2, m, n), 0.)

    def test_analytic_overlap_NM(self):
        DQ, w1, w2 = (0.00, 0.03, 0.03)
        for m, n in product(range(10), range(10)):
            if m == n:
                self.assertAlmostEqual(
                    analytic_overlap_NM(DQ, w1, w2, m, n), 1.)
            else:
                self.assertAlmostEqual(
                    analytic_overlap_NM(DQ, w1, w2, m, n), 0.)
        DQ, w1, w2 = (1.00, 0.03, 0.03)
        for m, n in product(range(10), range(10)):
            if m == n:
                self.assertNotAlmostEqual(
                    analytic_overlap_NM(DQ, w1, w2, m, n), 1.)
            else:
                self.assertNotAlmostEqual(
                    analytic_overlap_NM(DQ, w1, w2, m, n), 0.)
        DQ, w1, w2 = (1.00, 0.15, 0.03)
        for m, n in product(range(10), range(10)):
            if m == n:
                self.assertNotAlmostEqual(
                    analytic_overlap_NM(DQ, w1, w2, m, n), 1.)
            else:
                self.assertNotAlmostEqual(
                    analytic_overlap_NM(DQ, w1, w2, m, n), 0.)

    def test_compare_overlaps(self):
        for DQ, w1, w2 in product([0., 0.5, 3.14], [0.03, 0.1], [0.03, 0.5]):
            for m, n in product(range(10), range(10)):
                self.assertAlmostEqual(
                    overlap_NM(DQ, w1, w2, m, n),
                    analytic_overlap_NM(DQ, w1, w2, m, n), places=5)


# more robust tests for get_C would be ideal...
# we're a bit limited because there aren't too many "obvious" results to
# compare to
class GetCTest(unittest.TestCase):
    def setUp(self):
        self.args = {
            'DQ': 2.008,
            'DE': 1.0102,
            'w1': 0.0306775211118,
            'w2': 0.0339920265573,
            'V': 0.00669174,
            'Omega': 1.1e-21,
            'g': 1,
            'T': 300,
            'sigma': None,
            'overlap_method': 'analytic'
        }

    def test_normal_run(self):
        self.assertGreater(get_C(**self.args), 0.)

    def test_same_w(self):
        self.args['w2'] = self.args['w1']
        self.assertGreater(get_C(**self.args), 0.)
        self.args['DQ'] = 0.
        self.assertLess(get_C(**self.args), 1e-20)

    def test_integrate(self):
        self.args['overlap_method'] = 'integrate'
        self.assertGreater(get_C(**self.args), 0.)
        self.args['overlap_method'] = 'Integrate'
        self.assertGreater(get_C(**self.args), 0.)

    def test_gaussian(self):
        for sigma in np.linspace(0.1, 5, 5):
            self.args['sigma'] = sigma
            self.assertGreater(get_C(**self.args), 0.)

    def test_T(self):
        self.args['T'] = np.linspace(0.01, 1000, 100)
        cs = get_C(**self.args)
        self.assertEqual(type(cs), np.ndarray)
        self.assertEqual(len(cs), 100)
        for c in cs:
            self.assertGreater(c, 0.)
        self.args['T'] = [300]
        with self.assertRaises(TypeError):
            get_C(**self.args)

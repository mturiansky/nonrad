# pylint: disable=C0114,C0115,C0116

import unittest
from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import factorial

from nonrad.nonrad import analytic_overlap_NM, fact, get_C, herm, overlap_NM


class OverlapTest(unittest.TestCase):
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
            'dQ': 2.008,
            'dE': 1.0102,
            'wi': 0.0306775211118,
            'wf': 0.0339920265573,
            'Wif': 0.00669174,
            'volume': 1100,
            'g': 1,
            'T': 300,
            'sigma': None,
            'occ_tol': 1e-4,
            'overlap_method': 'Integrate'
        }

    def test_normal_run(self):
        self.assertGreater(get_C(**self.args), 0.)

    def test_same_w(self):
        self.args['wf'] = self.args['wi']
        self.assertGreater(get_C(**self.args), 0.)
        self.args['dQ'] = 0.
        self.assertLess(get_C(**self.args), 1e-20)

    def test_analytic(self):
        self.args['overlap_method'] = 'analytic'
        self.assertGreater(get_C(**self.args), 0.)
        self.args['overlap_method'] = 'Analytic'
        self.assertGreater(get_C(**self.args), 0.)

    def test_bad_overlap(self):
        self.args['overlap_method'] = 'blah'
        with self.assertRaises(ValueError):
            get_C(**self.args)

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

    def test_occ_tol(self):
        self.args['occ_tol'] = 1e-6
        self.assertGreater(get_C(**self.args), 0.)
        self.args['occ_tol'] = 1.
        self.args['dE'] = 150 * self.args['wf']
        with self.assertWarns(RuntimeWarning):
            get_C(**self.args)


class MathTest(unittest.TestCase):
    def test_fact(self):
        for i in range(171):
            exact = np.double(factorial(i, exact=True))
            self.assertAlmostEqual(fact(i)/exact - 1, 0.)

    def test_herm(self):
        for x in np.linspace(0.1, 1., 50):
            for i in range(70):
                exact = hermval(x, [0.]*i + [1.])
                self.assertAlmostEqual(herm(x, i)/exact - 1, 0.)

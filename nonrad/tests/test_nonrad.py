import unittest
from itertools import product
from nonrad.nonrad import fact, overlap_NM, analytic_overlap_NM


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

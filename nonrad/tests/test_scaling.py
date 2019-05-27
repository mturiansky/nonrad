import unittest
import numpy as np
from nonrad.scaling import sommerfeld_parameter


class ScalingTest(unittest.TestCase):
    def setUp(self):
        self.args = {
            'T': 300,
            'Z': 0,
            'm_eff': 1.,
            'eps0': 1.
        }

    def test_sommerfeld_neutral(self):
        self.assertAlmostEqual(sommerfeld_parameter(**self.args), 1.)

    def test_sommerfeld_attractive(self):
        self.args['Z'] = -1
        self.assertGreater(sommerfeld_parameter(**self.args), 1.)

    def test_sommerfeld_repulsive(self):
        self.args['Z'] = 1
        self.assertLess(sommerfeld_parameter(**self.args), 1.)

    def test_sommerfeld_list(self):
        self.args['T'] = np.linspace(0.1, 1000, 100)
        self.assertEqual(sommerfeld_parameter(**self.args), 1.)
        self.args['Z'] = -1
        self.assertTrue(np.all(sommerfeld_parameter(**self.args) > 1.))
        self.args['Z'] = 1
        self.assertTrue(np.all(sommerfeld_parameter(**self.args) < 1.))

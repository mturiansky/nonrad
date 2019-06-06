import unittest
import numpy as np
from scipy import constants as const
from nonrad.tests import TEST_FILES, FakeFig
from nonrad.scaling import sommerfeld_parameter, find_charge_center, \
    distance_PBC, radial_distribution, charged_supercell_scaling, \
    thermal_velocity


class SommerfeldTest(unittest.TestCase):
    def setUp(self):
        self.args = {
            'T': 300,
            'Z': 0,
            'm_eff': 1.,
            'eps0': 1.
        }

    def test_neutral(self):
        self.assertAlmostEqual(sommerfeld_parameter(**self.args), 1.)

    def test_attractive(self):
        self.args['Z'] = -1
        self.assertGreater(sommerfeld_parameter(**self.args), 1.)

    def test_repulsive(self):
        self.args['Z'] = 1
        self.assertLess(sommerfeld_parameter(**self.args), 1.)

    def test_list(self):
        self.args['T'] = np.linspace(0.1, 1000, 100)
        self.assertEqual(sommerfeld_parameter(**self.args), 1.)
        self.args['Z'] = -1
        self.assertTrue(np.all(sommerfeld_parameter(**self.args) > 1.))
        self.args['Z'] = 1
        self.assertTrue(np.all(sommerfeld_parameter(**self.args) < 1.))


class ChargedSupercellScalingTest(unittest.TestCase):
    def test_find_charge_center(self):
        lattice = np.eye(3)
        density = np.ones((50, 50, 50))
        self.assertTrue(
            np.allclose(find_charge_center(density, lattice), [0.49]*3)
        )
        density = np.zeros((50, 50, 50))
        density[0, 0, 0] = 1.
        self.assertTrue(
            np.allclose(find_charge_center(density, lattice), [0.]*3)
        )

    def test_distance_PBC(self):
        a = np.array([0.25]*3)
        b = np.array([0.5]*3)
        lattice = np.eye(3)
        self.assertEqual(distance_PBC(a, b, lattice), np.sqrt(3)*0.25)
        b = np.array([0.9]*3)
        self.assertEqual(distance_PBC(a, b, lattice), np.sqrt(3)*0.35)

    def test_radial_distribution(self):
        lattice = np.eye(3)
        density = np.zeros((50, 50, 50))
        density[0, 0, 0] = 1.
        point = np.array([0.]*3)
        dist = distance_PBC(np.zeros(3), point, lattice)
        r, n = radial_distribution(density, point, lattice)
        self.assertAlmostEqual(r[np.where(n == 1.)[0][0]], dist)
        point = np.array([0.25]*3)
        dist = distance_PBC(np.zeros(3), point, lattice)
        r, n = radial_distribution(density, point, lattice)
        self.assertAlmostEqual(r[np.where(n == 1.)[0][0]], dist)
        point = np.array([0.29, 0.73, 0.44])
        dist = distance_PBC(np.zeros(3), point, lattice)
        r, n = radial_distribution(density, point, lattice)
        self.assertAlmostEqual(r[np.where(n == 1.)[0][0]], dist)

    @unittest.skip('WAVECARs too large to share')
    def test_charged_supercell_scaling(self):
        f = charged_supercell_scaling(str(TEST_FILES / 'WAVECAR.C-'), 192, 189)
        self.assertAlmostEqual(f, 1.08)
        f = charged_supercell_scaling(str(TEST_FILES / 'WAVECAR.C-'), 192, 189,
                                      fig=FakeFig())
        self.assertAlmostEqual(f, 1.08)
        f = charged_supercell_scaling(str(TEST_FILES / 'WAVECAR.C-'), 192, 189,
                                      fig=FakeFig(), full_range=True)
        self.assertAlmostEqual(f, 1.08)


class ThermalVelocityTest(unittest.TestCase):
    def test_thermal_velocity(self):
        f = thermal_velocity(1., 1.)
        self.assertAlmostEqual(f, np.sqrt(3 * const.k / const.m_e) * 1e2)
        f = thermal_velocity(np.array([1.]), 1.)
        self.assertEqual(type(f), np.ndarray)

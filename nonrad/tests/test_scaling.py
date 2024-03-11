# pylint: disable=C0114,C0115,C0116

import unittest
from typing import Union

import numpy as np
from scipy import constants as const
from numpy.polynomial.laguerre import laggauss

from nonrad.scaling import (
    charged_supercell_scaling,
    charged_supercell_scaling_VASP,
    distance_PBC,
    find_charge_center,
    radial_distribution,
    sommerfeld_parameter,
    thermal_velocity,
)
from nonrad.tests import TEST_FILES, FakeFig


def _old_sommerfeld_parameter(
        T: Union[float, np.ndarray],
        Z: int,
        m_eff: float,
        eps0: float,
        method: str = 'Integrate'
) -> Union[float, np.ndarray]:
    if Z == 0:
        return 1.

    if method.lower()[0] == 'i':
        kT = const.k * T
        m = m_eff * const.m_e
        eps = (4 * np.pi * const.epsilon_0) * eps0
        f = -2 * np.pi * Z * m * const.e**2 / const.hbar**2 / eps

        def s_k(k):
            return f / k / (1 - np.exp(-f / k))

        t = 0.
        x, w = laggauss(64)
        for ix, iw in zip(x, w):
            t += iw * np.sqrt(ix) * s_k(np.sqrt(2 * m * kT * ix) / const.hbar)
        return t / np.sum(w * np.sqrt(x))

    # that 4*pi from Gaussian units....
    theta_b = np.pi**2 * (m_eff * const.m_e) * const.e**4 / \
        (2 * const.k * const.hbar**2 * (eps0 * 4*np.pi*const.epsilon_0)**2)
    zthetaT = Z**2 * theta_b / T

    if Z < 0:
        return 4 * np.sqrt(zthetaT / np.pi)
    return (8 / np.sqrt(3)) * \
        zthetaT**(2/3) * np.exp(-3 * zthetaT**(1/3))


class SommerfeldTest(unittest.TestCase):
    def setUp(self):
        self.args = {
            'T': 300,
            'Z': 0,
            'm_eff': 1.,
            'eps0': 1.,
            'method': 'Integrate'
        }

    def test_neutral(self):
        self.assertAlmostEqual(sommerfeld_parameter(**self.args), 1.)
        self.args['method'] = 'Analytic'
        self.assertAlmostEqual(sommerfeld_parameter(**self.args), 1.)

    def test_attractive(self):
        self.args['Z'] = -1
        self.assertGreater(sommerfeld_parameter(**self.args), 1.)
        self.args['method'] = 'Analytic'
        self.assertGreater(sommerfeld_parameter(**self.args), 1.)

    def test_repulsive(self):
        self.args['Z'] = 1
        self.assertLess(sommerfeld_parameter(**self.args), 1.)
        self.args['method'] = 'Analytic'
        self.assertLess(sommerfeld_parameter(**self.args), 1.)

    def test_list(self):
        self.args['T'] = np.linspace(0.1, 1000, 100)
        self.assertEqual(sommerfeld_parameter(**self.args), 1.)
        self.args['Z'] = -1
        self.assertTrue(np.all(sommerfeld_parameter(**self.args) > 1.))
        self.args['Z'] = 1
        self.assertTrue(np.all(sommerfeld_parameter(**self.args) < 1.))
        self.args['Z'] = 0
        self.args['method'] = 'Analytic'
        self.assertEqual(sommerfeld_parameter(**self.args), 1.)
        self.args['Z'] = -1
        self.assertTrue(np.all(sommerfeld_parameter(**self.args) > 1.))
        self.args['Z'] = 1
        self.assertTrue(np.all(sommerfeld_parameter(**self.args) < 1.))

    def test_compare_methods(self):
        self.args = {
            'T': 100,
            'Z': -1,
            'm_eff': 0.2,
            'eps0': 8.9,
            'method': 'Integrate'
        }

        f0 = sommerfeld_parameter(**self.args)
        self.args['method'] = 'Analytic'
        f1 = sommerfeld_parameter(**self.args)
        self.assertAlmostEqual(f0, f1, places=2)

        self.args['Z'] = 1
        self.args['T'] = 900
        f0 = sommerfeld_parameter(**self.args)
        self.args['method'] = 'Integrate'
        f1 = sommerfeld_parameter(**self.args)
        self.assertGreater(np.abs(f0-f1)/f1, 0.1)

    def test_old_sommerfeld(self):
        self.args = {'m_eff': 0.2, 'eps0': 8.9}
        for m in ['i', 'a']:
            self.args['method'] = m
            for t in [100, 300, 700, 900]:
                self.args['T'] = t
                for z in [0, 1, -1]:
                    self.args['Z'] = z
                    f0 = _old_sommerfeld_parameter(**self.args)
                    f1 = sommerfeld_parameter(**self.args)
                    self.assertAlmostEqual(f0, f1, places=2)

    def test_sommerfeld_dim(self):
        self.args = {
            'T': 200,
            'Z': -1,
            'm_eff': 0.2,
            'eps0': 8.9,
            'dim': 2,
            'method': 'Integrate'
        }

        self.assertAlmostEqual(sommerfeld_parameter(**self.args), 2., places=2)
        self.args['method'] = 'a'
        self.assertAlmostEqual(sommerfeld_parameter(**self.args), 2., places=5)
        self.args['method'] = 'i'
        self.args['Z'] = 1
        self.assertLess(sommerfeld_parameter(**self.args), 1.)
        self.args['method'] = 'a'
        self.assertLess(sommerfeld_parameter(**self.args), 1.)

        self.args['dim'] = 1
        with self.assertRaises(ValueError):
            self.assertLess(sommerfeld_parameter(**self.args), 1.)
        self.args['method'] = 'i'
        self.assertLess(sommerfeld_parameter(**self.args), 1.)
        self.args['Z'] = -1
        self.assertLess(sommerfeld_parameter(**self.args), 1.)

        self.args['Z'] = 0
        self.assertEqual(sommerfeld_parameter(**self.args), 1.)


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
    def test_charged_supercell_scaling_VASP(self):
        f = charged_supercell_scaling_VASP(
            str(TEST_FILES / 'WAVECAR.C-'),
            189,
            def_index=192
        )
        self.assertAlmostEqual(f, 1.08)

    def test_charged_supercell_scaling(self):
        # test that numbers work out for homogeneous case
        wf = np.ones((20, 20, 20))
        f = charged_supercell_scaling(wf, 10*np.eye(3), np.array([0.]*3))
        self.assertAlmostEqual(f, 1.00)

        # test the plotting stuff
        wf = np.ones((1, 1, 1))
        f = charged_supercell_scaling(wf, 10*np.eye(3), np.array([0.]*3),
                                      fig=FakeFig())
        self.assertAlmostEqual(f, 1.00)
        f = charged_supercell_scaling(wf, 10*np.eye(3), np.array([0.]*3),
                                      fig=FakeFig(), full_range=True)
        self.assertAlmostEqual(f, 1.00)


class ThermalVelocityTest(unittest.TestCase):
    def test_thermal_velocity(self):
        f = thermal_velocity(1., 1.)
        self.assertAlmostEqual(f, np.sqrt(3 * const.k / const.m_e) * 1e2)
        f = thermal_velocity(np.array([1.]), 1.)
        self.assertEqual(type(f), np.ndarray)

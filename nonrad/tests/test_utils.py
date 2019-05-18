import unittest
import warnings
import pymatgen as pmg
import numpy as np
from pathlib import Path
from itertools import product
from nonrad.nonrad import HBAR, EV2J, AMU2KG, ANGS2M
from nonrad.utils import get_cc_structures, get_dQ, get_Q_from_struct, \
    get_PES_from_vaspruns, get_omega_from_PES, _compute_matel


TEST_FILES = Path(__file__).absolute().parent / '..' / '..' / 'test_files'


class FakeAx:
    def plot(*args, **kwargs):
        pass

    def scatter(*args, **kwargs):
        pass

    def set_title(*args, **kwargs):
        pass


class FakeFig:
    def subplots(x, y, **kwargs):
        return [FakeAx() for _ in range(y)]


class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.gnd_real = pmg.Structure.from_file(TEST_FILES / 'POSCAR.C0.gz')
        self.exd_real = pmg.Structure.from_file(TEST_FILES / 'POSCAR.C-.gz')
        self.gnd_test = pmg.Structure(pmg.Lattice.cubic(1.), ['H'],
                                      [[0., 0., 0.]])
        self.exd_test = pmg.Structure(pmg.Lattice.cubic(1.), ['H'],
                                      [[0.5, 0.5, 0.5]])
        self.sct_test = pmg.Structure(pmg.Lattice.cubic(1.), ['H'],
                                      [[0.25, 0.25, 0.25]])

    def test_get_cc_structures(self):
        gs, es = get_cc_structures(self.gnd_real, self.exd_real, [0.])
        self.assertEqual(gs, [])
        self.assertEqual(es, [])
        gs, es = get_cc_structures(self.gnd_test, self.exd_test, [0.],
                                   remove_zero=False)
        self.assertEqual(self.gnd_test, gs[0])
        self.assertEqual(self.exd_test, es[0])
        gs, es = get_cc_structures(self.gnd_test, self.exd_test, [0.5])
        self.assertTrue(np.allclose(gs[0][0].coords, [0.25, 0.25, 0.25]))

    def test_get_dQ(self):
        self.assertEqual(get_dQ(self.gnd_test, self.gnd_test), 0.)
        self.assertEqual(get_dQ(self.exd_test, self.exd_test), 0.)
        self.assertEqual(get_dQ(self.gnd_real, self.gnd_real), 0.)
        self.assertEqual(get_dQ(self.exd_real, self.exd_real), 0.)
        self.assertAlmostEqual(get_dQ(self.gnd_test, self.exd_test), 0.86945,
                               places=4)
        self.assertAlmostEqual(get_dQ(self.gnd_real, self.exd_real), 1.68587,
                               places=4)

    def test_get_Q_from_struct(self):
        q = get_Q_from_struct(self.gnd_test, self.exd_test, self.sct_test)
        self.assertAlmostEqual(q, 0.5 * 0.86945, places=4)
        gs, es = get_cc_structures(self.gnd_real, self.exd_real,
                                   np.linspace(-0.5, 0.5, 100),
                                   remove_zero=False)
        Q = 1.68587 * np.linspace(-0.5, 0.5, 100)
        for s, q in zip(gs, Q):
            tq = get_Q_from_struct(self.gnd_real, self.exd_real, s)
            self.assertAlmostEqual(tq, q, places=4)
        for s, q in zip(es, Q + 1.68587):
            tq = get_Q_from_struct(self.gnd_real, self.exd_real, s)
            self.assertAlmostEqual(tq, q, places=4)

    def test_get_PES_from_vaspruns(self):
        vrs = [TEST_FILES / 'vasprun.xml.0.gz',
               TEST_FILES / 'vasprun.xml.20.gz']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q, en = get_PES_from_vaspruns(self.gnd_real, self.exd_real, vrs)
        self.assertEqual(len(q), 2)
        self.assertEqual(len(en), 2)
        self.assertEqual(np.min(en), 0.)
        self.assertEqual(en[0], 0.)

    def test_get_omega_from_PES(self):
        q = np.linspace(-0.5, 0.5, 20)
        for om, q0 in zip(np.linspace(0.01, 0.1, 10),
                          np.linspace(0.1, 3., 10)):
            omega = (om / HBAR)**2 * ANGS2M**2 * AMU2KG / EV2J
            en = 0.5 * omega * (q - q0)**2
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.assertAlmostEqual(get_omega_from_PES(q, en), om)
                self.assertAlmostEqual(get_omega_from_PES(q, en, Q0=q0), om)
        self.assertAlmostEqual(get_omega_from_PES(q, en, Q0=q0, ax=FakeAx()),
                               om)

    def test__compute_matel(self):
        N = 10
        H = np.random.rand(N, N).astype(np.complex) + \
            1j*np.random.rand(N, N).astype(np.complex)
        H = H + np.conj(H).T
        _, ev = np.linalg.eigh(H)
        for i, j in product(range(N), range(N)):
            if i == j:
                self.assertAlmostEqual(_compute_matel(ev[:, i], ev[:, j]), 1.)
            else:
                self.assertAlmostEqual(_compute_matel(ev[:, i], ev[:, j]), 0.)

# pylint: disable=C0114,C0115,C0116

import glob
import unittest
import warnings
from itertools import product

import numpy as np

import pymatgen as pmg
from nonrad.ccd import get_Q_from_struct
from nonrad.elphon import (_compute_matel, _read_WSWQ, get_Wif_from_wavecars,
                           get_Wif_from_WSWQ, get_Wif_from_UNK)
from nonrad.tests import TEST_FILES, FakeFig


class ElphonTest(unittest.TestCase):
    def setUp(self):
        self.gnd_real = pmg.Structure.from_file(TEST_FILES / 'POSCAR.C0.gz')
        self.exd_real = pmg.Structure.from_file(TEST_FILES / 'POSCAR.C-.gz')
        self.gnd_test = pmg.Structure(pmg.Lattice.cubic(1.), ['H'],
                                      [[0., 0., 0.]])
        self.exd_test = pmg.Structure(pmg.Lattice.cubic(1.), ['H'],
                                      [[0.5, 0.5, 0.5]])
        self.sct_test = pmg.Structure(pmg.Lattice.cubic(1.), ['H'],
                                      [[0.25, 0.25, 0.25]])
        self.vrs = [TEST_FILES / 'vasprun.xml.0.gz'] + \
            glob.glob(str(TEST_FILES / 'lower' / '*' / 'vasprun.xml.gz'))

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

    @unittest.skip('WAVECARs too large to share')
    def test_get_Wif_from_wavecars(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcrs = [
                (pmg.Structure.from_file(d+'/vasprun.xml.gz'), d+'/WAVECAR')
                for d in glob.glob(str(TEST_FILES / 'lower' / '*'))
            ]
            wcrs = list(map(
                lambda x: (
                    get_Q_from_struct(self.gnd_real, self.exd_real, x[0]),
                    x[1]
                ), wcrs))
        self.assertAlmostEqual(
            get_Wif_from_wavecars(wcrs, str(TEST_FILES / 'WAVECAR.C0'),
                                  192, [189], spin=1)[0][1],
            0.087, places=2
        )
        self.assertAlmostEqual(
            get_Wif_from_wavecars(wcrs, str(TEST_FILES / 'WAVECAR.C0'),
                                  192, [189], spin=1, fig=FakeFig())[0][1],
            0.087, places=2
        )

    def test_get_Wif_from_UNK(self):
        Wif = get_Wif_from_UNK(
            unks=[(1., str(TEST_FILES / 'UNK.1'))],
            init_unk_path=str(TEST_FILES / 'UNK.0'),
            def_index=2,
            bulk_index=[1],
            eigs=np.array([0., 1.])
        )
        self.assertEqual(Wif[0][0], 1)
        self.assertAlmostEqual(Wif[0][1], 1.)

    def test__read_WSWQ(self):
        wswq = _read_WSWQ(str(TEST_FILES / 'lower' / '10' / 'WSWQ.gz'))
        self.assertGreater(len(wswq), 0)
        self.assertGreater(len(wswq[(1, 1)]), 0)
        self.assertGreater(np.abs(wswq[(1, 1)][(1, 1)]), 0)
        self.assertEqual(type(wswq), dict)
        self.assertEqual(type(wswq[(1, 1)]), dict)

    def test_get_Wif_from_WSWQ(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wswqs = [
                (pmg.Structure.from_file(d+'/vasprun.xml.gz'), d+'/WSWQ.gz')
                for d in glob.glob(str(TEST_FILES / 'lower' / '*'))
            ]
            wswqs = list(map(
                lambda x: (
                    get_Q_from_struct(self.gnd_real, self.exd_real, x[0]),
                    x[1]
                ), wswqs))
        self.assertAlmostEqual(
            get_Wif_from_WSWQ(wswqs, str(TEST_FILES / 'vasprun.xml.0.gz'),
                              192, [189], spin=1)[0][1],
            0.094, places=2
        )
        self.assertAlmostEqual(
            get_Wif_from_WSWQ(wswqs, str(TEST_FILES / 'vasprun.xml.0.gz'),
                              192, [189], spin=1, fig=FakeFig())[0][1],
            0.094, places=2
        )

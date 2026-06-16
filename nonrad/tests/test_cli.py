# pylint: disable=C0114,C0115,C0116

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from click.testing import CliRunner

from nonrad.cli import nonrad as nonrad_cli
from nonrad.tests import TEST_FILES


# ---------------------------------------------------------------------------
# Tests for the top-level `nonrad` group
# ---------------------------------------------------------------------------
class TestNonradGroup(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('Nonradiative recombination', result.output)

    def test_no_args(self):
        result = self.runner.invoke(nonrad_cli, [])
        self.assertEqual(result.exit_code, 0, result.output)


# ---------------------------------------------------------------------------
# Tests for `prep-ccd`
# ---------------------------------------------------------------------------
class TestPrepCcd(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['prep-ccd', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('ground_path', result.output.lower())

    @patch('nonrad.ccd.get_cc_structures')
    @patch('pymatgen.core.structure.Structure.from_file')
    def test_basic(self, mock_from_file, mock_get_cc):
        mock_struct = MagicMock()
        mock_from_file.return_value = mock_struct

        # get_cc_structures returns (list_of_structs, list_of_structs)
        struct_a = MagicMock()
        struct_b = MagicMock()
        mock_get_cc.return_value = ([struct_a], [struct_b])

        with self.runner.isolated_filesystem():
            # Create fake ground/excited directories with required files
            os.makedirs('ground')
            Path('ground/CONTCAR').touch()
            for f in ['KPOINTS', 'POTCAR', 'INCAR', 'job_script.sh']:
                Path(f'ground/{f}').write_text('dummy')

            os.makedirs('excited')
            Path('excited/CONTCAR').touch()
            for f in ['KPOINTS', 'POTCAR', 'INCAR', 'job_script.sh']:
                Path(f'excited/{f}').write_text('dummy')

            result = self.runner.invoke(
                nonrad_cli,
                ['prep-ccd', 'ground', 'excited', 'ccd_output']
            )
            self.assertEqual(result.exit_code, 0,
                             f'exit_code={result.exit_code}\n{result.output}')
            # Verify output directories were created
            self.assertTrue(os.path.isdir('ccd_output'))
            self.assertTrue(os.path.isdir('ccd_output/ground'))
            self.assertTrue(os.path.isdir('ccd_output/excited'))
            # Verify struct.to() was called for each displacement struct
            struct_a.to.assert_called()
            struct_b.to.assert_called()

    @patch('nonrad.ccd.get_cc_structures')
    @patch('pymatgen.core.structure.Structure.from_file')
    def test_custom_displacements(self, mock_from_file, mock_get_cc):
        mock_struct = MagicMock()
        mock_from_file.return_value = mock_struct
        mock_get_cc.return_value = ([], [])

        with self.runner.isolated_filesystem():
            os.makedirs('ground')
            Path('ground/CONTCAR').touch()
            for f in ['KPOINTS', 'POTCAR', 'INCAR', 'job_script.sh']:
                Path(f'ground/{f}').write_text('dummy')

            os.makedirs('excited')
            Path('excited/CONTCAR').touch()
            for f in ['KPOINTS', 'POTCAR', 'INCAR', 'job_script.sh']:
                Path(f'excited/{f}').write_text('dummy')

            result = self.runner.invoke(
                nonrad_cli,
                ['prep-ccd', 'ground', 'excited', 'ccd_output',
                 '-d', '-0.3', '0.3', '5']
            )
            self.assertEqual(result.exit_code, 0,
                             f'exit_code={result.exit_code}\n{result.output}')
            # Verify get_cc_structures was called
            mock_get_cc.assert_called_once()
            # Check displacements were passed correctly (via linspace)
            call_args = mock_get_cc.call_args
            displacements = call_args[0][2]
            self.assertEqual(len(displacements), 5)

    @patch('nonrad.ccd.get_cc_structures')
    @patch('pymatgen.core.structure.Structure.from_file')
    def test_nonexistent_ground_path(self, mock_from_file, mock_get_cc):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                nonrad_cli,
                ['prep-ccd', 'nonexistent', 'also_nonexistent', 'out']
            )
            self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Tests for `pes`
# ---------------------------------------------------------------------------
class TestPes(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['pes', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('cc_dir', result.output.lower())

    @patch('nonrad.ccd.get_omega_from_PES', return_value=0.05)
    @patch('nonrad.ccd.get_PES_from_vaspruns',
           return_value=(np.array([0.0, 1.0]), np.array([0.0, 0.5])))
    @patch('nonrad.ccd.get_dQ', return_value=1.5)
    @patch('nonrad.cli.glob', return_value=[])
    @patch('pymatgen.core.structure.Structure.from_file')
    def test_basic(self, mock_from_file, mock_glob,
                   mock_get_dq, mock_get_pes, mock_get_omega):
        mock_struct = MagicMock()
        mock_from_file.return_value = mock_struct

        with self.runner.isolated_filesystem():
            # Create fake dirs / files
            os.makedirs('cc/ground/0', exist_ok=True)
            os.makedirs('cc/excited/0', exist_ok=True)
            os.makedirs('ground_files')
            Path('ground_files/CONTCAR').touch()
            Path('ground_files/vasprun.xml').touch()
            os.makedirs('excited_files')
            Path('excited_files/CONTCAR').touch()
            Path('excited_files/vasprun.xml').touch()

            result = self.runner.invoke(
                nonrad_cli,
                ['pes', 'cc', 'ground_files', 'excited_files', '--energy-diff', '1.0']
            )
            # Verify exit code is 0 and mock gets called
            self.assertEqual(result.exit_code, 0, result.output)
            mock_get_dq.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for `dq`
# ---------------------------------------------------------------------------
class TestDq(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['dq', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_basic_real_files(self):
        ground = str(TEST_FILES / 'POSCAR.C0.gz')
        excited = str(TEST_FILES / 'POSCAR.C-.gz')
        result = self.runner.invoke(
            nonrad_cli, ['dq', ground, excited]
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')
        # dQ ≈ 1.6859
        self.assertIn('1.6858', result.output)

    def test_nonexistent_file(self):
        result = self.runner.invoke(
            nonrad_cli, ['dq', 'nonexistent_file', 'also_nonexistent']
        )
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Tests for `q-from-struct`
# ---------------------------------------------------------------------------
class TestQFromStruct(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['q-from-struct', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_basic_real_files(self):
        ground = str(TEST_FILES / 'POSCAR.C0.gz')
        excited = str(TEST_FILES / 'POSCAR.C-.gz')
        struct = str(TEST_FILES / 'POSCAR.C0.gz')
        result = self.runner.invoke(
            nonrad_cli, ['q-from-struct', ground, excited, struct]
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')
        # Q for ground == ground should be ≈ 0
        self.assertIn('0.0', result.output)

    def test_nonexistent_file(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['q-from-struct', 'nonexistent', 'nonexistent', 'nonexistent']
        )
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Tests for `barrier`
# ---------------------------------------------------------------------------
class TestBarrier(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['barrier', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_crossing(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['barrier',
             '--dq', '1.0', '--de', '0.0', '--wi', '0.05', '--wf', '0.05']
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')
        # Should contain barrier height info
        output_lower = result.output.lower()
        self.assertTrue(
            'barrier' in output_lower or 'height' in output_lower
            or 'ev' in output_lower,
            f'Expected barrier output, got: {result.output}'
        )

    @patch('nonrad.ccd.get_barrier_harmonic', return_value=None)
    def test_no_crossing(self, mock_barrier):
        result = self.runner.invoke(
            nonrad_cli,
            ['barrier',
             '--dq', '1.0', '--de', '5.0', '--wi', '0.01', '--wf', '0.01']
        )
        # The CLI should handle None return (no crossing) gracefully
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')
        output_lower = result.output.lower()
        self.assertTrue(
            'no' in output_lower or 'none' in output_lower
            or 'crossing' in output_lower,
            f'Expected no-crossing message, got: {result.output}'
        )

    def test_missing_options(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['barrier', '--dq', '1.0']  # missing --de, --wi, --wf
        )
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Tests for `capture`
# ---------------------------------------------------------------------------
class TestCapture(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['capture', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)

    @patch('nonrad.nonrad.get_C', return_value=1.23e-8)
    def test_basic_single_temperature(self, mock_get_c):
        result = self.runner.invoke(
            nonrad_cli,
            ['capture',
             '--dq', '1.0', '--de', '1.0',
             '--wi', '0.03', '--wf', '0.03',
             '--wif', '0.01', '--volume', '1000',
             '-T', '300']
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')

    @patch('nonrad.nonrad.get_C',
           return_value=np.array([1e-8, 2e-8, 3e-8]))
    def test_temperature_range(self, mock_get_c):
        result = self.runner.invoke(
            nonrad_cli,
            ['capture',
             '--dq', '1.0', '--de', '1.0',
             '--wi', '0.03', '--wf', '0.03',
             '--wif', '0.01', '--volume', '1000',
             '--temperature-range', '100', '500', '3']
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')

    def test_missing_required_options(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['capture', '--dq', '1.0']  # missing many required options
        )
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Tests for `sommerfeld`
# ---------------------------------------------------------------------------
class TestSommerfeld(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['sommerfeld', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_neutral_z0(self):
        """Z=0 is the neutral special case: sommerfeld_parameter returns 1.0"""
        result = self.runner.invoke(
            nonrad_cli,
            ['sommerfeld',
             '--z', '0', '--m-eff', '1.0', '--eps0', '10.0',
             '-T', '300']
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')
        # Z=0 → sommerfeld = 1.0
        self.assertIn('1.0', result.output)

    def test_temperature_range(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['sommerfeld',
             '--z', '0', '--m-eff', '1.0', '--eps0', '10.0',
             '--temperature-range', '100', '500', '3']
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')

    def test_missing_required_options(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['sommerfeld', '-T', '300']  # missing --z, --m-eff, --eps0
        )
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Tests for `thermal-velocity`
# ---------------------------------------------------------------------------
class TestThermalVelocity(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(
            nonrad_cli, ['thermal-velocity', '--help']
        )
        self.assertEqual(result.exit_code, 0, result.output)

    def test_basic(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['thermal-velocity', '--m-eff', '1.0', '-T', '300']
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')
        self.assertIn('cm/s', result.output)

    def test_temperature_range(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['thermal-velocity', '--m-eff', '1.0',
             '--temperature-range', '100', '500', '3']
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')

    def test_missing_meff(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['thermal-velocity', '-T', '300']  # missing --m-eff
        )
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Tests for `charged-supercell`
# ---------------------------------------------------------------------------
class TestChargedSupercell(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(
            nonrad_cli, ['charged-supercell', '--help']
        )
        self.assertEqual(result.exit_code, 0, result.output)

    @patch('nonrad.scaling.charged_supercell_scaling_VASP', return_value=0.95)
    def test_basic_with_def_index(self, mock_scaling):
        with self.runner.isolated_filesystem():
            Path('WAVECAR').write_bytes(b'\x00' * 64)
            result = self.runner.invoke(
                nonrad_cli,
                ['charged-supercell', 'WAVECAR',
                 '--bulk-index', '189', '--def-index', '192']
            )
            self.assertEqual(result.exit_code, 0,
                             f'exit_code={result.exit_code}\n{result.output}')

    @patch('nonrad.scaling.charged_supercell_scaling_VASP', return_value=0.95)
    def test_basic_with_def_coord(self, mock_scaling):
        with self.runner.isolated_filesystem():
            Path('WAVECAR').write_bytes(b'\x00' * 64)
            result = self.runner.invoke(
                nonrad_cli,
                ['charged-supercell', 'WAVECAR',
                 '--bulk-index', '189',
                 '--def-coord', '0.5', '0.5', '0.5']
            )
            self.assertEqual(result.exit_code, 0,
                             f'exit_code={result.exit_code}\n{result.output}')

    def test_missing_def_index_and_def_coord(self):
        """Both --def-index and --def-coord are omitted → error."""
        with self.runner.isolated_filesystem():
            Path('WAVECAR').write_bytes(b'\x00' * 64)
            result = self.runner.invoke(
                nonrad_cli,
                ['charged-supercell', 'WAVECAR', '--bulk-index', '189']
            )
            # The CLI or underlying function should flag this as an error
            self.assertNotEqual(result.exit_code, 0,
                                f'Expected error, got: {result.output}')

    def test_nonexistent_wavecar(self):
        result = self.runner.invoke(
            nonrad_cli,
            ['charged-supercell', 'nonexistent_WAVECAR',
             '--bulk-index', '189', '--def-index', '192']
        )
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Tests for `elphon` group
# ---------------------------------------------------------------------------
class TestElphonGroup(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(nonrad_cli, ['elphon', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('elphon', result.output.lower())


# ---------------------------------------------------------------------------
# Tests for `elphon wavecars`
# ---------------------------------------------------------------------------
class TestElphonWavecars(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(
            nonrad_cli, ['elphon', 'wavecars', '--help']
        )
        self.assertEqual(result.exit_code, 0, result.output)

    @patch('nonrad.elphon.get_Wif_from_wavecars',
           return_value=[(189, 0.087)])
    @patch('nonrad.ccd.get_Q_from_struct', return_value=0.5)
    @patch('pymatgen.io.vasp.outputs.Vasprun')
    @patch('pymatgen.core.structure.Structure.from_file')
    def test_basic(self, mock_from_file, mock_vasprun,
                   mock_get_q, mock_get_wif):
        mock_struct = MagicMock()
        mock_from_file.return_value = mock_struct

        mock_vr = MagicMock()
        mock_vr.structures = [mock_struct]
        mock_vasprun.return_value = mock_vr

        with self.runner.isolated_filesystem():
            # Create fake directory structure
            os.makedirs('cc/ground/0', exist_ok=True)
            os.makedirs('cc/excited/0', exist_ok=True)
            Path('cc/ground/0/WAVECAR').touch()
            Path('cc/ground/0/vasprun.xml').touch()
            os.makedirs('ground_files')
            Path('ground_files/CONTCAR').touch()
            os.makedirs('excited_files')
            Path('excited_files/CONTCAR').touch()
            Path('WAVECAR_init').touch()

            result = self.runner.invoke(
                nonrad_cli,
                ['elphon', 'wavecars',
                 'cc', 'ground_files', 'excited_files', 'WAVECAR_init',
                 '--def-index', '192', '-b', '189']
            )
            self.assertEqual(result.exit_code, 0,
                             f'exit_code={result.exit_code}\n{result.output}')


# ---------------------------------------------------------------------------
# Tests for `elphon wswq`
# ---------------------------------------------------------------------------
class TestElphonWswq(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(
            nonrad_cli, ['elphon', 'wswq', '--help']
        )
        self.assertEqual(result.exit_code, 0, result.output)

    @patch('nonrad.elphon.get_Wif_from_WSWQ',
           return_value=[(189, 0.094)])
    @patch('nonrad.ccd.get_Q_from_struct', return_value=0.5)
    @patch('pymatgen.io.vasp.outputs.Vasprun')
    @patch('pymatgen.core.structure.Structure.from_file')
    def test_basic(self, mock_from_file, mock_vasprun,
                   mock_get_q, mock_get_wif):
        mock_struct = MagicMock()
        mock_from_file.return_value = mock_struct

        mock_vr = MagicMock()
        mock_vr.structures = [mock_struct]
        mock_vasprun.return_value = mock_vr

        with self.runner.isolated_filesystem():
            os.makedirs('cc/ground/0', exist_ok=True)
            os.makedirs('cc/excited/0', exist_ok=True)
            Path('cc/ground/0/WSWQ').touch()
            Path('cc/ground/0/vasprun.xml').touch()
            os.makedirs('ground_files')
            Path('ground_files/CONTCAR').touch()
            Path('ground_files/vasprun.xml').touch()
            os.makedirs('excited_files')
            Path('excited_files/CONTCAR').touch()
            Path('excited_files/vasprun.xml').touch()

            result = self.runner.invoke(
                nonrad_cli,
                ['elphon', 'wswq',
                 'cc', 'ground_files', 'excited_files',
                 'ground_files/vasprun.xml',
                 '--def-index', '192', '-b', '189']
            )
            self.assertEqual(result.exit_code, 0,
                             f'exit_code={result.exit_code}\n{result.output}')


# ---------------------------------------------------------------------------
# Tests for `elphon unk`
# ---------------------------------------------------------------------------
class TestElphonUnk(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(
            nonrad_cli, ['elphon', 'unk', '--help']
        )
        self.assertEqual(result.exit_code, 0, result.output)

    @patch('nonrad.elphon.get_Wif_from_UNK',
           return_value=[(1, 0.5)])
    def test_basic(self, mock_get_wif):
        init_unk = str(TEST_FILES / 'UNK.0')
        result = self.runner.invoke(
            nonrad_cli,
            ['elphon', 'unk', init_unk,
             '--def-index', '2', '-b', '1',
             '--eigs', '0.0,1.0',
             '-u', '1.0', str(TEST_FILES / 'UNK.1')]
        )
        self.assertEqual(result.exit_code, 0,
                         f'exit_code={result.exit_code}\n{result.output}')


if __name__ == '__main__':
    unittest.main()

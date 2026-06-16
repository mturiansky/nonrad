# Copyright (c) Chris G. Van de Walle
# Distributed under the terms of the MIT License.

"""Command-line interface for nonrad.

This module provides a Click-based CLI exposing the main functionality of
the ``nonrad`` package, including configuration-coordinate diagram (CCD)
utilities, the nonradiative capture coefficient calculation, scaling
helpers, and electron-phonon coupling tools.

Commands
--------
prep-ccd
    Prepare displaced structures for a CCD calculation.
pes
    Extract potential energy surfaces from CCD VASP outputs.
dq
    Compute the configuration coordinate displacement *dQ*.
q-from-struct
    Compute the *Q* value for an arbitrary structure.
barrier
    Compute the classical barrier height in the harmonic approximation.
capture
    Evaluate the nonradiative capture coefficient.
sommerfeld
    Compute the Sommerfeld enhancement factor.
thermal-velocity
    Compute the thermal velocity of a carrier.
charged-supercell
    Estimate the charged-supercell scaling factor.
elphon wavecars
    Compute Wif from WAVECAR files.
elphon wswq
    Compute Wif from WSWQ files.
elphon unk
    Compute Wif from UNK files.
"""

import os
from glob import glob
from pathlib import Path
from shutil import copyfile

import click
import numpy as np


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name='nonrad')
def nonrad():
    """Nonradiative recombination coefficient calculator.

    A command-line tool for computing nonradiative capture coefficients
    and related quantities from first-principles calculations.
    """


# ---------------------------------------------------------------------------
# CCD commands
# ---------------------------------------------------------------------------

@nonrad.command(name='prep-ccd')
@click.argument('ground_path', type=click.Path(exists=True))
@click.argument('excited_path', type=click.Path(exists=True))
@click.argument('cc_dir', type=click.Path())
@click.option(
    '--displace', '-d',
    nargs=3, type=float, default=(-0.5, 0.5, 9),
    show_default=True,
    help='Displacement range (min, max, count).',
)
@click.option(
    '--input-files', '-f',
    type=str, default='KPOINTS,POTCAR,INCAR,job_script.sh',
    show_default=True,
    help='Comma-separated list of input files to copy into each sub-directory.',
)
def prep_ccd(ground_path, excited_path, cc_dir, displace, input_files):
    """Prepare displaced structures for a configuration-coordinate diagram.

    Reads CONTCAR from GROUND_PATH and EXCITED_PATH, generates displaced
    structures, writes POSCAR files into CC_DIR/ground/N and
    CC_DIR/excited/N sub-directories, and copies the requested input files.

    Parameters
    ----------
    ground_path : str
        Directory containing the ground-state CONTCAR.
    excited_path : str
        Directory containing the excited-state CONTCAR.
    cc_dir : str
        Output directory for the CCD sub-directories.
    displace : tuple of float
        ``(min, max, count)`` passed to ``numpy.linspace``.
    input_files : str
        Comma-separated filenames to copy from each source directory.
    """
    from pymatgen.core import Structure

    from nonrad.ccd import get_cc_structures

    ground_files = Path(ground_path)
    excited_files = Path(excited_path)
    ground_struct = Structure.from_file(str(ground_files / 'CONTCAR'))
    excited_struct = Structure.from_file(str(excited_files / 'CONTCAR'))

    cc_dir = Path(cc_dir)
    os.makedirs(str(cc_dir), exist_ok=True)
    os.makedirs(str(cc_dir / 'ground'), exist_ok=True)
    os.makedirs(str(cc_dir / 'excited'), exist_ok=True)

    displacements = np.linspace(displace[0], displace[1], int(displace[2]))
    ground, excited = get_cc_structures(ground_struct, excited_struct,
                                        displacements)

    files_to_copy = [f.strip() for f in input_files.split(',')]

    for i, struct in enumerate(ground):
        working_dir = cc_dir / 'ground' / str(i)
        os.makedirs(str(working_dir), exist_ok=True)
        struct.to(filename=str(working_dir / 'POSCAR'), fmt='poscar')
        for f in files_to_copy:
            src = ground_files / f
            if src.exists():
                copyfile(str(src), str(working_dir / f))
            else:
                click.echo(f'Warning: {src} not found, skipping.')

    for i, struct in enumerate(excited):
        working_dir = cc_dir / 'excited' / str(i)
        os.makedirs(str(working_dir), exist_ok=True)
        struct.to(filename=str(working_dir / 'POSCAR'), fmt='poscar')
        for f in files_to_copy:
            src = excited_files / f
            if src.exists():
                copyfile(str(src), str(working_dir / f))
            else:
                click.echo(f'Warning: {src} not found, skipping.')

    click.echo(f'Prepared {len(ground)} ground and {len(excited)} excited '
               f'structures in {cc_dir}')


@nonrad.command(name='pes')
@click.argument('cc_dir', type=click.Path(exists=True))
@click.argument('ground_files', type=click.Path(exists=True))
@click.argument('excited_files', type=click.Path(exists=True))
@click.option(
    '--energy-diff', '-e',
    type=float, required=True,
    help='Energy difference dE between ground and excited states (eV).',
)
@click.option(
    '--plot', '-p',
    is_flag=True, default=False,
    help='Plot the potential energy surfaces.',
)
@click.option(
    '--plot-name', '-n',
    type=str, default='pes.png',
    show_default=True,
    help='Filename for the PES plot.',
)
def pes(cc_dir, ground_files, excited_files, energy_diff, plot, plot_name):
    """Extract potential energy surfaces from CCD calculations.

    Reads vasprun.xml files from CC_DIR/ground/*/vasprun.xml and
    CC_DIR/excited/*/vasprun.xml (including ``.gz`` variants), appends
    the equilibrium vaspruns from GROUND_FILES and EXCITED_FILES, and
    computes the PES and harmonic phonon frequencies.

    Parameters
    ----------
    cc_dir : str
        Root directory of the CCD calculation tree.
    ground_files : str
        Directory with the ground-state equilibrium calculation.
    excited_files : str
        Directory with the excited-state equilibrium calculation.
    energy_diff : float
        Energy offset *dE* between the two PES (eV).
    plot : bool
        Whether to save a PES plot.
    plot_name : str
        Output filename for the plot image.
    """
    from pymatgen.core import Structure

    from nonrad.ccd import (
        get_dQ,
        get_omega_from_PES,
        get_PES_from_vaspruns,
    )

    ground_struct = Structure.from_file(str(Path(ground_files) / 'CONTCAR'))
    excited_struct = Structure.from_file(str(Path(excited_files) / 'CONTCAR'))

    dQ = get_dQ(ground_struct, excited_struct)
    click.echo(f'dQ = {dQ:.2f} amu^{{1/2}} Angstrom')

    # Collect vasprun.xml (and .gz) from the CCD sub-directories
    ground_vaspruns = (
        glob(str(Path(cc_dir) / 'ground' / '*' / 'vasprun.xml')) +
        glob(str(Path(cc_dir) / 'ground' / '*' / 'vasprun.xml.gz'))
    )
    excited_vaspruns = (
        glob(str(Path(cc_dir) / 'excited' / '*' / 'vasprun.xml')) +
        glob(str(Path(cc_dir) / 'excited' / '*' / 'vasprun.xml.gz'))
    )

    # Append equilibrium vaspruns
    for gf in [str(Path(ground_files) / 'vasprun.xml'),
               str(Path(ground_files) / 'vasprun.xml.gz')]:
        if os.path.isfile(gf):
            ground_vaspruns.append(gf)
            break

    for ef in [str(Path(excited_files) / 'vasprun.xml'),
               str(Path(excited_files) / 'vasprun.xml.gz')]:
        if os.path.isfile(ef):
            excited_vaspruns.append(ef)
            break

    Q_ground, E_ground = get_PES_from_vaspruns(
        ground_struct, excited_struct, ground_vaspruns,
    )
    Q_excited, E_excited = get_PES_from_vaspruns(
        ground_struct, excited_struct, excited_vaspruns,
    )
    E_excited = energy_diff + E_excited

    # Always compute omega
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(Q_ground, E_ground, s=10)
        ax.scatter(Q_excited, E_excited, s=10)
        q = np.linspace(-1.0, 3.5, 100)
        ground_omega = get_omega_from_PES(Q_ground, E_ground, ax=ax, q=q)
        excited_omega = get_omega_from_PES(Q_excited, E_excited, ax=ax, q=q)
        ax.set_xlabel(r'$Q$ [amu$^{1/2}$ $\AA$]')
        ax.set_ylabel(r'$E$ [eV]')
        plt.savefig(plot_name, dpi=300, format=plot_name.split('.')[-1])
        click.echo(f'Plot saved to {plot_name}')
    else:
        ground_omega = get_omega_from_PES(Q_ground, E_ground)
        excited_omega = get_omega_from_PES(Q_excited, E_excited)

    click.echo(f'Ground state omega = {ground_omega:.2f} eV')
    click.echo(f'Excited state omega = {excited_omega:.2f} eV')


@nonrad.command(name='dq')
@click.argument('ground_path', type=click.Path(exists=True))
@click.argument('excited_path', type=click.Path(exists=True))
def dq_cmd(ground_path, excited_path):
    """Compute the configuration-coordinate displacement dQ.

    GROUND_PATH and EXCITED_PATH are structure files (e.g. POSCAR,
    CONTCAR, CIF) — *not* directories.

    Parameters
    ----------
    ground_path : str
        Path to the ground-state structure file.
    excited_path : str
        Path to the excited-state structure file.
    """
    from pymatgen.core import Structure

    from nonrad.ccd import get_dQ

    ground_struct = Structure.from_file(ground_path)
    excited_struct = Structure.from_file(excited_path)
    dQ = get_dQ(ground_struct, excited_struct)
    click.echo(f'dQ = {dQ:.6f} amu^{{1/2}} Angstrom')


@nonrad.command(name='q-from-struct')
@click.argument('ground_path', type=click.Path(exists=True))
@click.argument('excited_path', type=click.Path(exists=True))
@click.argument('struct_path', type=click.Path(exists=True))
@click.option(
    '--tol', '-t',
    type=float, default=1e-4,
    show_default=True,
    help='Distance tolerance for filtering coordinates.',
)
@click.option(
    '--nround',
    type=int, default=5,
    show_default=True,
    help='Number of decimal places for rounding Q.',
)
def q_from_struct(ground_path, excited_path, struct_path, tol, nround):
    """Compute the Q value of an arbitrary structure.

    Given the ground and excited endpoint structures, determine the Q
    coordinate of STRUCT_PATH assuming linear interpolation.

    Parameters
    ----------
    ground_path : str
        Path to the ground-state structure file.
    excited_path : str
        Path to the excited-state structure file.
    struct_path : str
        Path to the structure file to evaluate.
    tol : float
        Distance cutoff for discarding nearly-stationary sites.
    nround : int
        Decimal places used when rounding Q.
    """
    from pymatgen.core import Structure

    from nonrad.ccd import get_Q_from_struct

    ground_struct = Structure.from_file(ground_path)
    excited_struct = Structure.from_file(excited_path)
    Q = get_Q_from_struct(ground_struct, excited_struct, struct_path,
                          tol=tol, nround=nround)
    click.echo(f'Q = {Q}')


@nonrad.command(name='barrier')
@click.option('--dq', type=float, required=True,
              help='Displacement dQ (amu^{1/2} Angstrom).')
@click.option('--de', type=float, required=True,
              help='Energy offset dE (eV).')
@click.option('--wi', type=float, required=True,
              help='Initial-state frequency (eV).')
@click.option('--wf', type=float, required=True,
              help='Final-state frequency (eV).')
def barrier_cmd(dq, de, wi, wf):
    """Compute the classical barrier height in the harmonic approximation.

    Parameters
    ----------
    dq : float
        Configuration-coordinate displacement (amu^{1/2} Angstrom).
    de : float
        Energy offset between oscillators (eV).
    wi : float
        Initial harmonic-oscillator frequency (eV).
    wf : float
        Final harmonic-oscillator frequency (eV).
    """
    from nonrad.ccd import get_barrier_harmonic

    result = get_barrier_harmonic(dq, de, wi, wf)
    if result is not None:
        click.echo(f'Barrier height = {result:.6f} eV')
    else:
        click.echo('No crossing point found.')


# ---------------------------------------------------------------------------
# Core capture coefficient command
# ---------------------------------------------------------------------------

@nonrad.command(name='capture')
@click.option('--dq', type=float, required=True,
              help='Displacement dQ (amu^{1/2} Angstrom).')
@click.option('--de', type=float, required=True,
              help='Energy offset dE (eV).')
@click.option('--wi', type=float, required=True,
              help='Initial-state frequency (eV).')
@click.option('--wf', type=float, required=True,
              help='Final-state frequency (eV).')
@click.option('--wif', type=float, required=True,
              help='Electron-phonon coupling Wif (eV amu^{-1/2} Angstrom^{-1}).')
@click.option('--volume', type=float, required=True,
              help='Supercell volume (Angstrom^3).')
@click.option('--g', type=int, default=1, show_default=True,
              help='Degeneracy factor.')
@click.option('--temperature', '-T', type=float, default=300.,
              show_default=True,
              help='Temperature (K) for a single-point evaluation.')
@click.option('--temperature-range', nargs=3, type=float, default=None,
              help='Temperature range as (min, max, count) for scanning.')
@click.option('--sigma', type=str, default='pchip', show_default=True,
              help='Smearing parameter (float) or interpolation method '
                   '("pchip" or "cubic").')
@click.option('--occ-tol', type=float, default=1e-5, show_default=True,
              help='Occupation tolerance for Bose weights.')
@click.option('--overlap-method',
              type=click.Choice(['Analytic', 'Integral', 'HermiteGauss'],
                                case_sensitive=False),
              default='HermiteGauss', show_default=True,
              help='Method for evaluating vibrational overlaps.')
def capture_cmd(dq, de, wi, wf, wif, volume, g, temperature,
                temperature_range, sigma, occ_tol, overlap_method):
    """Compute the nonradiative capture coefficient.

    Evaluates the capture coefficient following Alkauskas *et al.*,
    Phys. Rev. B 90, 075202 (2014).  The result is **unscaled** and has
    units of cm^3 s^{-1}.

    Parameters
    ----------
    dq : float
        Configuration-coordinate displacement (amu^{1/2} Angstrom).
    de : float
        Energy offset between oscillators (eV).
    wi : float
        Initial-state frequency (eV).
    wf : float
        Final-state frequency (eV).
    wif : float
        Electron-phonon coupling matrix element
        (eV amu^{-1/2} Angstrom^{-1}).
    volume : float
        Supercell volume (Angstrom^3).
    g : int
        Degeneracy factor of the final state.
    temperature : float
        Single temperature (K).
    temperature_range : tuple of float or None
        ``(min, max, count)`` for a temperature scan.
    sigma : str
        Smearing parameter or interpolation keyword.
    occ_tol : float
        Bose-weight occupation tolerance.
    overlap_method : str
        Overlap evaluation method.
    """
    from nonrad.nonrad import get_C

    # Parse sigma: interpret as float when possible
    try:
        sigma_val = float(sigma)
    except ValueError:
        sigma_val = sigma

    if temperature_range is not None:
        temps = np.linspace(temperature_range[0], temperature_range[1],
                            int(temperature_range[2]))
    else:
        temps = temperature

    result = get_C(dq, de, wi, wf, wif, volume,
                   g=g, T=temps, sigma=sigma_val,
                   occ_tol=occ_tol, overlap_method=overlap_method)

    if isinstance(temps, np.ndarray):
        click.echo(f'{"T (K)":>12s}  {"C (cm^3/s)":>14s}')
        click.echo('-' * 28)
        for t, c in zip(temps, np.atleast_1d(result)):
            click.echo(f'{t:12.2f}  {c:14.6e}')
    else:
        click.echo(f'C = {result:.6e} cm^3/s  (T = {temps} K)')


# ---------------------------------------------------------------------------
# Scaling commands
# ---------------------------------------------------------------------------

@nonrad.command(name='sommerfeld')
@click.option('--temperature', '-T', type=float, default=300.,
              show_default=True, help='Temperature (K).')
@click.option('--temperature-range', nargs=3, type=float, default=None,
              help='Temperature range as (min, max, count).')
@click.option('--z', type=int, required=True,
              help='Charge ratio Z = Q/q (negative = attractive).')
@click.option('--m-eff', type=float, required=True,
              help='Effective mass in units of electron mass.')
@click.option('--eps0', type=float, required=True,
              help='Static dielectric constant.')
@click.option('--dim',
              type=click.Choice(['1', '2', '3'], case_sensitive=False),
              default='3', show_default=True,
              help='Dimensionality of the system.')
@click.option('--method',
              type=click.Choice(['Integrate', 'Analytic'],
                                case_sensitive=False),
              default='Integrate', show_default=True,
              help='Evaluation method.')
def sommerfeld_cmd(temperature, temperature_range, z, m_eff, eps0, dim,
                   method):
    """Compute the Sommerfeld enhancement factor.

    Evaluates the temperature-dependent Sommerfeld parameter following
    Pässler *et al.*, phys. stat. sol. (b) 78, 625 (1976).

    Parameters
    ----------
    temperature : float
        Temperature (K) for a single-point evaluation.
    temperature_range : tuple of float or None
        ``(min, max, count)`` for a temperature scan.
    z : int
        Charge ratio *Z = Q / q*.
    m_eff : float
        Carrier effective mass in units of *m_e*.
    eps0 : float
        Static dielectric constant.
    dim : str
        System dimensionality (``'1'``, ``'2'``, or ``'3'``).
    method : str
        ``'Integrate'`` or ``'Analytic'``.
    """
    from nonrad.scaling import sommerfeld_parameter

    dim_int = int(dim)

    if temperature_range is not None:
        temps = np.linspace(temperature_range[0], temperature_range[1],
                            int(temperature_range[2]))
    else:
        temps = temperature

    result = sommerfeld_parameter(temps, z, m_eff, eps0,
                                  dim=dim_int, method=method)

    if isinstance(temps, np.ndarray):
        click.echo(f'{"T (K)":>12s}  {"Sommerfeld":>14s}')
        click.echo('-' * 28)
        for t, s in zip(temps, np.atleast_1d(result)):
            click.echo(f'{t:12.2f}  {s:14.6e}')
    else:
        click.echo(f'Sommerfeld factor = {result:.6e}  (T = {temps} K)')


@nonrad.command(name='thermal-velocity')
@click.option('--temperature', '-T', type=float, default=300.,
              show_default=True, help='Temperature (K).')
@click.option('--temperature-range', nargs=3, type=float, default=None,
              help='Temperature range as (min, max, count).')
@click.option('--m-eff', type=float, required=True,
              help='Effective mass in units of electron mass.')
def thermal_velocity_cmd(temperature, temperature_range, m_eff):
    """Compute the thermal velocity of a carrier.

    Parameters
    ----------
    temperature : float
        Temperature (K) for a single-point evaluation.
    temperature_range : tuple of float or None
        ``(min, max, count)`` for a temperature scan.
    m_eff : float
        Carrier effective mass in units of *m_e*.
    """
    from nonrad.scaling import thermal_velocity

    if temperature_range is not None:
        temps = np.linspace(temperature_range[0], temperature_range[1],
                            int(temperature_range[2]))
    else:
        temps = temperature

    result = thermal_velocity(temps, m_eff)

    if isinstance(temps, np.ndarray):
        click.echo(f'{"T (K)":>12s}  {"v_th (cm/s)":>14s}')
        click.echo('-' * 28)
        for t, v in zip(temps, np.atleast_1d(result)):
            click.echo(f'{t:12.2f}  {v:14.6e}')
    else:
        click.echo(f'Thermal velocity = {result:.6e} cm/s  (T = {temps} K)')


@nonrad.command(name='charged-supercell')
@click.argument('wavecar_path', type=click.Path(exists=True))
@click.option('--bulk-index', type=int, required=True,
              help='Index of the bulk wavefunction (1-based).')
@click.option('--def-index', type=int, default=-1, show_default=True,
              help='Index of the defect wavefunction (1-based). '
                   'Provide this or --def-coord.')
@click.option('--def-coord', nargs=3, type=float, default=None,
              help='Cartesian coordinates (x y z) of the defect position.')
@click.option('--cutoff', type=float, default=0.02, show_default=True,
              help='Slope cutoff for plateau detection.')
@click.option('--limit', type=float, default=5., show_default=True,
              help='Upper radial limit (Angstrom) for the fitting window.')
@click.option('--spin', type=int, default=0, show_default=True,
              help='Spin channel (0 = up, 1 = down).')
@click.option('--kpoint', type=int, default=1, show_default=True,
              help='k-point index (1-based).')
def charged_supercell_cmd(wavecar_path, bulk_index, def_index, def_coord,
                          cutoff, limit, spin, kpoint):
    """Estimate the charged-supercell scaling factor.

    Compares the radial distribution of the bulk wavefunction around the
    defect position to a uniform distribution and returns a scaling
    factor for the capture coefficient.

    Parameters
    ----------
    wavecar_path : str
        Path to the WAVECAR file.
    bulk_index : int
        Band index of the bulk wavefunction (1-based).
    def_index : int
        Band index of the defect wavefunction (1-based), or -1 if
        ``def_coord`` is provided instead.
    def_coord : tuple of float or None
        Cartesian coordinates of the defect centre.
    cutoff : float
        Gradient cutoff for plateau identification.
    limit : float
        Upper radius limit for curve fitting (Angstrom).
    spin : int
        Spin channel (0 = up, 1 = down).
    kpoint : int
        k-point index (1-based).
    """
    from nonrad.scaling import charged_supercell_scaling_VASP

    if def_index == -1 and def_coord is None:
        raise click.UsageError(
            'Either --def-index (not -1) or --def-coord must be specified.'
        )

    coord = np.array(def_coord) if def_coord is not None else None

    scaling = charged_supercell_scaling_VASP(
        wavecar_path, bulk_index,
        def_index=def_index, def_coord=coord,
        cutoff=cutoff, limit=limit,
        spin=spin, kpoint=kpoint,
    )
    click.echo(f'Charged-supercell scaling factor = {scaling:.6f}')


# ---------------------------------------------------------------------------
# Electron-phonon sub-group
# ---------------------------------------------------------------------------

@nonrad.group('elphon')
def elphon():
    """Electron-phonon coupling utilities.

    Sub-commands for computing the electron-phonon coupling matrix
    element Wif using WAVECARs, WSWQ files, or UNK files.
    """


@elphon.command(name='wavecars')
@click.argument('cc_dir', type=click.Path(exists=True))
@click.argument('ground_path', type=click.Path(exists=True))
@click.argument('excited_path', type=click.Path(exists=True))
@click.argument('init_wavecar', type=click.Path(exists=True))
@click.option('--def-index', type=int, required=True,
              help='Defect wavefunction index (1-based).')
@click.option('--bulk-index', '-b', type=int, required=True, multiple=True,
              help='Bulk wavefunction index (1-based, repeatable).')
@click.option('--spin', type=int, default=0, show_default=True,
              help='Spin channel (0 = up, 1 = down).')
@click.option('--kpoint', type=int, default=1, show_default=True,
              help='k-point index (1-based).')
def elphon_wavecars(cc_dir, ground_path, excited_path, init_wavecar,
                    def_index, bulk_index, spin, kpoint):
    """Compute Wif from WAVECAR files along the CCD path.

    Automatically discovers sub-directories inside CC_DIR that contain
    both a ``vasprun.xml`` (or ``.gz``) and a ``WAVECAR`` file.

    Parameters
    ----------
    cc_dir : str
        Directory tree with CCD calculations.
    ground_path : str
        Directory containing the ground-state CONTCAR.
    excited_path : str
        Directory containing the excited-state CONTCAR.
    init_wavecar : str
        Path to the initial (reference) WAVECAR.
    def_index : int
        Band index of the defect wavefunction.
    bulk_index : tuple of int
        Band indices of the bulk wavefunctions.
    spin : int
        Spin channel.
    kpoint : int
        k-point index.
    """
    from pymatgen.core import Structure
    from pymatgen.io.vasp.outputs import Vasprun

    from nonrad.ccd import get_Q_from_struct
    from nonrad.elphon import get_Wif_from_wavecars

    ground_struct = Structure.from_file(str(Path(ground_path) / 'CONTCAR'))
    excited_struct = Structure.from_file(str(Path(excited_path) / 'CONTCAR'))

    # Auto-discover sub-directories with vasprun + WAVECAR
    wavecars = []
    for subdir in sorted(Path(cc_dir).iterdir()):
        if not subdir.is_dir():
            continue
        wavecar_file = subdir / 'WAVECAR'
        if not wavecar_file.exists():
            continue
        vr_path = None
        for vr_name in ('vasprun.xml', 'vasprun.xml.gz'):
            candidate = subdir / vr_name
            if candidate.exists():
                vr_path = candidate
                break
        if vr_path is None:
            continue
        vr = Vasprun(str(vr_path), parse_dos=False, parse_eigen=False)
        q = get_Q_from_struct(ground_struct, excited_struct,
                              vr.structures[-1])
        wavecars.append((q, str(wavecar_file)))

    results = get_Wif_from_wavecars(
        wavecars, init_wavecar, def_index, list(bulk_index),
        spin=spin, kpoint=kpoint,
    )

    click.echo(f'{"bulk_index":>12s}  {"Wif (eV amu^{-1/2} A^{-1})":>30s}')
    click.echo('-' * 44)
    for bi, wif_val in results:
        click.echo(f'{bi:12d}  {wif_val:30.6e}')


@elphon.command(name='wswq')
@click.argument('cc_dir', type=click.Path(exists=True))
@click.argument('ground_path', type=click.Path(exists=True))
@click.argument('excited_path', type=click.Path(exists=True))
@click.argument('init_vasprun', type=click.Path(exists=True))
@click.option('--def-index', type=int, required=True,
              help='Defect wavefunction index (1-based).')
@click.option('--bulk-index', '-b', type=int, required=True, multiple=True,
              help='Bulk wavefunction index (1-based, repeatable).')
@click.option('--spin', type=int, default=0, show_default=True,
              help='Spin channel (0 = up, 1 = down).')
@click.option('--kpoint', type=int, default=1, show_default=True,
              help='k-point index (1-based).')
def elphon_wswq(cc_dir, ground_path, excited_path, init_vasprun,
                def_index, bulk_index, spin, kpoint):
    """Compute Wif from WSWQ files along the CCD path.

    Automatically discovers sub-directories inside CC_DIR that contain
    both a ``vasprun.xml`` (or ``.gz``) and a ``WSWQ`` (or ``.gz``)
    file.

    Parameters
    ----------
    cc_dir : str
        Directory tree with CCD calculations.
    ground_path : str
        Directory containing the ground-state CONTCAR.
    excited_path : str
        Directory containing the excited-state CONTCAR.
    init_vasprun : str
        Path to the initial vasprun.xml for eigenvalue extraction.
    def_index : int
        Band index of the defect wavefunction.
    bulk_index : tuple of int
        Band indices of the bulk wavefunctions.
    spin : int
        Spin channel.
    kpoint : int
        k-point index.
    """
    from pymatgen.core import Structure
    from pymatgen.io.vasp.outputs import Vasprun

    from nonrad.ccd import get_Q_from_struct
    from nonrad.elphon import get_Wif_from_WSWQ

    ground_struct = Structure.from_file(str(Path(ground_path) / 'CONTCAR'))
    excited_struct = Structure.from_file(str(Path(excited_path) / 'CONTCAR'))

    # Auto-discover sub-directories with vasprun + WSWQ
    wswqs = []
    for subdir in sorted(Path(cc_dir).iterdir()):
        if not subdir.is_dir():
            continue
        wswq_file = None
        for wswq_name in ('WSWQ', 'WSWQ.gz'):
            candidate = subdir / wswq_name
            if candidate.exists():
                wswq_file = candidate
                break
        if wswq_file is None:
            continue
        vr_path = None
        for vr_name in ('vasprun.xml', 'vasprun.xml.gz'):
            candidate = subdir / vr_name
            if candidate.exists():
                vr_path = candidate
                break
        if vr_path is None:
            continue
        vr = Vasprun(str(vr_path), parse_dos=False, parse_eigen=False)
        q = get_Q_from_struct(ground_struct, excited_struct,
                              vr.structures[-1])
        wswqs.append((q, str(wswq_file)))

    results = get_Wif_from_WSWQ(
        wswqs, init_vasprun, def_index, list(bulk_index),
        spin=spin, kpoint=kpoint,
    )

    click.echo(f'{"bulk_index":>12s}  {"Wif (eV amu^{-1/2} A^{-1})":>30s}')
    click.echo('-' * 44)
    for bi, wif_val in results:
        click.echo(f'{bi:12d}  {wif_val:30.6e}')


@elphon.command(name='unk')
@click.argument('init_unk', type=click.Path(exists=True))
@click.option('--def-index', type=int, required=True,
              help='Defect wavefunction index (1-based).')
@click.option('--bulk-index', '-b', type=int, required=True, multiple=True,
              help='Bulk wavefunction index (1-based, repeatable).')
@click.option('--eigs', type=str, required=True,
              help='Comma-separated eigenvalues (eV).')
@click.option('--unk-pair', '-u', type=(float, str), required=True,
              multiple=True,
              help='(Q, unk_path) pair — repeatable.')
def elphon_unk(init_unk, def_index, bulk_index, eigs, unk_pair):
    """Compute Wif from UNK files.

    Parameters
    ----------
    init_unk : str
        Path to the initial (reference) UNK file.
    def_index : int
        Band index of the defect wavefunction.
    bulk_index : tuple of int
        Band indices of the bulk wavefunctions.
    eigs : str
        Comma-separated string of eigenvalues (eV).
    unk_pair : tuple of (float, str)
        One or more ``(Q, path)`` pairs specifying displaced UNK files.
    """
    from nonrad.elphon import get_Wif_from_UNK

    eigs_array = np.array([float(e) for e in eigs.split(',')])
    unk_list = [(q, path) for q, path in unk_pair]

    results = get_Wif_from_UNK(
        unk_list, init_unk, def_index, list(bulk_index),
        eigs=eigs_array,
    )

    click.echo(f'{"bulk_index":>12s}  {"Wif (eV amu^{-1/2} A^{-1})":>30s}')
    click.echo('-' * 44)
    for bi, wif_val in results:
        click.echo(f'{bi:12d}  {wif_val:30.6e}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    nonrad()

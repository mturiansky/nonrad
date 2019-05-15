import re
import numpy as np
from itertools import groupby
from pymatgen.io.vasp.outputs import Vasprun, BSVasprun, Wavecar
from pymatgen.electronic_structure.core import Spin
from scipy.optimize import curve_fit
from nonrad.nonrad import HBAR, EV2J, AMU2KG, ANGS2M


def get_cc_structures(ground, excited, displacements, remove_zero=True):
    """
    Generate the structures for a CC diagram

    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    excited : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state
    displacements : list(float)
        list of displacements to compute the perturbed structures. Note: the
        displacements are for only one potential energy surface and will be
        applied to both (e.g. displacements=np.linspace(-0.1, 0.1, 5)) will
        return 10 structures 5 of the ground state displaced at +-10%, +-5%,
        and 0% and 5 of the excited state displaced similarly)
    remove_zero : bool
        remove 0% displacement from list (default is True)

    Returns
    -------
    ground_structs = list(pymatgen.core.structure.Struture)
        a list of structures corresponding to the displaced ground state
    excited_structs = list(pymatgen.core.structure.Structure)
        a list of structures corresponding to the displaced excited state
    """
    displacements = np.array(displacements)
    if remove_zero:
        displacements = displacements[displacements != 0.]
    ground_structs = ground.interpolate(excited, nimages=displacements)
    excited_structs = ground.interpolate(excited, nimages=(displacements + 1.))
    return ground_structs, excited_structs


def get_dQ(ground, excited):
    """
    Calculate dQ from the initial and final structures

    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    excited : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state

    Returns
    -------
    float
        the dQ value (amu^{1/2} Angstrom)
    """
    return np.sqrt(np.sum(list(map(
        lambda x: x[0].distance(x[1])**2 * x[0].specie.atomic_mass,
        zip(ground, excited)
    ))))


def get_Q_from_struct(ground, excited, struct, tol=0.001):
    """
    Calculate the Q value for a given structure.

    This function calculates the Q value for a given structure, knowing the
    endpoints and assuming linear interpolation.

    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    excited : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state
    struct : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the structure we want to calculate
        the Q value for
    tol : float
        distance cutoff to throw away sites for determining Q (sites that
        don't move very far could introduce numerical noise)

    Returns
    -------
    float
        the Q value (amu^{1/2} Angstrom) of the structure
    """
    dQ = get_dQ(ground, excited)
    possible_x = []
    for i, site in enumerate(struct):
        if ground[i].distance(excited[i]) < 0.001:
            continue
        possible_x += ((site.coords - ground[i].coords) /
                       (excited[i].coords - ground[i].coords)).tolist()
    spossible_x = np.sort(np.round(possible_x, 6))
    return dQ * max(groupby(spossible_x), key=lambda x: len(list(x[1])))[0]


def get_PES_from_vaspruns(ground, excited, vasprun_paths):
    """
    Extract the potential energy surface (PES) from vasprun.xml files.

    This function reads in vasprun.xml files to extract the energy and Q value
    of each calculation and then returns it as a list.

    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    excited : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state
    vasprun_paths : list(strings)
        a list of paths to each of the vasprun.xml files that make up the PES.
        Note that the minimum (0% displacement) should be included in the list,
        and each path should end in 'vasprun.xml' (e.g. /path/to/vasprun.xml)

    Returns
    -------
    Q : np.array(float)
        array of Q values (amu^{1/2} Angstrom) corresponding to each vasprun
    energy : np.array(float)
        array of energies (eV) corresponding to each vasprun
    """
    num = len(vasprun_paths)
    Q, energy = (np.zeros(num), np.zeros(num))
    for i, vr_fname in enumerate(vasprun_paths):
        vr = Vasprun(vr_fname, parse_dos=False, parse_eigen=False)
        Q[i] = get_Q_from_struct(ground, excited, vr.structures[-1])
        energy[i] = vr.final_energy
    return Q, (energy - np.min(energy))


def get_omega_from_PES(Q, energy, Q0=None, ax=None):
    """
    Calculates the harmonic phonon frequency for the given PES.

    Parameters
    ----------
    Q : np.array(float)
        array of Q values (amu^{1/2} Angstrom) corresponding to each vasprun
    energy : np.array(float)
        array of energies (eV) corresponding to each vasprun
    Q0 : float
        fix the minimum of the parabola (default is None)
    ax : matplotlib.axes.Axes
        optional axis object to plot the resulting fit (default is None)

    Returns
    -------
    float
        harmonic phonon frequency from the PES in eV
    """
    def f(Q, omega, Q0, dE):
        return 0.5 * omega * (Q - Q0)**2 + dE

    # set bounds to restrict Q0 to the given Q0 value
    bounds = (-np.inf, np.inf) if Q0 is None else \
        ([-np.inf, Q0 - 1e-10, -np.inf], [np.inf, Q0, np.inf])
    popt, pcov = curve_fit(f, Q, energy, bounds=bounds)

    # optional plotting to check fit
    if ax is not None:
        q = np.linspace(np.min(Q), np.max(Q), 1000)
        ax.plot(q, f(q, *popt))

    return HBAR * np.sqrt(popt[0] * EV2J / (ANGS2M**2 * AMU2KG))


def _compute_matel(psi0, psi1):
    """
    Computes the inner product of the two wavefunctions.

    Parameters
    ----------
    psi0 : np.array
        first wavefunction
    psi1 : np.array
        second wavefunction

    Returns
    -------
    float
        inner product np.abs(<psi0 | psi1>)
    """
    npsi0 = psi0 / np.sqrt(np.abs(np.vdot(psi0, psi0)))
    npsi1 = psi1 / np.sqrt(np.abs(np.vdot(psi1, psi1)))
    return np.abs(np.vdot(npsi0, npsi1))


def get_Wif_from_wavecars(wavecars, init_wavecar_path, def_index, bulk_index,
                          spin=0, kpoint=1, fig=None):
    """
    Computes the electron-phonon matrix element using the WAVECAR's.

    This function reads in the pseudo-wavefunctions from the WAVECAR files and
    computes the overlaps necessary.

    *** WARNING: USE AT YOUR OWN RISK ***
    Because these are pseudo-wavefunctions, the core information from the PAWs
    is missing. As a result, the resulting Wif value may be unreliable. A good
    test of this is how close the Q=0 overlap is to 0. (it would be exactly 0.
    if you include the corrections from the PAWs). This should only be used
    to get a preliminary idea of the Wif value.
    ***************

    Parameters
    ----------
    wavecars : list((Q, wavecar_path))
        a list of tuples where the first value is the Q and the second is the
        path to the WAVECAR file
    init_wavecar_path : string
        path to the initial wavecar for computing overlaps
    def_index : int
        index corresponding to the defect wavefunction (1-based indexing)
    bulk_index : int, list(int)
        index or list of indices corresponding to the bulk wavefunction
        (1-based indexing)
    spin : int
        spin channel to read from (0 - up, 1 - down)
    kpoint : int
        kpoint to read from (defaults to the first kpoint)
    fig : matplotlib.figure.Figure
        optional figure object to plot diagnostic information

    Returns
    -------
    list((bulk_index, Wif))
        electron-phonon matrix element Wif in units of
        eV amu^{-1/2} Angstrom^{-1} for each bulk_index
    """
    bulk_index = np.array(bulk_index)
    initial_wavecar = Wavecar(init_wavecar_path)
    if initial_wavecar.spin == 2:
        psi_i = initial_wavecar.coeffs[spin][kpoint-1][def_index-1]
    else:
        psi_i = initial_wavecar.coeffs[kpoint-1][def_index-1]

    Nw, Nbi = (len(wavecars), len(bulk_index))
    Q, matels, deig = (np.zeros(Nw+1), np.zeros((Nbi, Nw+1)), np.zeros(Nbi))

    # first compute the Q = 0 values and eigenvalue differences
    for i, bi in enumerate(bulk_index):
        if initial_wavecar.spin == 2:
            psi_f = initial_wavecar.coeffs[spin][kpoint-1][bi-1]
            deig[i] = initial_wavecar.band_energy[spin][kpoint-1][bi-1][0] - \
                initial_wavecar.band_energy[spin][kpoint-1][def_index-1][0]
        else:
            psi_f = initial_wavecar.coeffs[kpoint-1][bi-1]
            deig[i] = initial_wavecar.band_energy[kpoint-1][bi-1][0] - \
                initial_wavecar.band_energy[kpoint-1][def_index-1][0]
        matels[i, Nw] = _compute_matel(psi_i, psi_f)
    deig = np.abs(deig)

    # now compute for each Q
    for i, (q, fname) in enumerate(wavecars):
        Q[i] = q
        final_wavecar = Wavecar(fname)
        for j, bi in enumerate(bulk_index):
            if final_wavecar.spin == 2:
                psi_f = final_wavecar.coeffs[spin][kpoint-1][bi-1]
            else:
                psi_f = final_wavecar.coeffs[kpoint-1][bi-1]
            matels[j, i] = _compute_matel(psi_i, psi_f)

    if fig is not None:
        ax = fig.subplots(1, Nbi)
        ax = np.array(ax)
        for a, i in zip(ax, range(Nbi)):
            a.scatter(Q, matels[i, :])
            a.set_title(f'{bulk_index[i]}')

    return [(bi, deig[i] * np.mean(np.abs(np.gradient(matels[i, :], Q))))
            for i, bi in enumerate(bulk_index)]


def _read_WSWQ(fname):
    """
    Reads the WSWQ file from VASP.

    Parameters
    ----------
    fname : string
        path to the WSWQ file to read

    Returns
    -------
    dict(dict)
        a dict of dicts that takes keys (spin, kpoint) and (initial, final) as
        indices and maps it to a complex number
    """
    wswq = {}
    current = None
    with open(fname, 'r') as f:
        for line in f:
            spin_kpoint = \
                re.search(r'\s*spin=(\d+), kpoint=\s*(\d+)', line)
            data = \
                re.search(r'i=\s*(\d+), '
                          r'j=\s*(\d+)\s*:\s*([0-9\-.]+)\s+([0-9\-.]+)', line)
            if spin_kpoint:
                current = \
                    (int(spin_kpoint.group(1)), int(spin_kpoint.group(2)))
                wswq[current] = {}
            elif data:
                wswq[current][(int(data.group(1)), int(data.group(2)))] = \
                    complex(float(data.group(3)), float(data.group(4)))
    return wswq


def get_Wif_from_WSWQ(wswqs, initial_vasprun, def_index, bulk_index,
                      spin=0, kpoint=1, fig=None):
    """

    Parameters
    ----------
    wswqs : list((Q, wswq_path))
        a list of tuples where the first value is the Q and the second is the
        path to the directory that contains the WSWQ file
    initial_vasprun : string
        path to the initial vasprun.xml to extract the eigenvalue difference
    def_index : int
        index corresponding to the defect wavefunction (1-based indexing)
    bulk_index : int, list(int)
        index or list of indices corresponding to the bulk wavefunction
        (1-based indexing)
    spin : int
        spin channel to read from (0 - up, 1 - down)
    kpoint : int
        kpoint to read from (defaults to the first kpoint)
    fig : matplotlib.figure.Figure
        optional figure object to plot diagnostic information

    Returns
    -------
    list((bulk_index, Wif))
        electron-phonon matrix element Wif in units of
        eV amu^{-1/2} Angstrom^{-1} for each bulk_index
    """
    bulk_index = np.array(bulk_index)

    Nw, Nbi = (len(wswqs), len(bulk_index))
    Q, matels, deig = (np.zeros(Nw+1), np.zeros((Nbi, Nw+1)), np.zeros(Nbi))

    # first compute the eigenvalue differences
    bvr = BSVasprun(initial_vasprun)
    for i, bi in enumerate(bulk_index):
        sp = Spin.up if spin == 0 else Spin.down
        deig[i] = bvr.eigenvalues[sp][kpoint-1][bi-1][0]
    deig = np.abs(deig)

    # now compute for each Q
    for i, (q, fname) in enumerate(wswqs):
        Q[i] = q
        wswq = _read_WSWQ(fname)
        for j, bi in enumerate(bulk_index):
            matels[j, i] = np.sign(q) * \
                np.abs(wswq[(spin+1, kpoint)][(bi, def_index)])

    if fig is not None:
        ax = fig.subplots(1, Nbi)
        ax = np.array(ax)
        for a, i in zip(ax, range(Nbi)):
            tq = np.linspace(np.min(Q), np.max(Q), 100)
            a.scatter(Q, matels[i, :])
            a.plot(tq, np.polyval(np.polyfit(Q, matels[i, :], 1), tq))
            a.set_title(f'{bulk_index[i]}')

    return [(bi, deig[i] * np.polyfit(Q, matels[i, :], 1)[0])
            for i, bi in enumerate(bulk_index)]

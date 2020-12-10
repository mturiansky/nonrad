# -*- coding: utf-8 -*-
# Copyright (c) Chris G. Van de Walle
# Distributed under the terms of the MIT License.

"""Utilities to compute electron-phonon coupling.

This module provides various utilities to evaluate electron-phonon coupling
strength using different first-principles codes.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from monty.io import zopen

from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BSVasprun, Wavecar
from pymatgen.io.wannier90 import Unk


def _compute_matel(psi0: np.ndarray, psi1: np.ndarray) -> float:
    """Compute the inner product of the two wavefunctions.

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


def get_Wif_from_wavecars(
        wavecars: List,
        init_wavecar_path: str,
        def_index: int,
        bulk_index: Sequence[int],
        spin: int = 0,
        kpoint: int = 1,
        fig=None
) -> List:
    """Compute the electron-phonon matrix element using the WAVECARs.

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


def get_Wif_from_UNK(
        unks: List,
        init_unk_path: str,
        def_index: int,
        bulk_index: Sequence[int],
        eigs: Sequence[float],
        fig=None
) -> List:
    """Compute the electron-phonon matrix element using UNK files.

    Evaluate the electron-phonon coupling matrix element using the information
    stored in the given UNK files. This is compatible with any first-principles
    code that write to the wannier90 UNK file format. The onus is on the user
    to ensure the wavefunctions are valid (i.e., norm-conserving).

    Parameters
    ----------
    unks: list((Q, unk_path))
        a list of tuples where the first value is the Q and the second is the
        path to the UNK file
    init_unk_path : string
        path to the initial unk file for computing overlaps
    def_index : int
        index corresponding to the defect wavefunction (1-based indexing)
    bulk_index : int, list(int)
        index or list of indices corresponding to the bulk wavefunction
        (1-based indexing)
    eigs : np.ndarray
        array of eigenvalues in eV where the indices correspond to those given
        by def_index and bulk_index
    fig : matplotlib.figure.Figure
        optional figure object to plot diagnostic information

    Returns
    -------
    list((bulk_index, Wif))
        electron-phonon matrix element Wif in units of
        eV amu^{-1/2} Angstrom^{-1} for each bulk_index
    """
    bulk_index = np.array(bulk_index)
    initial_unk = Unk.from_file(init_unk_path)
    psi_i = initial_unk.data[def_index-1].flatten()

    Nu, Nbi = (len(unks), len(bulk_index))
    Q, matels, deig = (np.zeros(Nu+1), np.zeros((Nbi, Nu+1)), np.zeros(Nbi))

    # first compute the Q = 0 values and eigenvalue differences
    for i, bi in enumerate(bulk_index):
        psi_f = initial_unk.data[bi-1].flatten()
        deig[i] = eigs[bi-1] - eigs[def_index-1]
        matels[i, Nu] = _compute_matel(psi_i, psi_f)
    deig = np.abs(deig)

    # now compute for each Q
    for i, (q, fname) in enumerate(unks):
        Q[i] = q
        final_unk = Unk.from_file(fname)
        for j, bi in enumerate(bulk_index):
            psi_f = final_unk.data[bi-1].flatten()
            matels[j, i] = _compute_matel(psi_i, psi_f)

    print(matels)

    if fig is not None:
        ax = fig.subplots(1, Nbi)
        ax = np.array(ax)
        for a, i in zip(ax, range(Nbi)):
            a.scatter(Q, matels[i, :])
            a.set_title(f'{bulk_index[i]}')

    return [(bi, deig[i] * np.mean(np.abs(np.gradient(matels[i, :], Q))))
            for i, bi in enumerate(bulk_index)]


def _read_WSWQ(fname: str) -> Dict:
    """Read the WSWQ file from VASP.

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
    # whoa, this is horrific
    wswq: Dict[Optional[Tuple[int, int]], Dict[Tuple[int, int], complex]] = {}
    current = None
    with zopen(fname, 'r') as f:
        for line in f:
            spin_kpoint = \
                re.search(r'\s*spin=(\d+), kpoint=\s*(\d+)', str(line))
            data = \
                re.search(r'i=\s*(\d+), '
                          r'j=\s*(\d+)\s*:\s*([0-9\-.]+)\s+([0-9\-.]+)',
                          str(line))
            if spin_kpoint:
                current = \
                    (int(spin_kpoint.group(1)), int(spin_kpoint.group(2)))
                wswq[current] = {}
            elif data:
                wswq[current][(int(data.group(1)), int(data.group(2)))] = \
                    complex(float(data.group(3)), float(data.group(4)))
    return wswq


def get_Wif_from_WSWQ(
        wswqs: List,
        initial_vasprun: str,
        def_index: int,
        bulk_index: Sequence[int],
        spin: int = 0,
        kpoint: int = 1,
        fig=None
) -> List:
    """Compute the electron-phonon matrix element using the WSWQ files.

    Read in the WSWQ files to obtain the overlaps. Then compute the electron-
    phonon matrix elements from the overlaps as a function of Q.

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
    sp = Spin.up if spin == 0 else Spin.down
    def_eig = bvr.eigenvalues[sp][kpoint-1][def_index-1][0]
    for i, bi in enumerate(bulk_index):
        deig[i] = bvr.eigenvalues[sp][kpoint-1][bi-1][0] - def_eig
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

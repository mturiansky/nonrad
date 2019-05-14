import numpy as np
from itertools import groupby
from pymatgen.io.vasp.outputs import Vasprun


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
    ground_structs = ground.interpolate(excited, ximages=displacements)
    excited_structs = ground.interpolate(excited, ximages=(displacements + 1.))
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
        the dQ value
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
        the Q value of the structure
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
        array of Q values corresponding to each vasprun
    energy : np.array(float)
        array of energies corresponding to each vasprun
    """
    num = len(vasprun_paths)
    Q, energy = (np.zeros(num), np.zeros(num))
    for i, vr_fname in enumerate(vasprun_paths):
        vr = Vasprun(vr_fname, parse_dos=False, parse_eigen=False)
        Q[i] = get_Q_from_struct(ground, excited, vr.structures[-1])
        energy[i] = vr.final_energy
    return Q, (energy - np.min(energy))


def get_omega_from_PES(Q, energy):
    pass

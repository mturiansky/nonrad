import numpy as np
from itertools import groupby


def get_cc_structures(ground, excited, displacements):
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

    Returns
    -------
    ground_structs = list(pymatgen.core.structure.Struture)
        a list of structures corresponding to the displaced ground state
    excited_structs = list(pymatgen.core.structure.Structure)
        a list of structures corresponding to the displaced excited state
    """
    displacements = np.array(displacements)
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
    Generate the structures for a CC diagram

    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    excited : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state
    struct : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the structure we want to calculate
        the Q value for

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

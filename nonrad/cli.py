# Copyright (c) Chris G. Van de Walle
# Distributed under the terms of the MIT License.

"""Functionality for the nonrad command-line interface."""

import shutil
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from nonrad.ccd import get_cc_structures


def _check_vasp_dir(path: Path, check_run: bool = False) -> None:
    """Check whether the path contains expected VASP files."""
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    for fname in ("INCAR", "KPOINTS", "POSCAR", "POTCAR"):
        if not (path / fname).exists():
            raise ValueError(f"VASP input file {path / fname} is missing")

    if check_run:
        for fname in ("vasprun.xml", "CONTCAR"):
            if not (path / fname).exists():
                raise ValueError(f"VASP output file {path / fname} is missing")


def _copy_vasp(from_path: Path, to_path: Path) -> None:
    """Copy VASP input files to a new directory."""
    for fname in ("INCAR", "KPOINTS", "POTCAR"):
        shutil.copyfile(from_path / fname, to_path / fname)


def generate_ccd(
    ground_path: Path,
    excited_path: Path,
    output_path: Path = Path("./ccd"),
    disp_mag: float = 0.2,
    num_steps: int = 5,
) -> None:
    """TODO."""
    _check_vasp_dir(ground_path, check_run=True)
    _check_vasp_dir(excited_path, check_run=True)

    if output_path.exists():
        raise ValueError(f"output path {output_path} already exists")

    ground_struct = Structure.from_file(ground_path / "CONTCAR")
    excited_struct = Structure.from_file(excited_path / "CONTCAR")

    displacements = np.linspace(-disp_mag, disp_mag, num_steps)
    gnd_ccd, exc_ccd = get_cc_structures(ground_struct, excited_struct, displacements)

    for gnd_or_exc in ("ground", "excited"):
        for i, struct in enumerate(gnd_ccd if gnd_or_exc[0] == "g" else exc_ccd):
            working_dir = output_path / gnd_or_exc / f"{i}"
            working_dir.mkdir(parents=True)

            _copy_vasp(ground_path if gnd_or_exc[0] == "g" else excited_path, working_dir)
            struct.to(filename=(working_dir / "POSCAR"), fmt="poscar")


def process_ccd():
    """TODO."""

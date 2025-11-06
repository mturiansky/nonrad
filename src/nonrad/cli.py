# Copyright (c) Chris G. Van de Walle
# Distributed under the terms of the MIT License.

"""Functionality for the nonrad command-line interface."""

import shutil
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from ruamel.yaml import YAML

from nonrad.ccd import get_cc_structures, get_dQ, get_omega_from_PES, get_PES_from_vaspruns
from nonrad.elphon import get_Wif_from_WSWQ


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


def _copy_vasp(from_path: Path, to_path: Path, no_relax: bool = False, elph: bool = False) -> None:
    """Copy VASP input files to a new directory."""
    for fname in ("KPOINTS", "POTCAR") + (() if no_relax or elph else ("INCAR",)):
        shutil.copyfile(from_path / fname, to_path / fname)

    if no_relax:
        with open(from_path / "INCAR") as fin, open(to_path / "INCAR") as fout:
            for line in fin:
                for tag in ("EDIFFG", "IBRION", "NSW"):
                    if tag in line:
                        break
                else:
                    fout.write(line)
    elif elph:
        (to_path / "WAVECAR").symlink_to(from_path / "WAVECAR")

        with open(from_path / "INCAR") as fin, open(to_path / "INCAR") as fout:
            for line in fin:
                # TODO: decide on appropriate tags here
                for tag in ("EDIFF", "ENCUT", "ISPIN", "PREC"):
                    if tag in line:
                        fout.write(line)
                        break
            fout.write("\nALGO = None\nNELM = 1\nLWSWQ = True\n")


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

    gnd_struct = Structure.from_file(ground_path / "CONTCAR")
    exc_struct = Structure.from_file(excited_path / "CONTCAR")

    displacements = np.linspace(-disp_mag, disp_mag, num_steps)
    gnd_ccd, exc_ccd = get_cc_structures(gnd_struct, exc_struct, displacements)

    for gnd_or_exc in ("ground", "excited"):
        for i, struct in enumerate(gnd_ccd if gnd_or_exc[0] == "g" else exc_ccd):
            working_dir = output_path / gnd_or_exc / f"{i}"
            working_dir.mkdir(parents=True)

            _copy_vasp(
                ground_path if gnd_or_exc[0] == "g" else excited_path, working_dir, no_relax=True
            )
            struct.to(filename=(working_dir / "POSCAR"), fmt="poscar")


def setup_elph(
    ground_path: Path,
    excited_path: Path,
    ccd_path: Path,
    use_excited: bool = False,
) -> None:
    """TODO."""
    _check_vasp_dir(ground_path, check_run=True)
    _check_vasp_dir(excited_path, check_run=True)

    for working_dir in (ccd_path / ("excited" if use_excited else "ground")).glob("*"):
        (wswq_dir := working_dir / "wswq").mkdir()
        (wswq_dir / "WAVECAR.qqq").symlink_to(working_dir / "WAVECAR")
        _copy_vasp(excited_path if use_excited else ground_path, wswq_dir, elph=True)


def process(
    ground_path: Path,
    excited_path: Path,
    ccd_path: Path,
    save_plot: Path | None = None,
    wif_defect: int | None = None,
    wif_bulk: list[int] | None = None,
    wif_spin: int = 0,
    wif_kpt: int = 1,
) -> None:
    """TODO."""
    _check_vasp_dir(ground_path, check_run=True)
    _check_vasp_dir(excited_path, check_run=True)

    gnd_struct = Structure.from_file(ground_path / "CONTCAR")
    exc_struct = Structure.from_file(excited_path / "CONTCAR")
    dQ = get_dQ(gnd_struct, exc_struct)

    gnd_vaspruns = [ground_path / "vasprun.xml"] + list(ccd_path.glob("ground/*/vasprun.xml"))
    exc_vaspruns = [excited_path / "vasprun.xml"] + list(ccd_path.glob("excited/*/vasprun.xml"))

    q_gnd, en_gnd = get_PES_from_vaspruns(gnd_struct, exc_struct, gnd_vaspruns)
    q_exc, en_exc = get_PES_from_vaspruns(gnd_struct, exc_struct, exc_vaspruns)

    if save_plot is not None:
        import matplotlib.pyplot as plt

        q = np.linspace(1.2 * q_gnd.min(), 1.2 * q_exc.max(), 250)
        _, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    else:
        q, ax = None, None

    wi = get_omega_from_PES(q_exc, en_exc, Q0=dQ, ax=ax, q=q)
    wf = get_omega_from_PES(q_gnd, en_gnd, Q0=0, ax=ax, q=q)

    if save_plot is not None:
        assert ax is not None
        ax.set_ylabel(r"Energy [eV]")
        ax.set_xlabel(r"$Q$ [amu$^{1/2}$ ${\rm \AA}$]")
        plt.savefig(save_plot)

    if wif_defect is not None and wif_bulk is not None:
        wswq_paths = list(ccd_path.glob("ground/*/wswq/WSWQ"))
        if len(wswq_paths) == 0:
            wswq_paths = list(ccd_path.glob("excited/*/wswq/WSWQ"))

            if len(wswq_paths) == 0:
                raise RuntimeError("expected wswq files, but couldn't find any")

            Wifs = get_Wif_from_WSWQ(
                wswq_paths, excited_path / "vasprun.xml", wif_defect, wif_bulk, wif_spin, wif_kpt
            )
        else:
            Wifs = get_Wif_from_WSWQ(
                wswq_paths, ground_path / "vasprun.xml", wif_defect, wif_bulk, wif_spin, wif_kpt
            )

    Wif = np.sqrt(np.mean([wif[1] ** 2 for wif in Wifs]))
    with open("ccd.yaml", "w") as f:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump({"dQ": dQ, "wi": wi, "wf": wf, "Wif": Wif, "volume": gnd_struct.volume}, f)

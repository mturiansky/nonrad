======================
Command Line Interface
======================

``nonrad`` includes a comprehensive Command Line Interface (CLI) built with `Click <https://click.palletsprojects.com/>`_. This tool allows you to perform CCD calculations, compute capture coefficients, scale coefficients, and extract electron-phonon coupling parameters directly from your terminal.

Usage
=====

The entry point for all commands is ``nonrad``. You can view the list of commands and general help by running:

.. code-block:: sh

    nonrad --help

To view the help and option details for a specific command, use:

.. code-block:: sh

    nonrad <command> --help

CCD Commands
============

These commands are used to prepare and process configuration-coordinate diagrams (CCDs).

prep-ccd
--------
Prepare displaced structures along the configuration coordinate.

**Usage:**

.. code-block:: sh

    nonrad prep-ccd [OPTIONS] GROUND_PATH EXCITED_PATH CC_DIR

**Arguments:**
* ``GROUND_PATH``: Path to the directory containing the ground-state calculation (with ``CONTCAR``).
* ``EXCITED_PATH``: Path to the directory containing the excited-state calculation (with ``CONTCAR``).
* ``CC_DIR``: Path to the directory where displaced structures should be written.

**Options:**
* ``-d, --displace <float float float>``: Displacement range specified as ``(min, max, count)``. (Default: ``-0.5 0.5 9``).
* ``-f, --input-files TEXT``: Comma-separated list of VASP input files to copy into each displacement folder. (Default: ``KPOINTS,POTCAR,INCAR,job_script.sh``).

pes
---
Extract potential energy surfaces (PES) from CCD calculations.

**Usage:**

.. code-block:: sh

    nonrad pes [OPTIONS] CC_DIR GROUND_FILES EXCITED_FILES

**Arguments:**
* ``CC_DIR``: Root directory containing the CCD displacement folders.
* ``GROUND_FILES``: Path to the ground-state equilibrium calculation directory (containing ``CONTCAR`` and ``vasprun.xml``).
* ``EXCITED_FILES``: Path to the excited-state equilibrium calculation directory (containing ``CONTCAR`` and ``vasprun.xml``).

**Options:**
* ``-e, --energy-diff FLOAT``: [Required] Energy difference *dE* between ground and excited states (in eV).
* ``-p, --plot``: Flag to plot the potential energy surfaces and save the figure.
* ``-n, --plot-name TEXT``: Filename for the saved plot. (Default: ``pes.png``).

dq
--
Compute the configuration-coordinate displacement *dQ* between two structures.

**Usage:**

.. code-block:: sh

    nonrad dq GROUND_PATH EXCITED_PATH

**Arguments:**
* ``GROUND_PATH``: Path to the ground-state structure file (e.g. ``POSCAR``, ``CONTCAR``).
* ``EXCITED_PATH``: Path to the excited-state structure file (e.g. ``POSCAR``, ``CONTCAR``).

q-from-struct
-------------
Compute the *Q* value of an arbitrary structure relative to ground and excited endpoint structures.

**Usage:**

.. code-block:: sh

    nonrad q-from-struct [OPTIONS] GROUND_PATH EXCITED_PATH STRUCT_PATH

**Arguments:**
* ``GROUND_PATH``: Path to the ground-state structure file.
* ``EXCITED_PATH``: Path to the excited-state structure file.
* ``STRUCT_PATH``: Path to the structure file to evaluate.

**Options:**
* ``-t, --tol FLOAT``: Distance tolerance for filtering coordinate differences. (Default: ``1e-4``).
* ``--nround INTEGER``: Number of decimal places to round *Q*. (Default: ``5``).

barrier
-------
Compute the classical barrier height using the harmonic approximation.

**Usage:**

.. code-block:: sh

    nonrad barrier [OPTIONS]

**Options (All Required):**
* ``--dq FLOAT``: Displacement *dQ* (in amu\ :sup:`1/2` Å).
* ``--de FLOAT``: Energy offset *dE* between states (in eV).
* ``--wi FLOAT``: Initial-state frequency (in eV).
* ``--wf FLOAT``: Final-state frequency (in eV).


Capture Commands
================

capture
-------
Evaluate the nonradiative capture coefficient *C* (in cm\ :sup:`3` s\ :sup:`-1`).

**Usage:**

.. code-block:: sh

    nonrad capture [OPTIONS]

**Options:**
* ``--dq FLOAT``: [Required] Configuration displacement *dQ* (in amu\ :sup:`1/2` Å).
* ``--de FLOAT``: [Required] Energy offset *dE* (in eV).
* ``--wi FLOAT``: [Required] Initial-state frequency (in eV).
* ``--wf FLOAT``: [Required] Final-state frequency (in eV).
* ``--wif FLOAT``: [Required] Electron-phonon coupling matrix element *Wif* (in eV amu\ :sup:`-1/2` Å\ :sup:`-1`).
* ``--volume FLOAT``: [Required] Supercell volume (in Å\ :sup:`3`).
* ``--g INTEGER``: Degeneracy factor. (Default: ``1``).
* ``-T, --temperature FLOAT``: Temperature in Kelvin for a single-point calculation. (Default: ``300.0``).
* ``--temperature-range <float float float>``: Temperature range as ``(min, max, count)`` for scanning multiple temperatures.
* ``--sigma TEXT``: Smearing parameter (float) or interpolation method (``pchip`` or ``cubic``). (Default: ``pchip``).
* ``--occ-tol FLOAT``: Bose-weight occupation tolerance. (Default: ``1e-5``).
* ``--overlap-method [Analytic|Integral|HermiteGauss]``: Method for evaluating vibrational overlaps. (Default: ``HermiteGauss``).


Scaling Commands
================

sommerfeld
----------
Compute the Sommerfeld enhancement factor.

**Usage:**

.. code-block:: sh

    nonrad sommerfeld [OPTIONS]

**Options:**
* ``-T, --temperature FLOAT``: Temperature in Kelvin. (Default: ``300.0``).
* ``--temperature-range <float float float>``: Temperature range as ``(min, max, count)``.
* ``--z INTEGER``: [Required] Charge ratio *Z = Q / q* (negative values represent attractive potential).
* ``--m-eff FLOAT``: [Required] Carrier effective mass in units of electron mass.
* ``--eps0 FLOAT``: [Required] Static dielectric constant.
* ``--dim [1|2|3]``: Dimensionality of the system. (Default: ``3``).
* ``--method [Integrate|Analytic]``: Evaluation method. (Default: ``Integrate``).

thermal-velocity
----------------
Compute the thermal velocity of a carrier.

**Usage:**

.. code-block:: sh

    nonrad thermal-velocity [OPTIONS]

**Options:**
* ``-T, --temperature FLOAT``: Temperature in Kelvin. (Default: ``300.0``).
* ``--temperature-range <float float float>``: Temperature range as ``(min, max, count)``.
* ``--m-eff FLOAT``: [Required] Carrier effective mass in units of electron mass.

charged-supercell
-----------------
Estimate the charged-supercell scaling factor for VASP calculations.

**Usage:**

.. code-block:: sh

    nonrad charged-supercell [OPTIONS] WAVECAR_PATH

**Arguments:**
* ``WAVECAR_PATH``: Path to the VASP WAVECAR file.

**Options:**
* ``--bulk-index INTEGER``: [Required] 1-based band index of the bulk wavefunction.
* ``--def-index INTEGER``: 1-based band index of the defect wavefunction. Provide this or ``--def-coord``. (Default: ``-1``).
* ``--def-coord <float float float>``: Cartesian coordinates (x, y, z) of the defect position.
* ``--cutoff FLOAT``: Slope cutoff for plateau detection. (Default: ``0.02``).
* ``--limit FLOAT``: Upper radial limit (in Å) for the fitting window. (Default: ``5.0``).
* ``--spin INTEGER``: Spin channel (0 = up, 1 = down). (Default: ``0``).
* ``--kpoint INTEGER``: 1-based k-point index. (Default: ``1``).


Electron-Phonon Commands
========================

These commands are nested under the ``elphon`` subcommand group and compute the electron-phonon matrix element *Wif*.

elphon wavecars
---------------
Compute *Wif* using VASP WAVECAR files along the CCD path.

**Usage:**

.. code-block:: sh

    nonrad elphon wavecars [OPTIONS] CC_DIR GROUND_PATH EXCITED_PATH INIT_WAVECAR

**Arguments:**
* ``CC_DIR``: Directory containing the CCD displacement folders.
* ``GROUND_PATH``: Path to the ground-state equilibrium directory.
* ``EXCITED_PATH``: Path to the excited-state equilibrium directory.
* ``INIT_WAVECAR``: Path to the initial (reference) WAVECAR file.

**Options:**
* ``--def-index INTEGER``: [Required] 1-based defect wavefunction index.
* ``-b, --bulk-index INTEGER``: [Required] 1-based bulk wavefunction index (repeatable for multiple bands).
* ``--spin INTEGER``: Spin channel (0 = up, 1 = down). (Default: ``0``).
* ``--kpoint INTEGER``: 1-based k-point index. (Default: ``1``).

elphon wswq
-----------
Compute *Wif* using VASP WSWQ files along the CCD path.

**Usage:**

.. code-block:: sh

    nonrad elphon wswq [OPTIONS] CC_DIR GROUND_PATH EXCITED_PATH INIT_VASPRUN

**Arguments:**
* ``CC_DIR``: Directory containing the CCD displacement folders.
* ``GROUND_PATH``: Path to the ground-state equilibrium directory.
* ``EXCITED_PATH``: Path to the excited-state equilibrium directory.
* ``INIT_VASPRUN``: Path to the reference ``vasprun.xml`` (used to extract eigenvalues).

**Options:**
* ``--def-index INTEGER``: [Required] 1-based defect wavefunction index.
* ``-b, --bulk-index INTEGER``: [Required] 1-based bulk wavefunction index (repeatable for multiple bands).
* ``--spin INTEGER``: Spin channel (0 = up, 1 = down). (Default: ``0``).
* ``--kpoint INTEGER``: 1-based k-point index. (Default: ``1``).

elphon unk
----------
Compute *Wif* using UNK files.

**Usage:**

.. code-block:: sh

    nonrad elphon unk [OPTIONS] INIT_UNK

**Arguments:**
* ``INIT_UNK``: Path to the initial (reference) UNK file.

**Options:**
* ``--def-index INTEGER``: [Required] 1-based defect wavefunction index.
* ``-b, --bulk-index INTEGER``: [Required] 1-based bulk wavefunction index (repeatable).
* ``--eigs TEXT``: [Required] Comma-separated list of eigenvalues (in eV).
* ``-u, --unk-pair <float text>``: [Required] A ``(Q, unk_path)`` pair. (Repeatable for all displacements).

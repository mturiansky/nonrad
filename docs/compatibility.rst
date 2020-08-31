=============
Compatibility
=============

`nonrad` was built to be sufficiently general for the study of nonradiative transitions, but invariably, some utilities are tied to a given first-principles DFT code.
Below, we enumerate the various functionalities provided by the `nonrad` code and specify the compatibility of each.
**General** means that the input parameters are general and do not depend on a given first-principles code.
**Pymatgen** means that function supports any of the first-principles codes that are supported by the `pymatgen` code, including VASP or quantum espresso, as well as general file formats, such as `.cif`, `.xyz`, etc.
**VASP** means that the function is specific to the VASP code.

.. list-table:: Compatibility
   :widths: 25 10 10 10
   :header-rows: 1

   * - Function
     - General
     - Pymatgen
     - VASP
   * - `get_C <nonrad.nonrad.html#nonrad.nonrad.get_C>`_
     - X
     -
     -
   * - `charged_supercell_scaling <nonrad.scaling.html#nonrad.scaling.charged_supercell_scaling>`_
     -
     -
     - X
   * - `sommerfeld_parameter <nonrad.scaling.html#nonrad.scaling.sommerfeld_parameter>`_
     - X
     -
     -
   * - `thermal_velocity <nonrad.scaling.html#nonrad.scaling.thermal_velocity>`_
     - X
     -
     -
   * - `get_PES_from_vaspruns <nonrad.utils.html#nonrad.utils.get_PES_from_vaspruns>`_
     -
     -
     - X
   * - `get_Q_from_struct <nonrad.utils.html#nonrad.utils.get_Q_from_struct>`_
     -
     - X
     -
   * - `get_Wif_from_WSWQ <nonrad.utils.html#nonrad.utils.get_Wif_from_WSWQ>`_
     -
     -
     - X
   * - `get_Wif_from_wavecars <nonrad.utils.html#nonrad.utils.get_Wif_from_wavecars>`_
     -
     -
     - X
   * - `get_cc_structures <nonrad.utils.html#nonrad.utils.get_cc_structures>`_
     -
     - X
     -
   * - `get_dQ <nonrad.utils.html#nonrad.utils.get_dQ>`_
     -
     - X
     -
   * - `get_omega_from_PES <nonrad.utils.html#nonrad.utils.get_omega_from_PES>`_
     - X
     -
     -

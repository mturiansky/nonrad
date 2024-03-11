![build badge](https://img.shields.io/github/actions/workflow/status/mturiansky/nonrad/ci.yml) [![docs badge](https://readthedocs.org/projects/nonrad/badge/?version=latest)](https://nonrad.readthedocs.io/en/latest/?badge=latest) [![codacy](https://app.codacy.com/project/badge/Grade/97df4e822c2349ff858a756b033c6041)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mturiansky/nonrad&amp;utm_campaign=Badge_Grade) [![codecov](https://codecov.io/gh/mturiansky/nonrad/branch/master/graph/badge.svg?token=N1IXIQK333)](https://codecov.io/gh/mturiansky/nonrad) [![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4274317.svg)](https://doi.org/10.5281/zenodo.4274317)

# NONRAD

An implementation of the methodology pioneered by [Alkauskas *et al.*](https://doi.org/10.1103/PhysRevB.90.075202) for computing nonradiative recombination rates from first principles.
The code includes various utilities for processing first principles calculations and preparing the input for computing capture coefficients.
More details on the implementation of the code can be found in [our recent paper]().
Documentation for the code is hosted on [Read the Docs](https://nonrad.readthedocs.io/en/latest).

## Installation
NONRAD is implemented in python and can be installed through `pip`.
Dependencies are kept to a minimum and include standard packages such as `numpy`, `scipy`, and `pymatgen`.

#### With pip
As always with python, it is highly recommended to use a virtual environment.
To install NONRAD, issue the following command,
```
$ pip install nonrad
```
or to install directly from github,
```
$ pip install git+https://github.com/mturiansky/nonrad
```

#### Going Fast (*Recommended*)
NONRAD can use `numba` to accelerate certain calculations.
If `numba` is already installed, it will be used;
otherwise, it can be installed by specifying `[fast]` during installation with pip, e.g.
```
$ pip install nonrad[fast]
```

#### For Development
To install NONRAD for development purposes, clone the repository
```
$ git clone https://github.com/mturiansky/nonrad && cd nonrad
```
then install the package in editable mode with development dependencies
```
$ pip install -e .[dev]
```
`pytest` is used for unittesting.
To run the unittests, issue the command `pytest nonrad` from the base directory.
Unittests should run correctly with and without `numba` installed.

## Usage
A tutorial notebook that describes the various steps is available [here](https://github.com/mturiansky/nonrad/blob/master/notebooks/tutorial.ipynb).
The basic steps are summarized below:

0. Perform a first-principles calculation of the target defect system. A good explanation of the methodology can be found in this [Review of Modern Physics](http://dx.doi.org/10.1103/RevModPhys.86.253). A high quality calculation is necessary as input for the nonradiative capture rate as the resulting values can differ by orders of magnitude depending on the input values.
1. Calculate the potential energy surfaces for the configuration coordinate diagram. This is facilitated using the `get_cc_structures` function. Extract the relevant parameters from the configuration coordinate diagram, aided by `get_dQ`, `get_PES_from_vaspruns`, and `get_omega_from_PES`.
2. Calculate the electron-phonon coupling matrix elements, using the method of your choice (see [our paper]() for details on this calculation with `VASP`). Extraction of the matrix elements are facilitated by the `get_Wif_from_wavecars` or the `get_Wif_from_WSWQ` function.
3. Calculate scaling coefficients using `sommerfeld_parameter` and/or `charged_supercell_scaling`.
4. Perform the calculation of the nonradiative capture coefficient using `get_C`.

## Contributing
To contribute, see the above section on installing [for development](#for-development).
Contributions are welcome and any potential change or improvement should be submitted as a pull request on [Github](https://github.com/mturiansky/nonrad/pulls).
Potential contribution areas are:
 - [ ] implement a command line interface
 - [ ] add more robust tests for various functions

## How to Cite
If you use our code to calculate nonradiative capture rates, please consider citing
```
@article{alkauskas_first-principles_2014,
	title = {First-principles theory of nonradiative carrier capture via multiphonon emission},
	volume = {90},
	doi = {10.1103/PhysRevB.90.075202},
	number = {7},
	journal = {Phys. Rev. B},
	author = {Alkauskas, Audrius and Yan, Qimin and Van de Walle, Chris G.},
	month = aug,
	year = {2014},
	pages = {075202},
}
```
and
```
@article{turiansky_nonrad_2021,
	title = {Nonrad: {Computing} nonradiative capture coefficients from first principles},
	volume = {267},
	doi = {10.1016/j.cpc.2021.108056},
	journal = {Comput. Phys. Commun.},
	author = {Turiansky, Mark E. and Alkauskas, Audrius and Engel, Manuel and Kresse, Georg and Wickramaratne, Darshana and Shen, Jimmy-Xuan and Dreyer, Cyrus E. and Van de Walle, Chris G.},
	month = oct,
	year = {2021},
	pages = {108056},
}
```
If you use the functionality for the Sommerfeld parameter in 2 and 1 dimensions, then please cite
```
@article{turiansky_dimensionality_2024,
    title = {Dimensionality Effects on Trap-Assisted Recombination: The {{Sommerfeld}} Parameter},
    shorttitle = {Dimensionality Effects on Trap-Assisted Recombination},
    author = {Turiansky, Mark E and Alkauskas, Audrius and Van De Walle, Chris G},
    year = {2024},
    month = may,
    journal = {J. Phys.: Condens. Matter},
    volume = {36},
    number = {19},
    pages = {195902},
    doi = {10.1088/1361-648X/ad2588},
}
```

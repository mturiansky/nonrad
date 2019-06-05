[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# NONRAD

An implementation of the methodology pioneered by [Alkauskas *et al.*](https://doi.org/10.1103/PhysRevB.90.075202) for computing nonradiative recombination rates from first principles.
The code includes various utilities for processing first principles calculations and preparing the input for computing capture coefficients.
More details on the implementation of the code can be found in [our recent paper]().

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
`nose2` is used for unittesting.
To run the unittests, issue the command `nose2 -v` from the base directory.
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
 - [ ] more numba support

## How to Cite
If you use our code to calculate nonradiative capture rates, please consider citing
```
@article{alkauskas_first-principles_2014,
	title = {First-principles theory of nonradiative carrier capture via multiphonon emission},
	volume = {90},
	url = {https://link.aps.org/doi/10.1103/PhysRevB.90.075202},
	doi = {10.1103/PhysRevB.90.075202},
	number = {7},
	journal = {Physical Review B},
	author = {Alkauskas, Audrius and Yan, Qimin and Van de Walle, Chris G.},
	month = aug,
	year = {2014},
	pages = {075202},
}
```
and
```
To be added...
```

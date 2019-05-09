[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# NONRAD

An implementation of the methodology pioneered by [Alkauskas *et al.*](https://doi.org/10.1103/PhysRevB.90.075202) for computing nonradiative recombination rates from first principles.
This code is written in Python and includes various improvements over the original implementation (in FORTRAN and also provided).

## Installation
NONRAD is implemented in python and can be installed through `pip` or directly with `setuptools`.
The code depends on various, standard libraries such as `numpy` and `scipy`.
NONRAD and its dependencies can be installed through various means.

#### From Github
First, clone the repository.
```
git clone https://github.com/mturiansky/nonrad
cd nonrad/
```
It is recommended to setup a new virtual environment when installing python packages.
The package and its dependencies can then be installed with
```
pip install .
```
If for some reason, you don't like `pip`, then you can use `setuptools`
```
python setup.py install
```

#### Development
To install NONRAD for development purposes, first clone the repository.
```
git clone https://github.com/mturiansky/nonrad
cd nonrad/
```
Next, install with pip in editable mode with development dependencies
```
pip install -e .[dev]
```

## Usage
## Contributing
## How to Cite
## Acknowledgements

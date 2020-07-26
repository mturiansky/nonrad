"""setup.py for nonrad."""

from setuptools import setup, find_packages


with open('nonrad/VERSION', 'r') as f:
    VERSION = f.readline().strip()

with open('README.md', 'r') as f:
    long_desc = f.read()

setup(
    name='nonrad',
    version=VERSION,
    author='Mark E. Turiansky',
    author_email='mturiansky@physics.ucsb.edu',
    description=('Implementation for computing nonradiative recombination '
                 'rates in semiconductors'),
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/mturiansky/nonrad',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'pymatgen>=v2020.6.8',
        'monty',
        'numba>=v0.50.1'],
    extras_require={
        'dev': [
            'pycodestyle',
            'pydocstyle',
            'pylint',
            'flake8',
            'mypy',
            'coverage',
            'pytest',
            'pytest-cov',
            'sphinx',
            'sphinx-rtd-theme'
        ],
    },
    keywords=['physics', 'materials', 'science', 'VASP', 'recombination',
              'Shockley-Read-Hall'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)

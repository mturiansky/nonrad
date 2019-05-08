from setuptools import setup, find_packages


VERSION = '0.0.1'

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
    install_requires=['numpy', 'scipy'],
    keywords=['physics', 'materials', 'science', 'VASP', 'recombination',
              'Shockley-Read-Hall'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)

# Copyright (c) Chris G. Van de Walle
# Distributed under the terms of the MIT License.

"""Constants used by various parts of the code."""

import scipy.constants as const

HBAR = const.hbar / const.e                     # in units of eV.s
EV2J = const.e                                  # 1 eV in Joules
AMU2KG = const.physical_constants['atomic mass constant'][0]
ANGS2M = 1e-10                                  # angstrom in meters

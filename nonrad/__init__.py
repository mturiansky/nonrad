# Copyright (c) Chris G. Van de Walle
# Distributed under the terms of the MIT License.

"""Init module for nonrad.

This module provides the main implementation to evaluate the nonradiative
capture coefficient from first-principles.
"""

import importlib.metadata

from nonrad.nonrad import get_C

__all__ = ['get_C']
__author__ = 'Mark E. Turiansky'
__email__ = 'mturiansky@physics.ucsb.edu'
__version__ = importlib.metadata.version('nonrad')

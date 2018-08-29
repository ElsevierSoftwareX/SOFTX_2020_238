# Qi Chu adapted from spawaveform.py
# Copyright (C) 2010  Kipp Cannon,  Drew Keppel
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


"""
This module is a wrapper of the _spiir module, supplementing the C
code in that module with additional features that are more easily
implemented in Python.  It is recommended that you import this module
rather than importing _spiir directly.
"""


import math

import lal
from ._spiir_decomp import *


__author__ = "Qi Chu <qi.chu@ligo.org>"


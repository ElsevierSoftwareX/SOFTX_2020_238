# Copyright (C) 2020  Amit Reza
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
This module contains functions for the calculation of the effective precessing spin.
The convention are taken from Phys. Rev. D 91, 024043
"""

"""
This module has been adapted from the lalinference (bayespputils.py). The original code can be seen in the following link
https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/python/lalinference/bayespputils.py#L3920
"""


#standard library imports
import os
import sys


import lal
import numpy as np
import scipy.linalg as linalg
from math import cos, ceil, floor, sqrt, pi as pi_constant


__author__ = "Amit Reza <amit.reza@ligo.org>"


def chi_precessing(m1, a1, tilt1, m2, a2, tilt2):

	q_inv = m1/m2
	A1 = 2. + (3.*q_inv/2.)
	A2 = 2. + 3./(2.*q_inv)
    	S1_perp = a1*np.sin(tilt1)*m1*m1
    	S2_perp = a2*np.sin(tilt2)*m2*m2
    	Sp = np.maximum(A1*S2_perp, A2*S1_perp)
    	chi_p = Sp/(A2*m1*m1)

	return chi_p




def orbital_momentum(fref, m1, m2, inclination):

	eta = m1*m2/( (m1+m2)*(m1+m2) )
	Lmag = orbital_momentum_mag(fref, m1, m2, eta)
	Lx, Ly, Lz = sph2cart(Lmag, inclination, 0.0)

	return np.hstack((Lx,Ly,Lz))


def orbital_momentum_mag(fref, m1, m2, eta):

	v0 = np.power((m1+m2) *pi_constant * lal.MTSUN_SI * fref, 1.0/3.0)

	PNFirst = (((m1+m2)**2)*eta)/v0
	PNSecond = 1+ (v0**2) * (3.0/2.0 +eta/6.0)
	Lmag = PNFirst*PNSecond

	return Lmag


def sph2cart(r, theta, phi):

	x = r*np.sin(theta)*np.cos(phi)
	y = r*np.sin(theta)*np.sin(phi)
	z = r*np.cos(theta)

	return x,y,z

def cart2sph(x,y,z):

	r = np.sqrt(x*x + y*y + z*z)
	theta = np.arccos(z/r)
	phi = np.fmod(2*pi_constant + np.arctan2(y, x), 2*pi_constant)

	return r, theta, phi


def array_ang_sep(vec1, vec2):

	vec1_mag = np.sqrt(array_dot(vec1, vec1))
	vec2_mag = np.sqrt(array_dot(vec2, vec2))

	return np.arccos(array_dot(vec1, vec2)/(vec1_mag*vec2_mag))


def array_dot(vec1, vec2):

	if vec1.ndim == 1:
		product = (vec1*vec2).sum()
	else:
		product = (vec1*vec2).sum(axis = 1).reshape(-1, 1)

	return product


def main(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, fref): 

	L  = orbital_momentum(fref, m1, m2, iota)

	tilt1 = array_ang_sep(L, np.array([s1x, s1y, s1z]))
	tilt2 = array_ang_sep(L, np.array([s2x, s2y, s2z]))


	a1, _, _ = cart2sph(s1x, s1y, s1z)
	a2, _, _ = cart2sph(s2x, s2y, s2z)


	chi_prec = chi_precessing(m1, a1, tilt1, m2, a2, tilt2)

	return chi_prec

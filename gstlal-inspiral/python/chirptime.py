# Copyright (C) 2015 Jolien Creighton
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

import sys
from math import pi
import numpy
import lalsimulation as lalsim
import warnings

def tmplttime(f0, m1, m2, j1, j2, approx):
	dt = 1.0 / 16384.0
	approximant = lalsim.GetApproximantFromString(approx)
	hp, hc = lalsim.SimInspiralChooseTDWaveform(
                	phiRef=0,
                	deltaT=dt,
                	m1=m1,
                	m2=m2,
                	s1x=0,
                	s1y=0,
                	s1z=j1,
                	s2x=0,
                	s2y=0,
                	s2z=j2,
                	f_min=f0,
                	f_ref=0,
                	r=1,
                	i=0,
                	lambda1=0,
                	lambda2=0,
                	waveFlags=None,
                	nonGRparams=None,
                	amplitudeO=0,
                	phaseO=0,
                	approximant=approximant)
	h = numpy.array(hp.data.data, dtype=complex)
	h += 1j * numpy.array(hc.data.data, dtype=complex)
	try:
		n = list(abs(h)).index(0)
	except ValueError:
		n = len(h)
	return n * hp.deltaT


GMsun = 1.32712442099e20 # heliocentric gravitational constant, m^2 s^-2
G = 6.67384e-11 # Newtonian constant of gravitation, m^3 kg^-1 s^-2
c = 299792458 # speed of light in vacuum (exact), m s^-1
Msun = GMsun / G # solar mass, kg

def velocity(f, M):
	"""
	Finds the orbital "velocity" corresponding to a gravitational
	wave frequency.
	"""
	return (pi * G * M * f)**(1.0/3.0) / c

def chirptime(f, M, nu, chi):
	"""
	Over-estimates the chirp time in seconds from a certain frequency.
	Uses 2 pN chirp time from some starting frequency f in Hz
	where all negative contributions are dropped.
	"""
	v = velocity(f, M)
	tchirp = 1.0
	tchirp += ((743.0/252.0) + (11.0/3.0)*nu) * v**2
	tchirp += (226.0/15.0) * chi * v**3
	tchirp += ((3058673.0/508032.0) + (5429.0/504.0)*nu + (617.0/72.0)*nu**2) * v**4
	tchirp *= (5.0/(256.0*nu)) * (G*M/c**3) / v**8
	return tchirp

def ringf(M, j):
	"""
	Computes the approximate ringdown frequency in Hz.
	Here j is the reduced Kerr spin parameter, j = c^2 a / G M.
	"""
	return (1.5251 - 1.1568 * (1 - j)**0.1292) * (c**3 / (2.0 * pi * G * M))

def ringtime(M, j):
	"""
	Computes the black hole ringdown time in seconds.
	Here j is the reduced Kerr spin parameter, j = c^2 a / G M.
	"""
	efolds = 11
	# these are approximate expressions...
	f = ringf(M, j)
	Q = 0.700 + 1.4187 * (1 - j)**-0.4990
	T = Q / (pi * f)
	return efolds * T

def mergetime(M):
	"""
	Over-estimates the time from ISCO to merger in seconds.
	Assumes one last orbit at the maximum ISCO radius.
	"""
	# The maximum ISCO is for orbits about a Kerr hole with
	# maximal counter rotation.  This corresponds to a radius
	# of 9GM/c^2 and a orbital speed of ~ c/3.  Assume the plunge
	# and merger takes one orbit from this point.
	norbits = 1.0
	v = c / 3.0
	r = 9.0 * G * M / c**2
	return norbits * (2.0 * pi * r / v)

def overestimate_j_from_chi(chi):
	"""
	Overestimate final black hole spin
	formula is roughly based on
	Tichy and Marronetti Physical Review D 78 081501 (2008)
	"""
	return max(0.686 + 0.15 * chi, chi)

def imr_time(f, m1, m2, j1, j2, f_max = None):
	"""
	Returns an overestimate of the inspiral time and the
	merger-ringdown time, both in seconds.  Here, m1 and m2
	are the component masses in kg, j1 and j2 are the dimensionless
	spin parameters of the components.
	"""

	if f_max is not None and f_max < f:
		raise ValueError("f_max must either be None or greater than f")

	# compute total mass and symmetric mass ratio
	M = m1 + m2
	nu = m1 * m2 / M**2

	# compute spin parameters
	#chi_s = 0.5 * (j1 + j2)
	#chi_a = 0.5 * (j1 - j2)
	#chi = (1.0 - (76.0/113.0)) * chi_s + ((m1 - m2) / M) * chi_a
	# overestimate chi:
	chi = max(j1, j2)

	j = overestimate_j_from_chi(chi)

	if f > ringf(M, j):
		warnings.warn("f is greater than the ringdown frequency. This might not be what you intend to compute")

	if f_max is None or f_max > ringf(M, j):
		return chirptime(f, M, nu, chi) + mergetime(M) + ringtime(M, j)
	else:
		# Always be conservative and allow a merger time to be added to
		# the end in case your frequency is above the last stable orbit
		# but below the ringdown.
		return imr_time(f, m1, m2, j1, j2) - imr_time(f_max, m1, m2, j1, j2)

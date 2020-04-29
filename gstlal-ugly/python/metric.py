# Copyright (C) 2014 Miguel Fernandez, Chad Hanna
# Copyright (C) 2016,2017 Kipp Cannon, Miguel Fernandez, Chad Hanna, Stephen Privitera, Jonathan Wang
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

import lalsimulation as lalsim
import lal
from lal import series
from scipy import integrate
import numpy
from gstlal import reference_psd
from ligo.lw import utils as ligolw_utils
import itertools
import scipy
from lal import LIGOTimeGPS
import sys
import math

DELTA = 2e-7
EIGEN_DELTA_DET = 2e-7

# Round a number up to the nearest power of 2
def ceil_pow_2(x):
	x = int(math.ceil(x))
	x -= 1
	n = 1
	while n and (x & (x + 1)):
		x |= x >> n
		n *= 2
	return x + 1


def add_quadrature_phase(fseries, n):
	"""
	From the Fourier transform of a real-valued function of
	time, compute and return the Fourier transform of the
	complex-valued function of time whose real component is the
	original time series and whose imaginary component is the
	quadrature phase of the real part.  fseries is a LAL
	COMPLEX16FrequencySeries and n is the number of samples in
	the original time series.
	"""
	#
	# positive frequencies include Nyquist if n is even
	#

	have_nyquist = not (n % 2)

	#
	# shuffle frequency bins
	#

	positive_frequencies = numpy.array(fseries.data.data) # work with copy
	positive_frequencies[0] = 0	# set DC to zero
	zeros = numpy.zeros((len(positive_frequencies),), dtype = "cdouble")
	if have_nyquist:
		# complex transform never includes positive Nyquist
		positive_frequencies = positive_frequencies[:-1]

	#
	# prepare output frequency series
	#

	out_fseries = lal.CreateCOMPLEX16FrequencySeries(
		name = fseries.name,
		epoch = fseries.epoch,
		f0 = fseries.f0,	# caution: only 0 is supported
		deltaF = fseries.deltaF,
		sampleUnits = fseries.sampleUnits,
		length = len(zeros) + len(positive_frequencies) - 1
	)
	out_fseries.data.data = numpy.concatenate((zeros, 2 * positive_frequencies[1:]))

	return out_fseries


#
# Utility functions to help with the coordinate system for evaluating the waveform metric
#

def m1_m2_func(coords):
	return (coords[0], coords[1], 0., 0., 0., 0., 0., 0.)


def m1_m2_s2z_func(coords):
	return (coords[0], coords[1], 0., 0., 0., 0., 0., coords[2])

def m1_m2_s1z_s2z_func(coords):
	return (coords[0], coords[1], 0., 0., coords[2], 0., 0., coords[3])


# FIXME FIXME This assumes that the third and fourth coordinates are z1 and z2.  will not work for precessing.
def M_q_func(coords):
	mass2 = coords[1]
	mass1 = m1_from_mc_m2(coords[0], mass2)
	out = [mass1, mass2, 0., 0., 0., 0., 0., 0.]
	# FIXME assumes s1z s2z for the third and forth coordinates
	for i,c in enumerate(coords[2:]):
		out[i*3+4] = c
	return tuple(out)

def m1_from_mc_m2(mc, m2):
	a = mc**5 / m2**3
	q = -m2 * a
	p = -a

	def tk(p,q,k):
		t1 = 2 * (-p / 3.)**.5
		arg = 3. * q / (2. * p) * (-3. / p)**.5
		arg2 = math.acos(arg)
		t2 = math.cos(1./3. * arg2 - 2. * math.pi * k / 3.)
		return t1 * t2

	def t0(p,q):
		arg = -3. * abs(q) / 2. / p * (-3. / p)**.5
		arg = 1./3. * math.acosh(arg)
		return 2 * (-p / 3.)**.5 * math.cosh(arg)

	for k in (0,1,2):
		try:
			m1 = tk(p,q,k)
			if m1 >= m2:
				return m1
		except ValueError:
			pass

	return t0(p,q)

def mc_from_m1_m2(m1, m2):
	return (m1 * m2)**(.6) / (m1 + m2)**.2

# inspired by arXiv:gr-qc/9808076v1 27 Aug 1998

def x_y_z_zn_func(coords):
	s1y = s2y = s1x = s2x = 0.
	m1 = m1_from_x_y_z_zn(*coords)
	m2 = m2_from_x_y_z_zn(*coords)
	s1z = s1_from_x_y_z_zn(*coords)
	s2z = s2_from_x_y_z_zn(*coords)
	q = m1 / m2
	if False:#q < 1.05 and q > 0.95:
		M = m1 + m2
		q = 1.05
		m2 = (M / q) / 2.
		m1 = m2 * q
	return [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z]

def x_from_m1_m2_s1_s2(m1, m2, s1, s2):
	m1 = float(m1); m2 = float(m2); s1 = float(s1); s2 = float(s2)
	M = m1 + m2
	eta = m1*m2/(m1+m2)**2
	return M**(-5/3.) / eta

def y_from_m1_m2_s1_s2(m1, m2, s1, s2):
	m1 = float(m1); m2 = float(m2); s1 = float(s1); s2 = float(s2)
	M = m1 + m2
	eta = m1*m2/(m1+m2)**2
	return M**(-2/3.) / eta

def z_from_m1_m2_s1_s2(m1, m2, s1, s2):
	m1 = float(m1); m2 = float(m2); s1 = float(s1); s2 = float(s2)
	M = m1 + m2
	return (s1 * m1 + s2 * m2) / M

def zn_from_m1_m2_s1_s2(m1, m2, s1, s2):
	m1 = float(m1); m2 = float(m2); s1 = float(s1); s2 = float(s2)
	M = m1 + m2
	return (s1 * m1 - s2 * m2) / M

def m1_from_x_y_z_zn(x, y, z, zn):
	x = float(x); y = float(y)
	M = y / x
	eta = min((x**2 / y**5)**(1./3), 0.25)
	m1 = (M + (M**2 * (1 - 4 * eta))**.5) / 2.
	return m1

def m2_from_x_y_z_zn(x, y, z, zn):
	x = float(x); y = float(y)
	M = y / x
	eta = min((x**2 / y**5)**(1./3), 0.25)
	m2 = M - (M + (M**2 * (1 - 4 * eta))**.5) / 2.
	return m2

def s1_from_x_y_z_zn(x, y, z, zn):
	x = float(x); y = float(y); z = float(z); zn = float(zn)
	M = y / x
	m1 = m1_from_x_y_z_zn(x, y, z, zn)
	return (z + zn) * M / m1 / 2.

def s2_from_x_y_z_zn(x, y, z, zn):
	x = float(x); y = float(y); z = float(z); zn = float(zn)
	M = y / x
	m2 = m1_from_x_y_z_zn(x, y, z, zn)
	return (z - zn) * M / m2 / 2.

#
# Metric object that numerically evaluates the waveform metric
#


class Metric(object):
	def __init__(self, psd_xml, coord_func, duration = 4, flow = 30.0, fhigh = 512., approximant = "TaylorF2"):
		# FIXME expose all of these things some how
		self.duration = duration
		self.approximant = lalsim.GetApproximantFromString(approximant)
		self.coord_func = coord_func
		self.df = 1. / self.duration
		self.flow = flow
		self.fhigh = fhigh
		self.working_length = int(round(self.duration * 2 * self.fhigh)) + 1
		self.psd = reference_psd.interpolate_psd(series.read_psd_xmldoc(ligolw_utils.load_filename(psd_xml, verbose = True, contenthandler = series.PSDContentHandler)).values()[0], self.df)
		self.revplan = lal.CreateReverseCOMPLEX16FFTPlan(self.working_length, 1)
		self.delta_t = DELTA
		self.t_factor = numpy.exp(-2j * numpy.pi * (numpy.arange(self.working_length) * self.df - self.fhigh) * self.delta_t)
		self.neg_t_factor = numpy.exp(-2j * numpy.pi * (numpy.arange(self.working_length) * self.df - self.fhigh) * (-self.delta_t))
		self.tseries = lal.CreateCOMPLEX16TimeSeries(
			name = "workspace",
			epoch = 0,
			f0 = 0,
			deltaT = 1. / (2 * self.fhigh),
			length = self.working_length,
			sampleUnits = lal.Unit("strain")
		)
		self.w1w2 = lal.CreateCOMPLEX16FrequencySeries(
			name = "template",
			epoch = 0.0,
			f0 = 0.0,
			deltaF = self.df,
			sampleUnits = lal.Unit("strain"),
			length = self.working_length
		)


	def waveform(self, coords):
		# Generalize to different waveform coords
		p = self.coord_func(coords)

		try:
			parameters = {}
			parameters['m1'] = lal.MSUN_SI * p[0]
			parameters['m2'] = lal.MSUN_SI * p[1]
			parameters['S1x'] = p[2]
			parameters['S1y'] = p[3]
			parameters['S1z'] = p[4]
			parameters['S2x'] = p[5]
			parameters['S2y'] = p[6]
			parameters['S2z'] = p[7]
			parameters['distance'] = 1.e6 * lal.PC_SI
			parameters['inclination'] = 0.
			parameters['phiRef'] = 0.
			parameters['longAscNodes'] = 0.
			parameters['eccentricity'] = 0.
			parameters['meanPerAno'] = 0.
			parameters['deltaF'] = self.df
			parameters['f_min'] = self.flow
			parameters['f_max'] = self.fhigh
			parameters['f_ref'] = 0.
			parameters['LALparams'] = None
			parameters['approximant'] = self.approximant

			hplus, hcross = lalsim.SimInspiralFD(**parameters)

			fseries = hplus
			lal.WhitenCOMPLEX16FrequencySeries(fseries, self.psd)
			fseries = add_quadrature_phase(fseries, self.working_length)
			del hplus
			del hcross
		except RuntimeError:
			print(p)
			#raise
			return None
		return fseries

	def match_minus_1(self, w1, w2, t_factor = 1.0):
		def norm(w):
			n = numpy.abs(integrate.romb(numpy.conj(w) * w))**.5
			return n
		def ip(w1, w2):
			return numpy.abs(integrate.romb(numpy.conj(w1) * w2))

		try:
			x = numpy.copy(w1.data.data)
			y = numpy.copy(w2.data.data)
			y /= norm(y)
			x /= norm(x)
			y *= t_factor
			m = ip(x,y)
			if m < 1. -  1e-15:
				return m - 1.0
			d1 = x - y
			mm2 = ip(d1, d1)
			return -0.5 * mm2

		except AttributeError:
			return None


	def __set_diagonal_metric_tensor_component(self, i, wp, wm, deltas, g, w1):
		plus_match_minus_1 = self.match_minus_1(w1, wp[i,i])
		minus_match_minus_1 = self.match_minus_1(w1, wp[i,i])
		d2mbydx2 = (plus_match_minus_1 + minus_match_minus_1) / deltas[i]**2
		# - 1/2 the second partial derivative
		g[i,i] = -0.5 * d2mbydx2

	def __set_tt_metric_tensor_component(self, center, w1):
		minus_match_minus_1 = self.match_minus_1(w1, w1, t_factor = self.neg_t_factor)
		plus_match_minus_1 = self.match_minus_1(w1, w1, t_factor = self.t_factor)
		d2mbydx2 = (plus_match_minus_1 + minus_match_minus_1) / self.delta_t**2
		return -0.5 * d2mbydx2

	def __set_offdiagonal_metric_tensor_component(self, inds, wp, wm, deltas, g, w1):
		i, j = inds
		# evaluate symmetrically
		if j <= i:
			return None
		fii = -2 * g[i,i] * deltas[i]**2
		fjj = -2 * g[j,j] * deltas[j]**2
		d2mbydxdy = (self.match_minus_1(w1, wp[i,j]) + self.match_minus_1(w1, wm[i,j]) - fii - fjj) / 2. / deltas[i] / deltas[j]
		g[i,j] = g[j,i] = -0.5 * d2mbydxdy
		return None

	def __set_offdiagonal_time_metric_tensor_component(self, j, wp, wm, deltas, g, g_tt, delta_t, w1):
		fjj = -2 * g[j,j] * deltas[j]**2
		ftt = -2 * g_tt * delta_t**2
		# Second order
		plus_match_minus_1 = self.match_minus_1(w1, wp[j,j], t_factor = self.t_factor)
		minus_match_minus_1 = self.match_minus_1(w1, wm[j,j], t_factor = self.neg_t_factor)
		return -0.5 * (plus_match_minus_1 + minus_match_minus_1 - ftt - fjj) / 2 / delta_t / deltas[j]

	def __call__(self, center):#, deltas = None):
		center = numpy.array(center)
		g = numpy.zeros((len(center), len(center)), dtype=numpy.double)
		w1 = self.waveform(center)
		wp = {}
		wm = {}
		# FIXME assumes masses as first two and spins as the rest
		deltas = numpy.zeros(len(center))
		deltas[0:2] = abs(center[0:2]) * DELTA
		deltas[2:] += DELTA
		for i, x in enumerate(deltas):
			for j, y in enumerate(deltas):
				dx = numpy.zeros(len(deltas))
				dx[i] = x
				dx[j] = y
				if j < i:
					continue
				elif j == i:
					wp[i,i] = self.waveform(center + dx)
					wm[i,i] = self.waveform(center - dx) 
				else:
					wp[i,j] = wp[j,i] = self.waveform(center + dx) 
					wm[i,j] = wm[j,i] = self.waveform(center - dx) 
			
		g_tt = self.__set_tt_metric_tensor_component(center, w1)

		# First get the diagonal components
		[self.__set_diagonal_metric_tensor_component(i, wp, wm, deltas, g, w1) for i in range(len(center))]

		# Then the rest
		[self.__set_offdiagonal_metric_tensor_component(ij, wp, wm, deltas, g, w1) for ij in itertools.product(range(len(center)), repeat = 2)]

		g_tj = [self.__set_offdiagonal_time_metric_tensor_component(j, wp, wm, deltas, g, g_tt, self.delta_t, w1) for j in range(len(deltas))]

		# project out the time component Owen 2.28
		for i, j in itertools.product(range(len(deltas)), range(len(deltas))):
			g[i,j] = g[i,j] -  g_tj[i] * g_tj[j] / g_tt

		U, S, V = numpy.linalg.svd(g)
		condition = S < max(S) * EIGEN_DELTA_DET
		S[condition] = EIGEN_DELTA_DET * max(S)
		g = numpy.dot(U, numpy.dot(numpy.diag(S), V))
		det = numpy.product(S[S>0])

		return g, det


	def _distancesq(self, metric_tensor, x, y):
		delta = x - y
		X = numpy.dot(delta, metric_tensor)
		Y = delta
		d2 = numpy.sum((X * Y), axis = 1)
		d2[d2<0.] = 0.
		return d2

	def distancesq(self, metric_tensor, x, y):
		"""
		Compute the distance squared between to points inside the cube using
		the metric tensor, but assuming it is constant
		"""
		if len(y.shape) > 1:
			return self._distancesq(metric_tensor, x, y)
		def dot(x, y, metric):
			return numpy.dot(numpy.dot(x.T, metric), y)

		delta = x-y
		# always return floating point epsilon distance
		return max(1e-7, (dot(delta, delta, metric_tensor)))

	def volume_element(self, metric_tensor):
		return abs(numpy.linalg.det(metric_tensor))**.5

	def metric_match(self, metric_tensor, c1, c2):
		d2 = self.distancesq(metric_tensor, c1, c2)
		if d2 < 1 and d2 >= 0:
			return 1 - d2
		else:
			return 0.

	def pseudo_match(self, metric_tensor, c1, c2):
		d2 = self.distancesq(metric_tensor, c1, c2)
		match = 1. / (1. + numpy.log(1. + d2))
		try:
			if match > 1.0:
				return 1.0
			if match < 0.0:
				return 0.0
			return match
		except ValueError:# have an array
			match[match > 1.0] = 1.0
			match[match < 0.0] = 0.0
			return match

	def explicit_match(self, c1, c2):
		def fftmatch(w1, w2):
			def norm(w):
				n = numpy.real((numpy.conj(w) * w).sum())**.5 / self.duration**.5
				return n

			self.w1w2.data.data[:] = numpy.conj(w1.data.data) * w2.data.data
			lal.COMPLEX16FreqTimeFFT(self.tseries, self.w1w2, self.revplan)
			m = numpy.real(numpy.abs(numpy.array(self.tseries.data.data)).max()) / norm(w1.data.data) / norm(w2.data.data)
			if m > 1.0000001:
				raise ValueError("Match is greater than 1 : %f" % m)
			return m

		return fftmatch(self.waveform(c1), self.waveform(c2))

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
import numpy
from gstlal import reference_psd
from glue.ligolw import utils as ligolw_utils
import itertools
import scipy
from lal import LIGOTimeGPS
import sys
import math

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


def M_q_func(coords):
	mass2 = coords[1]
	mass1 = m1_from_mc_m2(coords[0], mass2)
	return (mass1, mass2, 0., 0., 0., 0., 0., 0.)

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
		self.working_length = int(round(self.duration * 2 * self.fhigh))
		self.psd = reference_psd.interpolate_psd(series.read_psd_xmldoc(ligolw_utils.load_filename(psd_xml, verbose = True, contenthandler = series.PSDContentHandler)).values()[0], self.df)
		self.metric_tensor = None
		self.metric_is_valid = False
		self.revplan = lal.CreateReverseCOMPLEX16FFTPlan(self.working_length, 1)
		# FIXME NOTE this code is written to allow different spacing
		# depending on mass if that is a good idea, but right now it
		# just does the same spacing
		self.delta_t = {}
		self.t_factor = {}
		self.neg_t_factor = {}
		delta_t = numpy.finfo(numpy.float32).eps * 2
		t_factor = numpy.exp(-2j * numpy.pi * (numpy.arange(self.working_length) * self.df - self.fhigh) * delta_t)
		neg_t_factor = numpy.exp(-2j * numpy.pi * (numpy.arange(self.working_length) * self.df - self.fhigh) * (-delta_t))
		for t in numpy.array([1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,1024]):
			#self.delta_t[t] = 3e-6 * t # 1 M time spacing
			#self.t_factor[t] = numpy.exp(-2j * numpy.pi * (numpy.arange(self.working_length) * self.df - self.fhigh) * self.delta_t[t])
			#self.neg_t_factor[t] = numpy.exp(-2j * numpy.pi * (numpy.arange(self.working_length) * self.df - self.fhigh) * (-self.delta_t[t]))
			self.delta_t[t] = delta_t
			self.t_factor[t] = t_factor
			self.neg_t_factor[t] = neg_t_factor
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

		def fmin(m1, m2):
			mc = (m1*m2)**.6 / (m1+m2)**.2 * 5e-6
			return  1./numpy.pi / mc * (5./256. * mc / 1.)**(3./8.)

		flow = self.flow
		#flow = max(min(fmin(p[0], p[1]), self.flow), 10)

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
			parameters['f_min'] = flow
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
			print p
			#raise
			return None
		return fseries

	def match(self, w1, w2, t_factor = None):
		def norm(w):
			n = numpy.real((numpy.conj(w) * w).sum())**.5
			return n

		try:
			if t_factor is None:
				self.w1w2.data.data[:] = numpy.conj(w1.data.data) * w2.data.data
			else:
				self.w1w2.data.data[:] = numpy.conj(w1.data.data) * w2.data.data * t_factor
			m = numpy.real(numpy.abs(self.w1w2.data.data[:].sum())) / norm(w1.data.data) / norm(w2.data.data)
			if m > 1.0000001:
				raise ValueError("Match is greater than 1 : %f" % m)
			return m
		except AttributeError:
			return None


	#def __set_diagonal_metric_tensor_component(self, i, center, deltas, g, w1, min_d2 = numpy.finfo(numpy.float32).eps * 5, max_d2 = numpy.finfo(numpy.float32).eps * 100):
	def __set_diagonal_metric_tensor_component(self, i, center, deltas, g, w1):

		# make the vector to solve for the metric by choosing
		# either a principle axis or a bisector depending on if
		# this is a diagonal component or not
		x = numpy.zeros(len(deltas))
		x[i] = deltas[i]

		# get the positive side of the central difference
		plus_match = self.match(w1, self.waveform(center+x))
		if plus_match is None:
			# random new nearby point
			return self.__set_diagonal_metric_tensor_component(i, center + numpy.abs(numpy.random.randn()) * x, deltas, g, w1)

		# get the negative side of the central difference
		minus_match = self.match(w1, self.waveform(center-x))
		if minus_match is None:
			# random new nearby point
			return self.__set_diagonal_metric_tensor_component(i, center + numpy.abs(numpy.random.randn()) * x, deltas, g, w1)

		# fourth order
		minus_match2 = self.match(w1, self.waveform(center-2*x))
		plus_match2 = self.match(w1, self.waveform(center+2*x))
		d2mbydx2 = (4./3. * (plus_match + minus_match) - 1./12. * (plus_match2 + minus_match2) - 5./2.) / x[i]**2

		# - 1/2 the second partial derivative
		g[i,i] = -0.5 * d2mbydx2
		return x[i]

	def __set_tt_metric_tensor_component(self, center, w1):

		# FIXME FIXME assumes m1 and m2 are first coords
		M = center[0] + center[1]
		ix = ceil_pow_2(M)
		minus_match = self.match(w1, w1, t_factor = self.neg_t_factor[ix])
		plus_match = self.match(w1, w1, t_factor = self.t_factor[ix])
		d2mbydx2 = (plus_match + minus_match - 2.0) / self.delta_t[ix]**2
		return -0.5 * d2mbydx2, self.delta_t[ix]

	def __set_offdiagonal_metric_tensor_component(self, (i,j), center, deltas, g, w1):
		# evaluate symmetrically
		if j <= i:
			return None
		# Second order
		xmm = numpy.zeros(len(deltas))
		xmp = numpy.zeros(len(deltas))
		xpm = numpy.zeros(len(deltas))
		xpp = numpy.zeros(len(deltas))

		xmm[i] = -deltas[i]
		xmm[j] = -deltas[j]
		xmp[i] = -deltas[i]
		xmp[j] = +deltas[j]
		xpm[i] = +deltas[i]
		xpm[j] = -deltas[j]
		xpp[i] = +deltas[i]
		xpp[j] = +deltas[j]

		d2mbydxdy = (self.match(w1, self.waveform(center+xmm)) + self.match(w1, self.waveform(center+xpp)) - self.match(w1, self.waveform(center+xmp)) - self.match(w1, self.waveform(center+xpm))) / 4. / xpp[i] / xpp[j]

		g[i,j] = g[j,i] = -0.5 * d2mbydxdy
		return None

	def __set_offdiagonal_time_metric_tensor_component(self, j, center, deltas, g, g_tt, delta_t, w1):
		# FIXME FIXME assumes m1 and m2 are first coords
		M = center[0] + center[1]
		ix = ceil_pow_2(M)
		# make the vector to solve for the metric by choosing
		# either a principle axis or a bisector depending on if
		# this is a diagonal component or not
		x = numpy.zeros(len(deltas))
		x[j] = deltas[j]
		fjj = -2 * g[j,j] * deltas[j]**2 + 2.0
		ftt = -2 * g_tt * delta_t**2 + 2.0

		# Second order
		plus_match = self.match(w1, self.waveform(center+x), t_factor = self.t_factor[ix])
		minus_match = self.match(w1, self.waveform(center-x), t_factor = self.neg_t_factor[ix])
		return -0.5 * (plus_match + minus_match - ftt - fjj + 2.0) / 2 / delta_t / deltas[j]

	def __call__(self, center, deltas = None, thresh = 1. * numpy.finfo(numpy.float32).eps):

		g = numpy.zeros((len(center), len(center)), dtype=numpy.double)
		w1 = self.waveform(center)
		g_tt, delta_t = self.__set_tt_metric_tensor_component(center, w1)

		# First get the diagonal components
		deltas = numpy.array([self.__set_diagonal_metric_tensor_component(i, center, deltas, g, w1) for i in range(len(center))])

		# Then the rest
		[self.__set_offdiagonal_metric_tensor_component(ij, center, deltas, g, w1) for ij in itertools.product(range(len(center)), repeat = 2)]
		g_tj = [self.__set_offdiagonal_time_metric_tensor_component(j, center, deltas, g, g_tt, delta_t, w1) for j in range(len(deltas))]

		# project out the time component Owen 2.28
		for i, j in itertools.product(range(len(deltas)), range(len(deltas))):
			g[i,j] = g[i,j] -  g_tj[i] * g_tj[j] / g_tt

		# find effective dimension
		U, S, V = numpy.linalg.svd(g)
		condition = S < max(S) * thresh
		eff_dimension = len(S) - len(S[condition])
		S[condition] = 0.0
		g = numpy.dot(U, numpy.dot(numpy.diag(S), V))

		return g, eff_dimension, numpy.product(S[S>0])


	def distance(self, metric_tensor, x, y):
		"""
		Compute the distance between to points inside the cube using
		the metric tensor, but assuming it is constant
		"""

		def dot(x, y, metric):
			return numpy.dot(numpy.dot(x.T, metric), y)

		delta = x-y
		return (dot(delta, delta, metric_tensor))**.5


	def metric_match(self, metric_tensor, c1, c2):
		d2 = self.distance(metric_tensor, c1, c2)**2
		if d2 < 1:
			return 1 - d2
		else:
			return 0.


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

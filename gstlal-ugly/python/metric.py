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
	mass2 = coords[0] / (coords[1] + 1)
	mass1 = coords[0] - mass2
	return (mass1, mass2, 0., 0., 0., 0., 0., 0.)


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
		self.tseries = lal.CreateCOMPLEX16TimeSeries(
			name = "workspace",
			epoch = 0,
			f0 = 0,
			deltaT = 1. / (2 * self.fhigh),
			length = self.working_length,
			sampleUnits = lal.Unit("strain")
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
		except RuntimeError:
			print p
			raise
		return fseries

	def match(self, w1, w2):
		def norm(w):
			n = numpy.real((numpy.conj(w) * w).sum())**.5 / self.duration**.5
			return n

		w1w2 = lal.CreateCOMPLEX16FrequencySeries(
			name = "template",
			epoch = w1.epoch,
			f0 = 0.0,
			deltaF = self.df,
			sampleUnits = lal.Unit("strain"),
			length = self.working_length
		)
		w1w2.data.data[:] = numpy.conj(w1.data.data) * w2.data.data
		lal.COMPLEX16FreqTimeFFT(self.tseries, w1w2, self.revplan)
		m = numpy.real(numpy.abs(numpy.array(self.tseries.data.data)).max()) / norm(w1.data.data) / norm(w2.data.data)
		if m > 1.0000001:
			raise ValueError("Match is greater than 1 : %f" % m)
		return m


	def metric_tensor_component(self, (i,j), center, deltas, g = None, w1 = None):
		if w1 is None:
			w1 = self.waveform(center)

		# evaluate symmetrically
		if j < i:
			return

		if g is None:
			if i !=j:
				raise ValueError("Must provide metric tensor to compute off-diagonal terms")
			g = numpy.zeros((center.shape[0], center.shape[0]))

		# make the vector to solve for the metric by choosing
		# either a principle axis or a bisector depending on if
		# this is a diagonal component or not
		x = numpy.zeros(len(deltas))
		x[i] = deltas[i]
		x[j] = deltas[j]

		# Compute the diagonal
		if j == i:
			# Check the match
			d2 = 1. - self.match(w1, self.waveform(center+x))
			# The match must like close but not beyond machine
			# epsilon to be a good approximation
			if (d2 > numpy.finfo(numpy.float32).eps * 32):
				return self.metric_tensor_component((i,j), center = center, deltas = deltas / 10., g = g, w1 = w1)
			if (d2 < numpy.finfo(numpy.float32).eps * 8):
				return self.metric_tensor_component((i,j), center = center, deltas = deltas * 2., g = g, w1 = w1)

			g[i,i] = d2 / deltas[i] / deltas[i]
			return g[i,i], deltas[i]
		# NOTE Assumes diagonal parts are already computed!!!
		else:
			# Check the match
			d2 = 1 - self.match(w1, self.waveform(center+x))
			g[i,j] = g[j,i] = d2 / (2 * deltas[i] * deltas[j]) - g[i,i] * deltas[i] / 2. / deltas[j] - g[j,j] * deltas[j] / 2. / deltas[i]
			return g[i,j], None


	def set_metric_tensor(self, center, deltas):

		g = numpy.zeros((len(center), len(center)), dtype=numpy.double)
		w1 = self.waveform(center)
		# First get the diagonal components
		deltas = numpy.array([self.metric_tensor_component((i,i), center, deltas / 2., g, w1)[1] for i in range(len(center))])

		# Then the rest
		[self.metric_tensor_component(ij, center, deltas, g, w1) for ij in itertools.product(range(len(center)), repeat = 2)]

		# FIXME this is a hack to get rid of negative eigenvalues
		w, v = numpy.linalg.eigh(g)
		mxw = numpy.max(w)
		self.metric_is_valid = True
		if numpy.any(w < 0):
			w[w<0.] = 1e-4 * mxw
			g = numpy.dot(numpy.dot(v, numpy.abs(numpy.diag(w))), v.T)
			self.metric_is_valid = False
		return g




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
		return self.match(self.waveform(c1), self.waveform(c2))

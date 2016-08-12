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
			hplus,hcross = lalsim.SimInspiralChooseFDWaveform(
				0., # phase
				self.df, # sampling interval
				lal.MSUN_SI * p[0],
				lal.MSUN_SI * p[1],
				p[2],
				p[3],
				p[4],
				p[5],
				p[6],
				p[7],
				self.flow,
				self.fhigh,
				0,
				1.e6 * lal.PC_SI, # distance
				0., # inclination
				0., # tidal deformability lambda 1
				0., # tidal deformability lambda 2
				None, # waveform flags
				None, # Non GR params
				0,
				7,
				self.approximant
				)

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

		# Check the match
		d2 = 1 - self.match(w1, self.waveform(center+x))
		# The match must lie in the range 0.9995 - 0.999999 to be valid for a metric computation, which means d is between 0.000001 and 0.0005
		if (d2 > 1e-5):
			return self.metric_tensor_component((i,j), center = center, deltas = deltas / 10., g = g, w1 = w1)
		if (d2 < 1e-8):
			return self.metric_tensor_component((i,j), center = center, deltas = deltas * 2., g = g, w1 = w1)

		# Compute the diagonal
		if j == i:
			g[i,i] = d2 / deltas[i] / deltas[i]
			return g[i,i]
		# NOTE Assumes diagonal parts are already computed!!!
		else:
			g[i,j] = g[j,i] = (d2 - g[i,i] * deltas[i]**2 - g[j,j] * deltas[j]**2) / (2 *  deltas[i] * deltas[j])
			return g[i,j]


	def metric_tensor(self, center, deltas):

		g = numpy.zeros((len(center), len(center)), dtype=numpy.double)
		w1 = self.waveform(center)
		# First get the diagonal components
		[self.metric_tensor_component((i,i), center, deltas / 2., g, w1) for i in range(len(center))]

		# Then the rest
		[self.metric_tensor_component(ij, center, deltas / 2., g, w1) for ij in itertools.product(range(len(center)), repeat = 2)]

		# FIXME this is a hack to get rid of negative eigenvalues
		w, v = numpy.linalg.eigh(g)
		mxw = numpy.max(w)
		if numpy.any(w < 0):
			w[w<0.] = 1e-4 * mxw
			g = numpy.dot(numpy.dot(v, numpy.abs(numpy.diag(w))), v.T)
		return g


def distance(metric_tensor, x, y):
	"""
	Compute the distance between to points inside the cube using
	the metric tensor, but assuming it is constant
	"""

	def dot(x, y, metric):
		return numpy.dot(numpy.dot(x.T, metric), y)

	return (dot(x-y, x-y, metric_tensor))**.5
		

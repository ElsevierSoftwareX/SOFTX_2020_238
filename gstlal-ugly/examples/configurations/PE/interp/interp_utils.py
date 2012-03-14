import sys
import copy
import numpy
from scipy import interpolate

def interp_mc_eta(mc, eta, M):
	interp = interpolate.RectBivariateSpline(mc, eta, M)
	#interp = interpolate.interp2d(mc, eta, M, kind = "quintic")
	return interp

def get_mc_eta_arrays_from_sngl_inspiral_table(sngl_inspiral_table):
	"""
	function assumes that the bank was built as a grid in chirp mass and eta
	"""
	t = sngl_inspiral_table
	etas = numpy.array(sorted(list(set([l.eta for l in t]))))
	mchirps = numpy.array(sorted(list(set([l.mchirp for l in t]))))
	return mchirps, etas

def pack_mixing_matrix_on_mchirp_and_eta_grid(mchirps, etas, m):
	"""
	function assumes that M contains mchirps * etas * 2 entries (2 is the
	complex template) in its second dimension and that the packing is correct
	"""
	mr = m[:,::2].reshape((m.shape[0], 20, 20))
	mi = m[:,1::2].reshape((m.shape[0], 20, 20))
	m = mr + 1.j * mi

	# solve for the phase of the first template
	phase = numpy.arctan2(numpy.imag(m[0,...]), numpy.real(m[0,...]))

	# rotate the phase of the coefficients
	# found to improve interpolation http://arxiv.org/pdf/1108.5618v1.pdf (12)
	m *= numpy.exp(-1.j * phase)

	# force them to be exactly 0 (cut roundoff)
	m[0,...] = numpy.real(m[0,...])
	return m

def waveform(mchirp, eta, interps, basis_templates):
	"""
	function takes an mchirp and eta value as well as a list of interp
	objects that is the same length as the basis functions.  The interp object,
	M(mchirp, eta) defines the new waveform as
	
	h = \sum_i M_i(mchirp, eta) * U_i

	whwere U_i is the ith basis
	"""
	out = numpy.zeros(basis_templates.shape[1])
	for interp, basis in zip(interps, basis_templates):
		out += interp(mchirp, eta)[0] * basis
	return out


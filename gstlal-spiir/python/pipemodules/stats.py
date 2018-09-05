# make a signal_feature:H1L1V1_lgsnr_lgchisq_pdf fitted from MDC detections
#
# copied from gstlal/gstlal/python/stats/__init__.py
#
# ============================================================================
#
#			Non-central chisquared pdf
#
# ============================================================================
#

#
# FIXME this is to work around a precision issue in scipy
# See: https://github.com/scipy/scipy/issues/1608
#


try:
	from fpconst import NaN, NegInf, PosInf
except ImportError:
	# fpconst is not part of the standard library and might not be
	# available
	NaN = float("nan")
	NegInf = float("-inf")
	PosInf = float("+inf")

import numpy
import math
from scipy.special import ive
def logiv(v, z):
	# from Abramowitz and Stegun (9.7.1), if mu = 4 v^2, then for large
	# z:
	#
	# Iv(z) ~= exp(z) / (\sqrt(2 pi z)) { 1 - (mu - 1)/(8 z) + (mu - 1)(mu - 9) / (2! (8 z)^2) - (mu - 1)(mu - 9)(mu - 25) / (3! (8 z)^3) ... }
	# Iv(z) ~= exp(z) / (\sqrt(2 pi z)) { 1 + (mu - 1)/(8 z) [-1 + (mu - 9) / (2 (8 z)) [1 - (mu - 25) / (3 (8 z)) ... ]]}
	# log Iv(z) ~= z - .5 log(2 pi) log z + log1p((mu - 1)/(8 z) (-1 + (mu - 9)/(16 z) (1 - (mu - 25)/(24 z) ... )))

	with numpy.errstate(divide = "ignore"):
		a = numpy.log(ive(v,z))

	# because this result will only be used for large z, to silence
	# divide-by-0 complaints from inside log1p() when z is small we
	# clip z to 1.
	mu = 4. * v**2.
	with numpy.errstate(divide = "ignore", invalid = "ignore"):
		b = -math.log(2. * math.pi) / 2. * numpy.log(z)
		z = numpy.clip(z, 1., PosInf)
		b += numpy.log1p((mu - 1.) / (8. * z) * (-1. + (mu - 9.) / (16. * z) * (1. - (mu - 25.) / (24. * z))))

	return z + numpy.where(z < 1e8, a, b)

# See: http://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
def ncx2logpdf(x, k, l):
	return -math.log(2.) - (x+l)/2. + (k/4.-.5) * (numpy.log(x) - numpy.log(l)) + logiv(k/2.-1., numpy.sqrt(l) * numpy.sqrt(x))

def ncx2pdf(x, k, l):
	return numpy.exp(ncx2logpdf(x, k, l))


# ==================================================================
#
#                        copy finish 
#
# ==================================================================

import re
from glue.ligolw import ligolw, lsctables, array, param, utils, types
from gstlal.pipemodules import pipe_macro
Attributes = ligolw.sax.xmlreader.AttributesImpl
# 
class DefaultContentHandler(ligolw.LIGOLWContentHandler):
    pass
array.use_in(DefaultContentHandler)
param.use_in(DefaultContentHandler)

def gen_signal_stats(ncx2_dof = 2, ncx2_mean_factor = 0.045):
	sgstats_pdf_mat = numpy.zeros((pipe_macro.xbin, pipe_macro.ybin), dtype = numpy.float64)
	sgstats_rate_mat = numpy.zeros((pipe_macro.xbin, pipe_macro.ybin), dtype = numpy.int64)
	lgchisq_tick = numpy.linspace(pipe_macro.ymin, pipe_macro.ymax, pipe_macro.ybin)
	lgsnr_tick = numpy.linspace(pipe_macro.xmin, pipe_macro.xmax, pipe_macro.xbin)
	for i_snr in range(0, pipe_macro.xbin):
		for i_chisq in range(0, pipe_macro.ybin):
			snr = 10**lgsnr_tick[i_snr]
			sgstats_pdf_mat[i_chisq, i_snr] = ncx2pdf(10**lgchisq_tick[i_chisq], ncx2_dof, 1 + snr*snr*ncx2_mean_factor*ncx2_mean_factor)
			sgstats_rate_mat[i_chisq, i_snr] = math.floor(1e10* sgstats_pdf_mat[i_chisq, i_snr])
	snr_rate_arr = numpy.sum(sgstats_rate_mat, axis = 0)
	chisq_rate_arr = numpy.sum(sgstats_rate_mat, axis = 1)
	return sgstats_pdf_mat, sgstats_rate_mat, snr_rate_arr, chisq_rate_arr

def signal_stats_to_xml(filename, ifos, ncx2_dof = 2, ncx2_mean_factor = 0.045, verbose = False):
	# make ifos like "H1L1" to a ifo list ["H1", "L1"]
	ifo_list = re.findall('..', ifos)
	ifo_combos = pipe_macro.get_ifo_combos(ifo_list)
	# Create new document
	xmldoc = ligolw.Document()
	lw = ligolw.LIGO_LW()

    # set up root for this sub bank
	root = ligolw.LIGO_LW(Attributes({u"Name": pipe_macro.STATS_XML_ID_NAME}))
	lw.appendChild(root)
	arr_zero_int = [0] * pipe_macro.xbin
	arr_zero_double = [0.0] * pipe_macro.xbin
	mat_zero_double = numpy.zeros((pipe_macro.xbin, pipe_macro.ybin), dtype = numpy.float64)
	# assemble the stats file, see e.g. marginalized_stats_1w.xml.gz
	pdf_mat, rate_mat, snr_rate_arr, chisq_rate_arr = gen_signal_stats(ncx2_dof = ncx2_dof, ncx2_mean_factor = ncx2_mean_factor)
	nevent = rate_mat.sum()
	if verbose:
		print "nevent %l" % nevent
	all_ifo_combos = pipe_macro.IFO_MAP + ifo_combos
	for this_ifo_combo in all_ifo_combos:
		# feature
		name = "%s:%s_%s" % (pipe_macro.SIGNAL_XML_FEATURE_NAME, this_ifo_combo, pipe_macro.SNR_RATE_SUFFIX)
		root.appendChild(array.Array.build(name, numpy.array(snr_rate_arr)))
		name = "%s:%s_%s" % (pipe_macro.SIGNAL_XML_FEATURE_NAME, this_ifo_combo, pipe_macro.CHISQ_RATE_SUFFIX)
		root.appendChild(array.Array.build(name, numpy.array(chisq_rate_arr)))
		name = "%s:%s_%s" % (pipe_macro.SIGNAL_XML_FEATURE_NAME, this_ifo_combo, pipe_macro.SNR_CHISQ_RATE_SUFFIX)
		root.appendChild(array.Array.build(name, numpy.array(rate_mat)))
 		name = "%s:%s_%s" % (pipe_macro.SIGNAL_XML_FEATURE_NAME, this_ifo_combo, pipe_macro.SNR_CHISQ_PDF_SUFFIX)
		root.appendChild(array.Array.build(name, numpy.array(pdf_mat)))
		# rank
  		name = "%s:%s_%s" % (pipe_macro.SIGNAL_XML_RANK_NAME, this_ifo_combo, pipe_macro.RANK_MAP_SUFFIX)
		root.appendChild(array.Array.build(name, numpy.array(mat_zero_double)))
   		name = "%s:%s_%s" % (pipe_macro.SIGNAL_XML_RANK_NAME, this_ifo_combo, pipe_macro.RANK_RATE_SUFFIX)
		root.appendChild(array.Array.build(name, numpy.array(arr_zero_int)))
   		name = "%s:%s_%s" % (pipe_macro.SIGNAL_XML_RANK_NAME, this_ifo_combo, pipe_macro.RANK_PDF_SUFFIX)
		root.appendChild(array.Array.build(name, numpy.array(arr_zero_double)))
  		name = "%s:%s_%s" % (pipe_macro.SIGNAL_XML_RANK_NAME, this_ifo_combo, pipe_macro.RANK_FAP_SUFFIX)
		root.appendChild(array.Array.build(name, numpy.array(arr_zero_double)))
		name = "%s:%s_nevent" % (pipe_macro.SIGNAL_XML_FEATURE_NAME, this_ifo_combo)
		root.appendChild(param.Param.build(name, types.FromPyType[long], nevent))
		name = "%s:%s_livetime" % (pipe_macro.SIGNAL_XML_FEATURE_NAME, this_ifo_combo)
		root.appendChild(param.Param.build(name, types.FromPyType[int], 0))

 	name = "%s:hist_trials" % pipe_macro.SIGNAL_XML_FEATURE_NAME
	root.appendChild(param.Param.build(name, types.FromPyType[int], 0))


    # add top level LIGO_LW to document
	xmldoc.appendChild(lw)

    # Write to file
	utils.write_filename(xmldoc, filename, gz = filename.endswith('.gz'), verbose = verbose)

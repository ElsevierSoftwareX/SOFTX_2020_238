# Copyright (C) 2014 Chris Pankow
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
import copy
import warnings

import numpy

import lal
from pylal import lalburst
from pylal.lalfft import XLALCreateForwardREAL8FFTPlan, XLALCreateReverseREAL8FFTPlan, XLALREAL8FreqTimeFFT
from pylal import datatypes as laltypes

from glue.ligolw import ligolw, utils, ilwd, lsctables

import gstlal.fftw
from gstlal.excesspower import utils

#
# =============================================================================
#
#                      Pipeline Utility Functions
#
# =============================================================================
#

def build_filter(psd, rate=4096, flow=64, fhigh=2000, filter_len=0, b_wind=16.0, corr=None):
	"""
	Build a set of individual channel Hann window frequency filters (with bandwidth 'band') and then transfer them into the time domain as a matrix. The nth row of the matrix contains the time-domain filter for the flow+n*band frequency channel. The overlap is the fraction of the channel which overlaps with the previous channel. If filter_len is not set, then it defaults to nominal minimum width needed for the bandwidth requested.

	Note: Anything that can be done with this function can be done in a more flexible manner with build_filter_from_xml, so this function is likely to disappear.
	"""
	warnings.warn("The use of excesspower.filters.build_filter is deprecated.", DeprecationWarning)

	# Filter length needs to be long enough to get the pertinent features in
	# the time domain
	rate = 2 * psd.deltaF * len(psd.data)

	if fhigh > rate/2:
		print >> sys.stderr, "WARNING: high frequency (%f) requested is higher than sampling rate / 2, adjusting to match." % fhigh
		fhigh = rate/2

	if fhigh == rate/2:
		print >> sys.stderr, "WARNING: high frequency (%f) is equal to Nyquist. Filters will probably be bad. Reduce the high frequency." % fhigh

	filter_len = 4*int(rate/b_wind)

	if filter_len <= 0:
		print >>sys.stderr, "Invalid filter length (%d). Is your filter bandwidth too small?" % filter_len
		exit(-1)
	
	# define number of band window
	bands = int((fhigh - flow) / b_wind) - 1

	# FFTW requires a thread lock for plans
	gstlal.fftw.lock()
	try:

		# Build spectral correlation function
		# NOTE: The default behavior is relative to the Hann window used in the
		# filter bank and NOT the whitener. It's just not right. Fair warning.
		# TODO: Is this default even needed anymore?
		if corr == None:
			spec_corr = lal.REAL8WindowTwoPointSpectralCorrelation(
				lal.CreateHannREAL8Window(filter_len),
				lal.CreateForwardREAL8FFTPlan(filter_len, 1)
			).data
		else:
			spec_corr = numpy.array(corr)

		# If no PSD is provided, set it equal to unity for all bins
		#if psd == None:
			#ifftplan = XLALCreateReverseREAL8FFTPlan( filter_len, 1 )
		#else:
		ifftplan = XLALCreateReverseREAL8FFTPlan((len(psd.data)-1)*2, 1)
		d_len = (len(psd.data)-1)*2

	finally:
		# Give the lock back
		gstlal.fftw.unlock()

	# FIXME: Move to main script
	if b_wind % psd.deltaF != 0:
		print >>sys.stderr, "WARNING: tile bandwidth is not a multiple of the PSD binning width. The filters (and thus tiles) will not be aligned exactly. This may lead to strange effects and imperfect event reconstruction."

	filters = numpy.zeros((filter_len-1)*bands)
	freq_filters = []
	for band in range(bands):

		# Check that the filter start is aligned with a PSD bin start:
		# Calculate an approximate integer ratio
		# the half window offset is omitted since the filter frequency
		# series is handed to CreateCOMPLEX16FrequencySeries with this
		# f0 and so this one must match the psd binning alignment
		freq_off = ((flow + band*b_wind) / psd.deltaF).as_integer_ratio()
		# If it's not a whole number, e.g. not divisible by deltaF
		if freq_off[1] != 1:
			# Subtract off the offending fractional part of deltaF
			freq_off = (freq_off[0] % freq_off[1])*psd.deltaF / freq_off[1] 
			print >>sys.stderr, "Warning: Requested frequency settings would not align the filter bins with the PSD bins. Adjusting filter frequencies by %f to compensate. Note that this may not work due to floating point comparisons that are calculated internally by the filter generation. Alternatively, use a low frequency which is a multiple of the PSD bin width (%f)" % (freq_off, psd.deltaF)
			# and make sure the offset won't take us below the
			# lowest frequency available
			assert freq_off < psd.deltaF
			freq_off = -freq_off + psd.deltaF
		else:
			freq_off = 0

		# Make sure everything is aligned now
		assert ((flow + band*b_wind + freq_off) % psd.deltaF) == 0
		try:
			# Create the EP filter in the FD
			h_wind = lalburst.XLALCreateExcessPowerFilter(
				#channel_flow =
				# The XLAL function's flow corresponds to the left side FWHM, not the near zero point. Thus, the filter *actually* begins at f_cent - band and ends at f_cent + band, and flow = f_cent - band/2 and fhigh = f_cent + band/2
				(flow + b_wind/2.0) + band*b_wind + freq_off,
				#channel_width =
				b_wind,
				#psd =
				psd,
				#correlation =
				spec_corr
			)
		except: # The XLAL wrapped function didn't work
			statuserr = "Filter generation failed for band %f with %d samples.\nPossible relevant bits and pieces that went into the function call:\n" % (band*b_wind, filter_len)
			statuserr += "PSD - deltaF: %f, f0 %f, npoints %d\n" % (psd.deltaF, psd.f0, len(psd.data))
			statuserr += "spectrum correlation - npoints %d\n" % len(spec_corr)
			statuserr += "Filter f0 %f (%f in sample length), bandwidth %f (%f in sample length)" % (flow + band*b_wind + freq_off, (flow + band*b_wind + freq_off)/psd.deltaF, b_wind, b_wind/psd.deltaF)
			sys.exit(statuserr)

		# save the frequency domain filters, if necessary
		# We make a deep copy here because we don't want the zero padding that
		# is about to be done to get the filters into the time domain
		h_wind_copy = laltypes.COMPLEX16FrequencySeries()
		h_wind_copy.f0 = h_wind.f0
		h_wind_copy.deltaF = h_wind.deltaF
		h_wind_copy.data = copy.deepcopy(h_wind.data)
		freq_filters.append(h_wind_copy)

		# Zero pad up to lowest frequency
		tmpdata = numpy.zeros(len(psd.data), dtype=numpy.complex128)
		offset = int(h_wind.f0/h_wind.deltaF)
		tmpdata[offset:offset+len(h_wind_copy.data)] = h_wind_copy.data
		h_wind.data = tmpdata
		h_wind.f0 = 0.0

		# DEBUG: Uncomment to dump FD filters
		#f = open( "filters_fd/hann_%dhz" % int( flow + band*b_wind ), "w" )
		#for freq, s in enumerate( h_wind.data ):
			#f.write( "%f %g\n" % (freq*h_wind.deltaF,s) )
		#f.close()

		# IFFT the window into a time series for use as a TD filter
		t_series = laltypes.REAL8TimeSeries()
		t_series.data = numpy.zeros((d_len,), dtype="double") 
		try:
			XLALREAL8FreqTimeFFT( 
				# t_series =
				t_series, 
				# window_freq_series =
				h_wind, 
				# ifft plan =
				ifftplan
			)
		except:
			sys.exit("Failed to get time domain filters. The usual cause of this is a filter length which is only a few PSD bins wide. Try increasing the fft-length property of the whitener.")

		td_filter = t_series.data
		# FIXME: This is a work around for a yet unfound timestamp
		# drift. Once it's found this should be reverted.
		#td_filter = numpy.roll( td_filter, filter_len/2 )[:filter_len]
		td_filter = numpy.roll(td_filter, filter_len/2)[:filter_len-1]
		## normalize the filters
		td_filter /= numpy.sqrt(numpy.dot(td_filter, td_filter))
		td_filter *= numpy.sqrt(b_wind/psd.deltaF)
		#filters = numpy.concatenate( (filters, td_filter) )
		filters[(filter_len-1)*band:(filter_len-1)*(band+1)] = td_filter
		
		# DEBUG: Uncomment to dump TD filters
		#f = open( "filters_td/hann_%dhz" % int( flow + band*b_wind ), "w" )
		#for t, s in enumerate( td_filter ):
			#f.write( "%g %g\n" % (t/rate,s) )
		#f.close()

	# Shape it into a "matrix-like" object
	#filters.shape = ( bands, filter_len )
	filters.shape = (bands, filter_len-1)
	return filters, freq_filters

def build_filter_from_xml(sb_table, psd, corr=None):
	"""
	Build a set of individual channel Hann window frequency filters (with bandwidth 'band') and then transfer them into the time domain as a matrix. The nth row of the matrix contains the time-domain filter for the flow+n*band frequency channel. The overlap is the fraction of the channel which overlaps with the previous channel. If filter_len is not set, then it defaults to nominal minimum width needed for the bandwidth requested.
	"""

	# FIXME: We'll need to create one plan for each duration for filters with
	# differing durations
	# FFTW requires a thread lock for plans
	gstlal.fftw.lock()
	try:
		# Build spectral correlation function
		spec_corr = numpy.array(corr)
		ifftplan = XLALCreateReverseREAL8FFTPlan( (len(psd.data)-1)*2, 1 )
		d_len = (len(psd.data)-1)*2
	finally:
		# Give the lock back
		gstlal.fftw.unlock()

	# Filter length needs to be long enough to get the pertinent features in
	# the time domain
	rate = 2 * psd.deltaF * len(psd.data)
	# FIXME: Since rate was fixed, do we need the 4 below?
	filter_len = 4*int(rate/sb_table[0].bandwidth)
	
	# TODO: For filters with different durations, we'll have to keep track of
	# the intended length of each filter
	filters = numpy.zeros((filter_len-1)*len(sb_table))
	freq_filters = []
	for i, row in enumerate(sb_table):
		# cfreq + band since the filters are actually 2*band wide
		if row.central_freq + row.bandwidth > rate/2:
			raise ValueError("Filter high frequency (%f) requested is higher than Nyquist (%f)." % (row.central_freq + row.bandwidth, rate/2.0))
			continue

		if row.central_freq + row.bandwidth == rate/2:
			warnings.warn("Filter high frequency (%f) is equal to Nyquist. Filters could potentially be bad. Suggest to reduce the high frequency." % (row.central_freq+ row.bandwidth))

		try:
			# Create the EP filter in the FD
			h_wind = lalburst.XLALCreateExcessPowerFilter(
				#channel_flow =
				# The XLAL function's flow corresponds to the left side FWHM, not the near zero point. Thus, the filter *actually* begins at f_cent - band and ends at f_cent + band, and flow = f_cent - band/2 and fhigh = f_cent + band/2
				row.central_freq - row.bandwidth/2.0,
				#channel_width =
				row.bandwidth,
				#psd =
				psd,
				#correlation =
				spec_corr
			)
		except: # The XLAL wrapped function didn't work
			statuserr = "Filter generation failed for band %f with %d samples.\nPossible relevant bits and pieces that went into the function call:\n" % (row.bandwidth, filter_len)
			statuserr += "PSD - deltaF: %f, f0 %f, npoints %d\n" % (psd.deltaF, psd.f0, len(psd.data))
			statuserr += "spectrum correlation - npoints %d" % len(spec_corr)
			sys.exit(statuserr)

		# NOTE: The filter that come out of this are not normalized to
		# band/deltaF, they are actually a factor of two too small.
		# This is probably okay, because we renormalize the time domain versions
		# and so we never see the factor of two difference
		#print numpy.dot(abs(h_wind.data), abs(h_wind.data))
		# save the frequency domain filters, if necessary
		# We make a deep copy here because we don't want the zero padding that
		# is about to be done to get the filters into the time domain
		h_wind_copy = laltypes.COMPLEX16FrequencySeries()
		h_wind_copy.f0 = h_wind.f0
		h_wind_copy.deltaF = h_wind.deltaF
		h_wind_copy.data = copy.deepcopy(h_wind.data)
		freq_filters.append(h_wind_copy)

		# Zero pad up to lowest frequency
		tmpdata = numpy.zeros(len(psd.data), dtype=numpy.complex128)
		offset = int(h_wind.f0/h_wind.deltaF)
		tmpdata[offset:offset+len(h_wind_copy.data)] = h_wind_copy.data
		h_wind.data = tmpdata
		h_wind.f0 = 0.0

		# DEBUG: Uncomment to dump FD filters
		#f = open( "filters_fd/hann_%dhz" % int(row.central_freq), "w" )
		#for freq, s in enumerate( h_wind.data ):
			#f.write( "%f %g\n" % (freq*h_wind.deltaF,s) )
		#f.close()

		# IFFT the window into a time series for use as a TD filter
		t_series = laltypes.REAL8TimeSeries()
		t_series.data = numpy.zeros((d_len,), dtype="double") 
		try:
			XLALREAL8FreqTimeFFT( 
				# t_series =
				t_series, 
				# window_freq_series =
				h_wind, 
				# ifft plan =
				ifftplan
			)
		except:
			sys.exit( "Failed to get time domain filters. The usual cause of this is a filter length which is only a few PSD bins wide. Try increasing the fft-length property of the whitener." )

		td_filter = t_series.data
		# FIXME: This is a work around for a yet unfound timestamp
		# drift. Once it's found this should be reverted.
		#td_filter = numpy.roll( td_filter, filter_len/2 )[:filter_len]
		td_filter = numpy.roll(td_filter, filter_len/2)[:filter_len-1]
		## normalize the filters
		td_filter /= numpy.sqrt(numpy.dot(td_filter, td_filter))
		td_filter *= numpy.sqrt(row.bandwidth/psd.deltaF)
		filters[(filter_len-1)*i:(filter_len-1)*(i+1)] = td_filter
		
		# DEBUG: Uncomment to dump TD filters
		#f = open( "filters_td/hann_%dhz" % int(row.central_freq), "w" )
		#for t, s in enumerate( td_filter ):
			#f.write( "%g %g\n" % (t/rate,s) )
		#f.close()

	# Shape it into a "matrix-like" object
	#filters.shape = ( bands, filter_len )
	filters.shape = (len(sb_table), filter_len-1)
	return filters, freq_filters

def build_chan_matrix(nchannels=1, frequency_overlap=0.0, up_factor=0, norm=None):
	"""
	Build the matrix to properly normalize nchannels coming out of the FIR filter. Norm should be an array of length equal to the number of output channels, with the proper normalization factor. up_factor controls the number of output channels. E.g. If thirty two input channels are indicated, and an up_factor of two is input, then an array of length eight corresponding to eight output channels are required. The output matrix uses 1/sqrt(A_i) where A_i is the element of the input norm array.
	"""

	if up_factor > int(numpy.log2(nchannels))+1:
		sys.exit("up_factor cannot be larger than log2(nchannels).")
	elif up_factor < 0:
		sys.exit("up_factor must be larger than or equal to 0.")

	# If no normalization coefficients are provided, default to unity
	if norm is None:
		norm = numpy.ones(nchannels >> up_factor)

	# Number of non-zero elements in that row
	n = 2**up_factor

	# Samples to skip for frequency overlap
	m = int(1.0/(1-frequency_overlap))

	# Matrix row
	mat = numpy.zeros((len(norm), nchannels))
	mset, i = 0, 0
	indices = numpy.linspace(i, i+n-1, n)*m
	for mu_sq in norm:
		if mu_sq > 0:
			mat[i, map(int,indices)] = numpy.sqrt(1.0/mu_sq)
			if mset == m-1:
				indices += n*m - mset
				mset = 0
			else:
				indices += 1
				mset += 1
		#else:  # End of the filter bank which we're killing
		i += 1

	return numpy.array(mat).T

# FIXME: Remove band
def build_wide_filter_norm(corr, freq_filters, level, frequency_overlap=0, band=None, psd=None):
	"""
	Determine the mu^2(f_low, n*b) for higher bandwidth channels from the base band. Requires the spectral correlation (corr), the frequency domain filters (freq_filters), and resolution level. The bandwidth of the wide channels to normalize is 2**level*band. Overlap of channels (frequency_overlap) will divide the channels into sets of 1/(1-frequency_overlap) channels and sum the sets together rather than adjacent channels.
	"""
	# TODO: This can be made even more efficient by using the calculation of
	# lower levels for higher levels

	# number of channels to combine
	n = 2**level
	# frequency overlap requires every mth sample
	# for frequency_overlap = 0.0, sets = 1 (adjacent)
	# for frequency_overlap = 0.5 (50%), sets = 2
	# for frequency_overlap = 0.25 (75%), sets = 4
	# FIXME: What happens when frequency_overlap isn't a negative power of two?
	sets = int(1/(1-frequency_overlap))
	
	# prefactor
	if band is None:
		# NOTE: The filter is 2 bandwidths long
		band = len(freq_filters[0].data)/2*freq_filters[0].deltaF
	del_f = freq_filters[0].deltaF
	mu_sq = n*band/del_f

	# This is the default normalization for the base band
	if level == 0:
		return numpy.ones(len(freq_filters))*mu_sq

	filter_norm = numpy.zeros(len(freq_filters)/n)
	corr = numpy.array(corr)

	# Construct the normalization for the i'th wide filter at this level by
	# summing over n base band filters
	for s in range(sets):
		# This is the mth set of filters to sum
		filter_set = filter(lambda i: i % sets == s, range(len(freq_filters)))
		# Divided into n bands
		filter_n = 0
		for bands in [filter_set[i:i+n] for i in range(0, len(filter_set), n)]:
			if len(bands) != n:
				continue

			ip_sum = 0
			# Sum over n base band filters
			for j in range(len(bands)-1):
				#if psd is None:
				ip_sum += lalburst.XLALExcessPowerFilterInnerProduct( 
					freq_filters[bands[j]], freq_filters[bands[j+1]], corr
				)
				# TODO: fix if better hrss is required
				#else:
					#ip_sum += lalburst.XLALExcessPowerFilterInnerProduct( 
						#freq_filters[i*n+j], freq_filters[i*n+j+1], corr, psd
					#)

			filter_norm[sets*filter_n+s] = mu_sq + 2*ip_sum
			filter_n += 1

	return filter_norm

def build_fir_sq_adder(nsamp, padding=0):
	"""
	Just a square window of nsamp long. Used to sum samples in time. Setting the padding will pad the end with that many 0s. Padding is required in many cases because the audiofirfilter element in gstreamer defaults to time domain convolution for filters < 32 samples. However, TD convolution in audiofirfilter is broken, thus we invoke the FFT based convolution with a filter > 32 samples.

	Note: now that excesspower has switched to using lal_mean, this is no longer necessary, and is likely to be removed soon.
	"""
	warnings.warn("excesspower.build_fir_sq_adder is deprecated and could be removed soon.", DeprecationWarning)
	return numpy.hstack((numpy.ones(nsamp), numpy.zeros(padding)))

def create_bank_xml(flow, fhigh, band, duration, level=0, ndof=1, frequency_overlap=0, detector=None, units=utils.EXCESSPOWER_UNIT_SCALE['Hz']):
	"""
	Create a bank of sngl_burst XML entries. This file is then used by the trigger generator to do trigger generation. Takes in the frequency parameters and filter duration and returns an ligolw entity with a sngl_burst Table which can be saved to a file.
	"""

	xmldoc = ligolw.Document()
	xmldoc.appendChild(ligolw.LIGO_LW())
	bank = lsctables.New(lsctables.SnglBurstTable,
	["peak_time_ns", "start_time_ns", "stop_time_ns",
	"process_id", "ifo", "peak_time", "start_time", "stop_time",
	"duration", "time_lag", "peak_frequency", "search",
	"central_freq", "channel", "amplitude", "snr", "confidence",
	"chisq", "chisq_dof",
	"flow", "fhigh", "bandwidth", "tfvolume", "hrss", "event_id"])
	bank.sync_next_id()

	# The first frequency band actually begins at flow, so we offset the 
	# central frequency accordingly
	if level == 0: # Hann windows
		edge = band / 2
		cfreq = flow + band
	else: # Tukey windows
		edge = band / 2**(level+1)
		# The sin^2 tapering comes from the Hann windows, so we need to know 
		# how far they extend to account for the overlap at the ends
		cfreq = flow + edge + (band / 2)

	while cfreq + edge + band/2 <= fhigh:
		row = bank.RowType()
		row.search = u"gstlal_excesspower"
		row.duration = duration * ndof
		row.bandwidth = band
		row.peak_frequency = cfreq
		row.central_freq = cfreq
		# This actually marks the 50 % overlap point
		row.flow = cfreq - band / 2.0
		# This actually marks the 50 % overlap point
		row.fhigh = cfreq + band / 2.0
		row.ifo = detector
		row.chisq_dof = 2*band*row.duration
		row.duration *= units

		# Stuff that doesn't matter, yet
		row.peak_time_ns = 0
		row.peak_time = 0
		row.start_time_ns = 0
		row.start_time = 0
		row.stop_time_ns = 0
		row.stop_time = 0
		row.tfvolume = 0
		row.time_lag = 0
		row.amplitude = 0
		row.hrss = 0
		row.snr = 0
		row.chisq = 0
		row.confidence = 0
		row.event_id = bank.get_next_id()
		row.channel = "awesome full of GW channel"
		row.process_id = ilwd.ilwdchar( u"process:process_id:0" )

		bank.append(row)
		#cfreq += band #band is half the full width of the window, so this is 50% overlap
		cfreq += band * (1-frequency_overlap)

	xmldoc.childNodes[0].appendChild(bank)
	return xmldoc

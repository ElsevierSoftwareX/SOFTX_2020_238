#!/usr/bin/python
#
# Copyright (C) 2012 Chris Pankow
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
import os
import StringIO
import subprocess
import shlex
import re
import json
import copy
from bisect import bisect_left
from itertools import chain, ifilter

import numpy

from scipy.stats import chi2, poisson, mannwhitneyu, norm

from pylal import lalburst
from pylal.lalfft import XLALCreateForwardREAL8FFTPlan, XLALCreateReverseREAL8FFTPlan, XLALREAL8FreqTimeFFT
from pylal.xlal.datatypes.real8frequencyseries import REAL8FrequencySeries
from pylal.xlal.datatypes.complex16frequencyseries import COMPLEX16FrequencySeries
from pylal.xlal.datatypes.real8timeseries import REAL8TimeSeries

from glue.ligolw import ligolw
from glue.ligolw import ilwd
from glue.ligolw import utils
from glue.ligolw import lsctables
from glue import lal
from glue.segments import segment, segmentlist

from gstlal.pipeutil import gst
from gstlal import pipeparts 

import gstlal.fftw

#
# =============================================================================
#
#                                Utility Functions
#
# =============================================================================
#

def subdivide( seglist, length, min_length=0 ):
	newlist = []
	for seg in seglist:
		while abs(seg) - min_length > length + min_length:
			newlist.append( segment(seg[0], seg[0]+length ) )
			seg = segment(seg[0] + length, seg[1])

		if abs(seg) > 0:
			newlist.append( segment(seg[0], seg[1] - min_length) )
			newlist.append( segment(seg[1] - min_length, seg[1]) )

	return segmentlist(newlist)	


#
# =============================================================================
#
#                                Unit Handling
#
# =============================================================================
#

EXCESSPOWER_UNIT_SCALE = {
	'Hz':  1024**0,
	'kHz': 1024**1,
	'MHz': 1024**2,
	'GHz': 1024**3,
	'mHz': 1024**-1,
	'uHz': 1024**-2,
	'nHz': 1024**-3,
}
# This is really only for the strain channels where one might anticipate needing a state vector.
EXCESSPOWER_DQ_VECTOR = {
	# 1  Science mode on 
	"SCIENCE_MODE" : 0b1,
	# 2  ITF fully locked 
	"ITF_LOCK" : 0b10,
	# 3  h(t) reconstruction ok 
	"HREC_OK": 0b100,
	# 4  Reserved for future use
	"RESERVED" : 0b1000,
	# 5  CBC Injection 
	"CBC_INJECTION" : 0b10000,
	# 6  CBC_CAT1 
	"CBC_CAT1" : 0b100000,
	# 7  CBC_CAT2 [not used]
	"CBC_CAT2" : 0b1000000,
	# 8  CBC_CAT3 [not used]
	"CBC_CAT3" : 0b10000000,
	# 9  Burst injection 
	"BURST_INJECTION" : 0b100000000,
	# 10 Burst_CAT1 
	"BURST_CAT1" : 0b1000000000,
	# 11 Burst_CAT2 
	"BURST_CAT2" : 0b10000000000,
	# 12 Burst_CAT3 
	"BURST_CAT3" : 0b100000000000,
	# 13 CW injection 
	"CW_INJECTION" : 0b1000000000000,
	# 14 Reserved for future use, possibly by CW 
	# 15 Reserved for future use, possibly by CW 
	# 16 Reserved for future use, possibly by CW 
	# 17 Stochastic injection 
	"STOCHASTIC_INJECTION" : 0b10000000000000000
	# 18 Reserved for future use, possibly by stochastic 
	# 19 Reserved for future use, possibly by stochastic [not used]
	# 20 Reserved for future use, possibly by stochastic [not used]
}
DEFAULT_DQ_VECTOR_ON = EXCESSPOWER_DQ_VECTOR["SCIENCE_MODE"] | EXCESSPOWER_DQ_VECTOR["ITF_LOCK"] | EXCESSPOWER_DQ_VECTOR["HREC_OK"] # | EXCESSPOWER_DQ_VECTOR["BURST_CAT1"]

# This is to ensure that when the DQ vector goes out of lock that we follow it
DEFAULT_DQ_VECTOR_OFF = EXCESSPOWER_DQ_VECTOR["RESERVED"]

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
	"""

	if fhigh > rate/2:
		print >> sys.stderr, "WARNING: high frequency (%f) requested is higher than sampling rate / 2, adjusting to match." % fhigh
		fhigh = rate/2

	if fhigh == rate/2:
		print >> sys.stderr, "WARNING: high frequency (%f) is equal to Nyquist. Filters will probably be bad. Reduce the high frequency." % fhigh

	# Filter length needs to be long enough to get the pertinent features in
	# the time domain
	rate = psd.deltaF * len(psd.data)
	filter_len = 4*int(rate/b_wind)

	if filter_len <= 0:
		print >>sys.stderr, "Invalid filter length (%d). Is your filter bandwidth too small?" % filter_len
		exit(-1)
	
	# define number of band window
	bands = int( (fhigh - flow) / b_wind ) - 1

	# FFTW requires a thread lock for plans
	gstlal.fftw.lock()
	try:

		# Build spectral correlation function
		# NOTE: The default behavior is relative to the Hann window used in the
		# filter bank and NOT the whitener. It's just not right. Fair warning.
		# TODO: Is this default even needed anymore?
		if corr == None:
			wfftplan = XLALCreateForwardREAL8FFTPlan( filter_len, 1 )
			spec_corr = lalburst.XLALREAL8WindowTwoPointSpectralCorrelation(
				XLALCreateHannREAL8Window( filter_len ),
				wfftplan 
			)
		else:
			spec_corr = numpy.array( corr )

		# If no PSD is provided, set it equal to unity for all bins
		#if psd == None:
			#ifftplan = XLALCreateReverseREAL8FFTPlan( filter_len, 1 )
		#else:
		ifftplan = XLALCreateReverseREAL8FFTPlan( (len(psd.data)-1)*2, 1 )
		d_len = (len(psd.data)-1)*2

	finally:
		# Give the lock back
		gstlal.fftw.unlock()

	filters = numpy.array([])
	freq_filters = []
	for band in range( bands ):

		try:
			# Create the EP filter in the FD
			h_wind = lalburst.XLALCreateExcessPowerFilter(
				#channel_flow =
				# The XLAL function's flow corresponds to the left side FWHM, not the near zero point. Thus, the filter *actually* begins at f_cent - band and ends at f_cent + band, and flow = f_cent - band/2 and fhigh = f_cent + band/2
				(flow + b_wind/2.0) + band*b_wind,
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
			statuserr += "spectrum correlation - npoints %d" % len(spec_corr)
			sys.exit( statuserr )

		# save the frequency domain filters, if necessary
		# We make a deep copy here because we don't want the zero padding that
		# is about to be done to get the filters into the time domain
		h_wind_copy = COMPLEX16FrequencySeries()
		h_wind_copy.f0 = h_wind.f0
		h_wind_copy.deltaF = h_wind.deltaF
		h_wind_copy.data = copy.deepcopy(h_wind.data)
		freq_filters.append( h_wind_copy )

		# Zero pad up to lowest frequency
		h_wind.data = numpy.hstack((numpy.zeros((int(h_wind.f0 / h_wind.deltaF), ), dtype = "complex"), h_wind.data))
		h_wind.f0 = 0.0
		d = h_wind.data
		# Zero pad window to get up to Nyquist
		h_wind.data = numpy.hstack((d, numpy.zeros((len(psd.data) - len(d),), dtype = "complex")))

		# DEBUG: Uncomment to dump FD filters
		#f = open( "filters_fd/hann_%dhz" % int( flow + band*b_wind ), "w" )
		#for freq, s in enumerate( h_wind.data ):
			#f.write( "%f %g\n" % (freq*h_wind.deltaF,s) )
		#f.close()

		# IFFT the window into a time series for use as a TD filter
		try:
			t_series = REAL8TimeSeries()
			t_series.data = numpy.zeros( (d_len,), dtype="double" ) 
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
		td_filter = numpy.roll( td_filter, filter_len/2 )[:filter_len-1]
		## normalize the filters
		td_filter /= numpy.sqrt( numpy.dot(td_filter, td_filter) )
		td_filter *= numpy.sqrt(b_wind/psd.deltaF)
		filters = numpy.concatenate( (filters, td_filter) )
		
		# DEBUG: Uncomment to dump TD filters
		#f = open( "filters_td/hann_%dhz" % int( flow + band*b_wind ), "w" )
		#for t, s in enumerate( td_filter ):
			#f.write( "%g %g\n" % (t/rate,s) )
		#f.close()

	# Shape it into a "matrix-like" object
	#filters.shape = ( bands, filter_len )
	filters.shape = ( bands, filter_len-1 )
	return filters, freq_filters

def build_chan_matrix( nchannels=1, up_factor=0, norm=None ):
	"""
	Build the matrix to properly normalize nchannels coming out of the FIR filter. Norm should be an array of length equal to the number of output channels, with the proper normalization factor. up_factor controls the number of output channels. E.g. If thirty two input channels are indicated, and an up_factor of two is input, then an array of length eight corresponding to eight output channels are required. The output matrix uses 1/sqrt(A_i) where A_i is the element of the input norm array.
	"""

	if up_factor > int(numpy.log2(nchannels))+1:
		sys.exit( "up_factor cannot be larger than log2(nchannels)." )
	elif up_factor < 0:
		sys.exit( "up_factor must be larger than or equal to 0." )

	# If no normalization coefficients are provided, default to unity
	if norm is None:
		norm = numpy.ones( nchannels >> up_factor )

	# Number of non-zero elements in that row
	n = 2**up_factor

	# Matrix row
	r0 = numpy.zeros(nchannels)
	m = []
	for i, mu_sq in enumerate(norm):
		r = r0.copy()
		if mu_sq > 0:
			r[i*n:(i+1)*n] = numpy.sqrt(1.0/mu_sq)
		else:  # End of the filter bank which we're killing
			r[i*n:(i+1)*n] = 0
		m.append( r )

	return numpy.array(m).T

def build_wide_filter_norm( corr, freq_filters, level, band=None, psd=None ):
	"""
	Determine the mu^2(f_low, n*b) for higher bandiwdth channels from the base band. Requires the spectral correlation (corr) and the frequency domain filters (freq_filters), and resolution level. The bandwidth of the wide channels to normalize is 2**level*band.
	"""
	# TODO: This can be made even more efficient by using the calculation of
	# lower levels for higher levels

	# number of channels to combine
	n = 2**level

	# prefactor
	if band is None:
		band = len(freq_filters[0].data)/freq_filters[0].deltaF
	del_f = freq_filters[0].deltaF
	mu_sq = n*band/del_f

	# This is the default normalization for the base band
	if level == 0:
		return numpy.ones(len(freq_filters))*mu_sq

	filter_norm = numpy.zeros( len(freq_filters)/n )
	corr = numpy.array(corr)

	# Construct the normalization for the i'th wide filter at this level by
	# summing over n base band filters
	for i in range(len(filter_norm)):

		ip_sum = 0
		# Sum over n base band filters
		for j in range(n-1):
			#if psd is None:
			ip_sum += lalburst.XLALExcessPowerFilterInnerProduct( 
				freq_filters[i*n+j], freq_filters[i*n+j+1], corr
			)
			# TODO: fix if better hrss is required
			#else:
				#ip_sum += lalburst.XLALExcessPowerFilterInnerProduct( 
					#freq_filters[i*n+j], freq_filters[i*n+j+1], corr, psd
				#)

		filter_norm[i] = mu_sq + 2*ip_sum

	return filter_norm

def build_fir_sq_adder( nsamp, padding=0 ):
	"""
	Just a square window of nsamp long. Used to sum samples in time. Setting the padding will pad the end with that many 0s. Padding is required in many cases because the audiofirfilter element in gstreamer defaults to time domain convolution for filters < 32 samples. However, TD convolution in audiofirfilter is broken, thus we invoke the FFT based convolution with a filter > 32 samples.
	"""
	return numpy.hstack( (numpy.ones(nsamp), numpy.zeros(padding)) )

def create_bank_xml(flow, fhigh, band, duration, level=1, ndof=1, detector=None):
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

	# The first frequency band actually begins at flow, so we offset the central 
	# frequency accordingly
	if level == 0: # Hann windows
		cfreq = flow + band
	else: # Tukey windows
		# The sin^2 tapering comes from the Hann windows, so we need to know how far
		# they extend to account for the overlap at the ends
		cfreq = flow + (int(band) >> (level+1)) + band/2

	# This might overestimate the number of output channels by one, but that
	# shoudn't be a problem since the last channel would simply never be used.
	while cfreq <= fhigh:
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

		bank.append( row )
		cfreq += band #band is half the full width of the window, so this is 50% overlap

	xmldoc.childNodes[0].appendChild(bank)
	return xmldoc

def duration_from_cache( cachef ):
	cache = lal.Cache.fromfile( open( cachef ) )
	duration = cache[0].segment
	for entry in cache[1:]:
		duration |= entry.segment

	return duration[0], abs(duration)

def determine_thresh_from_fap( fap, ndof = 2 ):
	"""
	Given a false alarm probability desired, and a given number of degrees of freedom (ndof, default = 2), calculate the proper amplitude snr threshold for samples of tiles with that ndof. This is obtained by solving for the statistical value of a CDF for a chi_squared with ndof degrees of freedom at a given probability.
	"""

	return numpy.sqrt( chi2.ppf( 1-fap, ndof ) )

#
# =============================================================================
#
#                          Utility Functions
#
# =============================================================================
#

def determine_segment_with_whitening( analysis_segment, whiten_seg ):
	"""
	Determine the difference between the segment requested to be analyzed and the segment which was actually analyzed after the whitener time is dropped. This is equivalent to the "in" and "out" segs, respectively, in the search_summary.
	"""
	if whiten_seg.intersects( analysis_segment ):
		if analysis_segment in whiten_seg:
			# All the analyzed time is within the settling time
			# We make this explicit because the segment constructor will just reverse the arguments if arg2 < arg1 and create an incorrect segment
			analysis_segment = segment( analysis_segment[1], analysis_segment[1] )
		else:
			analysis_segment -= whiten_seg
	return analysis_segment
	
def append_formatted_output_path( fmt, handler, bdir="./", mkdir=True ):
	"""
	Append a formatted output path to the base directory (default is pwd). Acceptable options are:
	i: instrument[0] like 'H' for 'H1'
	I: instrument
	S: subsystem prefix
	c: channel without subsystem
	C: full channel (eg. LSC-STRAIN)
	G#: first # GPS digits

	Example: fmt="I/S/c/G5"
	=> H1/PSL/ISS_PDA_OUT_DQ/10340

	Options in the format which are unrecoginze will pass through without being modified. If mkdir is set to True (default), the directory will be created if it does not already exist.
	"""
	gps_reg = re.compile( "^G(\d*)$" )
	def converter( ic ):
		gps = re.search( gps_reg, ic )
		if gps is not None and len(gps.group(1)) > 0:
			gps = int( gps.group(1) )
		if ic == "i": return handler.inst[0]
		elif ic == "I": return handler.inst
		elif ic == "S": return handler.channel.split("-")[0]
		elif ic == "c": return handler.channel.split("-")[-1]
		elif ic == "C": return handler.channel
		elif type(gps) is int: return str(int(handler.time_since_dump))[:gps]
		elif gps is not None: return str(handler.time_since_dump)
		return ic

	subdir = bdir + "/".join( [ converter(seg) for seg in fmt.strip().split("/") ] )
	if mkdir and not os.path.exists( subdir ):
		os.makedirs( subdir )
	return subdir

def make_cache_parseable_name( inst, tag, start, stop, ext, dir="./" ):
	"""
	Make a LIGO cache parseable name for a segment of time spanning start to stop from instrument inst and file extension ext. If a directory shoudl be prepended to the path, indicate it with dir. The duration is calculated as prescirbed in the technical document describing the LIGO cache format.
	"""
	dur =  numpy.ceil(stop) - numpy.floor(start)
	tag = tag.replace("-","_")

	name = "%s/%s-%s_excesspower-%d-%d.%s" % (dir, inst, tag, start, dur, ext)
	
	return name

def upload_to_db( sb_event_table, search = "EP", type = "GlitchTrigger", db = "glitchdb" ):
	"""
	Upload a sngl_burst event to glitchdb. The 'search' and 'type' variables will be supplied to glitchdb for its search and type respectively. If no type is specified, the function will attempt to determine it from the channel. If it can't, it will default to GlitchTirgger.
	"""
	try: 
		type = sb_event_table[0].channel.split("-")[0]
	# FIXME: Default to glitchtrigger if the subsystem isn't in the channel name
	except AttributeError:
		pass

	cmd = "%s %s %s -" % (db, search, type)

	xmldoc = ligolw.Document()
	xmldoc.appendChild( ligolw.LIGO_LW() )
	xmldoc.childNodes[0].appendChild( sb_event_table )
	strbuf = StringIO.StringIO()
	table_str = utils.write_fileobj( xmldoc, strbuf, trap_signals=None )

	# Open a pipe to the process and pipe in the XML as stdin
	proc = subprocess.Popen( shlex.split(str(cmd)), stdin=subprocess.PIPE )
	proc.communicate( strbuf.getvalue() )
	if proc.returncode != 0:
		print >>sys.stderr, "Warning, failed to upload to gracedb. Process returned %d" % proc.returncode


#
# =============================================================================
#
#                          Visualization Routines
#
# =============================================================================
#

def stream_tfmap_video( pipeline, head, handler, filename=None, split_on=None, snr_max=None, history=4, framerate=5 ):
	"""
	Stream the time frequency channel map to a video source. If filename is None and split_on is None (the default), then the pipeline will attempt to stream to a desktop based (xvimagesink or equivalent) video sink. If filename is not None, but no splitting behavior is specified, video will be encoded and saved to the filename plus ".ogg" in Ogg Vorbis format. If split_on is specified to be 'keyframe', then the encoded video will be split between multiple files based on the keyframes being emitted by the ogg muxer. If no file name is specifed a default will be used, otherwise, an index and ".ogg" will be appended to the file name. Specifying amp_max will set the top of the colorscale for the amplitude SNR, the default is 10. History is the amount of time to retain in the video buffer (in seconds), the default is 4. The frame rate is the number of frames per second to output in the video stream.
	"""

	if snr_max is None:
		snr_max = 10 # arbitrary
		z_autoscale = True 
	# Tee off the amplitude stream
	head = chtee = pipeparts.mktee( pipeline, head )
	head = pipeparts.mkqueue( pipeline, head )
	head = pipeparts.mkgeneric( pipeline, head, "cairovis_waterfall",
			title = "TF map %s:%s" % (handler.inst, handler.channel),
			z_autoscale = z_autoscale,
			z_min = 0,
			z_max = snr_max,
			z_label = "SNR",
			#y_autoscale = True,
			#y_min = handler.flow,
			#y_max = handler.fhigh,
			y_data_autoscale = False,
			y_data_min = handler.flow,
			y_data_max = handler.fhigh,
			y_label = "frequency (Hz)",
			x_label = "time (s)",
			colormap = "jet",
			colorbar = True,
			history = gst.SECOND*history
	)

	# Do some format conversion
	head = pipeparts.mkcapsfilter( pipeline, head, "video/x-raw-rgb,framerate=%d/1" % framerate )
	head = pipeparts.mkprogressreport( pipeline, head, "video sink" )

	# TODO: Explore using different "next file" mechanisms
	if split_on == "keyframe":

		# Muxer
		head = pipeparts.mkcolorspace( pipeline, head )
		head = pipeparts.mkcapsfilter( pipeline, head, "video/x-raw-yuv,framerate=5/1" )
		head = pipeparts.mkoggmux( pipeline, pipeparts.mktheoraenc( pipeline, head ) )

		if filename is None: 
			filename = handler.inst + "_tfmap_%d.ogg"
		else: 
			filename = filename + "_%d.ogg"

		print >>sys.stderr, "Streaming TF maps to %s\n" % filename
		pipeparts.mkgeneric( pipeline, head, "multifilesink",
			next_file = 2, location = filename, sync = False, async = False )

	elif filename is not None:
		# Muxer
		head = pipeparts.mkcolorspace( pipeline, head )
		head = pipeparts.mkcapsfilter( pipeline, head, "video/x-raw-yuv,framerate=5/1" )
		head = pipeparts.mkoggmux( pipeline, pipeparts.mktheoraenc( pipeline, head ) )
		filename = filename + ".ogg"
		pipeparts.mkfilesink( pipeline, head, filename )

	else: # No filename and no splitting options means stream to desktop
		pipeparts.mkgeneric( pipeline, head, "autovideosink", filter_caps=gst.caps_from_string("video/x-raw-rgb") )

	return chtee

def compute_amplitude( sb, psd ):
	"""
	Compute the band-limited channel amplitudes from the whitened channel smaples. Note that this is only a rough approximation of the true band-limited amplitude since it ignores the effect of the filters. They are approximated as square windows which pick up the values of the PSD across the entire bandwidth of the tile.
	"""
	flow = int((sb.central_freq - sb.bandwidth/2.0 - psd.f0)/psd.deltaF)
	fhigh = int((sb.central_freq + sb.bandwidth/2.0 - psd.f0)/psd.deltaF)
	pow = sum(psd.data[flow:fhigh])
	sb.amplitude = numpy.sqrt(pow*sb.snr*psd.deltaF/sb.bandwidth)

class SBStats(object):
	"""
	Keep a "running history" of events seen. Useful for detecting statistically outlying sets of events.
	"""
	def __init__( self ):
		self.offsource = {}
		self.onsource = {}
		self.offsource_interval = 6000
		self.onsource_interval = 60

	def event_rate( self, nevents=10 ):
		"""
		Calculate the Poissonian significance of the 'on source' trial set for up to the loudest nevents.
		"""

		offtime = float(abs(segmentlist(self.offsource.keys())))
		offsource = sorted( chain(*self.offsource.values()), key=lambda sb: -sb.snr )
		offrate = zip( offsource, map( lambda i:i/offtime, range(1, len(offsource)+1) ) )
		offrate = offrate[::-1]
		offsource = offsource[::-1]
		offsnr = [sb.snr for sb in offsource]

		ontime = float(abs(segmentlist(self.onsource.keys())))
		if ontime == 0:
			return []
		onsource = sorted( chain(*self.onsource.values()), key=lambda sb: -sb.snr )
		onsnr = [sb.snr for sb in onsource]
		onrate = []
		for snr in onsnr:
			try:
				onrate.append( offrate[bisect_left( offsnr, snr )][1] )
			except IndexError: # on SNR > max off SNR
				onrate.append( 0 )

		return onrate

	# FIXME: Have event_sig call event_rate
	def event_significance( self, nevents=10, rank_fcn=None):
		"""
		Calculate the Poissonian significance of the 'on source' trial set for up to the loudest nevents.
		"""
		if rank_fcn is None:
			rank_fcn = lambda e: e.snr

		offtime = float(abs(segmentlist(self.offsource.keys())))
		offsource = sorted( chain(*self.offsource.values()), key=lambda sb: -sb.snr )
		offrate = zip( offsource, map( lambda i:i/offtime, range(1, len(offsource)+1) ) )
		offrate = offrate[::-1]
		offsource = offsource[::-1]
		offsnr = map(rank_fcn, offsource)

		ontime = float(abs(segmentlist(self.onsource.keys())))
		if ontime == 0:
			return []
		onsource = sorted( chain(*self.onsource.values()), key=lambda sb: -sb.snr )
		onsnr = map(rank_fcn, onsource)
		onrate = []
		for snr in onsnr:
			try:
				onrate.append( offrate[bisect_left( offsnr, snr )][1] )
			except IndexError: # on SNR > max off SNR
				onrate.append( 0 )

		onsource_sig = []
		for i, sb in enumerate(onsource[:nevents]):
			# From Gaussian
			#exp_num = chi2.cdf(sb.chisq_dof, sb.snr)*len(onsource)
			# From off-source
			exp_num = onrate[i]*ontime
			# FIXME: requires scipy >= 0.10
			#onsource_sig.append( [sb.snr, -poisson.logsf(i, exp_num)] )
			onsource_sig.append( [rank_fcn(sb), -numpy.log(poisson.sf(i, exp_num))] )

		return onsource_sig

	def mann_whitney_pval( self ):
		offsource = sorted( chain(*self.offsource.values()), key=lambda sb: -sb.snr )
		offsnr = [sb.snr for sb in offsource]

		onsource = sorted( chain(*self.onsource.values()), key=lambda sb: -sb.snr )
		onsnr = [sb.snr for sb in onsource]

		ranks = [(s, "B") for s in offsnr]
		ranks.extend( [(s, "F") for s in onsnr] )
		ranks = sorted( ranks, key=lambda st: s[0] )
		ranks_fg = [ s for s, t in ranks if t == "F" ]
		ranks_bg = [ s for s, t in ranks if t == "B" ]
		if len(ranks) <= 20:
			n = len(ranks)
			nt = len(ranks_fg)
                	u_fg = sum() - nt*(nt+1)/2.0
                	u = min( (n-nt)*nt - u_fg, u_fg )
			m_u = nt*(n-nt)/2.0
			sig_u = numpy.sqrt( m_u/6.0*(n+1) )
			zval = (u-m_u)/sig_u
		else:
			u, pval = scipy.stats.mannwhitneyu( ranks_fg, ranks_bg )
			# FIXME: tail or symmetric?
			zval = abs(scipy.stats.norm( pval ))

		return zval
		

	def normalize( self ):
		"""
		Redistribute events to offsource and onsource based on current time span.
		"""
		all_segs = segmentlist( self.onsource.keys() )
		if len(all_segs) == 0:
			return

		if len(self.offsource.keys()) > 0:
			all_segs += segmentlist( self.offsource.keys() )
		all_segs.coalesce()
		begin, end = all_segs[0][0], all_segs[-1][1] 
		span = float(end-begin)
		if span < self.onsource_interval:
			# Not much we can do.
			return

		if span > self.offsource_interval + self.onsource_interval:
			begin = end - (self.offsource_interval + self.onsource_interval)

		onsource_seg = segment( end-self.onsource_interval, end)
		offsource_seg = segment( begin, end-self.onsource_interval)

		for seg, sbt in self.offsource.items():
			try:
				seg & offsource_seg 
			except ValueError: # offsource segment is out of the current window
				del self.offsource[seg]
				continue
			
			newseg = seg & offsource_seg
			if seg != newseg:
				del self.offsource[seg]
				self.offsource[newseg] = filter( lambda sb: (sb.peak_time + 1e-9*sb.peak_time_ns) in newseg, sbt )

		for seg, sbt in self.onsource.items():
			if seg in onsource_seg:
				continue
			elif offsource_seg.disjoint( seg ) == 1:
				# segment ran off the span since last check
				del self.onsource[seg]
				continue

			offseg = seg & offsource_seg
			del self.onsource[seg]

			try:
				onseg = seg & onsource_seg
				self.onsource[onseg] = filter(lambda sb: (sb.peak_time + 1e-9*sb.peak_time_ns) in onseg, sbt)
			except ValueError: # onsource segment completely out of new segment
				pass

			self.offsource[offseg] = filter(lambda sb: (sb.peak_time + 1e-9*sb.peak_time_ns) in offseg, sbt)

	def add_events( self, sbtable, inseg=None ):
		"""
		Add a trial to the current running tally. If segment is provided, then the key in the trial table is set to be this. Otherwise, the segment is determined from the peak times of the snglbursts
		"""

		# If no events are provided and no segment is indicated, there is no
		# operation to map this into a trial, so we do nothing
		if len(sbtable) == 0 and inseg is None:
			return

		if inseg is None:
			inseg = []
			for sb in sbtable:
				start = sb.start_time + 1e-9*sb.start_time_ns
				stop = sb.start_time + sb.duration
				inseg.append(segment(start, stop))
			inseg = segmentlist( inseg ).coalesce()
			inseg = segment( inseg[0][0], inseg[-1][1] )

		oldsegs = filter(lambda s: s.intersects(inseg), self.onsource.keys())

		# FIXME: Is it possible for this to be > 1?
		# Yes, but the reorganization logic is tricky. 
		# Call normalize often (like everytime you add a new segment).
		if len(oldsegs) == 1:
			oldseg = oldsegs[0]
			sbtable += self.onsource[oldseg] 
			del self.onsource[oldseg]
			inseg = oldseg | inseg

		self.onsource[inseg] = sbtable

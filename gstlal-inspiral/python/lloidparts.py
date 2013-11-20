# Copyright (C) 2009--2013  Kipp Cannon, Chad Hanna, Drew Keppel
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


import math
import sys
import numpy
import warnings
import StringIO
import threading


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


from glue import iterutils
from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw.utils import process as ligolw_process
from gstlal import bottle
from gstlal import datasource
from gstlal import multirate_datasource
from gstlal import pipeparts
from gstlal import simplehandler
from gstlal import simulation
from pylal.datatypes import LIGOTimeGPS


#
# =============================================================================
#
#                              Pipeline Metadata
#
# =============================================================================
#


class DetectorData(object):
	# default block_size = 16384 samples/second * 8 bytes/sample * 512
	# second
	def __init__(self, frame_cache, channel, block_size = 16384 * 8 * 512):
		self.frame_cache = frame_cache
		self.channel = channel
		self.block_size = block_size


#
# =============================================================================
#
#                              Pipeline Elements
#
# =============================================================================
#


def seek_event_for_gps(gps_start_time, gps_end_time, flags = 0):
	"""Create a new seek event for a given gps_start_time and gps_end_time,
	with optional flags.  gps_start_time and gps_end_time may be provided as
	instances of LIGOTimeGPS, as doubles, or as floats."""

	def seek_args_for_gps(gps_time):
		"""Convenience routine to convert a GPS time to a seek type and a
		GStreamer timestamp."""

		if gps_time is None or gps_time == -1:
			return (gst.SEEK_TYPE_NONE, -1) # -1 == gst.CLOCK_TIME_NONE
		elif hasattr(gps_time, 'ns'):
			return (gst.SEEK_TYPE_SET, gps_time.ns())
		else:
			return (gst.SEEK_TYPE_SET, long(float(gps_time) * gst.SECOND))

	start_type, start_time = seek_args_for_gps(gps_start_time)
	stop_type, stop_time   = seek_args_for_gps(gps_end_time)

	return gst.event_new_seek(1., gst.FORMAT_TIME, flags, start_type, start_time, stop_type, stop_time)


#
# sum-of-squares aggregator
#


def mkcontrolsnksrc(pipeline, rate, verbose = False, suffix = None, inj_seg_list = None, seekevent = None, control_peak_samples = None):
	#
	# start with an adder and caps filter to select a sample rate
	#

	snk = gst.element_factory_make("lal_adder")
	snk.set_property("sync", True)
	pipeline.add(snk)
	src = pipeparts.mkcapsfilter(pipeline, snk, "audio/x-raw-float, rate=%d" % rate)

	#
	# Add a peak finder on the control signal sample number
	#

	if control_peak_samples > 0:
		src = pipeparts.mkpeak(pipeline, src, control_peak_samples)

	#
	# optionally add a segment src and gate to only reconstruct around
	# injections
	#

	if inj_seg_list is not None:
		src = datasource.mksegmentsrcgate(pipeline, src, inj_seg_list, seekevent = seekevent, invert_output = False)

	#
	# verbosity and a tee
	#

	logname = suffix and "_%s" % suffix or ""
	if verbose:
		src = pipeparts.mkprogressreport(pipeline, src, "progress_sumsquares%s" % logname)
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps%s_sumsquares" % logname)
	src = pipeparts.mktee(pipeline, src)

	#
	# return the adder and tee
	#

	return snk, src


class Handler(simplehandler.Handler):
	def __init__(self, mainloop, pipeline, gates = {}, tag = "", dataclass = None, verbose = False):
		"""
		here gates is a dict of gate names and messages for example
		gates = {"my_gate_name": "my message"}

		my_gate_name should refer to a gate element's name property that can be retrieved in this pipeline by name
		"""
		super(Handler, self).__init__(mainloop, pipeline)

		self.dataclass = dataclass
		self.lock = threading.Lock()

		self.segments = segments.segmentlistdict()
		self.current_segment_start = {}
		self.gates = gates
		self.tag = tag
		self.verbose = verbose
		for (name,msg) in self.gates.items():
			self.segments[name] = segments.segmentlist([])
			elem = self.pipeline.get_by_name(name)
			elem.set_property("emit-signals", True)
			elem.connect("start", self.gatehandler, "on")
			elem.connect("stop", self.gatehandler, "off")

		bottle.route("/segments.xml")(self.web_get_segments_xml)

	def do_on_message(self, bus, message):
		if message.type == gst.MESSAGE_APPLICATION:
			if message.structure.get_name() == "CHECKPOINT":
				# FIXME make a custom parser for CHECKPOINT messages?
				self.flush_segments_to_disk(message.structure["timestamp"])
				try:
					self.dataclass.snapshot_output_file("%s_LLOID" % self.tag, "xml.gz", verbose = self.verbose)
				except TypeError as te:
					print >>sys.stderr, "Warning: couldn't build output file on checkpoint, probably there aren't any triggers: %s" % te
				return True
		return False

	def flush_segments_to_disk(self, timestamp):
		self.lock.acquire()
		try:
			# close out existing segments
			for name, elem in [(name, self.pipeline.get_by_name(name)) for name in self.gates]:
				if name in self.current_segment_start:
					# By construction these gates should be
					# in the on state.  We fake a state
					# transition to off in order to flush
					# the segments
					self.gatehandler(elem, timestamp, "off")
					# But we have to remember to put it back
					self.gatehandler(elem, timestamp, "on")
			xmldoc = self.gen_segments_doc()
			# FIXME Can't use extent_all() since one list might be empty.
			ext = self.segments.union(self.segments.keys()).extent()
			fname = "%s-%s_SEGMENTS-%d-%d.xml.gz" % ("".join(sorted(self.dataclass.instruments)), self.tag, int(ext[0]), int(abs(ext)))
			utils.write_filename(xmldoc, fname, gz = fname.endswith('.gz'), verbose = self.verbose)

			# Reset the segment lists
			for name in self.segments:
				self.segments[name] = segments.segmentlist([])
		except ValueError:
			print >>sys.stderr, "Warning: couldn't build segment list on checkpoint, probably there aren't any segments"
		finally:
			self.lock.release()

	def gatehandler(self, elem, timestamp, segment_type):
		timestamp = LIGOTimeGPS(0, timestamp)	# timestamp is in nanoseconds
		name = elem.get_name()

		if self.verbose:
			print >>sys.stderr, "%s: %s state transition: %s @ %s" % (name, self.gates[name], segment_type, str(timestamp))

		# If there is a current_segment_start for this then the state transition has to be off
		if name in self.current_segment_start:
			self.segments[name] |= segments.segmentlist([segments.segment(self.current_segment_start.pop(name), timestamp)])
		if segment_type == "on":
			self.current_segment_start[name] = timestamp
		elif segment_type != "off":
			raise AssertionError

	def gen_segments_doc(self):
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		ligolwsegments = ligolw_segments.LigolwSegments(xmldoc)
		process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_inspiral", {})
		ligolwsegments.insert_from_segmentlistdict(self.segments, name = "datasegments", version = None, insertion_time = None, comment = "LLOID snapshot")
		ligolwsegments.finalize(process)
		return xmldoc

	def web_get_segments_xml(self):
		with self.lock:
			output = StringIO.StringIO()
			utils.write_fileobj(self.gen_segments_doc(), output, trap_signals = None)
			outstr = output.getvalue()
			output.close()
		return outstr


#
# one instrument, one template bank
#


def mkLLOIDbranch(pipeline, src, bank, bank_fragment, (control_snk, control_src), gate_attack_length, gate_hold_length, block_duration, nxydump_segment = None, fir_stride = None, control_peak_time = None):
	logname = "%s_%.2f.%.2f" % (bank.logname, bank_fragment.start, bank_fragment.end)

	#
	# FIR filter bank.  low frequency branches use time-domain
	# convolution, high-frequency branches use FFT convolution with a
	# block stride of 4 s.
	#
	# FIXME:  why the -1?  without it the pieces don't match but I
	# don't understand where this offset comes from.  it might really
	# need to be here, or it might be a symptom of a bug elsewhere.
	# figure this out.

	latency = -int(round(bank_fragment.start * bank_fragment.rate)) - 1
	block_stride = fir_stride * bank_fragment.rate
	
	# we figure an fft costs ~5 logN flops where N is duration + block
	# stride.  For each chunk you have to do a forward and a reverse fft.
	# Time domain costs N * block_stride. So if block stride is less than
	# about 10logN you might as well do time domain filtering
	# FIXME This calculation should probably be made more rigorous
	time_domain = 10 * numpy.log2((bank_fragment.end - bank_fragment.start) * bank_fragment.rate + block_stride) > block_stride

	src = pipeparts.mkfirbank(pipeline, src, latency = latency, fir_matrix = bank_fragment.orthogonal_template_bank, block_stride = block_stride, time_domain = time_domain)
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_firbank" % logname)
	# uncomment reblock if you ever use really big ffts and want to cut them down a bit
	#src = pipeparts.mkreblock(pipeline, src, block_duration = control_peak_time * gst.SECOND)
	#src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_firbank_reblock" % logname)
	#src = pipeparts.mktee(pipeline, src)	# comment-out the tee below if this is uncommented
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, src), "orthosnr_%s.dump" % logname, segment = nxydump_segment)

	#pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogramplot(pipeline, src), "video/x-raw-rgb, width=640, height=480, framerate=1/4"))
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, src), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "orthosnr_channelgram_%s.ogv" % logname, verbose = True)

	#
	# compute weighted sum-of-squares, feed to sum-of-squares
	# aggregator
	#

	if control_snk is not None and control_src is not None:
		src = pipeparts.mktee(pipeline, src)	# comment-out if the tee above is uncommented
		elem = pipeparts.mkqueue(pipeline, pipeparts.mksumsquares(pipeline, src, weights = bank_fragment.sum_of_squares_weights), max_size_buffers = 0, max_size_bytes = 0, max_size_time = block_duration)
		elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare" % logname)
		# FIXME:  the capsfilter shouldn't be needed, the adder
		# should intersect it's downstream peer's format with the
		# sink format
		elem = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, elem, quality = 9), "audio/x-raw-float, rate=%d" % max(bank.get_rates()))
		elem = pipeparts.mknofakedisconts(pipeline, elem)	# FIXME:  remove when resampler is patched
		elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare_resampler" % logname)
		elem.link(control_snk)

		#
		# use sum-of-squares aggregate as gate control for orthogonal SNRs
		#
		# FIXME This queue has to be large for the peak finder on the control
		# signal if that element gets smarter maybe this could be made smaller
		# It should be > 1 * control_peak_time * gst.SECOND + 4 * block_duration
		#
		# FIXME since peakfinding is done, or control is based on
		# injections only, we ignore the bank.gate_threshold parameter
		# and just use 1e-100

		src = pipeparts.mkgate(pipeline, pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = (1 * control_peak_time * gst.SECOND + 10 * block_duration)), threshold = 1e-100, attack_length = gate_attack_length, hold_length = gate_hold_length, control = pipeparts.mkqueue(pipeline, control_src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = (1 * control_peak_time * gst.SECOND + 10 * block_duration)))
		src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_gate" % logname)

	#
	# buffer orthogonal SNRs / or actual SNRs if SVD is not used
	#
	# FIXME:  teach the collectpads object not to wait for buffers on pads
	# whose segments have not yet been reached by the input on the other
	# pads.  then this large queue buffer will not be required because
	# streaming can begin through the downstream adders without waiting for
	# input from all upstream elements.

	src = pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 1 * block_duration)

	#
	# reconstruct physical SNRs
	#

	if bank_fragment.mix_matrix is not None:
		src = pipeparts.mkmatrixmixer(pipeline, src, matrix = bank_fragment.mix_matrix)
		src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_matrixmixer" % logname)

	#
	# done
	#
	# FIXME:  find a way to use less memory without this hack

	del bank_fragment.orthogonal_template_bank
	del bank_fragment.sum_of_squares_weights
	del bank_fragment.mix_matrix

	return src


def mkLLOIDhoftToSnrSlices(pipeline, hoftdict, bank, control_snksrc, block_duration, verbose = False, logname = "", nxydump_segment = None, fir_stride = None, control_peak_time = None, snrslices = None):
	"""Build the pipeline fragment that creates the SnrSlices associated with different sample rates from hoft."""
	#
	# parameters
	#

	rates = sorted(bank.get_rates())
	output_rate = max(rates)

	# work out the upsample factors for the attack and hold calculations below
	upsample_factor = dict((rate, rates[i+1] / rate) for i, rate in list(enumerate(rates))[:-1])
	upsample_factor[output_rate] = 0

	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	#
	# loop over template bank slices
	#

	branch_heads = dict((rate, set()) for rate in rates)
	for bank_fragment in bank.bank_fragments:
		# The attack and hold width has three parts
		#
		# 1) The audio resampler filter: 16 comes from the size of
		# the audioresampler filter in samples for the next highest
		# rate at quality 1. Note it must then be converted to the size
		# at the current rate using the upsample_factor dictionary
		# (which is 0 if you are at the max rate).
		#
		# 2) The chisq latency.  You must have at least latency number
		# of points before and after (at the maximum sample rate) to
		# compute the chisq
		#
		# 3) A fudge factor to get the width of the peak.  FIXME this
		# is just set to 1/8th of a second
		peak_half_width = upsample_factor[bank_fragment.rate] * 16 + int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))) + int(math.ceil(bank_fragment.rate / 8.))
		branch_heads[bank_fragment.rate].add(mkLLOIDbranch(
			pipeline,
			# FIXME:  the size isn't ideal:  the correct value
			# depends on how much data is accumulated in the
			# firbank element, and the value here is only
			# approximate and not tied to the fir bank
			# parameters so might not work if those change
			pipeparts.mkqueue(pipeline, pipeparts.mkdrop(pipeline, hoftdict[bank_fragment.rate], int(round((bank.filter_length - bank_fragment.end) * bank_fragment.rate))), max_size_bytes = 0, max_size_buffers = 0, max_size_time = (1 * fir_stride + int(math.ceil(bank.filter_length))) * gst.SECOND),
			bank,
			bank_fragment,
			control_snksrc,
			peak_half_width,
			peak_half_width,
			block_duration,
			nxydump_segment = nxydump_segment,
			fir_stride = fir_stride,
			control_peak_time = control_peak_time
		))

	#
	# if the calling code has requested copies of the snr
	# slices, sum together the highest sample rate streams and tee
	# them off here.  this needs to be done before constructing the
	# adder network below in order to have access to this snr slice by
	# itself.  if we put this off until later it'll have all the other
	# snrs added into it before we get a chance to tee it off
	#

	if snrslices is not None:
		rate, heads = output_rate, branch_heads[output_rate]
		if len(heads) > 1:
			#
			# this sample rate has more than one snr stream.
			# sum them together in an adder, which becomes the
			# head of the stream at this sample rate
			#

			branch_heads[rate] = gst.element_factory_make("lal_adder")
			branch_heads[rate].set_property("sync", True)
			pipeline.add(branch_heads[rate])
			for head in heads:
				pipeparts.mkqueue(pipeline, head, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration).link(branch_heads[rate])
		else:
			#
			# this sample rate has only one stream.  it's the
			# head of the stream at this sample rate
			#

			branch_heads[rate], = heads
		branch_heads[rate] = pipeparts.mktee(pipeline, branch_heads[rate])
		snrslices[rate] = pipeparts.mktogglecomplex(pipeline, branch_heads[rate])

		#
		# the code below expects an interable of elements
		#

		branch_heads[rate] = set([branch_heads[rate]])

	#
	# sum the snr streams
	#

	if True:	# FIXME:  make conditional on time-slice \chi^{2}
		next_rate = dict(zip(rates, rates[1:]))
	else:
		next_rate = dict((rate, output_rate) for rate in rates if rate != output_rate)

	for rate, heads in sorted(branch_heads.items()):
		if len(heads) > 1:
			#
			# this sample rate has more than one snr stream.
			# sum them together in an adder, which becomes the
			# head of the stream at this sample rate
			#

			branch_heads[rate] = gst.element_factory_make("lal_adder")
			branch_heads[rate].set_property("sync", True)
			pipeline.add(branch_heads[rate])
			for head in heads:
				pipeparts.mkqueue(pipeline, head, max_size_bytes = 0, max_size_buffers = 0, max_size_time = 1 * block_duration).link(branch_heads[rate])
			# FIXME capsfilter shouldn't be needed remove when adder is fixed
			branch_heads[rate] = pipeparts.mkcapsfilter(pipeline, branch_heads[rate], "audio/x-raw-float, rate=%d" % rate)
			branch_heads[rate] = pipeparts.mkchecktimestamps(pipeline, branch_heads[rate], "timestamps_%s_after_%d_snr_adder" % (logname, rate))
		else:
			#
			# this sample rate has only one stream.  it's the
			# head of the stream at this sample rate
			#

			branch_heads[rate], = heads

		#
		# resample if needed
		#

		if rate in next_rate:
			# Note quality = 1 requires that the template slices
			# are padded such that the Nyquist frequency is 1.5
			# times the highest frequency of the time slice
			branch_heads[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, branch_heads[rate], quality = 1), "audio/x-raw-float, rate=%d" % next_rate[rate])
			branch_heads[rate] = pipeparts.mknofakedisconts(pipeline, branch_heads[rate])	# FIXME:  remove when resampler is patched
			branch_heads[rate] = pipeparts.mkchecktimestamps(pipeline, branch_heads[rate], "timestamps_%s_after_%d_to_%d_snr_resampler" % (logname, rate, next_rate[rate]))

		#
		# if the calling code has requested copies of the snr
		# slices, tee that off here.  remember that we've already
		# got the highest sample rate slice from above
		#

		if snrslices is not None and rate != output_rate:
			branch_heads[rate] = pipeparts.mktee(pipeline, branch_heads[rate])
			snrslices[rate] = pipeparts.mktogglecomplex(pipeline, branch_heads[rate])

		#
		# chain to next adder if this isn't the final answer
		#

		if rate in next_rate:
			branch_heads[next_rate[rate]].add(branch_heads.pop(rate))

	#
	# done
	#

	snr, = branch_heads.values()	# make sure we've summed down to one stream
	return pipeparts.mktogglecomplex(pipeline, snr)
	#return pipeparts.mkcapsfilter(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkcapsfilter(pipeline, snr, "audio/x-raw-float, rate=%d" % output_rate)), "audio/x-raw-complex, rate=%d" % output_rate)


def mkLLOIDSnrSlicesToTimeSliceChisq(pipeline, branch_heads, bank, block_duration):
	"""Build pipeline fragment that computes the TimeSliceChisq from SnrSlices."""
	#
	# parameters
	#

	rates = sorted(bank.get_rates())

	#
	# compute the chifacs for each rate, store in ascending order in rate
	#

	chifacsdict = dict((rate, []) for rate in rates)
	for bank_fragment in bank.bank_fragments:
		chifacsdict[bank_fragment.rate].append(bank_fragment.chifacs)
	chifacs = []
	for rate, facs in sorted(chifacsdict.items()):
		chifacs.append(facs[0][0::2])
		chifacs[-1] += facs[0][1::2]
		for fac in facs[1:]:
			chifacs[-1] += fac[0::2]
			chifacs[-1] += fac[1::2]
		chifacs[-1] /= 2.

	#
	# create timeslicechisq element and add chifacs as a property
	#

	chisq = gst.element_factory_make("lal_timeslicechisq")
	pipeline.add(chisq)

	#
	# link the snrslices to the timeslicechisq element in ascending order in rate
	#

	for rate, snrslice in sorted(branch_heads.items()):
		pipeparts.mkqueue(pipeline, snrslice,  max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration).link(chisq)

	#
	# set chifacs-matrix property, needs to be done after snrslices are linked in
	#

	chisq.set_property("chifacs-matrix", chifacs)

	return pipeparts.mkqueue(pipeline, chisq, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration)


def mkLLOIDSnrChisqToTriggers(pipeline, snr, chisq, bank, verbose = False, nxydump_segment = None, logname = ""):
	"""Build pipeline fragment that converts single detector SNR and Chisq into triggers."""
	#
	# trigger generator and progress report
	#

	head = pipeparts.mktriggergen(pipeline, snr, chisq, template_bank_filename = bank.template_bank_filename, snr_threshold = bank.snr_threshold, sigmasq = bank.sigmasq)
	# FIXME:  add ability to choose this
	# "lal_blcbctriggergen", {"bank-filename": bank.template_bank_filename, "snr-thresh": bank.snr_threshold, "sigmasq": bank.sigmasq}
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % logname)

	#
	# done
	#

	return head


#
# many instruments, many template banks
#


def mkLLOIDmulti(pipeline, detectors, banks, psd, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, chisq_type = 'autochisq', track_psd = False, fir_stride = 16, control_peak_time = 16, block_duration = gst.SECOND, blind_injections = None):
	#
	# check for unrecognized chisq_types, non-unique bank IDs
	#

	if chisq_type not in ['autochisq', 'timeslicechisq']:
		raise ValueError("chisq_type must be either 'autochisq' or 'timeslicechisq', given %s" % chisq_type)
	if any(tuple(iterutils.nonuniq(bank.bank_id for bank in banklist)) for banklist in banks.values()):
		raise ValueError("bank IDs are not unique: %s" % "; ".join("for %s: %s" % (instrument, iterutils.nonuniq(bank.bank_id for bank in banklist)) for instrument, banklist in banks.items()))

	#
	# extract segments from the injection file for selected reconstruction
	#

	if detectors.injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(detectors.injection_filename)
	else:
		inj_seg_list = None
		#
		# Check to see if we are specifying blind injections now that we know
		# we don't want real injections. Setting this
		# detectors.injection_filename will ensure that injections are added
		# but won't only reconstruct injection segments.
		#
		detectors.injection_filename = blind_injections

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams
	#

	hoftdicts = {}
	for instrument in detectors.channel_dict:
		rates = set(rate for bank in banks[instrument] for rate in bank.get_rates()) # FIXME what happens if the rates are not the same?
		src = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose)
		if veto_segments is not None:
			hoftdicts[instrument] = multirate_datasource.mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument], seekevent = detectors.seekevent, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = 0, width = 32)
		else:
			hoftdicts[instrument] = multirate_datasource.mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = None, seekevent = detectors.seekevent, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = 0, width = 32)

	#
	# build gate control branches
	#

	if control_peak_time > 0 or inj_seg_list is not None:
		control_branch = {}
		for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
			suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
			if instrument != "H2":
				control_branch[(instrument, bank.bank_id)] = mkcontrolsnksrc(pipeline, max(bank.get_rates()), verbose = verbose, suffix = suffix, inj_seg_list= inj_seg_list, seekevent = detectors.seekevent, control_peak_samples = control_peak_time * max(bank.get_rates()))
				#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control_branch[(instrument, bank.bank_id)][1]), "control_%s.dump" % suffix, segment = nxydump_segment)

	else:
		control_branch = None
	#
	# construct trigger generators
	#

	triggersrcs = set()
	for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
		if control_branch is not None:
			if instrument != "H2":
				control_snksrc = control_branch[(instrument, bank.bank_id)]
			else:
				control_snksrc = (None, control_branch[("H1", bank.bank_id)][1])
		else:
			control_snksrc = (None, None)
		if chisq_type == 'timeslicechisq':
			snrslices = {}
		else:
			snrslices = None
		snr = mkLLOIDhoftToSnrSlices(
			pipeline,
			hoftdicts[instrument],
			bank,
			control_snksrc,
			block_duration,
			verbose = verbose,
			logname = suffix,
			nxydump_segment = nxydump_segment,
			control_peak_time = control_peak_time,
			fir_stride = fir_stride,
			snrslices = snrslices
		)
		snr = pipeparts.mkchecktimestamps(pipeline, snr, "timestamps_%s_snr" % suffix)
		if chisq_type == 'autochisq':
			# FIXME don't hardcode
			# peak finding window (n) in samples is 1 second at max rate, ie max(rates)
			head = pipeparts.mkitac(pipeline, snr, 1 * max(rates), bank.template_bank_filename, autocorrelation_matrix = bank.autocorrelation_bank, mask_matrix = bank.autocorrelation_mask, snr_thresh = bank.snr_threshold, sigmasq = bank.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % suffix)
			triggersrcs.add(head)
		else:
			raise NotImplementedError("Currently only 'autochisq' is supported")
		# FIXME:  find a way to use less memory without this hack
		del bank.autocorrelation_bank
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % suffix, segment = nxydump_segment)
		#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(bank.get_rates())], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# done
	#

	assert len(triggersrcs) > 0
	return triggersrcs

#
# SPIIR many instruments, many template banks
#


def mkSPIIRmulti(pipeline, detectors, banks, psd, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, chisq_type = 'autochisq', track_psd = False, block_duration = gst.SECOND, blind_injections = None):
	#
	# check for recognized value of chisq_type
	#

	if chisq_type not in ['autochisq']:
		raise ValueError("chisq_type must be either 'autochisq', given %s" % chisq_type)

	#
	# extract segments from the injection file for selected reconstruction
	#

	if detectors.injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(injection_filename)
	else:
		inj_seg_list = None
		#
		# Check to see if we are specifying blind injections now that we know
		# we don't want real injections. Setting this
		# detectors.injection_filename will ensure that injections are added
		# but won't only reconstruct injection segments.
		#
		detectors.injection_filename = blind_injections

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams
	#

	hoftdicts = {}
	for instrument in detectors.channel_dict:
		rates = set(rate for bank in banks[instrument] for rate in bank.get_rates()) # FIXME what happens if the rates are not the same?
		src = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose)
		if veto_segments is not None:
			hoftdicts[instrument] = multirate_datasource.mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument], seekevent = detectors.seekevent, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = 0, width = 32)
		else:
			hoftdicts[instrument] = multirate_datasource.mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = None, seekevent = detectors.seekevent, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = 0, width = 32)

	#
	# construct trigger generators
	#

	triggersrcs = set()
	for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))

		snr = mkSPIIRhoftToSnrSlices(
			pipeline,
			hoftdicts[instrument],
			bank,
			instrument,
			verbose = verbose,
			nxydump_segment = nxydump_segment,
			quality = 4
		)
		snr = pipeparts.mkchecktimestamps(pipeline, snr, "timestamps_%s_snr" % suffix)

		snr = pipeparts.mktogglecomplex(pipeline, snr)
		snr = pipeparts.mktee(pipeline, snr)
		# FIXME you get a different trigger generator depending on the chisq calculation :/
		if chisq_type == 'autochisq':
			# FIXME don't hardcode
			# peak finding window (n) in samples is one second at max rate, ie max(rates)
			head = pipeparts.mkitac(pipeline, snr, max(rates), bank.template_bank_filename, autocorrelation_matrix = bank.autocorrelation_bank, mask_matrix = bank.autocorrelation_mask, snr_thresh = bank.snr_threshold, sigmasq = bank.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % suffix)
			triggersrcs.add(head)
		# FIXME:  find a way to use less memory without this hack
		del bank.autocorrelation_bank
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % suffix, segment = nxydump_segment)
		#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(bank.get_rates())], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# done
	#

	assert len(triggersrcs) > 0
	return triggersrcs


def mkSPIIRhoftToSnrSlices(pipeline, src, bank, instrument, verbose = None, nxydump_segment = None, quality = 4, sample_rates = None, max_rate = None):
	if sample_rates is None:
		sample_rates = sorted(bank.get_rates())
	else:
		sample_rates = sorted(sample_rates)
	#FIXME don't upsample everything to a common rate
	if max_rate is None:
		max_rate = max(sample_rates)
	prehead = None

	for sr in sample_rates:
		head = pipeparts.mkqueue(pipeline, src[sr], max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
		head = pipeparts.mkreblock(pipeline, head)
		head = pipeparts.mkiirbank(pipeline, head, a1 = bank.A[sr], b0 = bank.B[sr], delay = bank.D[sr], name = "gstlaliirbank_%d_%s_%s" % (sr, instrument, bank.logname))
		head = pipeparts.mkqueue(pipeline, head, max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
		if prehead is not None:
			adder = gst.element_factory_make("lal_adder")
			adder.set_property("sync", True)
			pipeline.add(adder)
			head.link(adder)
			prehead.link(adder)
			head = adder
		# FIXME:  this should get a nofakedisconts after it until the resampler is patched
		head = pipeparts.mkresample(pipeline, head, quality = quality)
		if sr == max_rate:
			head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % max_rate)
		else:
			head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % (2 * sr))
		prehead = head

	return head


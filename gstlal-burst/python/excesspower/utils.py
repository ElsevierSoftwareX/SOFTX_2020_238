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
"""Utilities for gstlal_excesspower"""

import os
import types
import re
from bisect import bisect_left
from itertools import chain, ifilter

import numpy

from scipy.stats import chi2, poisson, mannwhitneyu, norm

from pylal import datatypes as laltypes

from lal.utils import CacheEntry
from ligo import segments

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

class SBStats(object):
	"""
	Keep a "running history" of events seen. Useful for detecting statistically outlying sets of events.
	"""
	def __init__(self):
		self.offsource = {}
		self.onsource = {}
		self.offsource_interval = 6000
		self.onsource_interval = 60

	def event_rate(self, nevents=10):
		"""
		Calculate the Poissonian significance of the 'on source' trial set for up to the loudest nevents.
		"""

		offtime = float(abs(segments.segmentlist(self.offsource.keys())))
		offsource = sorted(chain(*self.offsource.values()), key=lambda sb: -sb.snr )
		offrate = zip(offsource, map(lambda i:i/offtime, range(1, len(offsource)+1)))
		offrate = offrate[::-1]
		offsource = offsource[::-1]
		offsnr = [sb.snr for sb in offsource]

		ontime = float(abs(segments.segmentlist(self.onsource.keys())))
		if ontime == 0:
			return []
		onsource = sorted(chain(*self.onsource.values()), key=lambda sb: -sb.snr)
		onsnr = [sb.snr for sb in onsource]
		onrate = []
		for snr in onsnr:
			try:
				onrate.append(offrate[bisect_left( offsnr, snr )][1])
			except IndexError: # on SNR > max off SNR
				onrate.append(0)

		return onrate

	# FIXME: Have event_sig call event_rate
	def event_significance(self, nevents=10, rank_fcn=None):
		"""
		Calculate the Poissonian significance of the 'on source' trial set for up to the loudest nevents.
		"""
		if rank_fcn is None:
			rank_fcn = lambda e: e.snr

		offtime = float(abs(segments.segmentlist(self.offsource.keys())))
		offsource = sorted(chain(*self.offsource.values()), key=lambda sb: -sb.snr)
		offrate = zip(offsource, map( lambda i:i/offtime, range(1, len(offsource)+1)))
		offrate = offrate[::-1]
		offsource = offsource[::-1]
		offsnr = map(rank_fcn, offsource)

		ontime = float(abs(segments.segmentlist(self.onsource.keys())))
		if ontime == 0:
			return []
		onsource = sorted(chain(*self.onsource.values()), key=lambda sb: -sb.snr)
		onsnr = map(rank_fcn, onsource)
		onrate = []
		for snr in onsnr:
			try:
				onrate.append(offrate[bisect_left(offsnr, snr)][1])
			except IndexError: # on SNR > max off SNR
				onrate.append(0)

		onsource_sig = []
		for i, sb in enumerate(onsource[:nevents]):
			# From Gaussian
			#exp_num = chi2.cdf(sb.chisq_dof, sb.snr)*len(onsource)
			# From off-source
			exp_num = onrate[i]*ontime
			# FIXME: requires scipy >= 0.10
			#onsource_sig.append([sb.snr, -poisson.logsf(i, exp_num)])
			onsource_sig.append([rank_fcn(sb), -numpy.log(poisson.sf(i, exp_num))])

		return onsource_sig

	def mann_whitney_pval(self):
		offsource = sorted(chain(*self.offsource.values()), key=lambda sb: -sb.snr)
		offsnr = [sb.snr for sb in offsource]

		onsource = sorted(chain(*self.onsource.values()), key=lambda sb: -sb.snr)
		onsnr = [sb.snr for sb in onsource]

		ranks = [(s, "B") for s in offsnr]
		ranks.extend([(s, "F") for s in onsnr])
		ranks = sorted(ranks, key=lambda st: s[0])
		ranks_fg = [s for s, t in ranks if t == "F"]
		ranks_bg = [s for s, t in ranks if t == "B"]
		if len(ranks) <= 20:
			n = len(ranks)
			nt = len(ranks_fg)
			u_fg = sum() - nt*(nt+1)/2.0
			u = min((n-nt)*nt - u_fg, u_fg)
			m_u = nt*(n-nt)/2.0
			sig_u = numpy.sqrt(m_u/6.0*(n+1))
			zval = (u-m_u)/sig_u
		else:
			u, pval = scipy.stats.mannwhitneyu(ranks_fg, ranks_bg)
			# FIXME: tail or symmetric?
			zval = abs(scipy.stats.norm(pval))

		return zval

	def normalize(self):
		"""
		Redistribute events to offsource and onsource based on current time span.
		"""
		all_segs = segments.segmentlist(self.onsource.keys())
		if len(all_segs) == 0:
			return

		if len(self.offsource.keys()) > 0:
			all_segs += segments.segmentlist(self.offsource.keys())
		all_segs.coalesce()
		begin, end = all_segs[0][0], all_segs[-1][1] 
		span = float(end-begin)
		if span < self.onsource_interval:
			# Not much we can do.
			return

		if span > self.offsource_interval + self.onsource_interval:
			begin = end - (self.offsource_interval + self.onsource_interval)

		onsource_seg = segments.segment(end-self.onsource_interval, end)
		offsource_seg = segments.segment(begin, end-self.onsource_interval)

		for seg, sbt in self.offsource.items():
			try:
				seg & offsource_seg 
			except ValueError: # offsource segment is out of the current window
				del self.offsource[seg]
				continue
			
			newseg = seg & offsource_seg
			if seg != newseg:
				del self.offsource[seg]
				self.offsource[newseg] = filter(lambda sb: (sb.peak_time + 1e-9*sb.peak_time_ns) in newseg, sbt)

		for seg, sbt in self.onsource.items():
			if seg in onsource_seg:
				continue
			elif offsource_seg.disjoint(seg) == 1:
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

	def add_events(self, sbtable, inseg=None):
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
				inseg.append(segments.segment(start, stop))
			inseg = segments.segmentlist(inseg).coalesce()
			inseg = segments.segment(inseg[0][0], inseg[-1][1])

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

#
# =============================================================================
#
#                                Utility Functions
#
# =============================================================================
#

def subdivide(seglist, length, min_length=0):
	"""
	Subdivide a segent list into smaller segments of length, allowing for a minimum length (default = 0).
	"""
	newlist = []
	for seg in seglist:
		while abs(seg) - min_length > length + min_length:
			newlist.append(segments.segment(seg[0], seg[0]+length))
			seg = segments.segment(seg[0] + length, seg[1])

		if abs(seg) > 0:
			newlist.append(segments.segment(seg[0], seg[1] - min_length))
			newlist.append(segments.segment(seg[1] - min_length, seg[1]))

	return segments.segmentlist(newlist)	

def duration_from_cache(cachen):
	"""
	Determine the spanned duration of a cachefile
	"""
	segs = segments.segmentlistdict()
	for entry in map(CacheEntry, open(cachef)):
		segs |= entry.segmentlistdict
	segs = segs.union(segs)
	return segs[0], abs(segs)

def determine_thresh_from_fap(fap, ndof = 2):
	"""
	Given a false alarm probability desired, and a given number of degrees of freedom (ndof, default = 2), calculate the proper amplitude snr threshold for samples of tiles with that ndof. This is obtained by solving for the statistical value of a CDF for a chi_squared with ndof degrees of freedom at a given probability.
	"""

	return numpy.sqrt(chi2.ppf(1-fap, ndof))

def determine_segment_with_whitening(analysis_segment, whiten_seg):
	"""
	Determine the difference between the segment requested to be analyzed and the segment which was actually analyzed after the whitener time is dropped. This is equivalent to the "in" and "out" segs, respectively, in the search_summary.
	"""
	if whiten_seg.intersects(analysis_segment):
		if analysis_segment in whiten_seg:
			# All the analyzed time is within the settling time
			# We make this explicit because the segment constructor will just reverse the arguments if arg2 < arg1 and create an incorrect segment
			analysis_segment = segments.segment(analysis_segment[1], analysis_segment[1])
		else:
			analysis_segment -= whiten_seg
	return analysis_segment
	
def append_formatted_output_path(fmt, handler, bdir="./", mkdir=False):
	"""
	Append a formatted output path to the base directory (default is pwd). Acceptable options are:
	%i: instrument[0] like 'H' for 'H1'
	%I: instrument
	%S: subsystem prefix
	%c: channel without subsystem
	%C: full channel (eg. LSC-STRAIN)
	%G#: first # GPS digits

	Example: fmt="%I/%S/%c/%G5"
	=> H1/PSL/ISS_PDA_OUT_DQ/10340

	Example: fmt="%I/%C_excesspower/%G5"
	=> H1/PSL-ISS_PDA_OUT_DQ_excesspower/10340

	Options in the format which are unrecoginzed will pass through without being modified. If mkdir is set to True, the directory will be created if it does not already exist.
	"""
	gps_reg = re.compile("^%G(\d*)$")
	def converter(ic):
		gps = re.search(gps_reg, ic)
		if gps is not None and len(gps.group(1)) > 0:
			gps = int(gps.group(1))
		if "%i" in ic: ic = ic.replace("%i", handler.inst[0])
		if "%I" in ic: ic = ic.replace("%I", handler.inst)
		if "%S" in ic: ic = ic.replace("%S", handler.channel.split("-")[0])
		if "%c" in ic: ic = ic.replace("%c", handler.channel.split("-")[-1])
		if "%C" in ic: ic = ic.replace("%C", handler.channel)
		# FIXME: This will replace the entire path segment
		if type(gps) is int: return str(int(handler.time_since_dump))[:gps]
		elif gps is not None: return str(handler.time_since_dump)
		return ic

	subdir = os.path.join(bdir, *[converter(seg) for seg in fmt.strip().split("/")] )
	if mkdir and not os.path.exists(subdir):
		os.makedirs(subdir)
	return os.path.abspath(subdir)

def make_cache_parseable_name(inst, tag, start, stop, ext, dir="./"):
	"""
	Make a LIGO cache parseable name for a segment of time spanning start to stop from instrument inst and file extension ext. If a directory shoudl be prepended to the path, indicate it with dir. The duration is calculated as prescirbed in the technical document describing the LIGO cache format.
	"""
	dur =  numpy.ceil(stop) - numpy.floor(start)
	tag = tag.replace("-","_")

	name = "%s/%s-%s_excesspower-%d-%d.%s" % (dir, inst, tag, start, dur, ext)
	
	return name

def compute_amplitude(sb, psd):
	"""
	Compute the band-limited channel amplitudes from the whitened channel smaples. Note that this is only a rough approximation of the true band-limited amplitude since it ignores the effect of the filters. They are approximated as square windows which pick up the values of the PSD across the entire bandwidth of the tile.
	"""
	flow = int((sb.central_freq - sb.bandwidth/2.0 - psd.f0)/psd.deltaF)
	fhigh = int((sb.central_freq + sb.bandwidth/2.0 - psd.f0)/psd.deltaF)
	snr = sb.snr * sb.chisq_dof
	sb.amplitude = numpy.sqrt(1./2.*snr/((1/psd.data[flow:fhigh]*psd.deltaF).sum()/sb.bandwidth))

# Do this once per module load, since we may end up calling it a lot
__validattrs = [k for k, v in laltypes.SnglBurst.__dict__.iteritems() if isinstance(v, types.MemberDescriptorType) or isinstance(v, types.GetSetDescriptorType)]
def convert_sngl_burst(snglburst, sb_table):
	"""
	Convert the snglburst object (presumed to be a pylal.xlal SnglBurst type) into lsctables version, as provided by the RowType() function of the supplied sb_table.
	"""
	event = sb_table.RowType()  # lsctables version
	for attr in __validattrs:
		# FIXME: This is probably slow
		setattr(event, attr, getattr(snglburst, attr))
	return event

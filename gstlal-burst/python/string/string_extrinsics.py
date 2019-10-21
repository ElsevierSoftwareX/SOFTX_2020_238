# Copyright (C) 2016,2017  Kipp Cannon
# Copyright (C) 2011--2014  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2013  Jacob Peoples
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


import numpy

from ligo.lw import array as ligolw_array
from gstlal import stats as gstlalstats
import lal
from lal import rate


__doc__ = """

The goal of this module is to implement the probability of getting a given set
of extrinsic parameters for a set of detectors parameterized by n-tuples of
trigger parameters: (snr, chi2) assuming that the event is a gravitational wave
signal, *s*, coming from an isotropic distribution in location, orientation and
the volume of space.  The implementation of this in the calling code can be 
found in :py:mod:`string_lr_far`.
"""


#
# =============================================================================
#
#                               SNR, \chi^2 PDF
#
# =============================================================================
#


class NumeratorSNRCHIPDF(rate.BinnedLnPDF):
	"""
	Reports ln P(chi^2/rho^2 | rho, signal)
	"""
	# NOTE:  by happy coincidence numpy's broadcasting rules allow the
	# inherited .at_centres() method to be used as-is
	def __init__(self, *args, **kwargs):
		super(rate.BinnedLnPDF, self).__init__(*args, **kwargs)
		self.volume = self.bins[1].upper() - self.bins[1].lower()
		# FIXME: instead of .shape and repeat() use .broadcast_to()
		# when we can rely on numpy >= 1.8
		#self.volume = numpy.broadcast_to(self.volume, self.bins.shape)
		self.volume.shape = (1, len(self.volume))
		self.volume = numpy.repeat(self.volume, len(self.bins[0]), 0)
		self.norm = numpy.zeros((len(self.bins[0]), 1))

	def __getitem__(self, coords):
		return numpy.log(super(rate.BinnedLnPDF, self).__getitem__(coords)) - self.norm[self.bins(*coords)[0]]

	def marginalize(self, *args, **kwargs):
		raise NotImplementedError

	def __iadd__(self, other):
		# the total count is meaningless, it serves only to set the
		# scale by which the density estimation kernel chooses its
		# size, so we preserve the count across this operation.  if
		# the two arguments have different counts, use the
		# geometric mean unless one of the two is 0 in which case
		# don't screw with the total count
		self_count, other_count = self.array.sum(), other.array.sum()
		super(rate.BinnedLnPDF, self).__iadd__(other)
		if self_count and other_count:
			self.array *= numpy.exp((numpy.log(self_count) + numpy.log(other_count)) / 2.) / self.array.sum()
		self.norm = numpy.log(numpy.exp(self.norm) + numpy.exp(other.norm))
		return self

	def normalize(self):
		# replace the vector with a new one so that we don't
		# interfere with any copies that might have been made
		with numpy.errstate(divide = "ignore"):
			self.norm = numpy.log(self.array.sum(axis = 1))
		self.norm.shape = (len(self.norm), 1)

	@staticmethod
	def add_signal_model(lnpdf, n, prefactors_range, df, inv_snr_pow = 4., snr_min = 3.5, progressbar = None):
		if df <= 0.:
			raise ValueError("require df >= 0: %s" % repr(df))
		pfs = numpy.linspace(prefactors_range[0], prefactors_range[1], 100)
		if progressbar is not None:
			progressbar.max = len(pfs)

		# FIXME:  except for the low-SNR cut, the slicing is done
		# to work around various overflow and loss-of-precision
		# issues in the extreme parts of the domain of definition.
		# it would be nice to identify the causes of these and
		# either fix them or ignore them one-by-one with a comment
		# explaining why it's OK to ignore the ones being ignored.
		# for example, computing snrchi2 by exponentiating the sum
		# of the logs of the terms might permit its evaluation
		# everywhere on the domain.  can ncx2pdf() be made to work
		# everywhere?
		snrindices, rcossindices = lnpdf.bins[snr_min:1e10, 1e-10:1e10]
		snr, dsnr = lnpdf.bins[0].centres()[snrindices], lnpdf.bins[0].upper()[snrindices] - lnpdf.bins[0].lower()[snrindices]
		rcoss, drcoss = lnpdf.bins[1].centres()[rcossindices], lnpdf.bins[1].upper()[rcossindices] - lnpdf.bins[1].lower()[rcossindices]

		snr2 = snr**2.
		snrchi2 = numpy.outer(snr2, rcoss) * df

		arr = numpy.zeros_like(lnpdf.array)
		for pf in pfs:
			if progressbar is not None:
				progressbar.increment()
			arr[snrindices, rcossindices] += gstlalstats.ncx2pdf(snrchi2, df, numpy.array([pf * snr2]).T)

		# convert to counts by multiplying by bin volume, and also
		# multiply by an SNR powr law
		arr[snrindices, rcossindices] *= numpy.outer(dsnr / snr**inv_snr_pow, drcoss)

		# normalize to a total count of n
		arr *= n / arr.sum()

		# add to lnpdf
		lnpdf.array += arr

	def to_xml(self, *args, **kwargs):
		elem = super(rate.BinnedLnPDF, self).to_xml(*args, **kwargs)
		elem.appendChild(ligolw_array.Array.build("norm", self.norm))
		return elem

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = super(rate.BinnedLnPDF, cls).from_xml(xml, name)
		self.norm = ligolw_array.get_array(xml, "norm").array
		return self

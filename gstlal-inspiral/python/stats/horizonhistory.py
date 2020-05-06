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

## @file
# The python module to implement false alarm probability and false alarm rate
#
# ### Review Status
#
# STATUS: reviewed with actions
#
# | Names                                                                                 | Hash                                        | Date       | Diff to Head of Master      |
# | -------------------------------------------                                           | ------------------------------------------- | ---------- | --------------------------- |
# | Hanna, Cannon, Meacher, Creighton J, Robinet, Sathyaprakash, Messick, Dent, Blackburn | 7fb5f008afa337a33a72e182d455fdd74aa7aa7a | 2014-11-05 |<a href="@gstlal_inspiral_cgit_diff/python/far.py?id=HEAD&id2=7fb5f008afa337a33a72e182d455fdd74aa7aa7a">far.py</a> |
# | Hanna, Cannon, Meacher, Creighton J, Sathyaprakash,                                   | 72875f5cb241e8d297cd9b3f9fe309a6cfe3f716 | 2015-11-06 |<a href="@gstlal_inspiral_cgit_diff/python/far.py?id=HEAD&id2=72875f5cb241e8d297cd9b3f9fe309a6cfe3f716">far.py</a> |
#
# #### Action items
#

# - Address the fixed SNR PDF using median PSD which could be pre-computed and stored on disk. (Store database of SNR pdfs for a variety of horizon)
# - The binning parameters are hard-coded too; Could it be a problem?
# - Chisquare binning hasn't been tuned to be a good representation of the PDFs; could be improved in future

## @package far


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import bisect
import copy
try:
	from fpconst import NaN, NegInf, PosInf
except ImportError:
	# fpconst is not part of the standard library and might not be
	# available
	NaN = float("nan")
	NegInf = float("-inf")
	PosInf = float("+inf")
import math
import numpy
import random


from ligo.lw import ligolw
from ligo.lw import array as ligolw_array


__all__ = ["NearestLeafTree", "HorizonHistories"]


#
# =============================================================================
#
#                       Horizon Distance Record Keeping
#
# =============================================================================
#


class NearestLeafTree(object):
	"""
	A simple binary tree in which look-ups return the value of the
	closest leaf.  Only float objects are supported for the keys and
	values.  Look-ups raise KeyError if the tree is empty.

	Example:

	>>> x = NearestLeafTree()
	>>> bool(x)
	False
	>>> x[100.0] = 120.
	>>> bool(x)
	True
	>>> x[104.0] = 100.
	>>> x[102.0] = 110.
	>>> x[90.]
	120.0
	>>> x[100.999]
	120.0
	>>> x[101.001]
	110.0
	>>> x[200.]
	100.0
	>>> del x[104]
	>>> x[200.]
	110.0
	>>> x.keys()
	[100.0, 102.0]
	>>> 102 in x
	True
	>>> 103 in x
	False
	>>> import sys
	>>> x.to_xml(u"H1").write(sys.stdout) # doctest: +NORMALIZE_WHITESPACE
	<Array Type="real_8" Name="H1:nearestleaftree:array">
		<Dim>2</Dim>
		<Dim>2</Dim>
		<Stream Type="Local" Delimiter=" ">
			100 102
			120 110
		</Stream>
	</Array>
	"""
	def __init__(self, items = ()):
		"""
		Initialize a NearestLeafTree.

		Example:

		>>> x = NearestLeafTree()
		>>> x = NearestLeafTree([(100., 120.), (104., 100.), (102., 110.)])
		>>> y = {100.: 120., 104.: 100., 102.: 100.}
		>>> x = NearestLeafTree(y.items())
		"""
		# make a copy to ensure we have a stable object
		self.tree = list(map(tuple, items))
		if any(len(item) != 2 for item in self.tree):
			raise ValueError("items must be sequence of two-element sequences")
		self.tree.sort()

	def __setitem__(self, x, val):
		"""
		Example:

		>>> x = NearestLeafTree()
		>>> x[100.:200.] = 0.
		>>> x[150.] = 1.
		>>> x
		NearestLeafTree([(100, 0), (150, 1), (200, 0)])
		>>> x[210:] = 5.
		>>> x[:90] = 5.
		>>> x
		NearestLeafTree([(90, 5), (100, 0), (150, 1), (200, 0), (210, 5)])
		"""
		if isinstance(x, slice):
			# replace all entries in the requested range of
			# co-ordiantes with two entries, each with the
			# given value, one at the start of the range and
			# one at the end of the range.  thus, after this
			# all queries within that range will return this
			# value.
			if x.step is not None:
				raise ValueError("%s: step not supported" % repr(x))
			if x.start is None:
				if not self.tree:
					raise IndexError("open-ended slice not supported with empty tree")
				x = slice(min(self.minkey(), x.stop), x.stop)
			if x.stop is None:
				if not self.tree:
					raise IndexError("open-ended slice not supported with empty tree")
				x = slice(x.start, max(self.maxkey(), x.start))
			if x.start > x.stop:
				raise ValueError("%s: bounds out of order" % repr(x))
			lo = bisect.bisect_left(self.tree, (x.start, NegInf))
			hi = bisect.bisect_right(self.tree, (x.stop, PosInf))
			self.tree[lo:hi] = ((x.start, val), (x.stop, val)) if x.start != x.stop else ((x.start, val),)
		else:
			# replace all entries having the same co-ordinate
			# with this one
			lo = bisect.bisect_left(self.tree, (x, NegInf))
			hi = bisect.bisect_right(self.tree, (x, PosInf))
			self.tree[lo:hi] = ((x, val),)

	def __getitem__(self, x):
		if not self.tree:
			raise KeyError(x)
		if isinstance(x, slice):
			raise ValueError("slices not supported")
		hi = bisect.bisect_right(self.tree, (x, PosInf))
		try:
			x_hi, val_hi = self.tree[hi]
		except IndexError:
			x_hi, val_hi = self.tree[-1]
		if hi:
			x_lo, val_lo = self.tree[hi - 1]
		else:
			x_lo, val_lo = x_hi, val_hi
		return val_lo if abs(x - x_lo) < abs(x_hi - x) else val_hi

	def __delitem__(self, x):
		"""
		Example:

		>>> x = NearestLeafTree([(100., 0.), (150., 1.), (200., 0.)])
		>>> del x[150.]
		>>> x
		NearestLeafTree([(100, 0), (200, 0)])
		>>> del x[190.]
		Traceback (most recent call last):
			...
		IndexError: 190.0
		>>> del x[:]
		>>> x
		NearestLeafTree([])
		"""
		if isinstance(x, slice):
			if x.step is not None:
				raise ValueError("%s: step not supported" % repr(x))
			if x.start is None:
				if not self.tree:
					# no-op
					return
				x = slice(self.minkey(), x.stop)
			if x.stop is None:
				if not self.tree:
					# no-op
					return
				x = slice(x.start, self.maxkey())
			if x.stop < x.start:
				# no-op
				return
			lo = bisect.bisect_left(self.tree, (x.start, NegInf))
			hi = bisect.bisect_right(self.tree, (x.stop, PosInf))
			del self.tree[lo:hi]
		elif not self.tree:
			raise IndexError(x)
		else:
			lo = bisect.bisect_left(self.tree, (x, NegInf))
			if self.tree[lo][0] != x:
				raise IndexError(x)
			del self.tree[lo]

	def __bool__(self):
		"""
		True if the tree is not empty, False otherwise.
		"""
		return bool(self.tree)

	def __iadd__(self, other):
		"""
		For every (key, value) pair in other, assign self[key]=value.
		"""
		for x, val in other.tree:
			self[x] = val
		return self

	def keys(self):
		return [x for x, val in self.tree]

	def values(self):
		return [val for x, val in self.tree]

	def items(self):
		return list(self.tree)

	def min(self):
		"""
		Return the minimum value stored in the tree.  This is O(n).
		"""
		if not self.tree:
			raise ValueError("empty tree")
		return min(val for x, val in self.tree)

	def minkey(self):
		"""
		Return the minimum key stored in the tree.  This is O(1).
		"""
		if not self.tree:
			raise ValueError("empty tree")
		return self.tree[0][0]

	def max(self):
		"""
		Return the maximum value stored in the tree.  This is O(n).
		"""
		if not self.tree:
			raise ValueError("empty tree")
		return max(val for x, val in self.tree)

	def maxkey(self):
		"""
		Return the maximum key stored in the tree.  This is O(1).
		"""
		if not self.tree:
			raise ValueError("empty tree")
		return self.tree[-1][0]

	def __contains__(self, x):
		"""
		True if a key in self equals x, False otherwise.
		"""
		try:
			return bool(self.tree) and self.tree[bisect.bisect_left(self.tree, (x, NegInf))][0] == x
		except IndexError:
			return False

	def __len__(self):
		"""
		The number of (key, value) pairs in self.
		"""
		return len(self.tree)

	def __repr__(self):
		return "NearestLeafTree([%s])" % ", ".join("(%g, %g)" % item for item in self.tree)

	def functional_integral(self, lohi, w = lambda f: f):
		"""
		Given the function f(x) = self[x], compute

		\int_{lo}^{hi} w(f(x)) dx

		The arguments are the lo and hi bounds and the functional
		w(f).  The default functional is w(f) = f.

		Example:

		>>> x = NearestLeafTree([(100., 0.), (150., 2.), (200., 0.)])
		>>> x.functional_integral((130., 170.))
		80.0
		>>> x.functional_integral((100., 150.))
		50.0
		>>> x.functional_integral((300., 500.))
		0.0
		>>> x.functional_integral((100., 150.), lambda f: f**3)
		200.0
		"""
		lo, hi = lohi

		if not self.tree:
			raise ValueError("empty tree")
		if lo < hi:
			swapped = False
		elif lo == hi:
			return NaN if math.isinf(w(self[lo])) else 0.
		else:
			# lo > hi. remove a factor of -1 from the integral
			lo, hi = hi, lo
			swapped = True
		# now we are certain that lo < hi and that there is at
		# least one entry in the tree

		# construct an array of (x,y) pairs such that f(x) = y and
		# continues to equal y until the next x.  ensure the 0th
		# and last entries in the array are the left and right
		# edges of the integration domain.
		i = bisect.bisect_right(self.tree, (lo, NegInf))
		j = bisect.bisect_right(self.tree, (hi, PosInf))
		if i > 0:
			i -= 1
		samples = self.tree[i:j+1]
		samples = [((a_key + b_key) / 2., b_val) for (a_key, a_val), (b_key, b_val) in zip(samples[:-1], samples[1:])]
		if not samples:
			samples = [(lo, self[lo])]
		elif samples[0][0] > lo:
			samples.insert(0, (lo, self[lo]))
		else:
			samples[0] = lo, self[lo]
		if samples[-1][0] < hi:
			samples.append((hi, self[hi]))
		else:
			samples[-1] = hi, self[hi]
		# return the integral
		result = sum((b_key - a_key) * w(a_val) for (a_key, a_val), (b_key, b_val) in zip(samples[:-1], samples[1:]))
		return -result if swapped else result

	def weighted_mean(self, lohi, weight = lambda y: 1.):
		"""
		Given the function f(x) = self[x], compute

		\int_{lo}^{hi} w(f(x)) f(x) dx
		------------------------------
		   \int_{lo}^{hi} w(f(x)) dx

		where w(y) is a weighting function.  The default weight
		function is w(y) = 1.

		If the numerator is identically 0 and the denominator is
		also 0 then a value of 0 is returned rather than raising a
		divide-by-0 error.

		Example:

		>>> x = NearestLeafTree([(100., 0.), (150., 2.), (200., 0.)])
		>>> x.weighted_mean((130., 170.))
		2.0
		>>> x.weighted_mean((100., 150.))
		1.0
		>>> x.weighted_mean((300., 500.))
		0.0
		>>> x.weighted_mean((100., 150.), lambda x: x**3)
		2.0
		"""
		lo, hi = lohi
		if not self.tree:
			raise ValueError("empty tree")
		if lo > hi:
			# remove a common factor of -1 from the numerator
			# and denominator
			lo, hi = hi, lo
		elif lo == hi:
			return self[lo]
		# now we are certain that lo < hi and that there is at
		# least one entry in the tree

		# return the ratio of the two integrals
		num = self.functional_integral((lo, hi), lambda f: weight(f) * f)
		den = self.functional_integral((lo, hi), weight)
		return num / den if num or den else 0.0

	@classmethod
	def from_xml(cls, xml, name):
		return cls(map(tuple, ligolw_array.get_array(xml, u"%s:nearestleaftree" % name).array[:]))

	def to_xml(self, name):
		return ligolw_array.Array.build(u"%s:nearestleaftree" % name, numpy.array(self.tree, dtype = "double"))


class HorizonHistories(dict):
	def copy(self):
		"""
		Override of dict's .copy() that (a) returns the correct
		type and (b) makes copies of the HorizonHistories'
		contents.
		"""
		return type(self)((key, copy.deepcopy(value)) for key, value in self.items())

	def __iadd__(self, other):
		for key, history in other.items():
			try:
				self[key] += history
			except KeyError:
				self[key] = copy.deepcopy(history)
		return self

	def minkey(self):
		"""
		Return the minimum key stored in the trees.
		"""
		minkeys = tuple(history.minkey() for history in self.values() if history)
		if not minkeys:
			raise ValueError("empty trees")
		return min(minkeys)

	def maxkey(self):
		"""
		Return the maximum key stored in the trees.
		"""
		maxkeys = tuple(history.maxkey() for history in self.values() if history)
		if not maxkeys:
			raise ValueError("empty trees")
		return max(maxkeys)

	def getdict(self, x):
		return dict((key, value[x]) for key, value in self.items())

	def randhorizons(self):
		"""
		Generator yielding a sequence of random horizon distance
		dictionaries chosen by drawing random times uniformly
		distributed between the lowest and highest times recorded
		in the history and returning the dictionary of horizon
		distances for each of those times.
		"""
		x_min = self.minkey()
		x_max = self.maxkey()
		getdict = self.getdict
		rnd = random.uniform
		while 1:
			yield getdict(rnd(x_min, x_max))

	def all(self):
		"""
		Returns a list of the unique sets of horizon distances
		recorded in the histories.
		"""
		# unique times for which a horizon distance measurement is
		# available
		all_x = set(x for value in self.values() for x in value.keys())

		# the unique horizon distances from those times, expressed
		# as frozensets of instrument/distance pairs
		result = set(frozenset(self.getdict(x).items()) for x in all_x)

		# return a list of the results converted back to
		# dictionaries
		return map(dict, result)

	def functional_integral_dict(self, *args, **kwargs):
		"""
		Return a dictionary of the result of invoking
		.functional_integral() on each of the histories.  args and
		kwargs are passed to the .functional_integral() method of
		the histories objects, see their documentation for more
		information.
		"""
		return dict((key, value.functional_integral(*args, **kwargs)) for key, value in self.items())

	def functional_integral(self, lohi, w = lambda f: max(f.values())):
		"""
		Given the function f(x) = self.getdict(x), compute

		\int_{lo}^{hi} w(f(x)) dx

		The arguments are the lo and hi bounds and the functional
		w(f).  The default functional is w(f) = max(f.values()).

		Example:

		>>> x = HorizonHistories({
		...	"H1": NearestLeafTree([(100., 0.), (150., 2.), (200., 0.)]),
		...	"L1": NearestLeafTree([(100., 0.), (150., 20.), (200., 0.)]),
		... })
		>>> x.functional_integral((130., 170.))
		800.0
		>>> x.functional_integral((100., 150.))
		500.0
		>>> x.functional_integral((300., 500.))
		0.0
		>>> x["H1"].functional_integral((100., 150.), w = lambda f: f**3)
		200.0
		"""
		lo, hi = lohi
		if not self or not all(self.values()):
			raise ValueError("empty tree or no trees")
		if lo < hi:
			swapped = False
		elif lo == hi:
			return NaN if math.isinf(w(self.getdict(lo))) else 0.
		else:
			# lo > hi. remove a factor of -1 from the integral
			lo, hi = hi, lo
			swapped = True
		# now we are certain that lo < hi and that there is at
		# least one entry in the tree

		# construct an array of (x,y) pairs such that f(x) = y and
		# continues to equal y until the next x.  ensure the 0th
		# and last entries in the array are the left and right
		# edges of the integration domain.

		samples = sorted(set(x for value in self.values() for x in value.keys()))
		i = bisect.bisect_right(samples, lo)
		j = bisect.bisect_right(samples, hi)
		if i > 0:
			i -= 1
		samples = samples[i:j+1]
		samples = [((a + b) / 2., self.getdict(b)) for a, b in zip(samples[:-1], samples[1:])]
		if not samples:
			samples = [(lo, self.getdict(lo))]
		elif samples[0][0] > lo:
			samples.insert(0, (lo, self.getdict(lo)))
		else:
			samples[0] = lo, self.getdict(lo)
		if samples[-1][0] < hi:
			samples.append((hi, self.getdict(hi)))
		else:
			samples[-1] = hi, self.getdict(hi)

		# return the integral
		result = sum((b_key - a_key) * w(a_val) for (a_key, a_val), (b_key, b_val) in zip(samples[:-1], samples[1:]))
		return -result if swapped else result

	def weighted_mean_dict(self, *args, **kwargs):
		"""
		Return a dictionary of the result of invoking
		.weighted_mean() on each of the histories.  args and kwargs
		are passed to the .weighted_mean() method of the histories
		objects, see their documentation for more information.
		"""
		return dict((key, value.weighted_mean(*args, **kwargs)) for key, value in self.items())

	def compress(self, threshold = 0.03, remove_deviations = False, deviation_percent = 0.50, verbose = False):
		"""
		Remove distances that are non-zero and differ
		fractionally from both neighbours by less than
		the selected threshold.

		Also allows removal of uncharacteristic horizon
		distance values.
		"""
		abs_ln_thresh = math.log1p(threshold)
		for instrument, horizon_history in list(self.items()):
			# GPS time / distance pairs
			items = horizon_history.items()
			if remove_deviations:
				values = numpy.array(items)[:,1]
				mean_horizon = values[values!=0].mean()
				items = [item for item in items if item[1] < (mean_horizon * (1. + deviation_percent))]

			# compress array
			j = 1
			for i in range(1, len(items) - 1):
				values = items[j - 1][1], items[i][1], items[i + 1][1]
				# remove distances that are non-zero and differ
				# fractionally from both neighbours by less than
				# the selected threshold.  always keep the first
				# and last values
				if (values[0] > 0. and values[1] > 0. and values[2] > 0. and
				    abs(math.log(values[1] / values[0])) < abs_ln_thresh and
				    abs(math.log(values[1] / values[2])) < abs_ln_thresh):
					continue
				# remove distances that are 0 and surrounded by 0
				# on both sides (basically the same as the last
				# test, but we can't take log(0)).
				if values == (0., 0., 0.):
					continue
				items[j] = items[i]
				j += 1
			del items[j:]
			if verbose:
				print >>sys.stderr, "\"%s\":  %s horizon history reduced to %.3g%% of original size" % (filename, instrument, 100. * j / (i + 1.))

			# replace
			self[instrument] = type(horizon_history)(items)

	@classmethod
	def from_xml(cls, xml, name):
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == u"%s:horizonhistories" % name]
		try:
			xml, = xml
		except ValueError:
			raise ValueError("document must contain exactly 1 HorizonHistories named '%s'" % name)
		keys = [elem.Name.replace(u":nearestleaftree", u"") for elem in xml.getElementsByTagName(ligolw.Array.tagName) if elem.hasAttribute(u"Name") and elem.Name.endswith(u":nearestleaftree")]
		self = cls()
		for key in keys:
			self[key] = NearestLeafTree.from_xml(xml, key)
		return self

	def to_xml(self, name):
		xml = ligolw.LIGO_LW({u"Name": u"%s:horizonhistories" % name})
		for key, value in self.items():
			xml.appendChild(value.to_xml(key))
		return xml

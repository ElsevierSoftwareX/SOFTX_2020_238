# Copyright (C) 2017,2018  Kipp Cannon
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


from bisect import bisect_right
from ligo import segments
import math
import numpy
import random


from ligo.lw import ligolw
from ligo.lw import array as ligolw_array


__all__ = ["ratebin", "ratebinlist", "triggerrates"]


#
# =============================================================================
#
#               segment and segmentlist with event rate tracking
#
# =============================================================================
#


class ratebin(segments.segment):
	"""
	A version of the segment class (see ligo.segments) that carries a
	count of things.  Arithmetic operations update the count by
	interpreting the count as a uniform density of things throughout
	the interval spanned by the segment.  For example, the intersection
	of two ratebins is the interval they have in common, and the
	intersection's count is given by the sum of the densities of the
	two multiplied by the size of the intersection.

	The implementation does not enforce the use of numeric values for
	boundaries and counts, but is unlikely to behave sensibly unless
	numeric values are used.  One will also encounter problems with
	(semi-)open intervals due to the mean density being 0 for all
	finite counts in such intervals, and undefined otherwise.

	Negative counts are allowed, and in particular the subtraction
	operation can yield a ratebin with a negative count if the density
	in the ratebin being subtracted is sufficiently high.  If this is a
	concern, it is left as an excercise for the calling code to check
	for this condition.

	Some behaviour has been selected to accomodate the specific needs
	of the intended application, and might not be sensible for general
	use outside of that application.  For example, if the count is None
	then the density reported is 0, not None.

	Like the segment class from which it is derived, ratebin objects
	are immutable:  neither the boundaries, nor the count, can be
	modified after creation.

	Example:

	>>> # initialize from two boundary values without count
	>>> x = ratebin(0, 10)
	>>> x
	None@[0,10)
	>>> print(x.count)
	None
	>>> x.density	# count of None --> density = 0, not None
	0.0
	>>> # initialize from two boundary values with count
	>>> x = ratebin(0, 10, count = 5)
	>>> x
	5@[0,10)
	>>> x.density
	0.5
	>>> # intialize from a sequence without count
	>>> y = ratebin((5, 15))
	>>> y
	None@[5,15)
	>>> # intialize from a sequence with count
	>>> y = ratebin((5, 15), count = 2.5)
	>>> y.density
	0.25
	>>> # initialize from a ratebin without count
	>>> ratebin(y)
	2.5@[5,15)
	>>> # initialize from a ratebin with count
	>>> ratebin(y, count = 5)
	5@[5,15)
	>>> # arithmetic examples
	>>> x | y
	7.5@[0,15)
	>>> x + y
	7.5@[0,15)
	>>> (x + y).density
	0.5
	>>> x - y
	3.75@[0,5)
	>>> (x - y).density
	0.75
	>>> x & y
	3.75@[5,10)
	>>> (x & y).density
	0.75
	>>> x & x
	10@[0,10)
	>>> x
	5@[0,10)

	BUGS:  comparisons and hash behaviour are inherited from the
	segment class, specifically this means the count attribute is
	ignored for the purpose of comparisons.  For example, a set will
	believe a ratebin object is already in the set if its boundaries
	match those of a ratebin in the set, regardless of what their
	counts are.  A dictionary will do key look-up using the ratebin
	boundaries, only.

	Example:

	>>> x = ratebin(0, 10, count = 5)
	>>> s = set([x])
	>>> y = ratebin(0, 10, count = 10)
	>>> y in s
	True
	>>> y == x
	True
	>>> x
	5@[0,10)
	>>> y
	10@[0,10)
	>>> # x and y compare as equal, so d gets only one entry
	>>> d = {x: "a", y: "b"}
	>>> d
	{5@[0,10): 'b'}
	"""

	# basic methods

	def __init__(self, arg0 = None, arg1 = None, count = None):
		# parent class' .__init__() is a no-op, so don't bother
		# chaining to it.  need to accomodate 0, 1, or 2 optional
		# arguments before the optional count keyword
		if isinstance(arg0, self.__class__) and count is None:
			self._count = arg0._count
		else:
			self._count = count

	@property
	def count(self):
		return self._count

	@property
	def density(self):
		return (self._count or 0) / float(abs(self))

	def __str__(self):
		return "%s@[%s,%s)" % (("%.16g" % self._count if self._count is not None else "None"), str(self[0]), str(self[1]))

	__repr__ = __str__

	# arithmetic

	def __and__(self, other):
		new = super(ratebin, self).__and__(other)
		if self._count is not None or other._count is not None:
			return type(self)(new, count = (self.density + other.density) * float(abs(new)))
		return new

	def __or__(self, other):
		new = super(ratebin, self).__or__(other)
		if self._count is not None or other._count is not None:
			return type(self)(new, count = (self._count or 0) + (other._count or 0))
		return new

	__add__ = __or__

	def __sub__(self, other):
		new = super(ratebin, self).__sub__(other)
		if self._count is not None:
			return type(self)(new, count = self._count - other.density * (float(abs(self)) - float(abs(new))))
		return new

	#
	# overrides to preserve count
	#

	def protract(self, x):
		return self.__class__(self[0] - x, self[1] + x, count = self._count)

	def contract(self, x):
		return self.__class__(self[0] + x, self[1] - x, count = self._count)

	def shift(self, x):
		return self.__class__(self[0] + x, self[1] + x, count = self._count)


class ratebinlist(segments.segmentlist):
	"""
	Modified version of the segmentlist type (see ligo.segments) whose
	arithmetic operations implement the segment-and-count arithmetic
	operations defined by the ratebin type.

	This implemention has only been optimized for performance for the
	specific use cases for which it has been designed.  It will likely
	be found to have unacceptable performance if used as a
	general-purpose segmentlist implementation.

	Example:

	>>> x0 = ratebinlist([ratebin(0, 10, count = 5)])
	>>> x1 = ratebinlist([ratebin(5, 15, count = 2.5)])
	>>> x2 = ratebinlist([ratebin(15, 25, count = 2.5)])
	>>> x3 = ratebinlist([ratebin(20, 30, count = 5)])
	>>> x0 | x1
	[7.5@[0,15)]
	>>> x0 | x1 | x2 | x3
	[15@[0,30)]
	>>> x0 - x1
	[3.75@[0,5)]
	>>> x0 | x2
	[5@[0,10), 2.5@[15,25)]
	>>> (x0 | x2).density
	0.375
	>>> x0 - x1
	[3.75@[0,5)]
	>>> x0.density
	0.5
	>>> (x0 - x1).density
	0.75
	>>> x0 - x2
	[5@[0,10)]
	>>> y = ratebinlist(x0)
	>>> y.extend(x1)
	>>> y.extend(x2)
	>>> y.extend(x3)
	>>> y
	[5@[0,10), 2.5@[5,15), 2.5@[15,25), 5@[20,30)]
	>>> y.coalesce()
	[15@[0,30)]
	>>> # slices are ratebinlist objects
	>>> (x0 | x3)[1:].density
	0.5
	>>> # density is 0 at times with no segments
	>>> x0.density_at(15)
	0.0
	>>> # and empty lists have a mean density of 0 (not NaN)
	>>> (x0 & x2).density
	0.0

	NOTE:  the XML I/O feature of this class will only work correctly
	for float-valued boundaries and integer counts.
	"""
	def __getitem__(self, index):
		# make sure to return slices as ratebinlist objects, not
		# native list objects
		val = super(ratebinlist, self).__getitem__(index)
		return self.__class__(val) if isinstance(index, slice) else val

	# FIXME:  delete after porting to 3
	def __getslice__(self, *args, **kwargs):
		# make sure to return slices as ratebinlist objects, not
		# native list objects
		return self.__class__(super(ratebinlist, self).__getslice__(*args, **kwargs))

	@property
	def count(self):
		return sum(seg.count for seg in self)

	@property
	def density(self):
		# NOTE:  event density at times when there are no segments
		# is 0., not NaN!
		return self.count / float(abs(self)) if self else 0.

	def segmentlist(self):
		"""
		Construct and return a segments.segmentlist of
		segments.segment objects by type-casting the contents.
		"""
		return segments.segmentlist(segments.segment(seg) for seg in self)

	def add_ratebin(self, seg, count):
		"""
		Convenience method.  Equivalent to

		self |= type(self)([ratebin(seg, count = count)])

		Example:

		>>> x = ratebinlist()
		>>> x.add_ratebin((0, 10), 5)
		>>> x.add_ratebin((10, 20), 5)
		>>> x
		[10@[0,20)]
		"""
		seg = ratebin(seg, count = count)
		# tail optimization cases
		if not self:
			self.append(seg)
		elif not seg.disjoint(self[-1]):
			self[-1] |= seg
		elif seg.disjoint(self[-1]) > 0:
			self.append(seg)
		else:
			# general case.  implementation of .__ior__()
			# allows us to use a tuple here
			self |= (seg,)

	def find(self, item):
		"""
		Return the smallest i such that i is the index of an
		element that wholly contains item.  Raises ValueError if no
		such element exists.

		NOTE:  unlike the segmentlist class from which ratebinlist
		is derived, this implementation requires the ratebinlist to
		be coalesced.

		Example:

		>>> x = ratebinlist([ratebin(0, 10, count = 5), ratebin(15, 25, count = 2.5)])
		>>> x.find(-1)
		Traceback (most recent call last):
			...
		IndexError: -1
		>>> x.find(0)
		0
		>>> x.find(5)
		0
		>>> # upper bounds of segments not included in span
		>>> x.find(10)
		Traceback (most recent call last):
			...
		IndexError: 10
		>>> x.find(12)
		Traceback (most recent call last):
			...
		IndexError: 12
		>>> x.find(15)
		1
		>>> x.find(20)
		1
		>>> x.find(26)
		Traceback (most recent call last):
			...
		IndexError: 26
		"""
		i = bisect_right(self, item)
		if i and item in self[i - 1]:
			return i - 1
		raise IndexError(item)

	def density_at(self, x):
		try:
			i = self.find(x)
		except IndexError:
			# density is 0 at times not covered by segments
			return 0.
		return self[i].density

	def __iand__(self, other):
		if not other or not self:
			del self[:]
			return self
		if other is self:
			self[:] = (ratebin(seg, seg.count * 2) for seg in self)
			return self
		def andgen(self, other):
			self_next = iter(self).next
			other_next = iter(other).next
			self_seg = self_next()
			other_seg = other_next()
			try:
				while 1:
					while self_seg[1] <= other_seg[0]:
						self_seg = self_next()
					while other_seg[1] <= self_seg[0]:
						other_seg = other_next()
					if self_seg.intersects(other_seg):
						yield self_seg & other_seg
						if self_seg[1] > other_seg[1]:
							other_seg = other_next()
						else:
							self_seg = self_next()
			except StopIteration:
				pass
		i = bisect_right(self, other[0][0])
		self[:] = andgen(
			self[(i - 1 if i else 0):bisect_right(self, other[-1][1])],
			other
		)
		return self

	def __ior__(self, other):
		if other is self:
			self[:] = (ratebin(seg, count = 2 * seg.count) for seg in self)
			return self
		i = 0
		for seg in other:
			i = j = bisect_right(self, seg, i)
			lo, hi = seg
			if i and self[i - 1][1] >= lo:
				i -= 1
				lo = self[i][0]
			n = len(self)
			while j < n and self[j][0] <= hi:
				j += 1
			if j > i:
				self[i] = ratebin(lo, max(hi, self[j - 1][1]), count = sum((s.count for s in self[i:j]), seg.count))
				del self[i + 1 : j]
			else:
				self.insert(i, seg)
			i += 1
		return self

	def __isub__(self, other):
		if not other:
			return self
		if other is self:
			del self[:]
			return self
		i = j = 0
		other_lo, other_hi = other[j]
		while i < len(self):
			self_lo, self_hi = self[i]
			while other_hi <= self_lo:
				j += 1
				if j >= len(other):
					return self
				other_lo, other_hi = other[j]
			if self_hi <= other_lo:
				i += 1
			elif other_lo <= self_lo:
				if other_hi >= self_hi:
					del self[i]
				else:
					self[i] -= other[j]
			elif other_hi < self_hi:
				density = (self[i].count - other[j].count) / (float(abs(self[i])) - float(abs(other[j])))
				self[i:i+1] = (ratebin(self_lo, other_lo, count = (other_lo - self_lo) * density), ratebin(other_hi, self_hi, count = (self_hi - other_hi) * density))
				i += 1
			else:
				self[i] -= other[j]
				i += 1
		return self

	def __invert__(self):
		raise NotImplementedError

	def coalesce(self):
		"""
		Sort the elements of the list into ascending order, and merge
		continuous segments into single segments.  Segmentlist is
		modified in place.  This operation is O(n log n).
		"""
		self.sort()
		i = j = 0
		n = len(self)
		while j < n:
			lo, hi = self[j]
			count = self[j].count
			j += 1
			while j < n and hi >= self[j][0]:
				hi = max(hi, self[j][1])
				count += self[j].count
				j += 1
			if lo != hi:
				self[i] = ratebin(lo, hi, count = count)
				i += 1
		del self[i : ]
		return self

	@classmethod
	def from_xml(cls, xml, name):
		# need to cast start and stop to floats, otherwise they are
		# numpy.float64 objects and confuse things
		return cls(ratebin((float(start), float(stop)), count = int(count)) for (start, stop, count) in ligolw_array.get_array(xml, u"%s:ratebinlist" % name).array[:])

	def to_xml(self, name):
		# I/O is only safe for integer counts
		assert all(isinstance(seg.count, int) for seg in self), "counts must be type int for XML I/O"
		return ligolw_array.Array.build(u"%s:ratebinlist" % name, numpy.array([(seg[0], seg[1], seg.count) for seg in self], dtype = "double"))


#
# =============================================================================
#
#                 trigger rate tracking for a detector network
#
# =============================================================================
#


class triggerrates(segments.segmentlistdict):
	"""
	Example:

	>>> x = triggerrates({
	...		"H1": ratebinlist([
	...			ratebin((0, 10), count = 5),
	...			ratebin((20, 30), count = 5)
	...		]),
	...		"V1": ratebinlist([
	...			ratebin((0, 15), count = 7),
	...			ratebin((25, 35), count = 7)
	...		])
	...	})
	...
	>>> x
	{'H1': [5@[0,10), 5@[20,30)], 'V1': [7@[0,15), 7@[25,35)]}
	>>> y = x.copy()
	>>> y
	{'H1': [5@[0,10), 5@[20,30)], 'V1': [7@[0,15), 7@[25,35)]}
	>>> y == x
	True
	>>> y is x
	False
	>>> y = triggerrates.from_xml(x.to_xml("test"), "test")
	>>> y
	{'H1': [5@[0.0,10.0), 5@[20.0,30.0)], 'V1': [7@[0.0,15.0), 7@[25.0,35.0)]}
	>>> y == x
	True
	>>> y is x
	False
	"""
	@property
	def counts(self):
		return dict((key, value.count) for key, value in self.items())

	@property
	def densities(self):
		return dict((key, value.density) for key, value in self.items())

	def segmentlistdict(self):
		return segments.segmentlistdict((key, value.segmentlist()) for key, value in self.items())

	def __iand__(self, other):
		raise NotImplementedError

	def intersection(self, keys):
		raise NotImplementedError

	def union(self, keys):
		raise NotImplementedError

	def density_at(self, x):
		return dict((key, value.density_at(x)) for key, value in self.items())

	def random_uniform(self):
		lo, hi = self.extent_all()
		uniform = random.uniform
		lnP = -math.log(hi - lo)
		while 1:
			x = uniform(lo, hi)
			yield x, self.density_at(x), lnP

	@classmethod
	def from_xml(cls, xml, name):
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == u"%s:triggerrates" % name]
		try:
			xml, = xml
		except ValueError:
			raise ValueError("document must contain exactly 1 triggerrates named '%s'" % name)
		keys = [elem.Name.replace(u":ratebinlist", u"") for elem in xml.getElementsByTagName(ligolw.Array.tagName) if elem.hasAttribute(u"Name") and elem.Name.endswith(u":ratebinlist")]
		self = cls()
		for key in keys:
			self[key] = ratebinlist.from_xml(xml, key)
		return self

	def to_xml(self, name):
		xml = ligolw.LIGO_LW({u"Name": u"%s:triggerrates" % name})
		for key, value in self.items():
			xml.appendChild(value.to_xml(key))
		return xml

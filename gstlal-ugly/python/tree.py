# Copyright (C) 2014 Miguel Fernandez, Chad Hanna
# Copyright (C) 2016,2017 Kipp Cannon, Miguel Fernandez, Chad Hanna, Stephen Privitera, Jonathan Wang
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

import itertools
import metric as metric_module
import numpy
from numpy import random
from scipy.special import gamma

def uber_constraint(vertices, mtotal = 100, ns_spin = 0.05):
	# Assumes coords are m_1, m2, s1, s2
	for vertex in vertices:
		m1,m2,s1,s2 = vertex
		if m2 < m1 and ((m1 + m2) < mtotal) and ((m1 < 2. and abs(s1) < ns_spin) or m1 > 2.):
			return True
	return False

def mass_sym_constraint(vertices, mass_ratio  = float("inf"), total_mass = float("inf")):
	# Assumes m_1 and m_2 are first
	for vertex in vertices:
		m1,m2 = vertex[0:2]
		if m2 <= m1 and float(m1/m2) <= mass_ratio and (m1+m2) < total_mass:
			return True
	return False

def packing_density(n):
	# From: http://mathworld.wolfram.com/HyperspherePacking.html
	# return 1.25
	prefactor=0.50
	if n==1:
		return prefactor
	if n==2:
		return prefactor * numpy.pi / 6 * 3 **.5
	if n==3:
		return prefactor * numpy.pi / 6 * 2 **.5
	if n==4:
		return prefactor * numpy.pi**2 / 16
	if n==5:
		return prefactor * numpy.pi**2 / 30 * 2**.5
	if n==6:
		return prefactor * numpy.pi**3 / 144 * 3**.5
	if n==7:
		return prefactor * numpy.pi**3 / 105
	if n==8:
		return prefactor * numpy.pi**4 / 384
	
class HyperCube(object):

	def __init__(self, boundaries, mismatch, constraint_func = mass_sym_constraint, metric = None, metric_tensor = None, effective_dimension = None, det = None):
		"""
		Define a hypercube with boundaries given by boundaries, e.g.,

		boundaries = numpy.array([[1., 3.], [2., 12.], [4., 7.]])

		Where the numpy array has the min and max coordinates of each dimension.

		In order to compute the size or volume of the cube you have to
		provide a metric function which takes coordinates and returns a metric tensor, e.g.,

		metric.metric_tensor(coordinates)

		where coordinates is a 1xN dimensional array where N is the
		dimension of this HyperCube.
		"""
		self.boundaries = boundaries.copy()
		# The coordinate center, not the center as defined by the metric
		self.center = numpy.array([c[0] + (c[1] - c[0]) / 2. for c in boundaries])
		self.deltas = numpy.array([c[1] - c[0] for c in boundaries])
		self.metric = metric
		if self.metric is not None and metric_tensor is None:
			try:
				self.metric_tensor, self.effective_dimension, self.det = self.metric(self.center, self.deltas / 10000.)
			except RuntimeError:
				print "metric @", self.center, " failed, trying, ", self.center - self.deltas / numpy.pi
				self.metric_tensor, self.effective_dimension, self.det = self.metric(self.center - self.deltas / numpy.pi, self.deltas / 10000.)
		else:
			self.metric_tensor = metric_tensor
			self.effective_dimension = effective_dimension
			self.det = det
		self.size = self._size()
		self.tiles = []
		self.constraint_func = constraint_func
		self.__mismatch = mismatch
		self.neighbors = []
		self.vertices = list(itertools.product(*self.boundaries))

	def __eq__(self, other):
		# FIXME actually make the cube hashable and call that
		return (tuple(self.center), tuple(self.deltas)) == (tuple(other.center), tuple(other.deltas))

	def template_volume(self):
		#n = self.N()
		n = self.effective_dimension
		#print "template_volume ", (numpy.pi * self.__mismatch)**(n/2.) / gamma(n/2. +1)
		return (numpy.pi * self.__mismatch)**(n/2.) / gamma(n/2. +1)

	def N(self):
		return len(self.boundaries)

	def _size(self):
		"""
		Compute the size of the cube according to the metric through
		the center point for each dimension under the assumption of a constant metric
		evaluated at the center.
		"""
		size = numpy.empty(len(self.center))
		for i, sides in enumerate(self.boundaries):
			x = self.center.copy()
			y = self.center.copy()
			x[i] = self.boundaries[i,0]
			y[i] = self.boundaries[i,1]
			size[i] = self.metric.distance(self.metric_tensor, x, y)
		return size

	def num_tmps_per_side(self, mismatch):
		return self.size / self.dl(mismatch = mismatch)


	def split(self, dim, reuse_metric = False):
		leftbound = self.boundaries.copy()
		rightbound = self.boundaries.copy()
		leftbound[dim,1] = self.center[dim]
		rightbound[dim,0] = self.center[dim]
		if reuse_metric:
			return HyperCube(leftbound, self.__mismatch, self.constraint_func, metric = self.metric, metric_tensor = self.metric_tensor, effective_dimension = self.effective_dimension, det = self.det), HyperCube(rightbound, self.__mismatch, self.constraint_func, metric = self.metric, metric_tensor = self.metric_tensor, effective_dimension = self.effective_dimension, det = self.det)
		else:
			return HyperCube(leftbound, self.__mismatch, self.constraint_func, metric = self.metric), HyperCube(rightbound, self.__mismatch, self.constraint_func, metric = self.metric)

	def tile(self, mismatch, stochastic = False):
		self.tiles.append(self.center)
		return list(self.tiles)

	def __contains__(self, coords):
		size = self.size
		tmps = self.num_tmps_per_side(self.__mismatch)
		fraction = (tmps + 1.0) / tmps
		for i, c in enumerate(coords):
			# FIXME do something more sane to handle boundaries
			if not ((c >= self.center[i] - self.deltas[i] * fraction[i] / 2.) and (c <= self.center[i] + self.deltas[i] * fraction[i] / 2.)):
				return False
		return True

	def __repr__(self):
		return "coordinate center: %s\nedge lengths: %s" % (self.center, self.deltas)

	def dl(self, mismatch):
		# From Owen 1995
		return mismatch**.5

	def volume(self, metric_tensor = None):
		if metric_tensor is None:
			metric_tensor = self.metric_tensor
		#return numpy.product(self.deltas) * numpy.linalg.det(metric_tensor)**.5
		#print "volume ", numpy.product(self.deltas) * self.det**.5
		return numpy.product(self.deltas) * self.det**.5

	def coord_volume(self):
		# FIXME this assumes m_1 m_2 are the first coordinates, not necessarily true
		#return numpy.product(self.deltas[0:2])
		return numpy.product(self.deltas)

	def num_templates(self, mismatch):
		# Adapted from Owen 1995 (2.16). The ideal number of
		# templates required to cover the space with
		# non-overlapping spheres.
		return self.volume() / self.template_volume()

	def match(self, other):
		return self.metric.metric_match(self.metric_tensor, self.center, other.center)


class Node(object):
	"""
	A Node implements a node in a binary tree decomposition of the
	parameter space. A node is a container for one hypercube. It can
	have sub-nodes that split the hypercube.
	"""

	template_count = [1]
	bad_aspect_count = [0]

	def __init__(self, cube, parent = None):
		self.cube = cube
		self.right = None
		self.left = None
		self.parent = parent
		self.sibling = None

	def split(self, split_num_templates, mismatch, bifurcation = 0, verbose = True, vtol = 1.03, max_coord_vol = float(100)):
		size = self.cube.num_tmps_per_side(mismatch)
		splitdim = numpy.argmax(size)
		coord_volume = self.cube.coord_volume()
		aspect_ratios = size / min(size)
		aspect_factor = max(1., numpy.product(aspect_ratios[aspect_ratios > 2.0]))
		aspect_ratio = max(aspect_ratios)

		if not self.parent:
			numtmps = float("inf")
			vratio = float("inf")
			sib_aspect_factor = 1.0
			parent_aspect_factor = 1.0
		else:
			par_size = self.parent.cube.num_tmps_per_side(mismatch)
			par_aspect_ratios = par_size / min(par_size)
			par_aspect_factor = max(1., numpy.product(par_aspect_ratios[par_aspect_ratios > 2.0]))
			par_numtmps = self.parent.cube.num_templates(mismatch) * par_aspect_factor / 2.0

			sib_size = self.sibling.cube.num_tmps_per_side(mismatch)
			sib_aspect_ratios = sib_size / min(sib_size)
			sib_aspect_factor = max(1., numpy.product(sib_aspect_ratios[sib_aspect_ratios > 2.0]))
			sib_numtmps = self.sibling.cube.num_templates(mismatch) * sib_aspect_factor

			numtmps = self.cube.num_templates(mismatch) * aspect_factor
			par_vratio = numtmps / par_numtmps
			sib_vratio = numtmps / sib_numtmps
			volume_split_condition = (1./vtol < par_vratio < vtol) and (1./vtol < sib_vratio < vtol)

			# take the bigger of self and sibling
			numtmps = max(numtmps, par_numtmps)
		q = self.cube.center[0] / self.cube.center[1]
		if (coord_volume > max_coord_vol):
			numtmps *= 1
		if  (self.cube.constraint_func(self.cube.vertices + [self.cube.center]) and (numtmps > split_num_templates or ((numtmps > split_num_templates/3.) and not volume_split_condition))):
			self.template_count[0] = self.template_count[0] + 1
			bifurcation += 1
			if numtmps < 2**len(size) and volume_split_condition:
				left, right = self.cube.split(splitdim, reuse_metric = True)
			else:
				left, right = self.cube.split(splitdim)

			self.left = Node(left, self)
			self.right = Node(right, self)
			self.left.sibling = self.right
			self.right.sibling = self.left
			self.left.split(split_num_templates, mismatch = mismatch, bifurcation = bifurcation)
			self.right.split(split_num_templates, mismatch = mismatch, bifurcation = bifurcation)
		else:
			if verbose:
				print "%d tmps : level %03d @ %s : deltas %s : vol frac. %.2f : aspect ratio %.2f : coord vol %.2f" % (self.template_count[0], bifurcation, self.cube.center, self.cube.deltas, numtmps, aspect_ratio, self.cube.coord_volume())

	# FIXME can this be made a generator?
	def leafnodes(self, out = set()):
		"""
		Return a list of all leaf nodes that are ancestors of the given node
		and whose bounding box is not fully contained in the symmetryic region.
		"""
		if self.right:
			self.right.leafnodes(out)
		if self.left:
			self.left.leafnodes(out)

		if not self.right and not self.left and self.cube.constraint_func(self.cube.vertices + [self.cube.center]):
			out.add(self.cube)
		return out

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

def mass_sym_constraint(vertices):
	# Assumes m_1 and m_2 are first
	for vertex in vertices:
		m1,m2 = vertex[0:2]
		if m2 < m1:
			return True
	return False

def packing_density(n):
	# From: http://mathworld.wolfram.com/HyperspherePacking.html
	if n==1:
		return 1.
	if n==2:
		return numpy.pi / 6 * 3 **.5
	if n==3:
		return numpy.pi / 6 * 2 **.5
	if n==4:
		return numpy.pi**2 / 16
	if n==5:
		return numpy.pi**2 / 30 * 2**.5
	if n==6:
		return numpy.pi**3 / 144 * 3**.5
	if n==7:
		return numpy.pi**3 / 105
	if n==8:
		return numpy.pi**4 / 384
	
class HyperCube(object):

	def __init__(self, boundaries, mismatch, constraint_func = mass_sym_constraint, metric = None, metric_tensor = None):
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
			self.metric_tensor = self.metric.set_metric_tensor(self.center, self.deltas)
		else:
			self.metric_tensor = metric_tensor
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
		n = self.N()
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
			return HyperCube(leftbound, self.__mismatch, self.constraint_func, metric = self.metric, metric_tensor = self.metric_tensor), HyperCube(rightbound, self.__mismatch, self.constraint_func, metric = self.metric, metric_tensor = self.metric_tensor)
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
		# From Owen 1995 (2.15)
		return 2 * mismatch**.5 / self.N()**.5

	def volume(self, metric_tensor = None):
		if metric_tensor is None:
			metric_tensor = self.metric_tensor
		# Figure out the pseudo determinant
		# FIXME not sure if this is adequate
		w, v = numpy.linalg.eigh(self.metric_tensor)
		mxw = numpy.max(w)
		# use 4 times machine espilon to be safe
		volume_element = numpy.product(w[w > numpy.finfo(numpy.float32).eps * 4 * mxw])**.5
		#return numpy.product(self.deltas) * numpy.linalg.det(metric_tensor)**.5
		return numpy.product(self.deltas) * volume_element

	def mass_volume(self):
		# FIXME this assumes m_1 m_2 are the first coordinates, not necessarily true
		return numpy.product(self.deltas[0:2])

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

	def __init__(self, cube, parent = None):
		self.cube = cube
		self.right = None
		self.left = None
		self.parent = parent
		self.sibling = None

	def split(self, split_num_templates, mismatch, bifurcation = 0, verbose = True, min_mass_volume = float("inf"), vtol = 1.5):
		size = self.cube.size
		# FIXME this assumes m1/m2 are the first coordinates
		# Always split on the largest size unless the mass volume is less than 4 then prefer mass splitting
		if self.cube.mass_volume() <= min_mass_volume:
			splitdim = numpy.argmax(size)
		else:
			splitdim = numpy.argmax(size[0:2])
		# Figure out how many templates go inside
		numtmps = self.cube.num_templates(mismatch)
		# Artificially overcover the equal mass line
		if 0.9 < self.cube.center[0] / self.cube.center[1] < 1.1:
			numtmps *=2
		# NOTE we in an ad hoc way demand one template per unit mass squared
		if self.parent is not None:
			vratio = self.cube.volume() / self.sibling.cube.volume()
		if self.parent is None or (self.cube.constraint_func(self.cube.vertices) and (numtmps > split_num_templates or self.cube.mass_volume() > min_mass_volume or not (1./vtol < vratio < vtol))):
			bifurcation += 1
			if self.parent and (self.cube.mass_volume() < min_mass_volume) and (1./vtol < vratio < vtol):
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
				print "Reached final depth in level %03d at %s with edge lengths %s" % (bifurcation, self.cube.center, self.cube.deltas)

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

		if not self.right and not self.left and self.cube.constraint_func(self.cube.vertices):
			out.add(self.cube)
		return out

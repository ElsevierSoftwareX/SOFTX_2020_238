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
	Q = []
	M = []
	for vertex in vertices:
		m1,m2 = vertex[0:2]
		Q.append(m1/m2)
		M.append(m1+m2)
	minq_condition = all([q < 1 for q in Q])
	maxq_condition = all([q > mass_ratio for q in Q])
	mtotal_condition = all([m > total_mass for m in M])
	if minq_condition or maxq_condition or mtotal_condition:
		return False
	return True

def mass_sym_constraint_mc(vertices, mass_ratio  = float("inf"), total_mass = float("inf"), min_m1 = 0):
	# Assumes m_c and m_2 are first
	Q = []
	M = []
	M1 = []
	for vertex in vertices:
		mc,m2 = vertex[0:2]
		m1 = metric_module.m1_from_mc_m2(mc, m2)
		Q.append(m1/m2)
		M.append(m1+m2)
		M1.append(m1)
	minq_condition = all([q < 0.95 for q in Q])
	minm1_condition = all([m1 < min_m1 for m1 in M1])
	maxq_condition = all([q > mass_ratio for q in Q])
	mtotal_condition = all([m > total_mass for m in M])
	if minq_condition or minm1_condition or maxq_condition or mtotal_condition:
		return False
	return True

def packing_density(n):
	# this packing density puts two in a cell, we split if there is more
	# than this expected in a cell
	# From: http://mathworld.wolfram.com/HyperspherePacking.html
	prefactor = 1.0
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

def mc_m2_singularity(c):
	center = c.copy()
	F = 1. / 2**.2
	if F*.67 < center[0] / center[1] <= F:
		center[1] = 0.67 * F * center[0]
	if F*1.5 > center[0] / center[1] > F:
		center[1] = 1.5 * F * center[0]
	return center
	
def m1_m2_singularity(c):
	center = c.copy()
	F = 1.
	if F*.80 < center[0] / center[1] <= F:
		center[1] *= 1.25
	if F*1.25 > center[0] / center[1] > F:
		center[1] *= 0.80
	return center
	
class HyperCube(object):

	def __init__(self, boundaries, mismatch, constraint_func = mass_sym_constraint, metric = None, metric_tensor = None, effective_dimension = None, det = None, singularity = None):
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
		# FIXME don't assume m1 m2 and the spin coords are the coordinates we have here.
		deltas = 1e-6 * numpy.ones(len(self.center))
		#deltas = 5e-7 * numpy.ones(len(self.center))
		deltas[0:2] *= self.center[0:2]
		#deltas[2:] = 1.3e-4
		#deltas[2:] = 1.0e-5
		self.singularity = singularity

		if self.metric is not None and metric_tensor is None:
			#try:
			if self.singularity is not None:
				center = self.singularity(self.center)
			else:
				center = self.center
			try:
				self.metric_tensor, self.effective_dimension, self.det = self.metric(center, deltas)
			except ValueError:
				center *= 0.99
				self.metric_tensor, self.effective_dimension, self.det = self.metric(center, deltas)
			#	print "metric @", self.center, " failed, trying, ", self.center - self.deltas / 2.
			#	self.metric_tensor, self.effective_dimension, self.det = self.metric(self.center - self.deltas / 2., deltas)
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

	def template_volume(self, mismatch):
		#n = self.N()
		n = self.effective_dimension
		return (numpy.pi * self.__mismatch)**(n/2.) / gamma(n/2. +1)
		# NOTE code below assumes templates are cubes
		#a = 2 * mismatch**.5 / n**.5
		#return a**n

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
			return HyperCube(leftbound, self.__mismatch, self.constraint_func, metric = self.metric, metric_tensor = self.metric_tensor, effective_dimension = self.effective_dimension, det = self.det, singularity = self.singularity), HyperCube(rightbound, self.__mismatch, self.constraint_func, metric = self.metric, metric_tensor = self.metric_tensor, effective_dimension = self.effective_dimension, det = self.det, singularity = self.singularity)
		else:
			return HyperCube(leftbound, self.__mismatch, self.constraint_func, metric = self.metric, singularity = self.singularity), HyperCube(rightbound, self.__mismatch, self.constraint_func, metric = self.metric, singularity = self.singularity)

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
		return numpy.product(self.deltas)

	def num_templates(self, mismatch):
		# Adapted from Owen 1995 (2.16). The ideal number of
		# templates required to cover the space with
		# non-overlapping spheres.
		return self.volume() / self.template_volume(mismatch)

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

	def split(self, split_num_templates, mismatch, bifurcation = 0, verbose = True, vtol = 1.01, max_coord_vol = float(100)):
		size = self.cube.num_tmps_per_side(mismatch)
		splitdim = numpy.argmax(size)
		aspect_ratios = size / min(size)
		aspect_factor = max(1., numpy.product(aspect_ratios[aspect_ratios>2.0]) / 2.0**len(aspect_ratios[aspect_ratios>2.0]))
		if numpy.isnan(aspect_factor):
			aspect_factor = 1.0
		aspect_ratio = max(aspect_ratios)

		if not self.parent:
			numtmps = float("inf")
			sib_aspect_factor = 1.0
			parent_aspect_factor = 1.0
			volume_split_condition = False
			metric_diff = 1.0
		else:
			# Get the number of parent templates
			par_numtmps = self.parent.cube.num_templates(mismatch) / 2.0

			# get the number of sibling templates
			sib_numtmps = self.sibling.cube.num_templates(mismatch)

			# get our number of templates
			numtmps = self.cube.num_templates(mismatch)

			metric_diff = self.cube.metric_tensor - self.sibling.cube.metric_tensor
			metric_diff = numpy.linalg.norm(metric_diff) / numpy.linalg.norm(self.cube.metric_tensor)**.5 / numpy.linalg.norm(self.sibling.cube.metric_tensor)**.5

			# take the bigger of self, sibling and parent
			numtmps = max(max(numtmps, par_numtmps), sib_numtmps) * aspect_factor
		q = self.cube.center[1] / self.cube.center[0]

		metric_tol = 0.03
		if self.cube.constraint_func(self.cube.vertices + [self.cube.center]) and ((numtmps >= split_num_templates)):
			self.template_count[0] = self.template_count[0] + 1
			bifurcation += 1
			if metric_diff <= metric_tol:# and aspect_factor <= 1.0:
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

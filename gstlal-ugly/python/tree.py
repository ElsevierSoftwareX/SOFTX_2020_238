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
from gstlal import metric as metric_module
import numpy
from numpy import random
from scipy.special import gamma
from gstlal.metric import DELTA

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

def mass_sym_constraint_mc(vertices, mass_ratio  = float("inf"), total_mass = float("inf"), minmax_m1 = (0, float('inf')), minmax_m2 = (0, float('inf')), minmax_mc = (0, float('inf'))):
	# Assumes m_c and m_2 are first
	Q = []
	M = []
	M1 = []
	M2 = []
	MC = []
	for vertex in vertices:
		mc,m2 = vertex[0:2]
		m1 = metric_module.m1_from_mc_m2(mc, m2)
		Q.append(m1/m2)
		M.append(m1+m2)
		M1.append(m1)
		M2.append(m2)
		MC.append(mc)
	minq_condition = all([q < 1.0 for q in Q])
	maxq_condition = all([q > mass_ratio for q in Q])
	mtotal_condition = all([m > total_mass for m in M])

	minm1_condition = all([m1 < minmax_m1[0] for m1 in M1])
	maxm1_condition = all([m1 > minmax_m1[1] for m1 in M1])
	minm2_condition = all([m2 < minmax_m2[0] for m1 in M2])
	maxm2_condition = all([m2 > minmax_m2[1] for m1 in M2])
	minmc_condition = all([mc < minmax_mc[0] for m1 in MC])
	maxmc_condition = all([mc > minmax_mc[1] for m1 in MC])

	if minq_condition or maxq_condition or mtotal_condition or minm1_condition or maxm1_condition or minm2_condition or maxm2_condition or minmc_condition or maxmc_condition:
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

class HyperCube(object):

	def __init__(self, boundaries, mismatch, constraint_func = mass_sym_constraint, metric = None, metric_tensor = None, det = None):
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
			self.metric_tensor, self.det = self.metric(self.center)
		else:
			self.metric_tensor = metric_tensor
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
		n = self.N()
		return (numpy.pi * mismatch)**(n/2.) / gamma(n/2. +1)
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
			return HyperCube(leftbound, self.__mismatch, self.constraint_func, metric = self.metric, metric_tensor = self.metric_tensor, det = self.det), HyperCube(rightbound, self.__mismatch, self.constraint_func, metric = self.metric, metric_tensor = self.metric_tensor, det = self.det)
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
		#return 2. * (mismatch/len(self.center))**.5

	def volume(self, metric_tensor = None):
		if metric_tensor is None:
			metric_tensor = self.metric_tensor
		return numpy.product(self.deltas) * self.det**.5

	def coord_volume(self):
		return numpy.product(self.deltas)

	def num_templates(self, mismatch):
		# Adapted from Owen 1995 (2.16). The ideal number of
		# templates required to cover the space with
		# non-overlapping spheres.
		return self.volume() / self.template_volume(mismatch) / packing_density(self.N())

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

	def __init__(self, cube, parent = None, boundary = None):
		self.cube = cube
		self.right = None
		self.left = None
		self.parent = parent
		self.sibling = None
		self.splitdim = None
		self.boundary = boundary
		if self.boundary is None:
			assert self.parent is None
			self.boundary = cube
		self.on_boundary = False
		for vertex in self.cube.vertices:
			if vertex not in self.boundary:
				self.on_boundary = True
				print("\n\non boundary!!\n\n")

	def split(self, split_num_templates, mismatch, bifurcation = 0, verbose = True, metric_tol = 0.1, max_coord_vol = float(10)):
		if self.cube.constraint_func(self.cube.vertices + [self.cube.center]):
			size = self.cube.num_tmps_per_side(mismatch)
			self.splitdim = numpy.argmax(size)

			if not self.parent:
				numtmps = float("inf")
				par_numtmps = float("inf")
				volume_split_condition = False
				metric_diff = 1.0
				metric_cond = True
				tmp_diff = 1.0
			else:
				# Get the number of parent templates
				par_numtmps = self.parent.cube.num_templates(mismatch)

				# get the number of sibling templates
				sib_numtmps = self.sibling.cube.num_templates(mismatch)

				# get our number of templates
				numtmps = max(max(self.cube.num_templates(mismatch), par_numtmps / 2), sib_numtmps)

				metric_diff = self.cube.metric_tensor - self.sibling.cube.metric_tensor
				metric_diff = numpy.linalg.norm(metric_diff) / numpy.linalg.norm(self.cube.metric_tensor)**.5 / numpy.linalg.norm(self.sibling.cube.metric_tensor)**.5
				metric_diff2 = self.cube.metric_tensor - self.parent.cube.metric_tensor
				metric_diff2 = numpy.linalg.norm(metric_diff2) / numpy.linalg.norm(self.cube.metric_tensor)**.5 / numpy.linalg.norm(self.parent.cube.metric_tensor)**.5
				metric_diff = max(metric_diff, metric_diff2)

				reuse_metric = self.cube #max(mts)[1]

			tmps_per_side = self.cube.num_tmps_per_side(mismatch)
			# NOTE FIXME work out correct max templates per side
			if (numtmps >= split_num_templates) or bifurcation < 2:
				bifurcation += 1
				if metric_diff <= metric_tol:
					self.cube.metric_tensor = reuse_metric.metric_tensor
					self.cube.det = reuse_metric.det
					left, right = self.cube.split(self.splitdim, reuse_metric = True)
				else:
					left, right = self.cube.split(self.splitdim)

				self.left = Node(left, self, boundary = self.boundary)
				self.right = Node(right, self, boundary = self.boundary)
				self.left.sibling = self.right
				self.right.sibling = self.left
				self.left.split(split_num_templates, mismatch = mismatch, bifurcation = bifurcation)
				self.right.split(split_num_templates, mismatch = mismatch, bifurcation = bifurcation)
			else:
				self.template_count[0] = self.template_count[0] + 1
				if verbose and not (self.template_count[0] % 100):
					print "%d tmps : level %03d @ %s : num tmps per side %s: deltas %s : vol frac. %.2f : splits %s : det %.2f : size %s" % (self.template_count[0], bifurcation, self.cube.center, tmps_per_side, self.cube.deltas, numtmps, self.on_boundary, self.cube.det**.5, size)

	# FIXME can this be made a generator?
	#def leafnodes(self, out = set()):
	def leafnodes(self, out = None):
		"""
		Return a list of all leaf nodes that are ancestors of the given node
		and whose bounding box is not fully contained in the symmetryic region.
		"""
		if out is None:
			out = set([])
		if self.right:
			self.right.leafnodes(out)
		if self.left:
			self.left.leafnodes(out)

		if not self.right and not self.left and self.cube.constraint_func(self.cube.vertices + [self.cube.center]):
			out.add(self.cube)
		return out

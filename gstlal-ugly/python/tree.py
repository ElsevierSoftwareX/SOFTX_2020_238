import itertools
import metric as metric_module
import numpy
from numpy import random

def mass_sym(boundaries):
	# Assumes first two are m_1 m_2
	# Makes sure the entire hypercube is outside the symmetric region
	m1 = boundaries[0]
	m2 = boundaries[1]
	for corner in itertools.product(m1,m2):
		if corner[1] < corner[0]:
			return True
	return False

class HyperCube(object):

	def __init__(self, boundaries, mismatch, symmetry_func = mass_sym, metric = None, metric_tensor = None):
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
		self.symmetry_func = symmetry_func
		self.__mismatch = mismatch
		self.neighbors = []
		self.vertices = self.vertices()
		self.ACCEPT = 0
		self.REJECT = 0

	def __eq__(self, other):
		# FIXME actually make the cube hashable and call that
		return (tuple(self.center), tuple(self.deltas)) == (tuple(other.center), tuple(other.deltas))

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
			return HyperCube(leftbound, self.__mismatch, self.symmetry_func, metric = self.metric, metric_tensor = self.metric_tensor), HyperCube(rightbound, self.__mismatch, self.symmetry_func, metric = self.metric, metric_tensor = self.metric_tensor)
		else:
			return HyperCube(leftbound, self.__mismatch, self.symmetry_func, metric = self.metric), HyperCube(rightbound, self.__mismatch, self.symmetry_func, metric = self.metric)

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
			#if not (c >= 1. * self.boundaries[i,0] and c < (fraction[i] * self.deltas[i] + self.boundaries[i,0])):
				return False
		return True

	def __repr__(self):
		return "boundary: %s\ncoordinate center is: %s" % (self.boundaries, self.center)

	def dl(self, mismatch):
		# From Owen 1995 (2.15)
		return 2 * mismatch**.5 / self.N()**.5
		# FIXME the factor of 1.27 is just for a 2D hex lattice
		# return 2 * mismatch**.5 / self.N()**.5 * 1.27

	def volume(self, metric_tensor = None):
		if metric_tensor is None:
			metric_tensor = self.metric_tensor
		# FIXME check math
		return numpy.product(self.deltas) * numpy.linalg.det(metric_tensor)**.5

	def mass_volume(self):
		# FIXME this assumes m_1 m_2 are the first coordinates, not necessarily true
		return numpy.product(self.deltas[0:2])

	def num_templates(self, mismatch):
		# From Owen 1995 (2.16)
		# with an additional packaging fraction to account for the real random packing to be better
		# FIXME look this up it will depend on dimension
		return self.volume() / self.dl(mismatch)**self.N()

	def vertices(self):
		vertices = list(itertools.product(*self.boundaries))
		#print 'VERTICES:', vertices
		return vertices

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

	def split(self, split_num_templates, mismatch, bifurcation = 0, verbose = True):
		size = self.cube.size
		# Always split on the largest size
		splitdim = numpy.argmax(size)
		# Figure out how many templates go inside
		numtmps = self.cube.num_templates(mismatch)
		if self.parent is None or (self.cube.symmetry_func(self.cube.boundaries) and numtmps > split_num_templates): #or (self.cube.symmetry_func(self.cube.boundaries) and self.cube.mass_volume() > 1):
			bifurcation += 1
			#print 'LEVEL:', bifurcation
			if numtmps < 4**len(self.cube.deltas):
				left, right = self.cube.split(splitdim, reuse_metric = True)
			else:
				left, right = self.cube.split(splitdim)

			self.left = Node(left, self)
			self.right = Node(right, self)
			self.left.sibling = self.right
			self.right.sibling = self.left

			if verbose:
				print "Splitting parent with boundaries:"
				for row in self.cube.boundaries:
					print "\t", row
				print "\t\t(est. templates / split threshold: %04d / %04d)" % (numtmps, split_num_templates)
				print "\tLeft center: ", self.left.cube.center
				print "\tRight center:", self.right.cube.center

			self.left.split(split_num_templates, mismatch = mismatch, bifurcation = bifurcation)
			self.right.split(split_num_templates, mismatch = mismatch, bifurcation = bifurcation)
		else:
			if verbose:
				print "Not Splitting"

	def leafnodes(self):
		return list(self.walk())
		#return [x[1] for x in sorted([(random.random(), node) for node in self.walk()])]
		#return [x[1] for x in sorted([(chirptime(node.center[0], node.center[1]), node) for node in self.walk()])]

	def walk(self, out = set()):
		"""
		Return a list of all leaf nodes that are ancestors of the given node
		and whose bounding box is not fully contained in the symmetryic region.
		"""
		if self.right:
			self.right.walk(out)
		if self.left:
			self.left.walk(out)

		if not self.right and not self.left and self.cube.symmetry_func(self.cube.boundaries):
			out.add(self.cube)
		return out

import itertools
import metric as metric_module
import numpy
from numpy import random

# FIXME don't hardcode 30 hz, but what is the right time to use here??
def chirptime(m1, m2, flow = 30):
	mc = (m1 * m2)**.6 / (m1 + m2)**.2 * 5e-6 # Msun in seconds 
	return 5./256. * mc**(-5./3.) * (numpy.pi * flow)**(-8./3.)

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

	def match_size(self):
		"""
		Compute the size of the cube in match according to the metric through
		the center point for each dimension under the assumption of a constant metric
		evaluated at the center.
		"""
		size = numpy.empty(len(self.center))
		for i, sides in enumerate(self.boundaries):
			x = self.center.copy()
			y = self.center.copy()
			x[i] = self.boundaries[i,0]
			y[i] = self.boundaries[i,1]
			size[i] = self.metric.metric_match(self.metric_tensor, x, y)
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
		popcount = 0

		# Find the coordinate transformation matrix
		try:
			M = numpy.linalg.cholesky(self.metric_tensor)
			Minv = numpy.linalg.inv(M)
		except numpy.linalg.LinAlgError:
			self.tiles.append(self.center)
			print >>sys.stderr, "tiling failed: %f" % numpy.linalg.det(self.metric_tensor)
			raise
			return self.tiles, popcount

		if stochastic:
			# To Stephen with love
			# From Chad
			self.tiles.append(self.center)
			return list(self.tiles), popcount
			iters = 1
			neighbor_tiles = sum([n.tiles for n in self.neighbors], [])
			print "neighbor tiles %d neighbors %d" % (len(neighbor_tiles), len(self.neighbors))

			# FIXME don't hardcode this tolerance
			matches = []
			randcoord_list = []
			high_match_tolerance = 1. - len(self.deltas)**.5 * self.__mismatch
			#high_match_tolerance = 1. - self.__mismatch
			low_match_tolerance = 1. - 2 * self.__mismatch

			# FIXME consider tightening convergence
			def random_coordinate(self = self):
				randcoords = numpy.array([random.random()-0.5 for i in range(len(self.deltas))])
				randcoords *= self.deltas 
				randcoords += self.center
				assert randcoords in self
				yield randcoords
				
			def match_func(r, t, cubes = [self]):
				return max(c.metric.metric_match(c.metric_tensor, r, t) for c in cubes)

			def match_coords(this, other_list, neighbors = self.neighbors, tolerance = high_match_tolerance, matchfunc = match_func):
				match = None
				for other in other_list:
					if match is None:
						match = (matchfunc(this, other), other)
					else:
						match = max((matchfunc(this, other), other), match, key = lambda x: x[0])
					if match[0] >= tolerance:
						return match
				for neighbor in neighbors:
					for other in neighbor.tiles:
						if match is None:
							match = (matchfunc(this, other, cubes = [self, neighbor]), other)
						else:
							match = max((matchfunc(this, other, cubes = [self, neighbor]), other), match, key = lambda x: x[0])
						if match[0] >= tolerance:
							return match
				return match

			#while (float(1 + len(self.tiles)) / iters) > 0.01:
			while iters < 100:
				randcoord = random_coordinate().next()
				if len(self.tiles) == 0 and len(neighbor_tiles) == 0:
					self.tiles.append(randcoord)
					matches.append(0.)
					continue
				#self.tiles.append(randcoord)
				#continue
				match = match_coords(randcoord, self.tiles)
				if match[0] <= high_match_tolerance and match[0] > low_match_tolerance:
					match2 = self.metric.explicit_match(randcoord, match[1])
					if match2 <= high_match_tolerance:
						self.ACCEPT += 1
					#else: self.REJECT+=1
					#print self.ACCEPT, self.REJECT, match[0], match2
						self.tiles.append(randcoord)
						matches.append(match)
						iters = 0
					else:
						self.REJECT += 1
				iters += 1

			print "placed %d tiles with %d iterations and matches %s" % (len(self.tiles), iters, matches)

		else:
			# The bounding box has 2*N points to define it each point is
			# an N length vector.  Figure out the x' coordinates of the
			# bounding box in and divide by dl to get number of templates
			bounding_box = numpy.zeros((2*self.N(), self.N()))
			for i, (s,e) in enumerate(self.boundaries):
				Vs = numpy.zeros(self.N())
				Ve = numpy.zeros(self.N())
				Vs[i] = s - self.center[i]
				Ve[i] = e - self.center[i]
				Vsp = numpy.dot(M, Vs) / self.dl(mismatch)
				Vep = numpy.dot(M, Ve) / self.dl(mismatch)
				Vsp[Vsp<0] = numpy.floor(Vsp[Vsp<0])
				Vsp[Vsp>0] = numpy.ceil(Vsp[Vsp>0])
				Vep[Vep<0] = numpy.floor(Vep[Vep<0]) 
				Vep[Vep>0] = numpy.ceil(Vep[Vep>0])
				bounding_box[2*i,:] = Vsp
				bounding_box[2*i+1,:] = Vep

			grid = []
			for cnt, (s,e) in enumerate(zip(numpy.min(bounding_box,0), numpy.max(bounding_box,0))):
				assert s < e
				numtmps = 2**numpy.ceil(numpy.log2((numpy.ceil((e-s)) + 1) // 2))
				# FIXME hexagonal lattice in 2D
				grid.append((numpy.arange(-numtmps, numtmps))* self.dl(mismatch))
				#grid.append((numpy.arange(-numtmps, numtmps) + 0.5 * cnt % 2)* self.dl(mismatch))
				#grid.append(numpy.array((-numtmps, numtmps)))
			for coords in itertools.product(*grid):
				# check this math
				norm_coords = numpy.dot(Minv, coords)
				primed_coords = norm_coords + self.center

				# FIXME take care of ratty edges
				if primed_coords in self:
					self.tiles.append(primed_coords)

			# Gaurantee at least one
			if len(self.tiles) == 0:
				self.tiles.append(self.center)

		return list(self.tiles), popcount

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
		return self.metric.explicit_match(self.center, other.center)
		#return self.metric.metric_match(self.metric_tensor, self.center, other.center)


#	def returnneighbors(self):
#		revisedneighbors = []
#		for neighbor in self.neighbors:
#			if any([vertex in self for vertex in neighbor.vertices]) or any([vertex in neighbor for vertex in self.vertices]):
#				revisedneighbors.append(neighbor)
#		#print 'KEPT %d/%d NEIGHBORS' % (len(revisedneighbors), len(self.neighbors))
#		self.neighbors = set(revisedneighbors)


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
				#print "%30s: %04d : %04d" % ("Next Level of Splitting",numtmps, split_num_templates)
			#self.split_fine(mismatch, bifurcation, verbose)

	def split_fine(self, mismatch, bifurcation = 0, verbose = True):
		size = self.cube.size

		# Always split on the largest size
		splitdim = numpy.argmax(size)
		verr = 2.0

		if self.parent is not None:
			v1 = numpy.sqrt(numpy.linalg.det(self.cube.metric_tensor))
			v2 = numpy.sqrt(numpy.linalg.det(self.parent.cube.metric_tensor))
			avgv = (v2+v1)/2.
			deltav = abs(v2-v1)
			verr = deltav / avgv
			#print v1, v2, verr
		if self.parent is None or (self.cube.symmetry_func(self.cube.boundaries) and verr > 0.5):
			bifurcation += 1
			#print 'LEVEL:', bifurcation
			left, right = self.cube.split(splitdim)
			self.left = Node(left, self)
			self.right = Node(right, self)
			self.left.sibling = self.right
			self.right.sibling = self.left

			if verbose:
				print "%30s: %0.2f" % ("Splitting", verr)
			self.left.split_fine(mismatch = mismatch, bifurcation = bifurcation)
			self.right.split_fine(mismatch = mismatch, bifurcation = bifurcation)
		else:
			if verbose:
				print "%30s: %0.2f" % ("Not Splitting", verr)

			#print 'FINELEFT:', self.left.cube.neighbors
			#print 'FINERIGHT:', self.right.cube.neighbors

	def assign_neighbors(self, disable_neighbors = False):
		# FIXME don't hardcode tolerance
		if not disable_neighbors:
			tolerance = 1.0
		else:
			tolerance = float("inf")

		# FIXME assumes m1 m2 are the first two coordinates
		leaf_nodes = sorted([(chirptime(node.center[0], node.center[1]), node) for node in self.walk()])
		left_nodes = leaf_nodes
		right_nodes = leaf_nodes
		for i, (t, node) in enumerate(leaf_nodes):
			j = i
			# FIXME don't hardcode 1 second
			while j >= 0 and abs(leaf_nodes[j][0] - t) < tolerance:
				# Don't hard code
				#if disable_neighbors or node.match(leaf_nodes[j][1]) >= 0.0:
				node.neighbors.append(leaf_nodes[j][1])
				j -=1
			j = i
			# FIXME don't hardcode 1 second
			while j < len(leaf_nodes) and abs(leaf_nodes[j][0] - t) < tolerance:
				# Don't hard code
				#if disable_neighbors or node.match(leaf_nodes[j][1]) > 0.0:
				node.neighbors.append(leaf_nodes[j][1])
				j +=1

	def leafnodes(self):
		#return [x[1] for x in sorted([(random.random(), node) for node in self.walk()])]
		return [x[1] for x in sorted([(chirptime(node.center[0], node.center[1]), node) for node in self.walk()])]

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

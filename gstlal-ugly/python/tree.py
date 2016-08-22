import itertools
import metric as metric_module
import numpy

from lalinspiral.sbank.bank import Bank
from lalinspiral.sbank.waveforms import waveforms
waveform = waveforms["IMRPhenomD"]

def mass_sym(boundaries):
	# Assumes first two are m_1 m_2
	# Makes sure the entire hypercube is outside the symmetric region
	m1 = boundaries[0]
	m2 = boundaries[1]
	for corner in itertools.product(m1,m2):
		if corner[1] < corner[0]:
			return True
	return False

def find_neighbors_by_m1m2(cube, tiles):
	# Assumes first two are m_1 m_2
	m1 = cube.boundaries[0]
	m2 = cube.boundaries[1]
	neighbors = []
        for t in tiles:
                _t = [t.params[0], t.params[1]]
		if numpy.any(corner in _t for corner in itertools.product(m1,m2)):
			neighbors.append(t)
	return neighbors


class HyperCube(object):

	def __init__(self, boundaries, mismatch, symmetry_func = mass_sym, metric = None):
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
		self.metric_tensor = self.metric.metric_tensor(self.center, self.deltas)
		self.size = self._size()
		self.tiles = []
		self.symmetry_func = symmetry_func
		self.__mismatch = mismatch

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
			size[i] = metric_module.distance(self.metric_tensor, x, y)
		return size

	def num_tmps_per_side(self, mismatch):
		return numpy.ceil(self.size / self.dl(mismatch = mismatch))


	def split(self, dim):
		leftbound = self.boundaries.copy()
		rightbound = self.boundaries.copy()
		leftbound[dim,1] = self.center[dim]
		rightbound[dim,0] = self.center[dim]
		return HyperCube(leftbound, self.__mismatch, self.symmetry_func, metric = self.metric), HyperCube(rightbound, self.__mismatch, self.symmetry_func, metric = self.metric)

	def tile(self, mismatch, stochastic = False, neighbors = [], bank = None):

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
			#self.tiles = [self.center]
			N = self.N()
			target = numpy.ceil(self.num_templates(mismatch))
			dl = self.dl(mismatch)
			cnt = 0
			rand_coords = numpy.random.rand(1e4, len(self.deltas))
			for randcoord in rand_coords:

                                randcoord = (randcoord - 0.5) * self.deltas + self.center
                                wf = waveform(randcoord[0], randcoord[1], 0, 0, bank=bank)
                                match, matcher =  bank.covers(wf, 1 - mismatch, nhood = neighbors + self.tiles)
                                if match < 1 - mismatch:
                                        self.tiles.append(wf)

				if len(self.tiles) > 3*target:
					break

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
				grid.append((numpy.arange(-numtmps, numtmps))* self.dl(mismatch) / 5)
				#grid.append((numpy.arange(-numtmps, numtmps) + 0.5 * cnt % 2)* self.dl(mismatch))
				#grid.append(numpy.array((-numtmps, numtmps)))
			for coords in itertools.product(*grid):
				# check this math
				norm_coords = numpy.dot(Minv, coords)
				primed_coords = norm_coords + self.center

				if primed_coords in self:
                                        if len(primed_coords) == 2:
                                                primed_coords = [primed_coords[0], primed_coords[1], 0, 0]
                                        if -1 <= primed_coords[2] <= 1 and -1 <= primed_coords[3] <= 1:
                                                wf = waveform(primed_coords[0], primed_coords[1], primed_coords[2], primed_coords[3], bank=bank)
                                                match, matcher =  bank.covers(wf, 1 - mismatch, nhood = neighbors + self.tiles)
                                                if match < 1 - mismatch:
                                                        self.tiles.append(wf)

		# Gaurantee at least one
		if len(self.tiles) == 0:
                        self.tiles.append(waveform(self.center[0], self.center[1], 0, 0, bank=bank))

		return self.tiles, popcount

	def __contains__(self, coords):
		size = self.size
		tmps = self.num_tmps_per_side(self.__mismatch)
		fraction = (tmps + 10.0) / tmps
		for i, c in enumerate(coords):
			# FIXME do something more sane to handle boundaries
			if not ((c >= self.center[i] - self.deltas[i] * fraction[i] / 2.) and (c < self.center[i] + self.deltas[i] * fraction[i] / 2.)):
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
		return self.volume() / self.dl(mismatch)**self.N()



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

	def split(self, split_num_templates, mismatch, bifurcation = 0, verbose = True):
		size = self.cube.size

		# Always split on the largest size
		splitdim = numpy.argmax(size)

		# Figure out how many templates go inside
		numtmps = numpy.floor(self.cube.num_templates(mismatch))
		if self.parent is None or (self.cube.symmetry_func(self.cube.boundaries) and numtmps > split_num_templates) or (self.cube.symmetry_func(self.cube.boundaries) and self.cube.mass_volume() > 10):
			bifurcation += 1
			left, right = self.cube.split(splitdim)
			self.left = Node(left, self)
			self.right = Node(right, self)
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
				print "%30s: %04d : %04d" % ("Next Level of Splitting",numtmps, split_num_templates)
                        #self.split_fine(mismatch, bifurcation, verbose)

	def split_fine(self, mismatch, bifurcation = 0, verbose = True):
		size = self.cube.size

		# Always split on the largest size
		splitdim = numpy.argmax(size)
		derr = 2.0

		if self.parent is not None:
			d1 = metric_module.distance(self.cube.metric_tensor, self.cube.center, self.parent.cube.center)
			d2 = metric_module.distance(self.parent.cube.metric_tensor, self.cube.center, self.parent.cube.center)
			avgd = (d2+d1)/2.
			deltad = abs(d2-d1)
			derr = deltad / avgd

		if self.parent is None or (self.cube.symmetry_func(self.cube.boundaries) and derr > 0.05):
			bifurcation += 1
			left, right = self.cube.split(splitdim)
			self.left = Node(left, self)
			self.right = Node(right, self)
			if verbose:
				print "%30s: %0.2f" % ("Splitting", derr)
			self.left.split_fine(mismatch = mismatch, bifurcation = bifurcation)
			self.right.split_fine(mismatch = mismatch, bifurcation = bifurcation)
		else:
			if verbose:
				print "%30s: %0.2f" % ("Not Splitting", derr)

	def walk(self, out = []):
		"""
		Return a list of all leaf nodes that are ancestors of the given node
		and whose bounding box is not fully contained in the symmetryic region.
		"""
		if self.right:
			self.right.walk()
		if self.left:
			self.left.walk()

		if not self.right and not self.left and self.cube.symmetry_func(self.cube.boundaries):
			out.append(self.cube)

		return out

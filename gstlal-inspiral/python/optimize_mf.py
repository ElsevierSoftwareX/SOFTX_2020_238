# Copyright (C) 2015 Yan Wang (yan.wang@ligo.org)
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
# ==============================================================================
#
#         Classes and functions to optimize the templates and the filtering
#
# ==============================================================================
#


import commands
import sys
import os
import unittest
import matplotlib.pyplot as plt
import re
import numpy
import scipy
import warnings

import lalsimulation
#import pdb

import time
from functools import wraps


def timethis(func):
	'''
	Timing decorator
	'''
	def op_func(func):
		
		if not __debug__:
			return func
	
		@wrap(func)
		def wrapper(*args, **kwargs):
			start = time.time()
			result = func(*args, **kwargs)
			end = time.time()
			print func.__name__, end - start
			return result
		return wrapper
	return op_func


class FastMatchedFilter(object):
	pass

class Optimizer(object):


	@staticmethod
	def cnormalize(x):
		""" complex
		self norm=1

		Parameters
		----------
		x : numpy 1-D vector, complex value
		"""

		norm = numpy.sqrt(numpy.dot(x, numpy.conj(x)))
		return x/norm

	@staticmethod
	def cnorm(x):

		return numpy.sqrt(numpy.dot(x, numpy.conj(x)))

	@staticmethod
	def rnormalize(x):
		""" real
		self norm=1

		Parameters
		----------
		x : numpy 1-D vector, real value
		"""

		norm = numpy.sqrt(numpy.dot(x, numpy.conj(x)))
		return x/norm

	@staticmethod
	def rcnormalize(x):
		""" real signal and complex templates
		self norm=sqrt(2)

		Parameters
		----------
		x : numpy 1-D vector, complex value
		"""

		rnorm = numpy.dot(x.real, x.real)
		inorm = numpy.dot(x.imag, x.imag)

		if abs( (rnorm - inorm) / rnorm ) >=1e-3:
			warnings.warn(
				'''Waveform is too short or has strong modulations. 
				Using rcnormalize() may result in inaccuracy.''',
				stacklevel=2)
	
		norm = numpy.sqrt((rnorm + inorm)/2.0)
		return x/norm

	@staticmethod
	def innerProd(a,b):
		""" Inner product of two normalized time series
		"""
		return abs(numpy.dot(a,numpy.conj(b)))


class OptimizerIIR(Optimizer):
	
	def __init__(self, length, a1, b0, delay, *args, **kwargs):
		
		"""
		Parameters
		----------
		length : length of the original template

		a1 : a vector of a1 values for IIR filters
		b0 : a vector of b0 values for IIR filters
		delay : a vector of delays for IIR filters
		"""
		
		self.length = length
		self.a1 = a1
		self.b0 = b0
		self.delay = delay

		self.template = [] # original template
		self._iir_set_res = [] # columns are individual iir responses
		self._iir_sum_res = [] # a column vector
		self._iir_set_index = [] # columns are the beginning and ending indices for individual iir filters
		
		self.n_iir = len(self.a1)

		self.acc_flag = True  # acceleration flag

		assert self.n_iir == len(self.b0)
		assert self.n_iir == len(self.delay)
		
	def setTemplate(self, template):
		self.template = template

	@property
	def iir_set_res(self):
		
		self.getSetIIR()
		return self._iir_set_res

	@property
	def iir_sum_res(self):
		
		self.getSumIIR()
		return self._iir_sum_res

	def getSetIIR(self):
		""" Calculate the responses of a set of IIR filters.
		"""
		
		self._iir_set_res = numpy.zeros((self.length, self.n_iir), dtype=numpy.cdouble)
		self._iir_set_index = numpy.zeros((2, self.n_iir), dtype=numpy.int16)

		for ii in xrange(self.n_iir):
			length0 = numpy.round( numpy.log(1e-13) / numpy.log(numpy.abs(self.a1[ii])) )
			max_length = self.length - self.delay[ii]
			if length0 > max_length:
				length0 = max_length
			t_step = numpy.arange(length0)

			self._iir_set_res[ self.delay[ii] : self.delay[ii]+length0, ii] = self.b0[ii] * self.a1[ii]**t_step
			self._iir_set_res[:, ii] = self._iir_set_res[::-1, ii]
			
			self._iir_set_index[1, ii] = self.length - self.delay[ii]
			self._iir_set_index[0, ii] = self.length - self.delay[ii] - length0


	def getSumIIR(self):
		""" Calculate the sum responses of a set of IIR filters.
		"""
		if self._iir_set_res.shape == (self.length, self.n_iir):
			self._iir_sum_res = numpy.sum(self._iir_set_res, axis=1)
		else:
			self.getSetIIR()
			self._iir_sum_res = numpy.sum(self._iir_set_res, axis=1)

	def normalizeCoef(self):
		""" Normalize coefficients
		"""

		self.getSetIIR()
		self.getSumIIR()
		self.b0 /= self.cnorm(self._iir_sum_res)
		self.getSetIIR()
		self.getSumIIR()

	@property
	def percentSNR(self):
		""" percentage contribution of each IIR filter to SNR
		"""
		
		if self._iir_set_res.shape == (self.length, self.n_iir):
			pSNR = numpy.abs(numpy.dot(numpy.conj(self.template), self._iir_set_res)) 
		else:
			self.getSetIIR()
			pSNR = numpy.abs(numpy.dot(numpy.conj(self.template), self._iir_set_res)) 
		return pSNR

	@timethis
	def runLagrangeOp(self):
		""" Run Lagrangian optimization for the IIR coeficients.
		This is useful when the number of IIR filters is not too large,
		since large n_iir would result in the need of inversion of 
		large matrix, which in turn results in inaccuracy.

		Use the hierarchy opimizer for large number of IIR filters.
		"""
		
		if self.n_iir > 200:
			warnings.warn(
				''' Number of IIR filters larger than 200.
				Using Lagrange optimizer may result in inaccuracy.
				Better to use Hierarchical optimizer instead.''',
				stacklevel=2)
	
		# normalizations
		self.template = self.cnormalize(self.template)
		self._iir_sum_res = self.cnormalize(self._iir_sum_res)
		self._iir_set_res /= self.cnorm(self._iir_sum_res)


		if not self.acc_flag:
			G = numpy.dot(numpy.conj(self._iir_set_res.T), self._iir_set_res)
			H = numpy.dot(numpy.conj(self.template), self._iir_set_res)
		else:
			pass

		HH = numpy.multiply.outer(numpy.conj(H), H)

		GHH = numpy.dot(numpy.linalg.inv(G), HH)
		V, D = numpy.linalg.eig(2*GHH)
		V = abs(V)
		Vmax = max(V)
	
		Dmax = D[:, V==Vmax]
		self.b0 = self.b0 * Dmax.reshape((len(self.b0),))
		
		self.normalizeCoef()
		print "New overlap is ", numpy.sqrt(Vmax/2)

	def runHierarchyLagOp(self, nbatch=50):
		""" Run the Lagrangian optimization using a hierarchical topology.
		
		Parameters
		----------
		nbatch : number of IIR filters in each batch
		"""
		
		# total number of batches
		N_batch = numpy.ceil(self.n_iir/float(nbatch))
		N_batch = int(N_batch)
		
		# normalizations
		self.template = self.cnormalize(self.template)
		self._iir_sum_res = self.cnormalize(self._iir_sum_res)
		self._iir_set_res /= self.cnorm(self._iir_sum_res)

		H = numpy.dot(numpy.conj(self.template), self._iir_set_res)
		
		# sum response of each batch
		batch_sum_res = numpy.zeros((self.length, N_batch), dtype=numpy.cdouble)

		for ii in xrange(N_batch):
			if ii != N_batch - 1:
				batch_index = range(ii*nbatch, (ii+1)*nbatch)
			else:
				batch_index = range(ii*nbatch, self.n_iir)

			batch_set_res = self._iir_set_res[:, batch_index]

			G = numpy.dot(numpy.conj(batch_set_res.T), batch_set_res)
			H = numpy.dot(numpy.conj(self.template), batch_set_res)
			HH = numpy.multiply.outer(numpy.conj(H), H)

			GHH = numpy.dot(numpy.linalg.inv(G), HH)
			V, D = numpy.linalg.eig(2*GHH)
			V = abs(V)
			Vmax = max(V)

			Dmax = D[:, V==Vmax]

			self.b0[batch_index] = self.b0[batch_index] * Dmax.reshape((len(self.b0[batch_index]),))
		
			batch_sum_res[:, ii:ii+1] = numpy.dot(self._iir_set_res[:, batch_index], Dmax)

		if N_batch > 1:	
			G = numpy.dot(numpy.conj(batch_sum_res.T), batch_sum_res)
			H = numpy.dot(numpy.conj(self.template), batch_sum_res)
			HH = numpy.multiply.outer(numpy.conj(H), H)

			GHH = numpy.dot(numpy.linalg.inv(G), HH)
			V, D = numpy.linalg.eig(2*GHH)
		
			V = abs(V)
			Vmax = max(V)

			Dmax = D[:, V==Vmax]
			for ii in xrange(N_batch):
				if ii != N_batch - 1:
					batch_index = range(ii*nbatch, (ii+1)*nbatch)
				else:
					batch_index = range(ii*nbatch, self.n_iir)
				self.b0[batch_index] = self.b0[batch_index] * Dmax[ii]
			
		print 'The new overlap is ', numpy.sqrt(Vmax/2)
		self.normalizeCoef()


	def runIterLagOp(self, nbatch=100, n_iter=2):
		""" Run hierarchical Larangian optimization with iteration.
		Usually, use n_iter <=3.

		Parameters
		----------
		n_iter : number of iteration
		nbatch : number of IIR filters in each batch
		"""
		
		for ii in xrange(n_iter):
			self.runHierarchyLagOp(nbatch)
			self.getSetIIR()

def test():
	
	print opIIR.innerProd(opIIR.template, opIIR._iir_sum_res)
	print opIIR.innerProd(opIIR.template, opIIR.template)
	print opIIR.innerProd(opIIR._iir_sum_res, opIIR._iir_sum_res)
	
	plt.plot(numpy.arange(length), opIIR.template.real, 'b', numpy.arange(length), opIIR.iir_sum_res.real, 'r')
	plt.show()

if __name__=='__main__':
	import pickle
	with open('amp_phase_length_a1_b0_delay') as ff:
		amp, phase, length, a1, b0, delay = pickle.load(ff)

	print len(amp), len(phase), length, len(a1)

	opIIR = OptimizerIIR(length, a1, b0, delay)
	h = numpy.zeros(length, dtype=numpy.cdouble)
	h[-len(amp):] = amp * numpy.exp(1j * phase)

	opIIR.setTemplate(opIIR.cnormalize(h))

	opIIR.normalizeCoef()
	opIIR.runHierarchyLagOp(300)
	#opIIR.runLagrangeOp()

	test()





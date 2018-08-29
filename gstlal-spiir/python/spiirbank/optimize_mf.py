# Copyright (C) 2015-2016 Yan Wang (yan.wang@ligo.org)
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


from __future__ import division
import commands
import sys
import os
import unittest
import re
import numpy
import scipy
from scipy import sparse
import scipy.sparse.linalg
import warnings

import lalsimulation

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
		
		self.n_iir = len(self.a1)
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
                cols = []

		for ii in xrange(self.n_iir):
			length0 = numpy.round( numpy.log(1e-13) / numpy.log(numpy.abs(self.a1[ii])) )
			max_length = self.length - self.delay[ii]
			if length0 > max_length:
				length0 = max_length
			t_step = numpy.arange(length0)[::-1]

                        cols.append(sparse.csc_matrix((self.b0[ii] * self.a1[ii]**t_step,(numpy.arange(self.length-self.delay[ii]-length0,self.length-self.delay[ii]),numpy.zeros(length0))),shape=(self.length,1)))
                self._iir_set_res = sparse.hstack(cols)

	def getSumIIR(self):
		""" Calculate the sum responses of a set of IIR filters.
		"""
		if not self._iir_set_res.shape == (self.length, self.n_iir):
			self.getSetIIR()

                self._iir_sum_res = self._iir_set_res.sum(axis=1).A1
	def normalizeCoef(self):
		""" Normalize coefficients
		"""

		self.getSetIIR()
		self.getSumIIR()
		self.b0 /= self.cnorm(self._iir_sum_res)
		self.getSetIIR()
		self.getSumIIR()

	def runLagrangeOp(self):
		""" Run Lagrangian optimization for the IIR coeficients.
		"""
		# normalizations
		self.template = self.cnormalize(self.template)
		self._iir_sum_res = self.cnormalize(self._iir_sum_res)
		self._iir_set_res /= self.cnorm(self._iir_sum_res)
                T = self._iir_set_res.conj().T
                A = T.dot(self._iir_set_res)
                b = T.dot(self.template)
                Dmax = scipy.sparse.linalg.spsolve(A.tocsc(),b)
		self.b0 = self.b0 * Dmax
		
		self.normalizeCoef()

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
                        bsr=sparse.csr_matrix(batch_set_res)
                        Dmine,garbage = numpy.linalg.lstsq((bsr.conj().T*bsr).toarray(),bsr.conj().T.dot(self.template))
                        Dmax=Dmax.reshape((len(self.b0[batch_index]),))

			self.b0[batch_index] = self.b0[batch_index] * Dmax
		
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
	opIIR.runLagrangeOp()

	test()





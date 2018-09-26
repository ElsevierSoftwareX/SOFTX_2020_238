# Copyright (C) 2017-2018 Joel Bosveld
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

import numpy as np
cimport cython

cdef extern from "complex.h":
    double cabs(complex x)
    complex conj(complex x)
    double creal(complex x)
    double cimag(complex x)
cdef extern from "math.h":
    double log(double x)
    double sqrt(double x)


############
## y

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# y,dy,ddy should be initialized with np.empty(len(a1), dtype=complex)
# dy[i] = \frac{\mathrm{d} y_i}{\mathrm{d} a_i^*}
# ddy[i] = \frac{\mathrm{d}^2 y_i}{(\mathrm{d} a_i^*)^2}
def calc_y_dy_ddy(complex [:] a1, int [:] delay, complex [:] template, complex [:] y, complex [:] dy, complex [:] ddy, complex [:] a1_conj):
  cdef int a1_len = len(a1)
  cdef int template_len = len(template)
  cdef int j
  cdef int i
  cdef int decay_len
  cdef int end_pos
  for j in range(0, a1_len):
    end_pos = template_len - delay[j]
    decay_len = int(log(1e-13)/log(cabs(a1_conj[j])))
    y[j] = 0
    dy[j] = 0
    ddy[j] = 0
    for i in range(max(0,end_pos-decay_len), end_pos):
      ddy[j] = ddy[j] * a1_conj[j] + 2*dy[j]
      dy[j] = dy[j] * a1_conj[j] + y[j]
      y[j] = y[j] * a1_conj[j] + template[i]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def recalc_y_dy_ddy(complex [:] a1, int [:] delay, complex [:] template, complex [:] y, complex[:] dy, complex[:] ddy, int j, complex [:] a1_conj):
  cdef int template_len = len(template)
  cdef int end_pos = template_len - delay[j]
  cdef int decay_len = int(log(1e-13)/log(cabs(a1_conj[j])))
  cdef int i
  y[j] = 0
  dy[j] = 0
  ddy[j] = 0
  for i in range(max(0,end_pos-decay_len), end_pos):
    ddy[j] = ddy[j] * a1_conj[j] + 2*dy[j]
    dy[j] = dy[j] * a1_conj[j] + y[j]
    y[j] = y[j] * a1_conj[j] + template[i]



############
## Y

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# Y should be initialized with np.empty((len(a1),len(a1)), dtype=complex)
def calc_Y(complex [:] a1, int [:] delay, complex [:,:] Y, complex [:] a1_conj):
  cdef int a1_len = len(a1)
  cdef int i
  cdef int j
  for j in range(0, a1_len):
    Y[j,j] = 1 / (1 - creal(a1[j])**2 - cimag(a1[j])**2)
    for i in range(j+1, a1_len):
      Y[j,i] = a1_conj[j]**(delay[i] - delay[j]) / (1 - a1_conj[j]*a1[i])
      Y[i,j] = conj(Y[j,i])

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def recalc_Y(complex [:] a1, int [:] delay, complex [:,:] Y, int j, complex [:] a1_conj):
  cdef int a1_len = len(a1)
  cdef int i
  for i in range(0, j):
    Y[j,i] = a1[i]**(delay[j]-delay[i]) / (1 - a1_conj[j]*a1[i])
    Y[i,j] = conj(Y[j,i])
  Y[j,j] = 1 / (1 - creal(a1[j])**2 - cimag(a1[j])**2)
  for i in range(j+1, a1_len):
    Y[j,i] = a1_conj[j]**(delay[i] - delay[j]) / (1 - a1_conj[j]*a1[i])
    Y[i,j] = conj(Y[j,i])

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# dY should be initialized with np.empty((len(a1),len(a1)), dtype=complex)
# dY[j,i] = \frac{\partial Y_{ji}}{\partial a_i}
def calc_dY(complex [:] a1, int [:] delay, complex [:,:] dY, complex [:] a1_conj, complex [:,:] Y):
  cdef int a1_len = len(a1)
  cdef int i
  cdef int j
  for j in range(0, a1_len):
    for i in range(0, j):
      dY[j,i] = a1_conj[j]*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**2 + (delay[j] - delay[i])*a1[i]**(delay[j] - delay[i] - 1) / (1 - a1_conj[j]*a1[i])
    dY[j,j] = a1_conj[j] * Y[j,j]**2
    for i in range(j+1, a1_len):
      dY[j,i] = a1_conj[j]**(delay[i] - delay[j] + 1) / (1 - a1_conj[j]*a1[i])**2

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def recalc_dY(complex [:] a1, int [:] delay, complex [:,:] dY, int j, complex [:] a1_conj, complex [:,:] Y):
  cdef int a1_len = len(a1)
  cdef int i
  dY[j,j] = a1_conj[j] * Y[j,j]**2
  for i in range(0, j):
    dY[j,i] = a1_conj[j]*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**2 + (delay[j] - delay[i])*a1[i]**(delay[j] - delay[i] - 1) / (1 - a1_conj[j]*a1[i])
  for i in range(j+1, a1_len):
    dY[j,i] = a1_conj[j]**(delay[i] - delay[j] + 1) / (1 - a1_conj[j]*a1[i])**2
  i = j
  for j in range(0, i):
    dY[j,i] = a1_conj[j]**(delay[i] - delay[j] + 1) / (1 - a1_conj[j]*a1[i])**2
  for j in range(i+1, a1_len):
    dY[j,i] = a1_conj[j]*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**2 + (delay[j] - delay[i])*a1[i]**(delay[j] - delay[i] - 1) / (1 - a1_conj[j]*a1[i])

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# ddY should be initialized with np.empty((len(a1),len(a1)), dtype=complex)
# ddY[j,i] = \frac{\partial Y_{ji}}{\partial a_j^* \partial a_i}
def calc_ddY(complex [:] a1, int [:] delay, complex [:,:] ddY, complex [:] a1_conj, complex [:,:] Y):
  cdef int a1_len = len(a1)
  cdef int i
  cdef int j
  for j in range(0, a1_len):
    for i in range(0, j):
      ddY[j,i] = 2*a1_conj[j]*a1[i]**(delay[j] - delay[i] + 1) / (1 - a1_conj[j]*a1[i])**3 + (delay[j] - delay[i] + 1)*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**2
    ddY[j,j] = (1 + a1[j].real**2 + a1[j].imag**2) / (1 - a1[j].real**2 - a1[j].imag**2)**3
    for i in range(j+1, a1_len):
      ddY[j,i] = (delay[i] - delay[j] + 1)*a1_conj[j]**(delay[i] - delay[j]) / (1 - a1_conj[j]*a1[i])**2 + 2*a1_conj[j]**(delay[i] - delay[j] + 1)*a1[i] / (1 - a1_conj[j]*a1[i])**3

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def recalc_ddY(complex [:] a1, int [:] delay, complex [:,:] ddY, int j, complex [:] a1_conj, complex [:,:] Y):
  cdef int a1_len = len(a1)
  cdef int i
  ddY[j,j] = Y[j,j]**2 + 2*a1_conj[j]*a1[j] * Y[j,j]**3
  for i in range(0, j):
    ddY[j,i] = 2*a1_conj[j]*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**3 + (delay[j] - delay[i] + 1)*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**2
  for i in range(j+1, a1_len):
    ddY[j,i] = (delay[i] - delay[j] + 1)*a1_conj[j]**(delay[i] - delay[j]) / (1 - a1_conj[j]*a1[i])**2 + 2*a1_conj[j]**(delay[i] - delay[j] + 1)*a1[i] / (1 - a1_conj[j]*a1[i])**3
  i=j
  for j in range(0, i):
    ddY[j,i] = (delay[i] - delay[j] + 1)*a1_conj[j]**(delay[i] - delay[j]) / (1 - a1_conj[j]*a1[i])**2 + 2*a1_conj[j]**(delay[i] - delay[j] + 1)*a1[i] / (1 - a1_conj[j]*a1[i])**3
  for j in range(i+1, a1_len):
    ddY[j,i] = 2*a1_conj[j]*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**3 + (delay[j] - delay[i] + 1)*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**2

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# ddYd should be initialized with np.empty((len(a1),len(a1)), dtype=complex)
# ddYd[j,i] = \frac{\partial Y_{ji}}{\partial a_i \partial a_i}
def calc_ddYd(complex [:] a1, int [:] delay, complex [:,:] ddYd, complex [:] a1_conj, complex [:,:] Y):
  cdef int a1_len = len(a1)
  cdef int i
  cdef int j
  for j in range(0, a1_len):
    for i in range(0, j):
      ddYd[j,i] = 2*a1_conj[j]**2*a1[i]**(delay[j] - delay[i]) / (1 - a1_conj[j]*a1[i])**3 + 2*(delay[j] - delay[i])*a1_conj[j]*a1[i]**(delay[j] - delay[i]-1) / (1 - a1_conj[j]*a1[i])**2 + (delay[j] - delay[i] - 1)*(delay[j] - delay[i])*a1[i]**(delay[j] - delay[i] - 2) / (1 - a1_conj[j]*a1[i])
    ddYd[j,j] = 2*a1_conj[j]**2 / (1 - a1[j].real**2 - a1[j].imag**2)**3
    for i in range(j+1, a1_len):
      ddYd[j,i] = 2*a1_conj[j]**(delay[i] - delay[j] + 2) / (1 - a1_conj[j]*a1[i])**3



############
## Cholesky decomposition updates

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def chol_update(complex [:,:] L, complex [:] w):
    cdef int w_len = len(w)
    cdef double r
    cdef double c
    cdef complex s
    cdef complex sc
    cdef int i
    for i in range(w_len):
        r = sqrt(creal(L[i,i])**2 + (creal(w[i])**2 + cimag(w[i])**2))
        c = r / (L[i,i].real)
        s = w[i] / L[i,i]
        sc = conj(s)
        L[i,i] = r
        for j in range(i+1,w_len):
            L[j,i] = (L[j,i] + sc*w[j]) / c
            w[j] = c*w[j] - s*L[j,i]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def chol_downdate(complex [:,:] L, complex [:] w):
    cdef int w_len = len(w)
    cdef double r
    cdef double c
    cdef complex s
    cdef complex sc
    cdef int i
    for i in range(w_len):
        r = sqrt(creal(L[i,i])**2 -(creal(w[i])**2 + cimag(w[i])**2))
        c = r / (L[i,i].real)
        s = w[i] / L[i,i]
        sc = conj(s)
        L[i,i] = r
        for j in range(i+1,w_len):
            L[j,i] = (L[j,i] - sc*w[j]) / c
            w[j] = c*w[j] - s*L[j,i]



############
## Code for testing

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# y must be initialized with np.zeros(len(a1), dtype=complex)
# output should be initialized with np.empty(len(data)-max(delay), dtype=complex)
# data should have max(delay) zeros padded on left
def filter_data(complex [:] a1, complex[:] b0, int [:] delay, complex [:] data, complex [:] output, complex [:] y):
  cdef int a1_len = len(a1)
  cdef int data_len = len(data)
  cdef int max_delay = max(delay)
  cdef int j
  cdef int i
  cdef int decay_len
  cdef int end_pos
  for i in range(max_delay,data_len):
    output[i-max_delay] = 0
    for j in range(0, a1_len):
      y[j] = y[j] * a1[j] + b0[j] * data[i-delay[j]]
      output[i-max_delay]+=y[j]

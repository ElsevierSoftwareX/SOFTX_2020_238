# Copyright (C) 2017-2018 Joel Bosveld, David Weight
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

from __future__ import print_function
import sys
def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

import scipy.linalg as la
import scipy.signal as sg
import numpy as np
from optimizer_cy import calc_y_dy_ddy, calc_Y, calc_dY, calc_ddY, calc_ddYd
from optimizer_cy import recalc_y_dy_ddy, recalc_Y, recalc_dY, recalc_ddY
from optimizer_cy import chol_update, chol_downdate

# initializes memory for new template with n filters
def initialize_state(template, n):
  template = template

  a1_conj = np.empty(n, dtype=np.complex128)
  y = np.empty(n, dtype=np.complex128)
  dy = np.empty(n, dtype=np.complex128)
  ddy = np.empty(n, dtype=np.complex128)
  Y = np.empty((n,n), dtype=np.complex128)
  dY = np.empty((n,n), dtype=np.complex128)
  ddY = np.empty((n,n), dtype=np.complex128)
  ddYd = np.empty((n,n), dtype=np.complex128)
  fact = (np.empty((n,n), dtype=np.complex128), True)
  b0 = np.empty(n, dtype=np.complex128)
  norm_a = np.empty(1, dtype=np.float64)
  overlap_a = np.empty(1, dtype=np.float64)

  state= (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a)
  return state



############
## Cholesky decomposition updates

def chol_rank2_update(L,Y,j):
  if j==0:
    L21=L[j,:j]
    L22=np.sqrt(Y[j,j])
    L32 = Y[j+1:,j]/L22
  else:
    L21c = la.solve_triangular(L[:j,:j],Y[:j,j],lower=True)
    L21 = np.conj(L21c)
    L22 = np.sqrt(Y[j,j]-np.dot(L21,L21c))
    L32 = (Y[j+1:,j]-np.dot(L[j+1:,:j],L21c))/L22
  L32old = np.array(L[j+1:,j])
  L[j,:j]=L21
  L[j,j]=L22
  L[j+1:,j]=L32
  chol_update(L[j+1:,j+1:],L32old)
  chol_downdate(L[j+1:,j+1:],L32)

def chol_resync(state):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state
  tmp = la.cho_factor(Y,lower=fact[1])
  assert(np.sum(np.abs((tmp[0]-fact[0])[np.tril_indices(len(a1_conj))])) < 1e-9)
  fact[0][:,:]=tmp[0]



############
## b0 optimization

def optimize_b0(a1, delay, state):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state

  np.conj(a1, out=a1_conj)
  calc_y_dy_ddy(a1, delay, template, y, dy, ddy, a1_conj)
  calc_Y(a1, delay, Y, a1_conj)
  calc_dY(a1, delay, dY, a1_conj, Y)
  calc_ddY(a1, delay, ddY, a1_conj, Y)
  calc_ddYd(a1, delay, ddYd, a1_conj, Y)
  fact[0][:,:] = la.cho_factor(Y,lower=fact[1])[0]

  b0[:] = la.cho_solve(fact,y)
  norm_sq = np.dot(np.conj(b0), np.dot(Y, b0))
  assert(np.abs(norm_sq.imag) < 1e-14)
  norm_a[:] = np.sqrt(norm_sq.real)
  overlap = np.dot(np.conj(y),b0) / norm_a[0]
  assert(np.abs(overlap.imag) < 1e-14)
  overlap_a[:] = overlap.real

  return b0/norm_a, overlap_a[0]

def reoptimize_b0(a1, delay, state, j):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state

  a1_conj[j] = np.conj(a1[j])
  recalc_y_dy_ddy(a1, delay, template, y, dy, ddy, j, a1_conj)
  recalc_Y(a1, delay, Y, j, a1_conj)
  recalc_dY(a1, delay, dY, j, a1_conj, Y)
  recalc_ddY(a1, delay, ddY, j, a1_conj, Y)
  chol_rank2_update(fact[0], Y, j)

  b0[:] = la.cho_solve(fact,y)
  norm_sq = np.dot(np.conj(b0), np.dot(Y, b0))
  assert(np.abs(norm_sq.imag) < 1e-14)
  norm_a[:] = np.sqrt(norm_sq.real)
  overlap = np.dot(np.conj(y),b0) / norm_a
  assert(np.abs(overlap.imag) < 1e-14)
  overlap_a[:] = overlap.real

  return b0/norm_a, overlap_a[0]



############
## derivative with respect to a1

def calc_do(a1, delay, state):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state
  do2 = calc_do2(a1, delay, state)
  do = do2/2/overlap_a
  return do

def calc_do_indv(a1, delay, state, j):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state
  do2 = calc_do2_indv(a1, delay, state, j)
  do = do2/2/overlap_a[0]
  return do

def calc_do2(a1, delay, state):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state

  do2 = np.empty(len(a1), dtype=np.complex128)
  for i in range(len(a1)):
    do2[i] = 2*(np.conj(b0[i])*dy[i] - np.conj(b0[i]) * np.dot(np.conj(dY[:,i]),b0))
  return do2

def calc_do2_indv(a1, delay, state, j):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state

  do2 = 2*(np.conj(b0[j])*dy[j] - np.conj(b0[j]) * np.dot(np.conj(dY[:,j]),b0))
  return np.array([do2],dtype=np.complex128)



############
## functions for hessian

def calc_ddo(a1, delay, state):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state
  ddo2 = calc_ddo2(a1, delay, state)
  do2 = calc_do2(a1, delay, state).view(dtype=np.float64)
  ddo = ddo2/2/overlap_a - np.outer(do2,do2)/4/overlap_a**3
  return ddo

def calc_ddo_indv(a1, delay, state, j):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state
  ddo2 = calc_ddo2_indv(a1, delay, state, j)
  do2 = calc_do2_indv(a1, delay, state, j).view(dtype=np.float64)
  ddo = ddo2/2/overlap_a - np.outer(do2,do2)/4/overlap_a**3
  return ddo

def calc_ddo2(a1, delay, state):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state

  ddo2 = np.empty((2*len(a1),2*len(a1)), dtype=np.float64)
  Yinvdy=la.cho_solve(fact,np.diag(dy))
  dYbh = [np.dot(dY,np.diag(b0)) + np.diag(np.dot(b0,np.conj(dY))), 1j*np.dot(dY,np.diag(b0)) - 1j*np.diag(np.dot(b0,np.conj(dY)))]
  YinvdYbh = [la.cho_solve(fact,dYbh[0]), la.cho_solve(fact,dYbh[1])]
  for fi in range(2*len(a1)):
    for fj in range(fi,2*len(a1)):
      ci = 1 if fi%2 == 0 else 1j
      cj = 1 if fj%2 == 0 else 1j
      i=fi//2
      j=fj//2
      ddyij = ddy[j]*np.conj(ci*cj) if i == j else 0
      bhddYdbh = np.conj(b0[j])*np.dot(np.conj(ci*cj*ddYd[:,j]),b0) if i == j else 0
      dYibh = dYbh[fi%2][:,i]
      dYjbh = dYbh[fj%2][:,j]
      YinvdYibh = YinvdYbh[fi%2][:,i]
      YinvdYjbh = YinvdYbh[fj%2][:,j]

      ddo2[fi,fj] = 2*(np.conj(b0[j])*ddyij + np.conj(dy[j]*ci)*cj*Yinvdy[j,i] - np.dot(np.conj(dYjbh*ci),Yinvdy[:,i]) - np.dot(np.conj(dYibh*cj),Yinvdy[:,j]) + np.dot(np.conj(dYjbh),YinvdYibh) - np.conj(b0[j]*cj)*ddY[j,i]*b0[i]*ci - bhddYdbh).real
      ddo2[fj,fi]=ddo2[fi,fj]

  return ddo2

def calc_ddo2_indv(a1, delay, state, j):
  (template, a1_conj, y, dy, ddy, Y, dY, ddY, ddYd, fact, b0, norm_a, overlap_a) = state

  ddo2 = np.empty((2,2), dtype=np.float64)
  x=np.zeros(len(a1), dtype=np.complex128)
  x[j]=dy[j]
  Yinvdy=la.cho_solve(fact,x)
  x[j]=np.dot(np.conj(dY[:,j]),b0)
  dYbh=[b0[j]*dY[:,j] + x, 1j*b0[j]*dY[:,j] - 1j*x]
  YinvdYbh=[la.cho_solve(fact,dYbh[0]), la.cho_solve(fact,dYbh[1])]
  for fi in range(2):
    for fj in range(fi,2):
      ci = 1 if fi%2 == 0 else 1j
      cj = 1 if fj%2 == 0 else 1j
      ddyij = ddy[j]*np.conj(ci*cj)
      bhddYdbh = np.conj(b0[j])*np.dot(np.conj(ci*cj*ddYd[:,j]),b0)
      dYibh = dYbh[fi%2]
      dYjbh = dYbh[fj%2]
      YinvdYibh = YinvdYbh[fi%2]
      YinvdYjbh = YinvdYbh[fj%2]

      ddo2[fi,fj] = 2*(np.conj(b0[j])*ddyij + np.conj(dy[j]*ci)*cj*Yinvdy[j] - np.dot(np.conj(dYjbh*ci),Yinvdy) - np.dot(np.conj(dYibh*cj),Yinvdy) + np.dot(np.conj(dYjbh),YinvdYibh) - np.conj(b0[j]*cj)*ddY[j,j]*b0[j]*ci - bhddYdbh).real
      ddo2[fj,fi]=ddo2[fi,fj]

  return ddo2

def saddle_free_newton(do,ddo,lam=0):
  (val,vec)=la.eigh(ddo)
  return np.dot(vec,np.multiply(1/(np.abs(val)+lam),np.dot(vec.T,do.view(dtype=np.float64)))).view(dtype=np.complex128)



############
## a1 optimization

def optimize_a1(a1, delay, template, passes=0, indv=True, greedy=False, hessian=True, f="o2", eps=1e-3, lam=1e-1, verbose=False):
  calc_df = globals()['calc_d'+f]
  calc_df_indv = globals()['calc_d'+f+'_indv']
  calc_ddf_indv = globals()['calc_dd'+f+'_indv']
  calc_ddf = globals()['calc_dd'+f]

  state = initialize_state(template,len(a1))
  b0, overlap = optimize_b0(a1,delay,state)
  stop = False
  i = -1
  for i in range(passes):
    if verbose:
      eprint("Pass %d, overlap %f"%(i,overlap))
    if indv:
      for tj in range(len(a1)):
        if greedy:
          dff = calc_df(a1,delay,state)
          j = np.argmax(np.abs(dff))
          df = dff[j:j+1]
        else:
          j=tj
          df = calc_df_indv(a1,delay,state,j)
        if hessian:
          ddf = calc_ddf_indv(a1,delay,state,j)
          daj = saddle_free_newton(df,ddf,lam=lam)
        else:
          daj = eps*df
        aj = a1[j]
        while True:
          a1[j] = aj+daj
          if(np.abs(a1[j])<1):
            b0, over = reoptimize_b0(a1,delay,state,j)
            if(over>overlap):
              overlap = over
              break
          daj/=2
          if(np.abs(daj)<1e-9):
            a1[j] = aj
            b0, over = reoptimize_b0(a1,delay,state,j)
            break
    else: # not indv
      df = calc_df(a1,delay,state)
      if hessian:
        ddf = calc_ddf(a1,delay,state)
        da = saddle_free_newton(df,ddf,lam=lam)
      else:
        da = eps*df
      a = np.array(a1)
      while True:
        a1[:] = a + da
        if(np.max(np.abs(a1))<1):
          b0, over = optimize_b0(a1,delay,state)
          if(over>overlap):
            overlap = over
            break
        da/=2
        if(np.max(np.abs(da))<1e-9):
          a1[:] = a
          stop = True
    if stop:
      break
  if verbose:
    eprint("Pass %d, overlap %f"%(i+1,overlap))
  return a1,b0,overlap

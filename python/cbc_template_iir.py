######
#
# This code should read in a xml file and produce three matrices, a1, b0, delay
# that correspond to a bank of waveforms
#
# Copyright Shaun Hooper 2010-05-13
#######

from pylal import spawaveform
import numpy
import scipy
import pylab
import pdb
import csv
import time


def Theta(eta, Mtot, t):
	Tsun = 5.925491e-6
	theta = eta / (5.0 * Mtot * Tsun) * -t;
	return theta

def freq(eta, Mtot, t):
	theta = Theta(eta, Mtot, t)
	Tsun = 5.925491e-6
	f = 1.0 / (8.0 * Tsun * scipy.pi * Mtot) * (
		theta**(-3.0/8.0) +
		(743.0/2688.0 + 11.0 /32.0 * eta) * theta**(-5.0 /8.0) -
		3.0 * scipy.pi / 10.0 * theta**(-3.0 / 4.0) +
		(1855099.0 / 14450688.0 + 56975.0 / 258048.0 * eta + 371.0 / 2048.0 * eta**2.0) * theta**(-7.0/8.0))
	return f

def Phase(eta, Mtot, t, phic = 0.0):
	theta = Theta(eta, Mtot, t)
	phi = phic - 2.0 / eta * (
		theta**(5.0 / 8.0) +
		(3715.0 /8064.0 +55.0 /96.0 *eta) * theta**(3.0/8.0) -
		3.0 *scipy.pi / 4.0 * theta**(1.0/4.0) +
		(9275495.0 / 14450688.0 + 284875.0 / 258048.0 * eta + 1855.0 /2048.0 * eta**2) * theta**(1.0/8.0))
	return phi

def Amp(eta, Mtot, t):
	c = 3.0e8
	Tsun = 5.925491e-6
	f = freq(eta, Mtot, t)
	amp = - 2 * Tsun * c * (eta * Mtot ) * (Tsun * scipy.pi * Mtot * f)**(2.0/3.0);
	return amp

def waveform(m1, m2, fLow, fhigh, sampleRate):
	deltaT = 1.0 / sampleRate
	T = spawaveform.chirptime(m1, m2 , 4, fLow, fhigh)
	tc = -spawaveform.chirptime(m1, m2 , 4, fhigh)
	N = numpy.floor(T / deltaT)
	t = numpy.arange(tc-T, tc, deltaT)
	Mtot = m1 + m2
	eta = m1 * m2 / Mtot**2
	f = freq(eta, Mtot, t)
	amp = Amp(eta, Mtot, t);
	phase = Phase(eta, Mtot, t);
	return amp, phase, f

# FIX ME: Change the following to actually read in the XML file
#
# Start Code
#
	

	
def makeiirbank():

	fFinal = 1500.0
	sampleRate = 4096;
	maxlength = -10

	Amat = []
	Bmat = []
	Dmat = []
	for i in range(1, 2):
		for j in range(i, 2):
			m1 = (i - 1) * 0.2 + 0.8
			m2 = (j - 1) * 0.2 + 0.8
			amp, phase, f = waveform(m1, m2, 40, fFinal, sampleRate)
			a1, b0, delay = spawaveform.iir(amp, phase, 0.01, 0.2, 0.2)
			Amat.append(a1)
			Bmat.append(b0)
			Dmat.append(delay)
			

	dims = (len(Amat),max([len(i) for i in Amat]))
	A = numpy.zeros(dims)
	i = 0
	for a in Amat:
		A[i,0:len(a)] = a
		i = i+1

	B = numpy.zeros(dims)
	i = 0
	for a in Bmat:
		B[i,0:len(a)] = a
		i = i+1
				
	D = numpy.zeros(dims)
	i = 0
	for a in Dmat:
		D[i,0:len(a)] = a
		i = i+1
		


#psd = numpy.ones(amp.shape[0]/2)


	print A.shape, B.shape, D.shape

#ip = spawaveform.iirinnerproduct(a1, b0, delay, psd)
#print "inner product = ", ip

#pylab.plot()
#pylab.show()

	return A, B, D


def innerproduct(a,b):

	n = a.length
	a.append(zeros(n/2),complex)
	a.extend(zeros(n/2),complex)

	b.append(zeros(n/2),complex)
	b.extend(zeros(n/2),complex)

	af = fft(a)
	bf = fft(b)

	cf = af * bf
	c = ifft(cf)

	return max(abs(c))

	

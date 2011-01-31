######
#
# This code should read in a xml file and produce three matrices, a1, b0, delay
# that correspond to a bank of waveforms
#
# Copyright Shaun Hooper 2010-05-13
#######

from pylal import spawaveform
import sys
import time
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
	

def get_iir_sample_rate(xmldoc):
	pass

	
def makeiirbank(xmdoc=None, sampleRate=4096, fuzz=0.90, epsilon=0.04, alpha=.99, beta=0.4, pnorder=4, flower = 40, psd_interp=None):
	
	Amat = []
	Bmat = []
	Dmat = []
	mvec = numpy.linspace(1,3,20)
	snrvec = []
	for m1 in mvec:
		for m2 in mvec:
			if m1 > m2: continue

			start = time.time()
		
			#work out the waveform frequency
			fFinal = spawaveform.ffinal(m1,m2)
			if fFinal > fuzz * sampleRate / 2.0: fFinal = fuzz * sampleRate / 2.0 

			# make the waveform
			amp, phase, f = waveform(m1, m2, flower, fFinal, sampleRate)

			if psd_interp is not None:
				amp /= psd_interp(f)**0.5 * 1e23
			
			#print >> sys.stderr, "waveform %f" % (time.time() - start)
					
			# make the iir filter coeffs
			a1, b0, delay = spawaveform.iir(amp, phase, epsilon, alpha, beta)
			#print >> sys.stderr, "iir %f" % (time.time() - start)
			# get the chirptime
			duration = spawaveform.chirptime(m1,m2,pnorder,flower)
			length = 2**numpy.ceil(numpy.log2(duration * sampleRate))

			# get the IIR response
			out = spawaveform.iirresponse(length, a1, b0, delay)
			#print >> sys.stderr, "response %f" % (time.time() - start)
			out = out[::-1]
			vec1 = numpy.zeros(length * 2)
			vec1[-len(out):] = out
			norm1 = (vec1 * numpy.conj(vec1)).sum()**0.5
			vec1 /= norm1	
			#print >> sys.stderr, "norm 1 %f" % (time.time() - start)
		
			# get the original waveform
			out2 = amp * numpy.exp(1j * phase)
			vec2 = numpy.zeros(length * 2)
			vec2[-len(out2):] = out2
			vec2 /= (vec2 * numpy.conj(vec2)).sum()**0.5
			#print >> sys.stderr, "norm 2 %f" % (time.time() - start)
			
			#n = 2**numpy.ceil(numpy.log2(length))
			#print n, length
			snr = numpy.abs(scipy.ifft(scipy.fft(vec1) * numpy.conj(scipy.fft(vec2)))).max()
			snrvec.append(snr)
			print >>sys.stderr, m1, m2, snr, len(a1)
			#print >> sys.stderr, "done %f" % (time.time() - start)
			

	return snrvec


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

	

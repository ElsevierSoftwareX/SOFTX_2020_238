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
import pdb
import csv
import time
from glue.ligolw import ligolw, lsctables, array, param, utils, types
from gstlal.pipeio import repack_complex_array_to_real, repack_real_array_to_complex

def Theta(eta, Mtot, t):
	Tsun = 4.925491e-6
	theta = eta / (5.0 * Mtot * Tsun) * -t;
	return theta

def freq(eta, Mtot, t):
	theta = Theta(eta, Mtot, t)
	Tsun = 4.925491e-6
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
	theta = Theta(eta, Mtot, t)
	c = 3.0e10
	Tsun = 4.925491e-6
	f = 1.0 / (8.0 * Tsun * scipy.pi * Mtot) * (theta**(-3.0/8.0))
	amp = - 2 * Tsun * c * (eta * Mtot ) * (Tsun * scipy.pi * Mtot * f)**(2.0/3.0);
	return amp

def waveform(m1, m2, fLow, fhigh, sampleRate):
	deltaT = 1.0 / sampleRate
	T = spawaveform.chirptime(m1, m2 , 4, fLow, fhigh)
	tc = -spawaveform.chirptime(m1, m2 , 4, fhigh)
	t = numpy.arange(tc-T, tc, deltaT)
	#T = numpy.floor(-spawaveform.chirptime(m1, m2 , 4, fLow)*sampleRate+0.5)*deltaT
	#tc = numpy.ceil(-spawaveform.chirptime(m1, m2 , 4, fhigh)*sampleRate+0.5)*deltaT
	#t = numpy.arange(T, tc, deltaT)
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

def get_fir_matrix(xmldoc, sampleRate=4096, pnorder=4, psd_interp=None, output_to_xml = False, verbose=False, padding=1.1, autocorrelation_length=101):

	flower = param.get_pyvalue(xmldoc, 'flower')
	snrvec = []
	Mlist = []

	sngl_inspiral_table=lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)

	if not (autocorrelation_length % 2):
		raise ValueError, "autocorrelation_length must be odd (got %d)" % autocorrelation_length
	autocorrelation_bank = numpy.zeros((len(sngl_inspiral_table), autocorrelation_length), dtype = "cdouble")

	for tmp, row in enumerate(sngl_inspiral_table):
		m1 = row.mass1
		m2 = row.mass2

		# work out the waveform frequency
		fFinal = spawaveform.ffinal(m1,m2)
		if fFinal > sampleRate / 2.0 / padding: fFinal = sampleRate / 2.0 / padding

		# make the waveform
		amp, phase, f = waveform(m1, m2, flower, fFinal, sampleRate)
		if psd_interp is not None:
			amp /= psd_interp(f)**0.5 * 1e23

		length = 2**numpy.ceil(numpy.log2(amp.shape[0]))
		out = amp * numpy.exp(1j * phase)

		# normalize the fir coefficients
		vec1 = numpy.zeros(length * 2, dtype=numpy.cdouble)
		vec1[-len(out):] = out
		norm1 = (1.0/numpy.sqrt(2.0))*((vec1 * numpy.conj(vec1)).sum()**0.5)
		vec1 /= norm1
		#vec1 = vec1[::-1]

		# store the coeffs.
		Mlist.append(vec1.real)
		Mlist.append(vec1.imag)

		# compute the SNR
		corr = scipy.ifft(scipy.fft(vec1) * numpy.conj(scipy.fft(vec1)))

		#FIXME this is actually the cross correlation between the original waveform and this approximation
		autocorrelation_bank[tmp,:] = numpy.concatenate((corr[(-autocorrelation_length/2+2):],corr[:autocorrelation_length/2+2]))

	max_len = max([len(i) for i in Mlist])
	M = numpy.zeros((len(Mlist), max_len))

	for i, Am in enumerate(Mlist): M[i,:len(Am)] = Am

	return M, autocorrelation_bank

def makeiirbank(xmldoc, sampleRate=4096, padding=1.1, epsilon=0.02, alpha=.99, beta=0.25, pnorder=4, flower = 40, psd_interp=None, output_to_xml = False, autocorrelation_length=101, verbose=False):

	sngl_inspiral_table=lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
	Amat = []
	Bmat = []
	Dmat = []
	snrvec = []


	if not (autocorrelation_length % 2):
		raise ValueError, "autocorrelation_length must be odd (got %d)" % autocorrelation_length
	autocorrelation_bank = numpy.zeros((len(sngl_inspiral_table), autocorrelation_length), dtype = "cdouble")
	
	for tmp, row in enumerate(sngl_inspiral_table):

		m1 = row.mass1
		m2 = row.mass2
		
		start = time.time()
		
		# work out the waveform frequency
		fFinal = spawaveform.ffinal(m1,m2)
		if fFinal > sampleRate / 2.0 / padding: fFinal = sampleRate / 2.0 / padding

		# make the waveform
		amp, phase, f = waveform(m1, m2, flower, fFinal, sampleRate)

		if psd_interp is not None:
			amp /= psd_interp(f)**0.5 * 1e23
			
		#print >> sys.stderr, "waveform %f" % (time.time() - start)
					
		# make the iir filter coeffs
		a1, b0, delay = spawaveform.iir(amp, phase, epsilon, alpha, beta)

		if verbose: print>>sys.stderr, "row %4.0d - m1: %10.6f m2: %10.6f required %4.0d filters" % (tmp, m1,m2,len(a1))
		# get the chirptime
		length = 2**numpy.ceil(numpy.log2(amp.shape[0]))

		# get the IIR response
		out = spawaveform.iirresponse(length, a1, b0, delay)
		out = out[::-1]
		vec1 = numpy.zeros(length * 2, dtype=numpy.cdouble)
		vec1[-len(out):] = out
		norm1 = 1.0/numpy.sqrt(2.0)*((vec1 * numpy.conj(vec1)).sum()**0.5)
		vec1 /= norm1

		# normalize the iir coefficients
		b0 /= norm1

		# store the coeffs.
		Amat.append(a1)
		Bmat.append(b0)
		Dmat.append(delay)

		# get the original waveform
		out2 = amp * numpy.exp(1j * phase)
		vec2 = numpy.zeros(length * 2, dtype=numpy.cdouble)
		vec2[-len(out2):] = out2
		vec2 /= 1.0/numpy.sqrt(2.0)*((vec2 * numpy.conj(vec2)).sum()**0.5)


		# compute the SNR
		corr = scipy.ifft(scipy.fft(vec1) * numpy.conj(scipy.fft(vec2)))

		#FIXME this is actually the cross correlation between the original waveform and this approximation
		autocorrelation_bank[tmp,:] = numpy.concatenate((corr[(-autocorrelation_length/2+2):],corr[:autocorrelation_length/2+2]))
		
		snr = numpy.abs(corr).max()
		snrvec.append(snr)

		# store the match for later
		if output_to_xml: row.snr = snr
		print "dot product iir iir", numpy.abs((vec1) * numpy.conj(vec1)).sum()
		print "dot product fir fir", numpy.abs((vec2) * numpy.conj(vec2)).sum()
		print "dot product iir fir", numpy.abs((vec1) * numpy.conj(vec2)).sum()
		print "SNR ", snr
		##if verbose: print >>sys.stderr, m1, m2, snr, len(a1)
			
	# get ready to store the coefficients
	print [len(i) for i in Amat]
	max_len = max([len(i) for i in Amat])
	
	A = numpy.zeros((len(Amat), max_len),dtype=numpy.complex128)
	B = numpy.zeros((len(Amat), max_len), dtype=numpy.complex128)
	D = numpy.zeros((len(Amat), max_len), dtype=numpy.int32)

	for i, Am in enumerate(Amat): A[i,:len(Am)] = Am
	for i, Bm in enumerate(Bmat): B[i,:len(Bm)] = Bm
	for i, Dm in enumerate(Dmat): D[i,:len(Dm)] = Dm
	
	if output_to_xml: # Create new document and add them together
		root = xmldoc.childNodes[0]
		if param.get_param(xmldoc, 'sample_rate'):
			root.removeChild(param.get_param(xmldoc, 'sample_rate'))
		root.appendChild(param.new_param('sample_rate', types.FromPyType[int], sampleRate))
		if param.get_param(xmldoc, 'flower'):
			root.removeChild(param.get_param(xmldoc, 'flower'))
		root.appendChild(param.new_param('flower', types.FromPyType[float], flower))
		if array.get_array(xmldoc, 'a'):
			root.removeChild(array.get_array(xmldoc, 'a'))
		root.appendChild(array.from_array('a', repack_complex_array_to_real(A)))
		if array.get_array(xmldoc, 'b'):
			root.removeChild(array.get_array(xmldoc, 'b'))
		root.appendChild(array.from_array('b', repack_complex_array_to_real(B)))
		if array.get_array(xmldoc, 'd'):
			root.removeChild(array.get_array(xmldoc, 'd'))
		root.appendChild(array.from_array('d', D))
		if array.get_array(xmldoc, 'autocorrelation'):
			root.removeChild(array.get_array(xmldoc, 'autocorrelation'))
		root.appendChild(array.from_array('autocorrelation', repack_complex_array_to_real(autocorrelation_bank)))
	
	return A, B, D, snrvec


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

def smooth_and_interp(psd, width=1, length = 10):
	data = psd.data
	f = numpy.arange(len(psd.data)) * psd.deltaF
	ln = len(data)
	x = numpy.arange(-width*length, width*length)
	sfunc = numpy.exp(-x**2 / 2.0 / width**2) / (2. * numpy.pi * width**2)**0.5
	out = numpy.zeros(ln)
	for i,d in enumerate(data[width*length:ln-width*length]):
		out[i+width*length] = (sfunc * data[i:i+2*width*length]).sum()
	return scipy.interpolate.interp1d(f, out)

def get_matrices_from_xml(xmldoc):
	root = xmldoc
	A = repack_real_array_to_complex(array.get_array(root, 'a').array)
	B = repack_real_array_to_complex(array.get_array(root, 'b').array)
	D = array.get_array(root, 'd').array
	autocorrelation = repack_real_array_to_complex(array.get_array(root, 'autocorrelation').array)
	return A, B, D, autocorrelation

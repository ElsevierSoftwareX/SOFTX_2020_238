######
#
# This code should read in a xml file and produce three matrices, a1, b0, delay
# that correspond to a bank of waveforms
#
#######
#
# Copyright (C) 2010-2012 Shaun Hooper
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

from gstlal import spawaveform
import sys
import time
import numpy
import scipy
from scipy import integrate
from scipy import interpolate
import math
import lal
import pdb
import csv
from ligo.lw import ligolw, lsctables, array, param, utils, types
from gstlal.pipeio import repack_complex_array_to_real, repack_real_array_to_complex
import pdb

class XMLContentHandler(ligolw.LIGOLWContentHandler):
	pass

def Theta(eta, Mtot, t):
	Tsun = lal.MTSUN_SI #4.925491e-6
	theta = eta / (5.0 * Mtot * Tsun) * -t
	return theta

def freq(eta, Mtot, t):
        theta = Theta(eta, Mtot, t)
        Tsun = lal.MTSUN_SI #4.925491e-6
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
        c = lal.C_SI #3.0e10
        Tsun = lal.MTSUN_SI #4.925491e-6
        Mpc = 1e6 * lal.PC_SI #3.08568025e24
        f = 1.0 / (8.0 * Tsun * scipy.pi * Mtot) * (theta**(-3.0/8.0))
        amp = - 4.0/Mpc * Tsun * c * (eta * Mtot ) * (Tsun * scipy.pi * Mtot * f)**(2.0/3.0);
        return amp

def waveform(m1, m2, fLow, fhigh, sampleRate):
        deltaT = 1.0 / sampleRate
        T = spawaveform.chirptime(m1, m2 , 4, fLow, fhigh)
        tc = -spawaveform.chirptime(m1, m2 , 4, fhigh)
	# the last sampling point of any waveform is always set
	# at abs(t) >= delta. this is to avoid ill-condition of
	# frequency when abs(t) < 1e-5
        n_start = math.floor((tc-T) / deltaT + 0.5)
        n_end = min(math.floor(tc/deltaT), -1)
        t = numpy.arange(n_start, n_end+1, 1) * deltaT
        Mtot = m1 + m2
        eta = m1 * m2 / Mtot**2
        f = freq(eta, Mtot, t)
        amp = Amp(eta, Mtot, t);
        phase = Phase(eta, Mtot, t);
        return amp, phase, f

def sigmasq2(mchirp, fLow, fhigh, psd_interp):
	c = lal.C_SI #299792458
	G = lal.G_SI #6.67259e-11
	M = lal.MSUN_SI #1.98892e30
	Mpc =1e6 * lal.PC_SI #3.0856775807e22
	#mchirp = 1.221567#30787
	const = numpy.sqrt((5.0 * math.pi)/(24.*c**3))*(G*mchirp*M)**(5./6.)*math.pi**(-7./6.)/Mpc
	return  const * numpy.sqrt(4.*integrate.quad(lambda x: x**(-7./3.) / psd_interp(x), fLow, fhigh)[0])

# FIX ME: Change the following to actually read in the XML file
#
# Start Code
#


def get_iir_sample_rate(xmldoc):
        pass

def sample_rates_array_to_str(sample_rates):
        return ",".join([str(a) for a in sample_rates])

def sample_rates_str_to_array(sample_rates_str):
        return numpy.array([int(a) for a in sample_rates_str.split(',')])


def get_fir_matrix(xmldoc, fFinal=None, pnorder=4, flower = 40, psd_interp=None, autocorrelation_length=101, verbose=False):
	sngl_inspiral_table = lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
	fFinal = max(sngl_inspiral_table.getColumnByName("f_final"))
	sampleRate = int(2**(numpy.ceil(numpy.log2(fFinal)+1)))
	flower = param.get_pyvalue(xmldoc, 'flower')
	if verbose: print >> sys.stderr, "f_min = %f, f_final = %f, sample rate = %f" % (flower, fFinal, sampleRate)

	snrvec = []
	Mlist = []

	if not (autocorrelation_length % 2):
		raise ValueError("autocorrelation_length must be odd (got %d)".format(autocorrelation_length))
	autocorrelation_bank = numpy.zeros((len(sngl_inspiral_table), autocorrelation_length), dtype = "cdouble")

	for tmp, row in enumerate(sngl_inspiral_table):
		m1 = row.mass1
		m2 = row.mass2

                # make the waveform
		amp, phase, f = waveform(m1, m2, flower, fFinal, sampleRate)
		if psd_interp is not None:
			amp /= psd_interp(f)**0.5 * 1e23

		length = int(2**numpy.ceil(numpy.log2(amp.shape[0])))
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
		#corr = scipy.ifft(scipy.fft(vec1) * numpy.conj(scipy.fft(vec1)))

		#FIXME this is actually the cross correlation between the original waveform and this approximation
		#autocorrelation_bank[tmp,:] = numpy.concatenate((corr[(-autocorrelation_length/2+2):],corr[:autocorrelation_length/2+2]))

	max_len = max([len(i) for i in Mlist])
	M = numpy.zeros((len(Mlist), max_len))

	for i, Am in enumerate(Mlist): M[i,:len(Am)] = Am

	return M, autocorrelation_bank

def compute_autocorrelation_mask( autocorrelation ):
	'''
	Given an autocorrelation time series, estimate the optimal
	autocorrelation length to use and return a matrix which masks
	out the unwanted elements. FIXME TODO for now just returns
	ones
	'''
	return numpy.ones( autocorrelation.shape, dtype="int" )


def makeiirbank(xmldoc, sampleRate = None, padding=1.1, epsilon=0.02, alpha=.99, beta=0.25, pnorder=4, flower = 40, psd_interp=None, output_to_xml = False, autocorrelation_length=201, downsample=False, verbose=False):

	sngl_inspiral_table=lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
	fFinal = max(sngl_inspiral_table.getColumnByName("f_final"))
	if sampleRate is None:
		sampleRate = int(2**(numpy.ceil(numpy.log2(fFinal)+1)))

	if verbose: print >> sys.stderr, "f_min = %f, f_final = %f, sample rate = %f" % (flower, fFinal, sampleRate)

	Amat = {}
	Bmat = {}
	Dmat = {}
	snrvec = []


	# Check parity of autocorrelation length
	if autocorrelation_length is not None:
		if not (autocorrelation_length % 2):
			raise ValueError("autocorrelation_length must be odd (got %d)".format(autocorrelation_length))
		autocorrelation_bank = numpy.zeros((len(sngl_inspiral_table), autocorrelation_length), dtype = "cdouble")
		autocorrelation_mask = compute_autocorrelation_mask( autocorrelation_bank )
	else:
		autocorrelation_bank = None
		autocorrelation_mask = None


	for tmp, row in enumerate(sngl_inspiral_table):

		m1 = row.mass1
		m2 = row.mass2
		fFinal = row.f_final
		if verbose: start = time.time()

                # work out the waveform frequency
                #fFinal = spawaveform.ffinal(m1,m2)
                #if fFinal > sampleRate / 2.0 / padding: fFinal = sampleRate / 2.0 / padding

                # make the waveform

		amp, phase, f = waveform(m1, m2, flower, fFinal, sampleRate)
		if verbose:
			print >> sys.stderr, "waveform %f (T = %f)" % ((time.time() - start), float(amp.shape[0]/(float(sampleRate))))
			start = time.time()
		if psd_interp is not None:

                        amp /= psd_interp(f)**0.5

                # make the iir filter coeffs
		a1, b0, delay = spawaveform.iir(amp, phase, epsilon, alpha, beta, padding)
		if verbose:
			print("create IIR bank %f".format(time.time() - start),file=sys.stderr)
			start = time.time()
                # get the chirptime
		length = int(2**numpy.ceil(numpy.log2(amp.shape[0]+autocorrelation_length)))
		if verbose: print >> sys.stderr, "length = %d, amp length = %d, autocorrelation length = %d" % (length, amp.shape[0], autocorrelation_length)

                # get the IIR response
		out = spawaveform.iirresponse(length, a1, b0, delay)
		if verbose:
			print >> sys.stderr, "create IIR response %f" % (time.time() - start)
			start = time.time()
		out = out[::-1]
		u = numpy.zeros(length * 1, dtype=numpy.cdouble)
		u[-len(out):] = out
		norm1 = 1.0/numpy.sqrt(2.0)*((u * numpy.conj(u)).sum()**0.5)
		u /= norm1

                # normalize the iir coefficients
		b0 /= norm1

                # get the original waveform
		out2 = amp * numpy.exp(1j * phase)
		h = numpy.zeros(length * 1, dtype=numpy.cdouble)
		h[-len(out2):] = out2
		#norm2 = 1.0/numpy.sqrt(2.0)*((h * numpy.conj(h)).sum()**0.5)
                #h /= norm2
		#if output_to_xml: row.sigmasq = norm2**2*2.0*1e46/sampleRate/9.5214e+48

		norm2 = abs(numpy.dot(h, numpy.conj(h)))
		h *= numpy.sqrt(2 / norm2)
		if output_to_xml: row.sigmasq = 1.0 * norm2 / sampleRate
		if verbose:
			newsigma = sigmasq2(row.mchirp, flower, fFinal, psd_interp)
			print("norm2 = %e, sigma = %f, %f, %f".format(norm2, numpy.sqrt(row.sigmasq), newsigma, (numpy.sqrt(row.sigmasq)- newsigma)/newsigma), file=sys.stderr)
			print("mchirp %fm, fFinal %f, row.f_final %f".format(row.mchirp, fFinal, row.f_final), file=sys.stderr)
			start = time.time()

                #FIXME this is actually the cross correlation between the original waveform and this approximation
		autocorrelation_bank[tmp,:] = crosscorr(h, h, autocorrelation_length)/2.0
		if verbose:
			print("auto correlation %f".format(time.time() - start), file=sys.stderr)
			start = time.time()

		# compute the SNR
		snr = abs(numpy.dot(u, numpy.conj(h)))/2.0
		if verbose:
			print("dot product %f".format(time.time() - start), file=sys.stderr)
			start = time.time()
		snrvec.append(snr)
		if verbose: print("row %4.0d, m1 = %10.6f m2 = %10.6f, %4.0d filters, %10.8f match".format(tmp+1, m1,m2,len(a1), snr), file=sys.stderr)


                # store the match for later
		if output_to_xml: row.snr = snr

                # get the filter frequencies
		fs = -1. * numpy.angle(a1) / 2 / numpy.pi # Normalised freqeuncy
		a1dict = {}
		b0dict = {}
		delaydict = {}

		if downsample:
			# iterate over the frequencies and put them in the right downsampled bin
			for i, f in enumerate(fs):
				M = int(max(1, 2**-numpy.ceil(numpy.log2(f * 2.0 * padding)))) # Decimation factor
				a1dict.setdefault(sampleRate/M, []).append(a1[i]**M)
				newdelay = numpy.ceil((delay[i]+1)/(float(M)))
				b0dict.setdefault(sampleRate/M, []).append(b0[i]*M**0.5*a1[i]**(newdelay*M-delay[i]))
				delaydict.setdefault(sampleRate/M, []).append(newdelay)
				#print>>sys.stderr, "sampleRate %4.0d, filter %3.0d, M %2.0d, f %10.9f, delay %d, newdelay %d" % (sampleRate, i, M, f, delay[i], newdelay)

		else:
			a1dict[int(sampleRate)] = a1
			b0dict[int(sampleRate)] = b0
			delaydict[int(sampleRate)] = delay

		# store the coeffs
		for k in a1dict.keys():
			Amat.setdefault(k, []).append(a1dict[k])
			Bmat.setdefault(k, []).append(b0dict[k])
			Dmat.setdefault(k, []).append(delaydict[k])



		if verbose: print>>sys.stderr, "filters per sample rate (rate , num filters)\n",[(k,len(v)) for k,v in a1dict.items()]


	A = {}
	B = {}
	D = {}
	sample_rates = []

	max_rows = max([len(Amat[rate]) for rate in Amat.keys()])
	for rate in Amat.keys():
		sample_rates.append(rate)
		# get ready to store the coefficients
		max_len = max([len(i) for i in Amat[rate]])
		DmatMin = min([min(elem) for elem in Dmat[rate]])
		DmatMax = max([max(elem) for elem in Dmat[rate]])
		if verbose: print>>sys.stderr, "rate %d, dmin %d, dmax %d, max_row %d, max_len %d" % (rate, DmatMin, DmatMax, max_rows, max_len)
		A[rate] = numpy.zeros((max_rows, max_len), dtype=numpy.complex128)
		B[rate] = numpy.zeros((max_rows, max_len), dtype=numpy.complex128)
		D[rate] = numpy.zeros((max_rows, max_len), dtype=numpy.int)
		D[rate].fill(DmatMin)

		for i, Am in enumerate(Amat[rate]): A[rate][i,:len(Am)] = Am
		for i, Bm in enumerate(Bmat[rate]): B[rate][i,:len(Bm)] = Bm
		for i, Dm in enumerate(Dmat[rate]): D[rate][i,:len(Dm)] = Dm
		if output_to_xml:
			#print 'a_%d' % (rate)
			#print A[rate], type(A[rate])
			#print repack_complex_array_to_real(A[rate])
			#print array.from_array('a_%d' % (rate), repack_complex_array_to_real(A[rate]))
			root = xmldoc.childNodes[0]
			root.appendChild(array.from_array('a_%d' % (rate), repack_complex_array_to_real(A[rate])))
			root.appendChild(array.from_array('b_%d' % (rate), repack_complex_array_to_real(B[rate])))
			root.appendChild(array.from_array('d_%d' % (rate), D[rate]))

	if output_to_xml: # Create new document and add them together
		root = xmldoc.childNodes[0]
		root.appendChild(param.new_param('sample_rate', types.FromPyType[str], sample_rates_array_to_str(sample_rates)))
		root.appendChild(param.new_param('flower', types.FromPyType[float], flower))
		root.appendChild(param.new_param('epsilon', types.FromPyType[float], epsilon))
		root.appendChild(array.from_array('autocorrelation_bank_real', autocorrelation_bank.real))
		root.appendChild(array.from_array('autocorrelation_bank_imag', -autocorrelation_bank.imag))
		root.appendChild(array.from_array('autocorrelation_mask', autocorrelation_mask))

	return A, B, D, snrvec

def crosscorr(a, b, autocorrelation_length = 201):
	af = scipy.fft(a)
	bf = scipy.fft(b)
	corr = scipy.ifft( af * numpy.conj( bf ))
	return numpy.concatenate((corr[-(autocorrelation_length // 2):],corr[:(autocorrelation_length // 2 + 1)]))

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
        return interpolate.interp1d(f, out)

def get_matrices_from_xml(xmldoc):
	root = xmldoc
	sample_rates = [int(r) for r in param.get_pyvalue(root, 'sample_rate').split(',')]
	A = {}
	B = {}
	D = {}
	for sr in sample_rates:
		A[sr] = repack_real_array_to_complex(array.get_array(root, 'a_%d' % (sr,)).array)
		B[sr] = repack_real_array_to_complex(array.get_array(root, 'b_%d' % (sr,)).array)
		D[sr] = array.get_array(root, 'd_%d' % (sr,)).array
	autocorrelation_bank_real = array.get_array(root, 'autocorrelation_bank_real').array
	autocorrelation_bank_imag = array.get_array(root, 'autocorrelation_bank_imag').array
	autocorrelation_bank = autocorrelation_bank_real + (0+1j) * autocorrelation_bank_imag
	autocorrelation_mask = array.get_array(root, 'autocorrelation_mask').array

	sngl_inspiral_table=lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
	sigmasq  = sngl_inspiral_table.getColumnByName("sigmasq").asarray()

	return A, B, D, autocorrelation_bank, autocorrelation_mask, sigmasq

class Bank(object):
	def __init__(self, bank_xmldoc, snr_threshold, logname = None, verbose = False):
		self.template_bank_filename = None
		self.snr_threshold = snr_threshold
		self.logname = logname

		self.A, self.B, self.D, self.autocorrelation_bank, self.autocorrelation_mask, self.sigmasq = get_matrices_from_xml(bank_xmldoc)
		#self.sigmasq=numpy.ones(len(self.autocorrelation_bank)) # FIXME: make sigmasq correct

	def get_rates(self, verbose = False):
		bank_xmldoc = utils.load_filename(self.template_bank_filename, contenthandler = XMLContentHandler, verbose = verbose)
		return [int(r) for r in param.get_pyvalue(bank_xmldoc, 'sample_rate').split(',')]

	# FIXME: remove set_template_bank_filename when no longer needed
	# by trigger generator element
	def set_template_bank_filename(self,name):
		self.template_bank_filename = name

def load_iirbank(filename, snr_threshold, contenthandler = XMLContentHandler, verbose = False):
	bank_xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)

	bank = Bank.__new__(Bank)
	bank = Bank(
		bank_xmldoc,
		snr_threshold,
		verbose = verbose
	)

	bank.set_template_bank_filename(filename)
	return bank

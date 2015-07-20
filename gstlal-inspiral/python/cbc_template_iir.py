
# Copyright (C) 2010-2012 Shaun Hooper
# Copyright (C) 2013-2014 Qi Chu, David Mckenzie, Kipp Cannon, Chad Hanna, Leo Singer
# Copyright (C) 2015 Qi Chu, Shin Chung, David Mckenzie, Yan Wang
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

import os
import numpy
import scipy
from scipy import integrate
from scipy import interpolate
import math
import csv
import logging
import tempfile

import lal
import lalsimulation
from glue.ligolw import ligolw, lsctables, array, param, utils, types
from pylal import datatypes as laltypes
from pylal import lalfft
from gstlal.pipeio import repack_complex_array_to_real, repack_real_array_to_complex
from gstlal import cbc_template_fir
from gstlal import templates
import random
import pdb

Attributes = ligolw.sax.xmlreader.AttributesImpl

# will be DEPRECATED once the C SPIIR coefficient code be swig binded
from pylal import spawaveform


# FIXME:  require calling code to provide the content handler
class DefaultContentHandler(ligolw.LIGOLWContentHandler):
	pass
array.use_in(DefaultContentHandler)
param.use_in(DefaultContentHandler)
lsctables.use_in(DefaultContentHandler)

# DEPRECATED: The following code is ued to generate the amplitude and phase 
# 	      information of a waveform. NOW we use the standard lalsimulation
#             to obtain amp and phase.


class XMLContentHandler(ligolw.LIGOLWContentHandler):
	pass

def lefttukeywindow(data, samps = 200.):
       assert (len(data) >= 2 * samps) # make sure that the user is requesting something sane
       tp = float(samps) / len(data)
       wn = lal.CreateTukeyREAL8Window(len(data), tp).data.data
       wn[len(wn)//2:] = 1.0
       return wn

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


# For IIR bank construction we use LALSimulation waveforms
# FIR matrix code has not been updated but it is not used

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

# end of DEPRECATION

# Clean up the start and end frequencies
# Modifies the f array in place

def cleanFreq(f,fLower):
    i = 0;
    fStartFix = 0; #So if f is 0, -1, -2, 15, -3 15 16 17 18 then this will be 4
    while(i< 100): #This should be enough
	if(f[i] < fLower-5 or f[i] > fLower+5):  ##Say, 5 or 10 Hz)
	    fStartFix = i;
	i=i+1;

    if(fStartFix != 0):
	f[0:fStartFix+1] = fLower

    i=-100;
    while(i<-1):
	#require monotonicity
	if(f[i]>f[i+1]):
	    f[i+1:]=10; #effectively throw away the end
	    break;
	else:
	    i=i+1;

# Calculate the phase and amplitude from hc and hp
# Unwind the phase (this is slow - consider C extension or using SWIG
# if speed is needed)

def calc_amp_phase(hc,hp):
    amp = numpy.sqrt(hc*hc + hp*hp)
    phase = numpy.arctan2(hc,hp)
    
    #Unwind the phase
    #Based on the unwinding codes in pycbc
    #and the old LALSimulation interface
    count=0
    prevval = None
    phaseUW = phase 

    #Pycbc uses 2*PI*0.7 for some reason
    #We use the more conventional PI (more in line with MATLAB)
    thresh = lal.PI;
    for index, val in enumerate(phase):
	if prevval is None:
	    pass
	elif prevval - val >= thresh:
	    count = count+1
	elif val - prevval >= thresh:
	    count = count-1

	phaseUW[index] = phase[index] + count*lal.TWOPI
	prevval = val

    for index, val in enumerate(phase):
	    phaseUW[index] = phaseUW[index] - phaseUW[0]

    phase = phaseUW
    return amp,phase

def sigmasq2(mchirp, fLow, fhigh, psd_interp):
	c = lal.C_SI #299792458
	G = lal.G_SI #6.67259e-11
	M = lal.MSUN_SI #1.98892e30
	Mpc =1e6 * lal.PC_SI #3.0856775807e22
	#mchirp = 1.221567#30787
	const = numpy.sqrt((5.0 * math.pi)/(24.*c**3))*(G*mchirp*M)**(5./6.)*math.pi**(-7./6.)/Mpc
	return  const * numpy.sqrt(4.*integrate.quad(lambda x: x**(-7./3.) / psd_interp(x), fLow, fhigh)[0])

def sample_rates_array_to_str(sample_rates):
        return ",".join([str(a) for a in sample_rates])

def sample_rates_str_to_array(sample_rates_str):
        return numpy.array([int(a) for a in sample_rates_str.split(',')])

def compute_autocorrelation_mask( autocorrelation ):
	'''
	Given an autocorrelation time series, estimate the optimal
	autocorrelation length to use and return a matrix which masks
	out the unwanted elements. FIXME TODO for now just returns
	ones
	'''
	return numpy.ones( autocorrelation.shape, dtype="int" )

def normalized_crosscorr(a, b, autocorrelation_length = 201):
	af = scipy.fft(a)
	bf = scipy.fft(b)
	corr = scipy.ifft( af * numpy.conj( bf ))
	abs_corr = abs(corr)
	max_idx = numpy.where(abs_corr == max(abs_corr))[0][0]
	if max_idx != 0:
		raise ValueError, "max of autocorrelation must happen at position [0], got [%d]" % max_idx
	tmp_corr = corr
	corr = tmp_corr / tmp_corr[0]
	auto_bank = numpy.zeros(autocorrelation_length, dtype = 'cdouble')
	auto_bank[::-1] = numpy.concatenate((corr[-(autocorrelation_length // 2):],corr[:(autocorrelation_length // 2 + 1)]))
	return auto_bank

def normalized_convolv(a, b, autocorrelation_length = 201):
	af = scipy.fft(a)
	bf = scipy.fft(b)
	corr = scipy.ifft( af *  bf )
	abs_corr = abs(corr)
	max_idx = numpy.where(abs_corr == max(abs_corr))[0][0]
	if max_idx > len(abs_corr)/2:
		max_idx = max_idx - len(abs_corr)
	tmp_corr = corr
	corr = tmp_corr / tmp_corr[max_idx]

	#FIXME: The following code will raise type error
	auto_bank = numpy.concatenate(corr[max_idx -(autocorrelation_length // 2):min(max_idx,0)], corr[min(max_idx,0):max(max_idx, 0)])
	auto_bank = numpy.concatenate(auto, corr[max(max_idx, 0):max_idx + (autocorrelation_length // 2 + 1)])
	return auto_bank


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

def lalwhiten(psd, hplus, working_length, working_duration, sampleRate, length_max):

	"""	
	This function can be called to calculate a whitened waveform using lalwhiten.
	This is for comparison of whitening the waveform using lalwhiten in frequency domain
	and our own whitening in time domain. 
	and use this waveform to calculate a autocorrelation function.


	from pylal import datatypes as laltypes
	from pylal import lalfft
	lalwhiten_amp, lalwhiten_phase = lalwhiten(psd, hp, working_length, working_duration, sampleRate, length_max)
	lalwhiten_wave = lalwhiten_amp * numpy.exp(1j * lalwhiten_phase)
        auto_h = numpy.zeros(length_max * 1, dtype=numpy.cdouble)
        auto_h[-len(lalwhiten_wave):] = lalwhiten_wave
	auto_bank_new = normalized_crosscorr(auto_h, auto_h, autocorrelation_length)
	"""



	revplan = lalfft.XLALCreateReverseCOMPLEX16FFTPlan(working_length, 1)
	fwdplan = lalfft.XLALCreateForwardREAL8FFTPlan(working_length, 1)
	tseries = laltypes.COMPLEX16TimeSeries(
		data = numpy.zeros((working_length,), dtype = "cdouble")
	)
	fworkspace = laltypes.COMPLEX16FrequencySeries(
		name = "template",
		epoch = laltypes.LIGOTimeGPS(0),
		f0 = 0.0,
		deltaF = 1.0 / working_duration,
		data = numpy.zeros((working_length//2 + 1,), dtype = "cdouble")
	)


	data = numpy.zeros((sampleRate * working_duration,))
	norm = numpy.sqrt(numpy.dot(hplus.data.data, numpy.conj(hplus.data.data)))
	data[-hplus.data.length:] = hplus.data.data / norm


	tmptseries = laltypes.REAL8TimeSeries(
		name = "template",
		epoch = laltypes.LIGOTimeGPS(0),
		f0 = 0.0,
		deltaT = 1.0 / sampleRate,
		sampleUnits = laltypes.LALUnit("strain"),
		data = data
	)
		
	lalfft.XLALREAL8TimeFreqFFT(fworkspace, tmptseries, fwdplan)
	tmpfseries = numpy.copy(fworkspace.data)

	fseries = laltypes.COMPLEX16FrequencySeries(
		name = "template",
		epoch = laltypes.LIGOTimeGPS(0),
		f0 = 0.0,
		deltaF = 1.0 / working_duration,
		sampleUnits = laltypes.LALUnit("strain"),
		data = tmpfseries 
	)



	#
	# whiten and add quadrature phase ("sine" component)
	#

	if psd is not None:
		lalfft.XLALWhitenCOMPLEX16FrequencySeries(fseries, psd)
	fseries = templates.add_quadrature_phase(fseries, working_length)

	#
	# transform template to time domain
	#

	lalfft.XLALCOMPLEX16FreqTimeFFT(tseries, fseries, revplan)

	#
	# extract the portion to be used for filtering
	#

	#data = tseries.data[-length_max:]

	filter_len = min(length_max, 1.0 * len(hplus.data.data))
	data = tseries.data[-filter_len:]

	#pdb.set_trace()
	amp, phase = calc_amp_phase(numpy.imag(data), numpy.real(data))
	return amp, phase

def gen_whitened_amp_phase(psd, m1, m2, sampleRate, flower, is_freq_whiten, working_length, working_duration, length_max, spin1x=0., spin1y=0., spin1z=0., spin2x=0., spin2y=0., spin2z=0.):

        # generate the waveform
	# FIXME: waveform approximant should not be fixed.	
	hp,hc = lalsimulation.SimInspiralChooseTDWaveform(  0,				# reference phase, phi ref
		    				    1./sampleRate,			# delta T
						    m1*lal.MSUN_SI,			# mass 1 in kg
						    m2*lal.MSUN_SI,			# mass 2 in kg
						    spin1x, spin1y, spin1z,		# Spin 1 x, y, z
						    spin2x, spin2y, spin2z,		# Spin 2 x, y, z
						    flower,				# Lower frequency
						    0,					# Reference frequency 40?
						    1.e6*lal.PC_SI,			# r - distance in M (convert to MPc)
						    0,					# inclination
						    0,0,				# Lambda1, lambda2
						    None,				# Waveflags
						    None,				# Non GR parameters
						    0,7,				# Amplitude and phase order 2N+1
						    lalsimulation.GetApproximantFromString("SpinTaylorT4"))


	# The following code will plot the original autocorrelation function
	#ori_amp, ori_phase = calc_amp_phase(hc.data.data, hp.data.data)
	#ori_wave = ori_amp * numpy.exp(1j * ori_phase)
       	#auto_ori = numpy.zeros(working_length * 1, dtype=numpy.cdouble)
        #auto_ori[-len(ori_wave):] = ori_wave
	#auto_bank_ori = normalized_crosscorr(auto_ori, auto_ori, 201)

	#import matplotlib.pyplot as plt
	#axis_x = numpy.linspace(0, len(phase), len(phase))
	#plt.plot(axis_x, phase)
	#plt.show()


	if is_freq_whiten:
		hp.data.data *= lefttukeywindow(hp.data.data, samps = int(4 * sampleRate / flower))
		lalwhiten_amp, lalwhiten_phase = lalwhiten(psd, hp, working_length, working_duration, sampleRate, length_max)
		return lalwhiten_amp, lalwhiten_phase

	else:

		amp, phase = calc_amp_phase(hc.data.data, hp.data.data)
		amp = amp /numpy.sqrt(numpy.dot(amp,numpy.conj(amp))); 

		f = numpy.gradient(phase)/(2.0*numpy.pi * (1.0/sampleRate))
		cleanFreq(f,flower)


		# This is the key of SPIIR method. The whitening in
		# frequency domain
		# can be achieved in time domain

		if psd is not None:
       			fsampling = numpy.arange(len(psd.data)) * psd.deltaF
			# FIXME: which interpolation method should we choose,
			# currently we are using linear interpolation, splrep 
			# will generate negative values at the edge of psd. pchip is too slow
			#psd_interp = interpolate.splrep(fsampling, psd.data)
			#newpsd = interpolate.splev(f, psd_interp)
			#newpsd = interpolate.pchip_interpolate(fsampling, psd.data, f)
			psd_interp = interpolate.interp1d(fsampling, psd.data)
			newpsd = psd_interp(f)
			amp[0:len(f)] /= newpsd ** 0.5

		# The following code will plot the original (phase,amp) and
		# whitened (phase,amp)
		#import matplotlib.pyplot as plt
		#axis_x = numpy.linspace(0, len(phase), len(phase))
		#f, axarr = plt.subplots(2, sharex = True)
		#axarr[0].plot(axis_x, amp)
		#axarr[0].plot(axis_x, lalwhiten_amp)
		#axarr[1].plot(axis_x, phase)
		#axarr[1].plot(axis_x, lalwhiten_phase)
		#plt.show()

		#f2, axarr2 = plt.subplots(2, sharex = True)
		#axarr2[0].plot(axis_x, amp/lalwhiten_amp)
		#axarr2[1].plot(axis_x, phase - lalwhiten_phase)
		#plt.show()

		return amp, phase


class Bank(object):
	def __init__(self, logname = None):
		self.template_bank_filename = None
		self.bank_filename = None
		self.logname = logname
		self.sngl_inspiral_table = None
		self.sample_rates = []
		self.A = {} 
		self.B = {} 
		self.D = {}
		self.autocorrelation_bank = None
		self.autocorrelation_mask = None
		self.sigmasq = []
		self.matches = []
		self.flower = None
		self.epsilon = None

	def build_from_tmpltbank(self, filename, sampleRate = None, padding = 1.1, epsilon = 0.02, alpha = .99, beta = 0.25, pnorder = 4, flower = 40, all_psd = None, autocorrelation_length = 201, downsample = False, req_min_match = 0.0, verbose = False, contenthandler = DefaultContentHandler):
		# Open template bank file
		self.template_bank_filename = filename
		tmpltbank_xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)
	        sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(tmpltbank_xmldoc)
		fFinal = max(sngl_inspiral_table.getColumnByName("f_final"))
		self.flower = flower
		self.epsilon = epsilon
		self.alpha = alpha
		self.beta = beta

		epsilon_increment = 0.001

		if sampleRate is None:
			sampleRate = int(2**(numpy.ceil(numpy.log2(fFinal)+1)))

		if verbose: 
			logging.basicConfig(format='%(asctime)s %(message)s', level = logging.DEBUG)
			logging.info("fmin = %f,f_fin = %f, samplerate = %f" % (flower, fFinal, sampleRate))

	        Amat = {}
       		Bmat = {}
	        Dmat = {}

		# Check parity of autocorrelation length
		if autocorrelation_length is not None:
			if not (autocorrelation_length % 2):
				raise ValueError, "autocorrelation_length must be odd (got %d)" % autocorrelation_length
			self.autocorrelation_bank = numpy.zeros((len(sngl_inspiral_table), autocorrelation_length), dtype = "cdouble")
			self.autocorrelation_mask = compute_autocorrelation_mask( self.autocorrelation_bank )
		else:
			self.autocorrelation_bank = None
			self.autocorrelation_mask = None
		
		#This occasionally breaks with certain template banks
		#Can just specify a certain instrument as a hack fix
		psd = all_psd[sngl_inspiral_table[0].ifo]
		#psd = all_psd['H1']
		# smooth and create an interp object
		#psd_interp = smooth_and_interp(psd)


		# working f_low to actually use for generating the waveform
		working_f_low_extra_time, working_f_low = cbc_template_fir.joliens_function(flower, sngl_inspiral_table)

		# FIXME: This is a hack to calculate the maximum length of given table, we 
		# know that working_f_low_extra_time is about 1/10 of the maximum duration
		length_max = working_f_low_extra_time * 10 * sampleRate

		# Add 32 seconds to template length for PSD ringing, round up to power of 2 count of samples
		working_length = templates.ceil_pow_2(length_max + round((32.0 + working_f_low_extra_time) * sampleRate))
		working_duration = float(working_length) / sampleRate

		# Smooth the PSD and interpolate to required resolution
		if psd is not None:
			psd = cbc_template_fir.condition_psd(psd, 1.0 / working_duration, minfs = (working_f_low, flower), maxfs = (sampleRate / 2.0 * 0.90, sampleRate / 2.0))
			# This is to avoid nan amp when whitening the amp 
			tmppsd = psd.data
			tmppsd[numpy.isinf(tmppsd)] = 1.0
			psd.data = tmppsd

		if verbose:
			logging.info("condition of psd finished")

		#
		# condition the template if necessary (e.g. line up IMR
		# waveforms by peak amplitude)
		#

	
		original_epsilon = epsilon
	        for tmp, row in enumerate(sngl_inspiral_table):
		    spiir_match = -1
		    epsilon = original_epsilon

		    while(spiir_match < req_min_match and epsilon > 0):
			m1 = row.mass1
			m2 = row.mass2
			
			spin1x = row.spin1x
			spin1y = row.spin1y
			spin1z = row.spin1z

			spin2x = row.spin2x
			spin2y = row.spin2y
			spin2z = row.spin2z

			fFinal = row.f_final

			amp, phase = gen_whitened_amp_phase(psd, m1, m2, sampleRate, flower, 1, working_length, working_duration, length_max, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
			
			iir_type_flag = 1
	              	# make the iir filter coeffs
       	        	a1, b0, delay = spawaveform.iir(amp, phase, epsilon, alpha, beta, padding, iir_type_flag)
	
               		# get the chirptime (nearest power of two)
                	length = int(2**numpy.ceil(numpy.log2(amp.shape[0]+autocorrelation_length)))

                	# get the IIR response
                	u = spawaveform.iirresponse(length, a1, b0, delay)

	                #u_pad = numpy.zeros(length * 1, dtype=numpy.cdouble)
	                #u_pad[-len(u):] = u

			u_rev = u[::-1]
	                u_rev_pad = numpy.zeros(length * 1, dtype=numpy.cdouble)
	                u_rev_pad[-len(u_rev):] = u_rev

	                norm_u = 1.0/numpy.sqrt(2.0)*((u_rev_pad * numpy.conj(u_rev_pad)).sum()**0.5)
	                u_rev_pad /= norm_u
			#u_pad /= norm_u

	                # normalize the iir coefficients
	                b0 /= norm_u

	                # get the original waveform
	                h = amp * numpy.exp(1j * phase)
	                h_pad = numpy.zeros(length * 1, dtype=numpy.double)
	                h_pad[-len(h):] = h.real

			norm_h = numpy.sqrt(abs(numpy.dot(h_pad, numpy.conj(h_pad))))
	                h_pad /= norm_h
			self.sigmasq.append(1.0 * norm_h / sampleRate)


        	        # This is actually the cross correlation between the original waveform and this approximation
			self.autocorrelation_bank[tmp,:] = normalized_crosscorr(h_pad, u_rev_pad, autocorrelation_length)

			
			# compute the SNR
			spiir_match = abs(numpy.dot(u_rev_pad, h_pad))
			if(abs(original_epsilon - epsilon) < 1e-5):
			    original_match = spiir_match
			    original_filters = len(a1)

			if(spiir_match < req_minimum_match):
				epsilon -= epsilon_increment

		    self.matches.append(spiir_match)
		    self.sigmasq.append(1.0 * norm_h / sampleRate)

		    if verbose:
			    logging.info("template %4.0d/%4.0d, m1 = %10.6f m2 = %10.6f, epsilon = %1.4f:  %4.0d filters, %10.8f match. original_eps = %1.4f: %4.0d filters, %10.8f match" % (tmp+1, len(sngl_inspiral_table), m1,m2, epsilon, len(a1), spiir_match, original_epsilon, original_filters, original_match))	
		    # get the filter frequencies
		    fs = -1. * numpy.angle(a1) / 2 / numpy.pi # Normalised freqeuncy
		    a1dict = {}
		    b0dict = {}
		    delaydict = {}

		    if downsample:
			    min_M = 1
			    # iterate over the frequencies and put them in the right downsampled bin
			    for i, f in enumerate(fs):
				    M = int(max(min_M, 2**-numpy.ceil(numpy.log2(f * 2.0 * padding)))) # Decimation factor
				    a1dict.setdefault(sampleRate/M, []).append(a1[i]**M)
				    newdelay = numpy.ceil((delay[i]+1)/(float(M)))
				    b0dict.setdefault(sampleRate/M, []).append(b0[i]*M**0.5*a1[i]**(newdelay*M-delay[i]))
				    delaydict.setdefault(sampleRate/M, []).append(newdelay)
			    #logging.info("sampleRate %4.0d, filter %3.0d, M %2.0d, f %10.9f, delay %d, newdelay %d" % (sampleRate, i, M, f, delay[i], newdelay))
		    else:
			    a1dict[int(sampleRate)] = a1
			    b0dict[int(sampleRate)] = b0
			    delaydict[int(sampleRate)] = delay

		    # store the coeffs
		    for k in a1dict.keys():
			    Amat.setdefault(k, []).append(a1dict[k])
			    Bmat.setdefault(k, []).append(b0dict[k])
			    Dmat.setdefault(k, []).append(delaydict[k])


		max_rows = max([len(Amat[rate]) for rate in Amat.keys()])
		for rate in Amat.keys():
			self.sample_rates.append(rate)
			# get ready to store the coefficients
			max_len = max([len(i) for i in Amat[rate]])
			DmatMin = min([min(elem) for elem in Dmat[rate]])
			DmatMax = max([max(elem) for elem in Dmat[rate]])
			if verbose:
				logging.info("rate %d, dmin %d, dmax %d, max_row %d, max_len %d" % (rate, DmatMin, DmatMax, max_rows, max_len))

			self.A[rate] = numpy.zeros((max_rows, max_len), dtype=numpy.complex128)
			self.B[rate] = numpy.zeros((max_rows, max_len), dtype=numpy.complex128)
			self.D[rate] = numpy.zeros((max_rows, max_len), dtype=numpy.int)
			self.D[rate].fill(DmatMin)

			for i, Am in enumerate(Amat[rate]): self.A[rate][i,:len(Am)] = Am
			for i, Bm in enumerate(Bmat[rate]): self.B[rate][i,:len(Bm)] = Bm
			for i, Dm in enumerate(Dmat[rate]): self.D[rate][i,:len(Dm)] = Dm


	def write_to_xml(self, filename, contenthandler = DefaultContentHandler, write_psd = False, verbose = False):
		"""Write SPIIR banks to a LIGO_LW xml file."""
		# FIXME: does not support clipping and write psd.

		# Create new document
		xmldoc = ligolw.Document()
		lw = ligolw.LIGO_LW()

		# set up root for this sub bank
		root = ligolw.LIGO_LW(Attributes({u"Name": u"gstlal_iir_bank_Bank"}))
		lw.appendChild(root)

		# Open template bank file
		tmpltbank_xmldoc = utils.load_filename(self.template_bank_filename, contenthandler = contenthandler, verbose = verbose)

		# Get sngl inspiral table
		sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(tmpltbank_xmldoc)

		# put the bank table into the output document
		new_sngl_table = lsctables.New(lsctables.SnglInspiralTable)
		for row in sngl_inspiral_table:
			new_sngl_table.append(row)

		root.appendChild(new_sngl_table)

		root.appendChild(param.new_param('template_bank_filename', types.FromPyType[str], self.template_bank_filename))
		root.appendChild(param.new_param('sample_rate', types.FromPyType[str], sample_rates_array_to_str(self.sample_rates)))
		root.appendChild(param.new_param('flower', types.FromPyType[float], self.flower))
		root.appendChild(param.new_param('epsilon', types.FromPyType[float], self.epsilon))
		root.appendChild(param.new_param('alpha', types.FromPyType[float], self.alpha))
		root.appendChild(param.new_param('beta', types.FromPyType[float], self.beta))

		# FIXME:  ligolw format now supports complex-valued data
		root.appendChild(array.from_array('autocorrelation_bank_real', self.autocorrelation_bank.real))
		root.appendChild(array.from_array('autocorrelation_bank_imag', self.autocorrelation_bank.imag))
		root.appendChild(array.from_array('autocorrelation_mask', self.autocorrelation_mask))
		root.appendChild(array.from_array('sigmasq', numpy.array(self.sigmasq)))
		root.appendChild(array.from_array('matches', numpy.array(self.matches)))

		# put the SPIIR coeffs in
		for rate in self.A.keys():
			root.appendChild(array.from_array('a_%d' % (rate), repack_complex_array_to_real(self.A[rate])))
			root.appendChild(array.from_array('b_%d' % (rate), repack_complex_array_to_real(self.B[rate])))
			root.appendChild(array.from_array('d_%d' % (rate), self.D[rate]))

		# add top level LIGO_LW to document
		xmldoc.appendChild(lw)

		# Write to file
		utils.write_filename(xmldoc, filename, gz = filename.endswith('.gz'), verbose = verbose)


	def read_from_xml(self, filename, contenthandler = DefaultContentHandler, verbose = False):

		self.set_bank_filename(filename)
		# Load document
		xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)

		for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_iir_bank_Bank"):
			# FIXME: not Read sngl inspiral table

			# Read root-level scalar parameters
			self.template_bank_filename = param.get_pyvalue(root, 'template_bank_filename')
			if os.path.isfile(self.template_bank_filename):
				pass
			else:

				# FIXME teach the trigger generator to get this information a better way
				self.template_bank_filename = tempfile.NamedTemporaryFile(suffix = ".gz", delete = False).name

				self.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(root)
				tmpxmldoc = ligolw.Document()
				tmpxmldoc.appendChild(ligolw.LIGO_LW()).appendChild(self.sngl_inspiral_table)
				utils.write_filename(tmpxmldoc, self.template_bank_filename, gz = True, verbose = verbose)
				tmpxmldoc.unlink()	# help garbage collector


			self.autocorrelation_bank = array.get_array(root, 'autocorrelation_bank_real').array + 1j * array.get_array(root, 'autocorrelation_bank_imag').array
			self.autocorrelation_mask = array.get_array(root, 'autocorrelation_mask').array
			self.sigmasq = array.get_array(root, 'sigmasq').array

			# Read the SPIIR coeffs
			self.sample_rates = [int(r) for r in param.get_pyvalue(root, 'sample_rate').split(',')]
			for sr in self.sample_rates:
				self.A[sr] = repack_real_array_to_complex(array.get_array(root, 'a_%d' % (sr,)).array)
				self.B[sr] = repack_real_array_to_complex(array.get_array(root, 'b_%d' % (sr,)).array)
				self.D[sr] = array.get_array(root, 'd_%d' % (sr,)).array



	def get_rates(self, contenthandler = DefaultContentHandler, verbose = False):
		bank_xmldoc = utils.load_filename(self.bank_filename, contenthandler = contenthandler, verbose = verbose)
		return [int(r) for r in param.get_pyvalue(bank_xmldoc, 'sample_rate').split(',')]

	# FIXME: remove set_bank_filename when no longer needed
	# by trigger generator element
	def set_bank_filename(self, name):
		self.bank_filename = name

def load_iirbank(filename, snr_threshold, contenthandler = XMLContentHandler, verbose = False):
	tmpltbank_xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)

	bank = Bank.__new__(Bank)
	bank = Bank(
		tmpltbank_xmldoc,
		snr_threshold,
		verbose = verbose
	)

	bank.set_bank_filename(filename)
	return bank


def get_maxrate_from_xml(filename, contenthandler = DefaultContentHandler, verbose = False):
	xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)

	for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_iir_bank_Bank"):

		sample_rates = [int(r) for r in param.get_pyvalue(root, 'sample_rate').split(',')]
	
	return max(sample_rates)



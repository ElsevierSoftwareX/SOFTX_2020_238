
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
import cmath
from scipy import integrate
from scipy import interpolate
import math
import csv
import logging
import tempfile

import lal
import lalsimulation
import matplotlib
matplotlib.use('Agg')
from glue.ligolw import ligolw, lsctables, array, param, utils, types
from pylal import datatypes as laltypes
from pylal import lalfft
from gstlal.pipeio import repack_complex_array_to_real, repack_real_array_to_complex
from gstlal import cbc_template_fir
from gstlal import chirptime
from gstlal import templates
import random
import pdb
from gstlal.optimize_mf import OptimizerIIR

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
#           information of a waveform. NOW we use the standard lalsimulation
#             to obtain amp and phase.


class XMLContentHandler(ligolw.LIGOLWContentHandler):
    pass

def tukeywindow(data, samps = 200.):
    assert (len(data) >= 2 * samps) # make sure that the user is requesting something sane
    tp = float(samps) / len(data)
    return lal.CreateTukeyREAL8Window(len(data), tp).data.data


def ceil_pow_2(x):
    """ Return the smallest power of 2 that is larger than x.

    Parameters
    ----------
    x : int or float
    A positive integer number

    Returns
    -------
    Number : int
    The smallest power of 2 that is larger than x
    
    Example
    -------
    >>> ceil_pow_2(25)
    32
    """
    x = int(math.ceil(x))
    x -= 1
    n = 1
    while n and (x & (x+1)):
        x |= x >> n
        n <<= 1
    return x + 1

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
    
    n_temp = len(a)
    if autocorrelation_length > n_temp:
        raise ValueError, "autocorrelation length (%d) cannot be larger than the template length (%d)" % (autocorrelation_length, n_temp)
    if n_temp != len(b):
        raise ValueError, "len(a) should be the same as len(b)"
    
    af = scipy.fft(a)
    bf = scipy.fft(b)
    corr = scipy.ifft( af * numpy.conj( bf ))
    abs_corr = abs(corr)
    max_idx = numpy.where(abs_corr == max(abs_corr))[0][0]
    
    half_len = autocorrelation_length//2
    auto_bank = numpy.zeros(autocorrelation_length, dtype = 'cdouble')
    if max_idx == 0:
        auto_bank[::-1] = numpy.concatenate((corr[-half_len:],corr[:half_len+1]))
        auto_bank /= corr[max_idx]
    else:    
        print "Warning: max of autocorrelation happen at position [%d]" % max_idx
        temp_idx = (n_temp-1)//2
        temp_corr = numpy.concatenate((corr[-temp_idx:], corr[:-temp_idx]))
        max_idx = numpy.where(abs(temp_corr) == max(abs(temp_corr)))[0][0]
        
        if max_idx-half_len<0 or max_idx+half_len+1>n_temp:
            raise ValueError, "cannot generate cross-correlation of the given (autocorrelation) length, insufficient data"
        else:
            auto_bank[::-1] = temp_corr[max_idx-half_len:max_idx+half_len+1]
            auto_bank /= temp_corr[max_idx]
    
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


def matched_filt(template, strain, sampleRate = 4096.0):

    '''
    matched filtering using numpy fft
    template: complex
    data: time, real value
    the template is produced from using the gen_whitened_fir_template or gen_whitened_spiir_template_and_reconstructed_waveform.
    The unit of template is s^-1/2. the data is generated from gstlal_whiten where the unit is dimensionless.
    It needs to be normalized so the unit is s^-1/2, same as the template for unit consistency.
    '''
    # the data is generated from gstlal_play --whiten where the unit is dimensionless.
    # needs to be normalized so the unit is s^-1/2, i.e., *1/sqrt(dt). 
    time = strain[:, 0]
    data = strain[:, 1]
    data /= numpy.sqrt(2.0/sampleRate)
    # if the template inner product is normalized to 2, need to convert its unit by doing the following:
    #template /= numpy.sqrt(2.0/sampleRate)
    # need to extend to 2 times to avoid cyclic artifacts
    working_length = max(len(template), len(data)) * 2
    template_len = len(template)
    fs = float(sampleRate)
    df = 1.0/ (working_length/ fs)
    template_pad = numpy.zeros(working_length, dtype = "cdouble")
    template_pad[:len(template)] = template
    data_pad = numpy.zeros(working_length, dtype = "double")
    data_pad[:len(data)] = data

    data_pad *= tukeywindow(data_pad, samps = 32.)

    data_fft = numpy.fft.fft(data_pad)/ fs # times dt
    template_fft = numpy.fft.fft(template_pad)/ fs

    snr_fft = data_fft * template_fft.conjugate()
    snr_time = 2 * numpy.fft.ifft(snr_fft) * fs # times df then the unit is dimensionless, default ifft has the output scaled by 1/N 
    sigmasq = (template_fft * template_fft.conjugate()).sum() * df
    sigma = numpy.sqrt(abs(sigmasq))
    #pdb.set_trace()
    # normalize snr
    snr_time /= sigma
    # need to shift the SNR because cross-correlation FFT plays integration on cyclic template_pad,
    # so the first value of snr_time is out(0) = data(0:) times template(0:), we actually need out(0) = data(0:) times template(-1)
    # for the first value where N is the len of template, so that out(N-1) = data(0:) times template (-N:)
    roll_len = template_len - 1 # note here is N - 1
    snr_time = numpy.roll(snr_time, roll_len)
    # find the time and SNR value at maximum:
    SNR = abs(snr_time)
    indmax = numpy.argmax(SNR)
    try:
    	timemax = time[indmax]
    except:
	    raise ValueError("max SNR is outside the data, need to collect more data for a more correct SNR estimation")
    return SNR, sigma, indmax, timemax

# working length, length_max deprecated
def M_chi2_readline(flower=30., sampleRate=2048.):
    fh = open('m1_m2_mc_1xyz_2xyz.dat')
    line = fh.readline()
    #m1, m2, mc, s1x, s1y, s1z, s2x, s2y, s2z = line.strip().split(',')
    params = [float(pa) for pa in line.strip().split(',')]
    
    tchirp = lalsimulation.SimInspiralChirpTimeBound(flower, params[0] * lal.MSUN_SI, params[1] * lal.MSUN_SI, 0., 0.)

    # FIXME: This is a hack to calculate the maximum length of given table, we 
    # know that working_f_low_extra_time is about 1/10 of the maximum duration
    working_f_low_extra_time = .1 * tchirp + 1.0
    length_max = working_f_low_extra_time * 10 * sampleRate

    # Add 32 seconds to template length for PSD ringing, round up to power of 2 count of samples
    working_length = templates.ceil_pow_2(length_max + round((32.0 + working_f_low_extra_time) * sampleRate))
    working_duration = float(working_length) / sampleRate
    
    amp, phase = gen_whitened_amp_phase(None, params[0], params[1], sampleRate, flower, 0, working_length, working_duration, length_max, params[3], params[4], params[5], params[6], params[7], params[8] )
    fh.close()
    
    print len(amp)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(amp)), amp*numpy.cos(phase),'b')
    plt.show()


def gen_template_working_state(sngl_inspiral_table, f_low = 30., sampleRate = 2048.):
    # Some input checking to avoid incomprehensible error messages
    if not sngl_inspiral_table:
        raise ValueError("template list is empty")
    if f_low < 0.:
        raise ValueError("f_low must be >= 0.: %s" % repr(f_low))

    # working f_low to actually use for generating the waveform.  pick
    # template with lowest chirp mass, compute its duration starting
    # from f_low;  the extra time is 10% of this plus 3 cycles (3 /
    # f_low);  invert to obtain f_low corresponding to desired padding.
    # NOTE:  because SimInspiralChirpStartFrequencyBound() does not
    # account for spin, we set the spins to 0 in the call to
    # SimInspiralChirpTimeBound() regardless of the component's spins.
    template = min(sngl_inspiral_table, key = lambda row: row.mchirp)
    tchirp = lalsimulation.SimInspiralChirpTimeBound(f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI, 0., 0.)
    working_f_low = lalsimulation.SimInspiralChirpStartFrequencyBound(1.1 * tchirp + 3. / f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI)

    # FIXME: This is a hack to calculate the maximum length of given table, we 
    # know that working_f_low_extra_time is about 1/10 of the maximum duration
    working_f_low_extra_time = .1 * tchirp + 1.0
    length_max = int(round(working_f_low_extra_time * 10 * sampleRate))

    # Add 32 seconds to template length for PSD ringing, round up to power of 2 count of samples, this is for psd length to whiten the template later.
    working_length = templates.ceil_pow_2(length_max + round((16.0 + working_f_low_extra_time) * sampleRate))
    working_duration = float(working_length) / sampleRate

    working_state = {}
    working_state["working_f_low"] = working_f_low
    working_state["working_length"] = working_length
    working_state["working_duration"] = working_duration
    working_state["length_max"] = length_max
    return working_state


def M_chi2(flower=30., sampleRate=2048.):
    
    #m1, m2, mc, s1x, s1y, s1z, s2x, s2y, s2z = line.strip().split(',')
    params = numpy.loadtxt('test.dat')

    temp_list = []
    
    for ii in range(len(params[:,0])):
        tchirp = lalsimulation.SimInspiralChirpTimeBound(flower, params[ii, 0] * lal.MSUN_SI, params[ii, 1] * lal.MSUN_SI, 0., 0.)

        # FIXME: This is a hack to calculate the maximum length of given table, we 
        # know that working_f_low_extra_time is about 1/10 of the maximum duration
        working_f_low_extra_time = .1 * tchirp + 1.0
        length_max = working_f_low_extra_time * 10 * sampleRate

        # Add 32 seconds to template length for PSD ringing, round up to power of 2 count of samples
        working_length = templates.ceil_pow_2(length_max + round((32.0 + working_f_low_extra_time) * sampleRate))
        working_duration = float(working_length) / sampleRate
    
        amp, phase = gen_whitened_amp_phase(None, params[ii, 0], params[ii, 1], sampleRate, flower, 0, working_length, working_duration, length_max, params[ii, 3], params[ii, 4], params[ii, 5], params[ii, 6], params[ii, 7], params[ii, 8] )
        phase = phase - phase[-1]
        h_temp = amp*numpy.exp(phase*1j)
        h_temp /= numpy.sqrt(numpy.sum(amp*amp))
        temp_list.append(h_temp)
        # print len(amp)
    
    m_chi2 = numpy.zeros( (len(temp_list),len(temp_list)), dtype=complex )
    for ii in range(len(temp_list)):
        for jj in range(len(temp_list)):
            minlen = min(len(temp_list[ii][:]), len(temp_list[jj][:]))
            m_chi2[ii, jj] = numpy.dot( temp_list[ii][-minlen:], numpy.conj(temp_list[jj][-minlen:]))
    
    import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(range(len(temp_list)), numpy.abs(m_chi2[0,:]),'b')
    ax = plt.figure().gca()
    im = ax.imshow(numpy.abs(m_chi2))
    plt.colorbar(im)
    plt.show()

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

def lalwhitenFD_and_convert2TD(psd, fseries, sampleRate, working_state, flower):

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
    revplan = lalfft.XLALCreateReverseCOMPLEX16FFTPlan(working_state["working_length"], 1)
    tseries = laltypes.COMPLEX16TimeSeries(
        data = numpy.zeros((working_state["working_length"],), dtype = "cdouble")
    )


    #
    # whiten and add quadrature phase ("sine" component)
    #

    if psd is not None:
        lalfft.XLALWhitenCOMPLEX16FrequencySeries(fseries, psd)
    fseries = templates.add_quadrature_phase(fseries, working_state["working_length"])

    #
    # transform template to time domain
    #

    lalfft.XLALCOMPLEX16FreqTimeFFT(tseries, fseries, revplan)


    data = tseries.data

    # FIXME: need to condition for IMR wave templates
    data *= tukeywindow(data, samps = 32)
    # This is to normalize whitened template so it = h_{whitened at 1MPC}(t)
    # NOTE: because
    # XLALWhitenCOMPLEX16FrequencySeries() computed
    #
    # \tilde{h}'_{k} = \sqrt{2 \Delta f} \tilde{h}_{k} / \sqrt{S_{k}}
    # need to devide the time domain whitened waveform by \sqrt{2 \Delta f}
    data /= numpy.sqrt(2./working_state["working_duration"])

    #pdb.set_trace()    
    return data

# a modification from the cbc_template_fir.generate_templates
def gen_whitened_fir_template(template_table, approximant, irow, psd, f_low, time_slices, autocorrelation_length = 201, sampleRate = 4096., verbose = False):

    """!
    Generate a bank of templates, which are
     (3) whitened with the given psd.
    """
    sample_rate_max = sampleRate
    duration = max(time_slices['end'])
    length_max = int(round(duration * sample_rate_max))

    # Some input checking to avoid incomprehensible error messages
    if not template_table:
        raise ValueError("template list is empty")
    if f_low < 0.:
        raise ValueError("f_low must be >= 0.: %s" % repr(f_low))

    # working f_low to actually use for generating the waveform.  pick
    # template with lowest chirp mass, compute its duration starting
    # from f_low;  the extra time is 10% of this plus 3 cycles (3 /
    # f_low);  invert to obtain f_low corresponding to desired padding.
    # NOTE:  because SimInspiralChirpStartFrequencyBound() does not
    # account for spin, we set the spins to 0 in the call to
    # SimInspiralChirpTimeBound() regardless of the component's spins.
    template = min(template_table, key = lambda row: row.mchirp)
    tchirp = lalsimulation.SimInspiralChirpTimeBound(f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI, 0., 0.)
    working_f_low = lalsimulation.SimInspiralChirpStartFrequencyBound(1.1 * tchirp + 3. / f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI)

    # Add duration of PSD to template length for PSD ringing, round up to power of 2 count of samples
    working_length = templates.ceil_pow_2(length_max + round(1./psd.deltaF * sample_rate_max))
    working_duration = float(working_length) / sample_rate_max

    # Smooth the PSD and interpolate to required resolution
    if psd is not None:
        psd = cbc_template_fir.condition_psd(psd, 1.0 / working_duration, minfs = (working_f_low, f_low), maxfs = (sample_rate_max / 2.0 * 0.90, sample_rate_max / 2.0))

    if verbose:
        logging.basicConfig(format='%(asctime)s %(message)s', level = logging.DEBUG)
        logging.info("working_f_low %f, working_duration %f, flower %f, sampleRate %f" % (working_f_low, working_duration, f_low, sampleRate))
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

    # Check parity of autocorrelation length
    if autocorrelation_length is not None:
        if not (autocorrelation_length % 2):
            raise ValueError, "autocorrelation_length must be odd (got %d)" % autocorrelation_length
        autocorrelation_bank = numpy.zeros(autocorrelation_length, dtype = "cdouble")
    else:
        autocorrelation_bank = None

    # Multiply by 2 * length of the number of sngl_inspiral rows to get the sine/cosine phases.
    template_bank = [numpy.zeros((2 * len(template_table), int(round(rate*(end-begin)))), dtype = "double") for rate,begin,end in time_slices]

    # Store the original normalization of the waveform.  After
    # whitening, the waveforms are normalized.  Use the sigmasq factors
    # to get back the original waveform.
    sigmasq = []

    # Generate each template, downsampling as we go to save memory
    max_ringtime = max([chirptime.ringtime(row.mass1*lal.MSUN_SI + row.mass2*lal.MSUN_SI, chirptime.overestimate_j_from_chi(max(row.spin1z, row.spin2z))) for row in template_table])
    row = template_table[irow]
    if verbose:
        print "generating template %d/%d:  m1 = %g, m2 = %g, s1x = %g, s1y = %g, s1z = %g, s2x = %g, s2y = %g, s2z = %g, sample rate %d, working_duration %f" % (irow + 1, len(template_table), row.mass1, row.mass2, row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z, sample_rate_max, working_duration)

    #
    # generate "cosine" component of frequency-domain template.
    # waveform is generated for a canonical distance of 1 Mpc.
    #

    fseries = cbc_template_fir.generate_template(row, approximant, sample_rate_max, working_duration, f_low, sample_rate_max / 2., fwdplan = fwdplan, fworkspace = fworkspace)

    #
    # whiten and add quadrature phase ("sine" component)
    #

    if psd is not None:
        lalfft.XLALWhitenCOMPLEX16FrequencySeries(fseries, psd)
    fseries = templates.add_quadrature_phase(fseries, working_length)

    #
    # compute time-domain autocorrelation function
    #

    if autocorrelation_bank is not None:
        autocorrelation = templates.normalized_autocorrelation(fseries, revplan).data
        autocorrelation_bank[::-1] = numpy.concatenate((autocorrelation[-(autocorrelation_length // 2):], autocorrelation[:(autocorrelation_length // 2  + 1)]))

    #
    # transform template to time domain
    #

    lalfft.XLALCOMPLEX16FreqTimeFFT(tseries, fseries, revplan)

    data = tseries.data
    epoch_time = fseries.epoch.seconds + fseries.epoch.nanoseconds*1.e-9
    #
    # extract the portion to be used for filtering
    #


    #
    # condition the template if necessary (e.g. line up IMR
    # waveforms by peak amplitude)
    #

    if approximant in templates.gstlal_IMR_approximants:
        data, target_index = condition_imr_template(approximant, data, epoch_time, sample_rate_max, max_ringtime)
        # record the new end times for the waveforms (since we performed the shifts)
        row.end = laltypes.LIGOTimeGPS(float(target_index-(len(data) - 1.))/sample_rate_max)
    else:
        data *= tukeywindow(data, samps = 32)

    data = data[-length_max:]

    # This is to normalize whitened template so it = h_{whitened at 1MPC}(t)
    # NOTE: because
    # XLALWhitenCOMPLEX16FrequencySeries() computed
    #
    # \tilde{h}'_{k} = \sqrt{2 \Delta f} \tilde{h}_{k} / \sqrt{S_{k}}
    # need to devide the time domain whitened waveform by \sqrt{2 \Delta f}
    data /= numpy.sqrt(2./working_duration)

    #
    # normalize so that inner product of template with itself
    # is 2
    #
    
    #norm = abs(numpy.dot(data, numpy.conj(data)))
    #data *= cmath.sqrt(2 / norm)
    print "template length %d" % len(data)
    return data, autocorrelation_bank


def gen_whitened_spiir_template_and_reconstructed_waveform(sngl_inspiral_table, approximant, irow, psd, sampleRate = 4096, waveform_domain = "TD", epsilon = 0.02, epsilon_min = 0.0, alpha = .99, beta = 0.25, flower = 30, autocorrelation_length = 201, req_min_match = 0.99, verbose = False):
    
    working_state = gen_template_working_state(sngl_inspiral_table, flower, sampleRate = sampleRate)
    # Smooth the PSD and interpolate to required resolution
    if psd is not None:
        psd = cbc_template_fir.condition_psd(
                            psd, 
                            1.0 / working_state["working_duration"], 
                            minfs = (working_state["working_f_low"], flower), 
                            maxfs = (sampleRate / 2.0 * 0.90, sampleRate / 2.0)
                            )
 
    # This is to avoid nan amp when whitening the amp 
    #tmppsd = psd.data
    #tmppsd[numpy.isinf(tmppsd)] = 1.0
    #psd.data = tmppsd

    if verbose:
        logging.basicConfig(format='%(asctime)s %(message)s', level = logging.DEBUG)
        logging.info("condition of psd finished")
        logging.info("working_f_low %f, working_duration %f, flower %f, sampleRate %f" % (working_state["working_f_low"], working_state["working_duration"], flower, sampleRate))

    #
    # FIXME: condition the template if necessary (e.g. line up IMR
    # waveforms by peak amplitude)
    #

    original_epsilon = epsilon
    epsilon_increment = 0.001
    row = sngl_inspiral_table[irow]
    this_tchirp = lalsimulation.SimInspiralChirpTimeBound(flower, row.mass1 * lal.MSUN_SI, row.mass2 * lal.MSUN_SI, row.spin1z, row.spin2z)
    fFinal = row.f_final

    if verbose:
        logging.info("working_duration %f, chirp time %f, final frequency %f" % (working_state["working_duration"], this_tchirp, fFinal))

    amp, phase, data, data_full = gen_whitened_amp_phase(psd, approximant, waveform_domain, sampleRate, flower, working_state, row, is_frequency_whiten = 1, verbose = verbose)

    # This is to normalize whitened template so it = h_{whitened at 1MPC}(t)
    # NOTE: because
    # XLALWhitenCOMPLEX16FrequencySeries() computed
    #
    # \tilde{h}'_{k} = \sqrt{2 \Delta f} \tilde{h}_{k} / \sqrt{S_{k}}
    # need to devide the time domain whitened waveform by \sqrt{2 \Delta f}
    amp /= numpy.sqrt(2./working_state["working_duration"])

    iir_type_flag = 1
    spiir_match = -1
    n_filters = 0
    nround = 1

    while(spiir_match < req_min_match and epsilon > epsilon_min and n_filters < 2000):
        #a1, b0, delay, u_rev_pad, h_pad = gen_norm_spiir_coeffs(amp, phase, epsilon = epsilon)
        a1, b0, delay, u_rev_pad, h_pad = gen_spiir_coeffs(amp, phase, len(data_full), epsilon = epsilon)
            
        # compute the SNR
        # deprecated: spiir_match = abs(numpy.dot(u_rev_pad, numpy.conj(h_pad_real)))
        # the following definition is more close to the reality         
        norm_u = abs(numpy.dot(u_rev_pad, numpy.conj(u_rev_pad)))
        norm_h = abs(numpy.dot(h_pad, numpy.conj(h_pad)))

    	data_full_pad = numpy.zeros(len(u_rev_pad), dtype='cdouble')
	data_full_pad[-len(data_full):] = data_full
	norm_data_full = abs(numpy.dot(data_full_pad, numpy.conj(data_full_pad)))
	# overlap of spiir reconstructed waveform with cut template (spiir_template)
        spiir_match = abs(numpy.dot(u_rev_pad, numpy.conj(h_pad))/numpy.sqrt(norm_u * norm_h))
	# overlap of spiir reconstructed waveform with full template (data_full)
	u_full_match = abs(numpy.dot(u_rev_pad, numpy.conj(data_full_pad))/numpy.sqrt(norm_u * norm_data_full))
        n_filters = len(delay)

        if verbose:
            logging.info("number of rounds %d, epsilon %f, spiir overlap with cut template %f, spiir overlap with original template %f, number of filters %d" % (nround, epsilon, spiir_match, u_full_match, n_filters))

               
        if(abs(original_epsilon - epsilon) < 1e-5):
            original_match = spiir_match
            original_filters = len(a1)

        if(spiir_match < req_min_match):
            epsilon -= epsilon_increment

	nround += 1
    if verbose:
        logging.info("norm of the original template %f, norm of spiir template h_pad %f, norm of spiir response u_rev_pad %f" % (norm_data_full, norm_h, norm_u))

    # normalize the spiir response
    u_rev_pad = u_rev_pad * numpy.sqrt(norm_h / norm_u) * spiir_match
    return u_rev_pad, h_pad, data_full

def gen_lalsim_waveform(row, flower, sampleRate):
    # FIXME: currently only works for the non-spin or spin-aligned case,
    # the latest waveform for IMR is IMRPhenomD and SEOBR4_ROM
    if (row.mass1+row.mass2) <=4:
        approximant_string = "SpinTaylorT4"
    else:
        approximant_string = "IMRPhenomB"

    # NOTE: There is also ChooseFDWaveform. IMRPhenomB is FD
    hp,hc = lalsimulation.SimInspiralChooseTDWaveform(0,                # reference phase, phi ref
                                1./sampleRate,            # delta T
                            row.mass1*lal.MSUN_SI,            # mass 1 in kg
                            row.mass2*lal.MSUN_SI,            # mass 2 in kg
                            row.spin1x, row.spin1y, row.spin1z,        # Spin 1 x, y, z
                            row.spin2x, row.spin2y, row.spin2z,        # Spin 2 x, y, z
                            flower,                # Lower frequency
                            0,                    # Reference frequency 40?
                            1.e6*lal.PC_SI,            # r - distance in M (convert to MPc)
                            0,                    # inclination
                            0,0,                # Lambda1, lambda2
                            None,                # Waveflags
                            None,                # Non GR parameters
                            0,7,                # Amplitude and phase order 2N+1
                            lalsimulation.GetApproximantFromString(approximant_string))


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
    return hp, hc


def gen_whitened_amp_phase(psd, approximant, waveform_domain, sampleRate, flower, working_state, row, is_frequency_whiten = 1, verbose = False):
    """ Generates whitened waveform from given parameters and PSD, then returns the amplitude and the phase.
    
    Parameters
    ----------
    psd :
    Power spectral density
    sampleRate :
    Sampling rate in Hz
    flower :
    Low frequency cut-off
    is_freq_whiten :
    Whether perform the whitening in the frequency domain (if True),
    or perform the whitening in the time domain (otherwise).
    Time-domain whitening is quicker and better-conditioned, but less accurate.
    Use frequency-domain whitening by default.
    working_length :
    Number of samples pre-allocated for the template.
    working_duration :
    The period in seconds corresponding to working_length.
    length_max :
    Parameter for frequency-domain whitening.
    row: snglinspiral_table row include information:
    m1, m2:
        component mass
    spin1x, spin1y, spin1z :
    Spin parameters of compact object 1.
    spin2x, spin2y, spin2z :
    Spin parameters of compact object 2.

    Returns
    -------
    Amplitude : 
    The amplitude of the whitened template
    Phase :
    The phase of the whitened template
    """

    # prepare the working space for FD whitening
    fwdplan = lalfft.XLALCreateForwardREAL8FFTPlan(working_state["working_length"], 1)
    fworkspace = laltypes.COMPLEX16FrequencySeries(
        name = "template",
        epoch = laltypes.LIGOTimeGPS(0),
        f0 = 0.0,
        deltaF = 1.0 / working_state["working_duration"],
        data = numpy.zeros((working_state["working_length"]//2 + 1,), dtype = "cdouble")
    )


    if waveform_domain == "FD" and is_frequency_whiten == 1:
        #
        # generate "cosine" component of frequency-domain template.
        # waveform is generated for a canonical distance of 1 Mpc.
        #
        fseries = cbc_template_fir.generate_template(row, approximant, sampleRate, working_state["working_duration"], flower, sampleRate / 2., fwdplan = fwdplan, fworkspace = fworkspace)
    
        # whiten the FD waveform and transform it back to TD 
        data_full = lalwhitenFD_and_convert2TD(psd, fseries, sampleRate, working_state, flower)
        if verbose:
            logging.info("waveform chose from FD")

    elif waveform_domain == "TD" and is_frequency_whiten == 1:
	logging.error("TD waveform here not conditioned, caution to use.")
	# get the TD waveform
        hplus, hcross = gen_lalsim_waveform(row, flower, sampleRate)
        # transfomr the TD waveform to FD
        tmptdata = numpy.zeros(working_state["working_length"],)
        tmptdata[-hplus.data.length:] = hplus.data.data
    
    
        tmptseries = laltypes.REAL8TimeSeries(
            name = "template",
            epoch = laltypes.LIGOTimeGPS(0),
            f0 = 0.0,
            deltaT = 1.0 / sampleRate,
            sampleUnits = laltypes.LALUnit("strain"),
            data = tmptdata
        )
            
        lalfft.XLALREAL8TimeFreqFFT(fworkspace, tmptseries, fwdplan)
        tmpfseries = numpy.copy(fworkspace.data)
    
        fseries = laltypes.COMPLEX16FrequencySeries(
            name = "template",
            epoch = laltypes.LIGOTimeGPS(0),
            f0 = 0.0,
            deltaF = 1.0 / working_state["working_duration"],
            sampleUnits = laltypes.LALUnit("strain"),
            data = tmpfseries 
        )
        # whiten the FD waveform and transform it back to TD 
        data_full = lalwhitenFD_and_convert2TD(psd, fseries, sampleRate, working_state, flower)
        if verbose:
            logging.info("waveform chose from TD")
    else:
        # FIXME: the hp, hc are now in frequency domain. 
        # Need to transform them first into time domain to perform following whitening
        print >> sys.stderr, "Time domain whitening not supported"
        sys.exit()
 
        amp, phase = calc_amp_phase(hc.data.data, hp.data.data)
        amp = amp /numpy.sqrt(numpy.dot(amp,numpy.conj(amp))); 
 
        f = numpy.gradient(phase)/(2.0*numpy.pi * (1.0/sampleRate))
        cleanFreq(f,flower)


        # The whitening in frequency domain
        # can also be achieved in time domain,
        # when the frequency evolution is monotonic.
        # But the following are a bit obsolete.

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

        return amp, phase

    #
    # extract the portion to be used for filtering
    #
    hp, hc = gen_lalsim_waveform(row, flower, sampleRate)
    filter_len = min(working_state["length_max"], 1.0 * len(hp.data.data))
    # the time domain whitened data is chosen to be the same length as the original
    # template. The length_max from tchirp and the working_length are too
    # long, may cause too many spiir filters. 
    # FIXME: should we allow a little more time for low-frequency boundary?
    data = data_full[-filter_len:]
    amp_lalwhiten, phase_lalwhiten = calc_amp_phase(numpy.imag(data), numpy.real(data))

    if verbose:
        logging.info("original template length %d, cut to construct spiir coeffs %d" % (len(data_full), len(data)))

    return amp_lalwhiten, phase_lalwhiten, data, data_full

def gen_spiir_coeffs(amp, phase, data_full_len, padding = 1.3, epsilon = 0.02, alpha = .99, beta = 0.25, autocorrelation_length = 201, iir_type_flag = 1):
        # make the iir filter coeffs
        a1, b0, delay = spawaveform.iir(amp, phase, epsilon, alpha, beta, padding, iir_type_flag)
    
        # get the chirptime (nearest power of two)
        length = templates.ceil_pow_2(data_full_len+autocorrelation_length)

        # get the IIR response
        u = spawaveform.iirresponse(length, a1, b0, delay)

        u_pad = numpy.zeros(length * 1, dtype=numpy.cdouble)
        u_pad[-len(u):] = u

        u_rev = u[::-1]
        u_rev_pad = numpy.zeros(length * 1, dtype=numpy.cdouble)
        u_rev_pad[-len(u_rev):] = u_rev

        # get the original waveform
        h = amp * numpy.exp(1j * phase)
        h_pad = numpy.zeros(length * 1, dtype=numpy.cdouble)
        h_pad[-len(h):] = h

        return a1, b0, delay, u_rev_pad, h_pad
            

def gen_norm_spiir_coeffs(amp, phase, data_full_len, padding = 1.3, epsilon = 0.02, alpha = .99, beta = 0.25, autocorrelation_length = 201, iir_type_flag = 1):
        # make the iir filter coeffs
        a1, b0, delay = spawaveform.iir(amp, phase, epsilon, alpha, beta, padding, iir_type_flag)
    
	# pad the iir response to be nearest power of 2 of:
        length = templates.ceil_pow_2(data_full_len + autocorrelation_length)

        # get the IIR response
        u = spawaveform.iirresponse(length, a1, b0, delay)

        u_pad = numpy.zeros(length * 1, dtype=numpy.cdouble)
        u_pad[-len(u):] = u

        u_rev = u[::-1]
        u_rev_pad = numpy.zeros(length * 1, dtype=numpy.cdouble)
        u_rev_pad[-len(u_rev):] = u_rev

        # normalize the approximate waveform so its inner-product is 2
        norm_u = abs(numpy.dot(u_rev_pad, numpy.conj(u_rev_pad)))
        u_rev_pad *= cmath.sqrt(2 / norm_u)
        u_pad *= cmath.sqrt(2 / norm_u)

        # normalize the iir coefficients
        b0 *= cmath.sqrt(2 / norm_u)

        # get the original waveform
        h = amp * numpy.exp(1j * phase)
        h_pad = numpy.zeros(length * 1, dtype=numpy.cdouble)
        h_pad[-len(h):] = h

        # normalize the original waveform so its inner-product is 2
        norm_h = abs(numpy.dot(h_pad, numpy.conj(h_pad)))
        h_pad *= cmath.sqrt(2 / norm_h)
	#pdb.set_trace()
        return a1, b0, delay, u_rev_pad, h_pad
            
    

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

    def build_from_tmpltbank(self, filename, sampleRate = None, padding = 1.3, waveform_domain = "TD", epsilon = 0.02, epsilon_min = 0.0, alpha = .99, beta = 0.25, pnorder = 4, flower = 40, all_psd = None, autocorrelation_length = 201, downsample = False, req_min_match = 0.0, verbose = False, contenthandler = DefaultContentHandler):
        """
            Build SPIIR template bank from physical parameters, e.g. mass, spin.
            """
        
        # Open template bank file
        self.template_bank_filename = filename
        tmpltbank_xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)
        sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(tmpltbank_xmldoc)
        fFinal = max(sngl_inspiral_table.getColumnByName("f_final"))
        self.flower = flower
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta


        if sampleRate is None:
            sampleRate = int(2**(numpy.ceil(numpy.log2(fFinal)+1)))

        if verbose: 
            logging.basicConfig(format='%(asctime)s %(message)s', level = logging.DEBUG)
            logging.info("fmin = %f,f_fin = %f, samplerate = %f" % (flower, fFinal, sampleRate))


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


        working_state = gen_template_working_state(sngl_inspiral_table, flower, sampleRate = sampleRate)
        # Smooth the PSD and interpolate to required resolution
        if psd is not None:
            psd = cbc_template_fir.condition_psd(
                                psd, 
                                1.0 / working_state["working_duration"], 
                                minfs = (working_state["working_f_low"], flower), 
                                maxfs = (sampleRate / 2.0 * 0.90, sampleRate / 2.0)
                                )
            # This is to avoid nan amp when whitening the amp 
            #tmppsd = psd.data
            #tmppsd[numpy.isinf(tmppsd)] = 1.0
            #psd.data = tmppsd

        if verbose:
            logging.info("condition of psd finished")

        #
        # condition the template if necessary (e.g. line up IMR
        # waveforms by peak amplitude)
        #

        Amat = {}
        Bmat = {}
        Dmat = {}

        original_epsilon = epsilon
        epsilon_increment = 0.001
        for tmp, row in enumerate(sngl_inspiral_table):
            spiir_match = -1
            epsilon = original_epsilon
            n_filters = 0
                
            fFinal = row.f_final

            amp, phase, data, data_full = gen_whitened_amp_phase(psd, waveform_domain, sampleRate, flower, working_state, row, is_frequency_whiten = 1, verbose = verbose)
            
            iir_type_flag = 1
	    nround = 1

            while(spiir_match < req_min_match and epsilon > epsilon_min and n_filters < 2000):
                
                a1, b0, delay, u_rev_pad, h_pad = gen_norm_spiir_coeffs(amp, phase, len(data_full), epsilon = epsilon, alpha = alpha, beta = beta, padding = padding, iir_type_flag = iir_type_flag, autocorrelation_length = autocorrelation_length)
                    
                # compute the SNR
                # deprecated: spiir_match = abs(numpy.dot(u_rev_pad, numpy.conj(h_pad_real)))
                # the following definition is more close to the reality         
                spiir_match = abs(numpy.dot(u_rev_pad, numpy.conj(h_pad)))/2.0 # still need to use the spiir template

		# compute the expected SNR ratio compared to given the original template
		data_full_pad = numpy.zeros(len(u_rev_pad), dtype='cdouble')
		data_full_pad[-len(data_full):] = data_full
        	# normalize the original waveform so its inner-product is 2
        	norm_data_full = abs(numpy.dot(data_full_pad, numpy.conj(data_full_pad)))
        	data_full_pad *= cmath.sqrt(2 / norm_data_full)
		# overlap of spiir constructed waveform with original template
                u_full_match = abs(numpy.dot(u_rev_pad, numpy.conj(data_full_pad)))/2.0


 
                if(abs(original_epsilon - epsilon) < 1e-5):
                    original_match = spiir_match
                    original_filters = len(a1)

                if(spiir_match < req_min_match):
                    epsilon -= epsilon_increment

                n_filters = len(delay)
                if verbose:
            	    logging.info("number of rounds %d, epsilon %f, spiir overlap with cut template %f, spiir overlap with original template %f, number of filters %d" % (nround, epsilon, spiir_match, u_full_match, n_filters))
		nround += 1


            #if(spiir_match < req_min_match):
            #    opIIR = OptimizerIIR(length, a1, b0, delay)
            #    opIIR.setTemplate(opIIR.cnormalize(h_pad))
            #    opIIR.normalizeCoef()
            #    opIIR.runHierarchyLagOp(300)
            #    a1 = opIIR.a1
            #    b0 = opIIR.b0
            #    delay = opIIR.delay
            #    spiir_match = opIIR.innerProd(opIIR.template, opIIR._iir_sum_res)

            #
            # sigmasq = 2 \sum_{k=0}^{N-1} |\tilde{h}_{k}|^2 / S_{k} \Delta f
            #
            # XLALWhitenCOMPLEX16FrequencySeries() computed
            #
            # \tilde{h}'_{k} = \sqrt{2 \Delta f} \tilde{h}_{k} / \sqrt{S_{k}}
            #
            # and XLALCOMPLEX16FreqTimeFFT() computed
            #
            # h'_{j} = \Delta f \sum_{k=0}^{N-1} exp(2\pi i j k / N) \tilde{h}'_{k}
            #
            # therefore, "norm" is
            #
            # \sum_{j} |h'_{j}|^{2} = (\Delta f)^{2} \sum_{j} \sum_{k=0}^{N-1} \sum_{k'=0}^{N-1} exp(2\pi i j (k-k') / N) \tilde{h}'_{k} \tilde{h}'^{*}_{k'}
            #                       = (\Delta f)^{2} \sum_{k=0}^{N-1} \sum_{k'=0}^{N-1} \tilde{h}'_{k} \tilde{h}'^{*}_{k'} \sum_{j} exp(2\pi i j (k-k') / N)
            #                       = (\Delta f)^{2} N \sum_{k=0}^{N-1} |\tilde{h}'_{k}|^{2}
            #                       = (\Delta f)^{2} N 2 \Delta f \sum_{k=0}^{N-1} |\tilde{h}_{k}|^{2} / S_{k}
            #                       = (\Delta f)^{2} N sigmasq
            #
            # and \Delta f = 1 / (N \Delta t), so "norm" is
            #
            # \sum_{j} |h'_{j}|^{2} = 1 / (N \Delta t^2) sigmasq
            #
            # therefore
            #
            # sigmasq = norm * N * (\Delta t)^2
            #



            #self.sigmasq.append(1.0 * norm_h / sampleRate)
            norm_data = abs(numpy.dot(data, numpy.conj(data)))
            self.sigmasq.append(norm_data * spiir_match * spiir_match * working_state["working_length"] / sampleRate**2. )
            
            # This is actually the cross correlation between the original waveform and this approximation
            # FIXME: also update the waveform
            self.autocorrelation_bank[tmp,:] = normalized_crosscorr(h_pad, u_rev_pad, autocorrelation_length)
            
            self.matches.append(spiir_match)

            if verbose:
                logging.info("template %4.0d/%4.0d, m1 = %10.6f m2 = %10.6f, epsilon = %1.4f:  %4.0d filters, %10.8f match. original_eps = %1.4f: %4.0d filters, %10.8f match" % (tmp+1, len(sngl_inspiral_table), row.mass1,row.mass2, epsilon, n_filters, spiir_match, original_epsilon, original_filters, original_match))    
            # get the filter frequencies
            fs = -1. * numpy.angle(a1) / 2 / numpy.pi # Normalised freqeuncy
            a1dict = {}
            b0dict = {}
            delaydict = {}

            if downsample:
                min_M = 1
                max_M = int( 2**numpy.floor(numpy.log2(sampleRate/flower)))
                # iterate over the frequencies and put them in the right downsampled bin
                for i, f in enumerate(fs):
                    M = int(max(min_M, 2**-numpy.ceil(numpy.log2(f * 2.0 * padding) ) )) # Decimation factor
                    M = max(min_M, M)

                    if M > max_M:
                        continue

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
            # Get sngl inspiral table
            sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(root)

            # put the bank table in
            self.sngl_inspiral_table = lsctables.New(lsctables.SnglInspiralTable)
            for row in sngl_inspiral_table:
                self.sngl_inspiral_table.append(row)

            if os.path.isfile(self.template_bank_filename):
                pass
            else:

                # FIXME teach the trigger generator to get this information a better way
                self.template_bank_filename = tempfile.NamedTemporaryFile(suffix = ".gz", delete = False).name
                tmpxmldoc = ligolw.Document()
                tmpxmldoc.appendChild(ligolw.LIGO_LW()).appendChild(sngl_inspiral_table)
                utils.write_filename(tmpxmldoc, self.template_bank_filename, gz = True, verbose = verbose)
                tmpxmldoc.unlink()    # help garbage collector


            self.autocorrelation_bank = array.get_array(root, 'autocorrelation_bank_real').array + 1j * array.get_array(root, 'autocorrelation_bank_imag').array
            self.autocorrelation_mask = array.get_array(root, 'autocorrelation_mask').array
            self.sigmasq = array.get_array(root, 'sigmasq').array

            # Read the SPIIR coeffs
            self.sample_rates = [int(float(r)) for r in param.get_pyvalue(root, 'sample_rate').split(',')]
            for sr in self.sample_rates:
                self.A[sr] = repack_real_array_to_complex(array.get_array(root, 'a_%d' % (sr,)).array)
                self.B[sr] = repack_real_array_to_complex(array.get_array(root, 'b_%d' % (sr,)).array)
                self.D[sr] = array.get_array(root, 'd_%d' % (sr,)).array



    def get_rates(self, contenthandler = DefaultContentHandler, verbose = False):
        bank_xmldoc = utils.load_filename(self.bank_filename, contenthandler = contenthandler, verbose = verbose)
        return [int(float(r)) for r in param.get_pyvalue(bank_xmldoc, 'sample_rate').split(',')]

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

        sample_rates = [int(float(r)) for r in param.get_pyvalue(root, 'sample_rate').split(',')]
    
    return max(sample_rates)



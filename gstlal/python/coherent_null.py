#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import scipy.fftpack
import numpy

from pylal.xlal.datatypes.real8frequencyseries import REAL8FrequencySeries
from pylal import window

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

def factors_to_fir_kernel(coh_facs):
	"""
	Compute a finite impulse-response filter kernel from a power
	spectral density conforming to the LAL normalization convention,
	such that if zero-mean unit-variance Gaussian random noise is fed
	into an FIR filter using the kernel the filter's output will
	possess the given PSD.  The PSD must be provided as a
	REAL8FrequencySeries object (see
	pylal.xlal.datatypes.real8frequencyseries).

	The return value is the tuple (kernel, latency, sample rate).  The
	kernel is a numpy array containing the filter kernel, the latency
	is the filter latency in samples and the sample rate is in Hz.
        """
	#
	# this could be relaxed with some work
	#

	assert coh_facs.f0 == 0.0

	#
	# extract the PSD bins and determine sample rate for kernel
	#

	data = coh_facs.data
	sample_rate = 2 * int(round(coh_facs.f0 + len(data) * coh_facs.deltaF))

	#
	# compute the FIR kernel.  it always has an odd number of samples
	# and no DC offset.
	#

	data[0] = data[-1] = 0.0
	tmp = numpy.zeros((2 * len(data) - 1,), dtype = data.dtype)
	tmp[0] = data[0]
	tmp[1::2] = data[1:]
	data = tmp
	del tmp
	kernel = scipy.fftpack.irfft(data)
	kernel = numpy.hstack((kernel[::-1], kernel[1:]))

	#
	# apply a Tukey window whose flat bit is 50% of the kernel
	#

	kernel *= window.XLALCreateTukeyREAL8Window(len(kernel), .5).data

	#
	# the kernel's latency
	#

	latency = (len(kernel) + 1) / 2 - 1

	#
	# done
	#

	return kernel, latency, sample_rate

def psd_to_impulse_response(PSD1, PSD2):
	
	assert PSD1.f0 == PSD2.f0
	assert PSD1.deltaF == PSD2.deltaF
	assert len(PSD1.data) == len(PSD2.data)

	coh_facs_H1 = REAL8FrequencySeries()
	coh_facs_H1.name = PSD1.name
	coh_facs_H1.epoch = PSD1.epoch
	coh_facs_H1.f0 = PSD1.f0
	coh_facs_H1.deltaF = PSD1.deltaF
	coh_facs_H1.sampleUnits = PSD1.sampleUnits
	coh_facs_H1.data = PSD2.data/(PSD1.data + PSD2.data)
	# work around referencing vs. copying structure
	data = coh_facs_H1.data
	data[0] = data[-1] = 0.0
	coh_facs_H1.data = data

	coh_facs_H2 = REAL8FrequencySeries()
	coh_facs_H2.name = PSD2.name
	coh_facs_H2.epoch = PSD2.epoch
	coh_facs_H2.f0 = PSD2.f0
	coh_facs_H2.deltaF = PSD2.deltaF
	coh_facs_H2.sampleUnits = PSD2.sampleUnits
	coh_facs_H2.data = PSD1.data/(PSD1.data + PSD2.data)
	# work around referencing vs. copying structure
	data = coh_facs_H2.data
	data[0] = data[-1] = 0.0
	coh_facs_H2.data = data

	#
	# set up fir filter
	#

	H1_impulse, H1_latency, H1_srate = factors_to_fir_kernel(coh_facs_H1)
	H2_impulse, H2_latency, H2_srate = factors_to_fir_kernel(coh_facs_H2)

	assert H1_srate == H2_srate

	return H1_impulse, H1_latency, H2_impulse, H2_latency, H1_srate

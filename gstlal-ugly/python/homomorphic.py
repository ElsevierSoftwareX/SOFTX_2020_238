# This Program conversts an existing Linear Phase FIR to a Minimum phase FIR using Homomorphic algorithm #

#---------------------- Imports ---------------------------#
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import scipy
import math
#----------------------------------------------------------#



#------------------------------ Original Filter ------------------------------------#
file1 = open('original_filter.txt', 'r') #Requires a filter.txt containing the FIR filter
x = file1.readlines()
h = np.asarray(x, dtype=np.float32) #time domain of original
f = np.fft.fft(h,1024) #freq domain(1024 bins) of original
file1.close()
#-----------------------------------------------------------------------------------#



#------------------------------ Homomorphic Filtering -------------------------------#
n_fft = 10*len(h) #typically a lot greater than filter length
#Source code default - n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))
n_fft = int(n_fft)
n_half = len(h) // 2

# zero-pad; calculate the DFT
h_temp = np.abs(np.fft.fft(h, n_fft))
# take 0.25*log(|H|**2) = 0.5*log(|H|)
h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
np.log(h_temp, out=h_temp)
h_temp *= 0.5
# IDFT
h_temp = np.fft.ifft(h_temp).real
# multiply pointwise by the homomorphic filter
# lmin[n] = 2u[n] - d[n]
win = np.zeros(n_fft)
win[0] = 1
stop = (len(h) + 1) // 2
win[1:stop] = 2
if len(h) % 2:
    win[stop] = 1    
h_temp *= win
h_temp = np.fft.ifft(np.exp(np.fft.fft(h_temp)))
h_minimum = h_temp.real
n_out = n_half + len(h) % 2
filt = h_minimum[:n_out] #time domain of the min phase filter

#filt = np.pad(filt, (0, len(h)-len(filt)), 'constant')
f2 = np.fft.fft(filt,1024) #freq domain(1024 bins) of the min phase filter
#------------------------------------------------------------------------------------#



#------------------------------ Write Output ----------------------------------#
file2 = open('homomorphic_filter.txt', 'w')
for item in filt:
  file2.write("%f\n" % item)
file2.close()
#------------------------------------------------------------------------------#


#"""
#--------------------------- Plots/Filters --------------------------------#
plt.plot(filt,'r', label = 'Minimum Phase')
plt.plot(h,'b', label = 'Original Filter')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(loc='best')
plt.show()

plt.plot(abs(f)[:len(abs(f))/2+1],'b', label = 'Original Filter')
plt.plot(abs(f2)[:len(abs(f2))/2+1],'r', label = 'Minimum Phase')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend(loc='best')
plt.show()
#------------------------------------------------------------------#
#"""

"""
#--------------------------- Test on Noise ------------------------#
Fs = 1024
dt = 1./Fs
t = np.arange(0,1,dt)
nyquist = 512

mean = 0
std = 1
noise = np.random.normal(mean, std, size=len(t))
ftt = np.fft.fft(noise,Fs)
nout = np.convolve(filt, noise)
nout = nout[:len(noise)]
Sf = np.fft.fft(nout,Fs)
#------------------------------------------------------------------#
"""

"""
#----------------------------- Plots/Test ------------------------------#
plt.plot(abs(ftt)[:nyquist],'b', label = 'White Noise 1K Hz')
plt.plot(abs(Sf)[:nyquist],'r', label = 'Minimum Phase Filtered')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend(loc='best')
plt.show()
#------------------------------------------------------------------#
"""
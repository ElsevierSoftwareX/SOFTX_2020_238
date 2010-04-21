import matplotlib
matplotlib.use('Agg')
import numpy
import scipy
import pylab

def waveform(mchirp, duration, sr):
	# Finn, Chernoff PRD 47 2198-2219 (1993)
	msun = 4.92579497e-6 #mass of sun in seconds
	M = mchirp * msun
	pi = scipy.pi
	t = numpy.arange(-duration,0, 1. / sr)
	# STOP AT 6 M
	f = 1.0 / pi / M * (5.0 * M / 256.0 / (6*M - t))**(3./8.)
	phi = -2.0 * ((6*M - t) / 5 / M)**(5./8.)
	out = (pi*M*f)**(2./3.) * numpy.cos(phi)
	return out / numpy.sqrt(numpy.sum(out*out)), f

h,f = waveform(2.8*.25**.6, 30.5, 2048)
h /= max(abs(h))
t = numpy.arange(len(h))/2048.
boundaries = [(2048, 0, 0.5),
		(512, 0.5, 2.5),
		(256, 2.5, 6.5),
		(256, 6.5, 14.5),
		(128, 14.5, 30.5)]


fig = pylab.figure(figsize=(4,4))
fig.add_axes((.16,.63,.83,.32))
pylab.semilogy(t,f)
pylab.ylim(ymin=20)
pylab.xlim(t[0], t[-1])
pylab.xticks((0, 5, 10, 15, 20, 30), ('', '', '', '', '', '', ''), fontsize = 10)
pylab.yticks(fontsize=10)
pylab.ylabel('Frequency (Hz)', fontsize=12)

fig.add_axes((.16,.13,.83,.5))
pylab.plot(t[0:2048*16+16:16], h[0:2048*16+16:16], 'c', label='128 Hz')
pylab.plot(t[2048*16:2048*28+8:8], h[2048*16:2048*28+8:8], 'r', label='256 Hz')
pylab.plot(t[2048*28:2048*30+4:4], h[2048*28:2048*30+4:4], 'g', label='512 Hz')
pylab.plot(t[2048*30::1], h[2048*30::1], 'b', label='2048 Hz')
leg = pylab.legend(loc='upper left')
for txt in leg.get_texts():
	pylab.setp(txt, fontsize=12)
pylab.xlim(t[0], t[-1])
pylab.ylim(-1.1,1.1)
pylab.xticks(fontsize=10)
pylab.yticks(fontsize=10)
pylab.xlabel('Time (s)', fontsize=12)
pylab.ylabel('Amplitude', fontsize=12)

pylab.savefig('multiband.png', dpi=300)


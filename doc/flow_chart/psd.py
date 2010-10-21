import matplotlib
matplotlib.use('Agg')
import numpy
import scipy
import pylab

f,Sn = numpy.loadtxt('reference_psd.txt', unpack=True)

fig = pylab.figure(figsize=(6,4))
fig.add_axes((-0.05,-0.05,1.1,1.1))
pylab.loglog(f[40/0.0625:2048/0.0625],1./Sn[40/0.0625:2048/0.0625])
pylab.xticks([])
pylab.yticks([])

pylab.savefig('psd.png', dpi=300)


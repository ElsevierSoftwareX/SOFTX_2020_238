#!/usr/bin/env python

import pylab, numpy
from scipy import signal
tx, x = numpy.loadtxt('x.txt').T
ty, y = numpy.loadtxt('y.txt').T
kernel = numpy.loadtxt('kernel.txt')

downsample_factor = round(numpy.diff(ty).mean() / numpy.diff(tx).mean())
print downsample_factor

z = x.copy()
z = numpy.concatenate((z, numpy.zeros(len(kernel)/2)))
z = signal.lfilter(kernel, [1], z)
z = z[len(kernel)/2:][::downsample_factor]
tz = numpy.arange(len(z), dtype=float) / (64 / downsample_factor)

i = numpy.arange(-len(kernel)/2+1, len(kernel)/2+1)
pylab.vlines(i, 0, kernel)
pylab.plot(i, kernel, 'or')
pylab.ylim(-0.3, 1.1)
pylab.savefig('kernel.pdf')
pylab.close()

pylab.plot(tx, x, label='input')
pylab.plot(ty, y, 'k', label='output')
pylab.plot(tz, z, 'r', alpha=0.5, label='expected', lw=3)
pylab.legend()
pylab.savefig('test.pdf')
pylab.show()

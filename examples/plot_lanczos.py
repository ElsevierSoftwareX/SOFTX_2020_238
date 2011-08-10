#!/usr/bin/env python

import pylab, numpy
from scipy import signal
tx, x = numpy.loadtxt('x.txt').T
ty, y = numpy.loadtxt('y.txt').T
kernel = numpy.loadtxt('kernel.txt')

upsample_factor = round(numpy.diff(tx).mean() / numpy.diff(ty).mean())
print upsample_factor

z = numpy.vstack((x, numpy.zeros((upsample_factor - 1, len(x))))).T.flatten()
z = numpy.concatenate((z, numpy.zeros(len(kernel)/2)))
tz = numpy.arange(len(z), dtype=float) / (64 * upsample_factor)
z = signal.lfilter(kernel, [1], z)
z = z[len(kernel)/2:]
tz = tz[len(kernel)/2:]
tz -= tz[0]

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

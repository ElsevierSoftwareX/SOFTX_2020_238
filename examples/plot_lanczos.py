#!/usr/bin/env python

import pylab, numpy
from scipy import signal
tx, x = pylab.loadtxt('x.txt').T
ty, y = pylab.loadtxt('y.txt').T
kernel = pylab.loadtxt('kernel.txt')

upsample_factor = 3

z = pylab.vstack((x, pylab.zeros((upsample_factor - 1, len(x))))).T.flatten()
z = pylab.concatenate((z, pylab.zeros(len(kernel)/2)))
tz = pylab.arange(len(z), dtype=float) / (64 * upsample_factor)
z = signal.lfilter(kernel, [1], z)
z = z[len(kernel)/2:]
tz = tz[len(kernel)/2:]
tz -= tz[0]

i = pylab.arange(-len(kernel)/2+1, len(kernel)/2+1)
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
pylab.close()

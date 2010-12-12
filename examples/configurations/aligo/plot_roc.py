#!/usr/bin/env python

import pylab
import numpy as np
import bisect
from pylal import plotutils

noninj = np.load('noninjections.npy')
inj, dist = np.load('injections.npy').T
inj_weights = dist**3
inj_weights /= inj_weights.sum()

pylab.clf()
pylab.hist(noninj, bins=50, normed=True, histtype='step', label='noninjections')
pylab.hist(inj, bins=50, normed=True, histtype='step', label='injections')
pylab.xlabel('SNR')
pylab.ylabel('density')
pylab.legend()
pylab.savefig('hist.png')


rocplot = plotutils.ROCPlot('FAP', 'EFF', 'gstlal_inspiral ROC curve')
rocplot.add_content(noninj, inj, inj_weights, label=r"SNR, $\rho$")
rocplot.finalize()

ax = rocplot.ax
ax.plot( (0, 1), (0, 1), '--', label='no discrimination')
ax.legend(loc = 'upper left')

ax2 = ax.twinx()
ticklocs = pylab.arange(0., 1.1, 0.2)
ax2.set_yticks(ticklocs)
ax2.set_yticklabels(['%0.1f' % noninj[round(x * (len(noninj) - 1))] for x in reversed(ticklocs)])
ax2.set_ylabel(r'SNR threshold')

rocplot.savefig('roc.png')

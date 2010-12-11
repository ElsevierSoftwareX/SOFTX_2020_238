#!/usr/bin/env python

import pylab
import numpy as np
from pylal import plotutils

noninj = np.load('noninjections.npy')
inj, dist = np.load('injections.npy').T
inj_weights = dist**3
inj_weights /= inj_weights.sum()

if 0:
	pylab.clf()
	pylab.hist(noninj, bins=50, normed=True, histtype='step', label='noninjections')
	pylab.hist(inj, bins=50, normed=True, histtype='step', label='injections')
	pylab.xlabel('SNR')
	pylab.ylabel('density')
	pylab.legend()
	pylab.savefig('hist.png')


rocplot = plotutils.ROCPlot('FAP', 'EFF', 'gstlal\_inspiral ROC curve')
rocplot.add_content(noninj, inj, inj_weights, label=r"SNR, $\rho$")
rocplot.finalize()
rocplot.ax.plot( (0, 1), (0, 1), '--', label='no-discrimination line')
rocplot.ax.legend(loc = 'upper left')
rocplot.savefig('roc.png')

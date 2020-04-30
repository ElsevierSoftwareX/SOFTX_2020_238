#!/usr/bin/env python3

import sys
filename = sys.argv[1]

from gstlal.gstlal_svd_bank import read_bank
import pylab

bank = read_bank(filename)
frags = bank.bank_fragments

pylab.clf()
for frag in frags:
	pylab.plot(
		pylab.arange(-frag.end, -frag.start, 1./frag.rate),
		pylab.sqrt(float(frag.rate) / max(bank.get_rates())) *
		pylab.array(pylab.matrix(frag.mix_matrix.T) * pylab.matrix(frag.orthogonal_template_bank))[0, :].flatten(),
		label=str(frag.rate))

pylab.legend()
pylab.xlabel('Time (s)')
pylab.title("Real component of first template in file '%s'" % filename)
pylab.show()

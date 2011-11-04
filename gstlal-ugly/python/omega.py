import numpy

# sine-gaussian generator, given time vector
def sine_gaussian(t,q,f0):
	dt = t - t[len(t)/2]
	tau = ((numpy.pi*f0)/q)**2
	out = numpy.exp(-(dt**2.0)*tau) * numpy.sin(f0*numpy.pi*dt)
	return out / sum(out**2)**0.5 

def omega_bank(fvec, qvec, samplerate):
	tvec = numpy.arange(samplerate/6)/float(samplerate)
	
	qbank = numpy.zeros( (len(fvec) * len(qvec), len(tvec)) )
	
	for i,f in enumerate(fvec):
		for j,q in enumerate(qvec):
			qbank[i*len(qvec) + j, :] = sine_gaussian(tvec,q,f)

	return qbank
			

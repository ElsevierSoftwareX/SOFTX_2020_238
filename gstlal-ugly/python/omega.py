import numpy

def duration_from_f_and_q(f, q):
	#FIXME only approximate, Jamie make this right
	return q / f

def q_from_duration_and_f(duration, f):
	#FIXME only approximate, Jamie make this right
	return duration *  f

# sine-gaussian generator, given time vector
def sine_gaussian(t,q,f0, phase = 0):
	dt = t - t[len(t)/2]
	tau = ((numpy.pi*f0)/q)**2
	out = numpy.exp(-(dt**2.0)*tau) * numpy.sin(f0*numpy.pi*dt + phase)
	return out / sum(out**2)**0.5 

def omega_bank(table, samplerate):
	max_dur = 2 * max([row.duration for row in table])
	tvec = numpy.arange(0, 2**numpy.ceil(numpy.log2(max_dur)), 1.0 / samplerate)
	qbank = numpy.zeros((2 * len(table), len(tvec)))
	for i,row in enumerate(table):
		q = q_from_duration_and_f(row.duration, row.central_freq)
		f = row.central_freq
		qbank[i*2,:] = sine_gaussian(tvec, q, f) 
		qbank[i*2+1,:] = sine_gaussian(tvec, q, f, numpy.pi/2)
	return qbank

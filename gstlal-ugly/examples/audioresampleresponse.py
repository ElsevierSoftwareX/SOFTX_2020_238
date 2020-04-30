#!/usr/bin/env python3

from gstlal.pipeutil import *
import gstlal.pipeio as pipeio
from pylab import *

def measure_freq_response(num, den, quality, nsamples, ntrials=100):
	freqs = arange(ntrials)/float(ntrials)*0.5*num
	responses = zeros(len(freqs))

	def handoff_handler(fakesink, buffer, pad, (filtlen, n)):
		buf = pipeio.array_from_audio_buffer(buffer).flatten()
		outlatency = int(ceil(float(den)/num * filtlen))
		responses[n] = max(abs(buf[outlatency:-outlatency]))

	pipelines = []
	for n, freq in enumerate(freqs):
		pipeline = gst.Pipeline()
		elems = mkelems_in_bin(pipeline,
			('audiotestsrc', {'volume':1,'wave':'sine','freq':freq}),
			('capsfilter', {'caps':gst.Caps('audio/x-raw-float,width=64,rate=%d'%num)}),
			('audioresample', {'quality':quality}),
			('capsfilter', {'caps':gst.Caps('audio/x-raw-float,width=64,rate=%d'%den)}),
			('fakesink', {'signal-handoffs':True, 'num-buffers':1}))
		filtlen = elems[2].get_property('filter-length')
		elems[0].set_property('samplesperbuffer', 2*filtlen + nsamples)
		elems[-1].connect_after('handoff', handoff_handler, (filtlen, n))
		pipeline.set_state(gst.STATE_PLAYING)
		pipelines.append(pipeline)
	for pipeline in pipelines:
		pipeline.get_bus().poll(gst.MESSAGE_EOS, -1)
		pipeline.set_state(gst.STATE_NULL)
	return vstack((freqs, responses)).T

def measure_all_responses(num, den):
	figure()
	for quality in range(11):
		print 'den =', den, 'num =', num, 'quality =', quality
		resp = measure_freq_response(num, den, quality, 1024, 100)
		semilogy(resp[:,0]/num, resp[:,1], label='quality=%d'%quality)
		xlabel('frequency')
		ylabel('amplitude response')
		title('Frequency response of audioresample: %ssampling by factor %d/%d' % (('up','down')[den < num], den, num))
	legend(loc=3)
	savefig('resampler-%d-%d.pdf' % (den, num))
	savefig('resampler-%d-%d.png' % (den, num))

for x in range(1,4):
	measure_all_responses(1, 2**x)
	measure_all_responses(2**x, 1)

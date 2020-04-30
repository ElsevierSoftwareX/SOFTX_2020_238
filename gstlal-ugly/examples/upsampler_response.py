#!/usr/bin/env python3

# This example print out the different upsampling responses of test sine waves. 

# ntrials : determines the freqencies of test sine waves, set to 10
# freqs   : frequencies of sine waves, for ntrials=10, freqs=[0, 0.05, 0.1,..., 0.45]
# den     : upsampling ratio
# quality : upsampling quality, determines the filter length
# filtlen : determined by quality

from gstlal.pipeutil import *
import gstlal.pipeio as pipeio
from pylab import *


def output_response(num, den, quality, nsamples, ntrials=10):
	freqs = arange(ntrials)/float(ntrials)*0.5*num

	def handoff_handler(fakesink, buffer, pad, (filtlen, n)):
		buf = pipeio.array_from_audio_buffer(buffer).flatten()
		outlatency = int(ceil(float(den)/num * filtlen))
		response = buf[outlatency:-outlatency]
		print 'length of output data =', len(buf), ';latency =', outlatency, ';real response =', response 
		print '***************************************************************************'

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
		print 'filter length =', filtlen
		print 'freq =', freq
		elems[0].set_property('samplesperbuffer', 2*filtlen + nsamples)
		elems[-1].connect_after('handoff', handoff_handler, (filtlen, n))
		pipeline.set_state(gst.STATE_PLAYING)
		pipeline.get_bus().poll(gst.MESSAGE_EOS, -1)
		pipeline.set_state(gst.STATE_NULL)

def upsampler_responses(num, den):
	for quality in range(11):
		num_samples = 1024
		num_trials = 10
		print '\n'
		print 'upsampler factor =', den/num, ';quality =', quality, ';num of input samples =', num_samples, ';num of trials =', num_trials
		print '\n'
		output_response(num, den, quality, num_samples, num_trials)

for x in range(1,3):
	upsampler_responses(1, 2**x)

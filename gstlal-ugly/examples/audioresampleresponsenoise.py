#!/usr/bin/env python3

from gstlal.pipeutil import *
import gstlal.pipeio as pipeio
from pylab import *

filt_lens = (8, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256)


def measure_response(num, den, nsamples):
	responses = zeros(11)

	def handoff_handler(fakesink, buffer, pad, (filtlen, n)):
		outlatency = int(ceil(float(den)/num * filtlen))
		buf = pipeio.array_from_audio_buffer(buffer).flatten()[outlatency:-outlatency]
		responses[n] = var(buf)

	for qual in range(11):
		pipeline = gst.Pipeline()
		elems = mkelems_fast(pipeline,
			'audiotestsrc', {'volume':1,'wave':'gaussian-noise'},
			'capsfilter', {'caps':gst.Caps('audio/x-raw-float,width=64,rate=%d'%num)},
			'audioresample', {'quality':qual},
			'capsfilter', {'caps':gst.Caps('audio/x-raw-float,width=64,rate=%d'%den)},
			'fakesink', {'signal-handoffs':True, 'num-buffers':1})
		filtlen = elems[2].get_property('filter-length')
		elems[0].set_property('samplesperbuffer', 2*filtlen + nsamples)
		elems[-1].connect_after('handoff', handoff_handler, (filtlen, qual))
		pipeline.set_state(gst.STATE_PLAYING)
		pipeline.get_bus().poll(gst.MESSAGE_EOS, -1)
		pipeline.set_state(gst.STATE_NULL)
	return responses

def plot_response(num, den):
	resp = measure_response(num, den, 2**22)
	if num > den:
		resp *= float(num)/den
		args = {"label":"downsampling %d:%d" % (num, den), "linestyle":":", "color":"rgbm"[int(log2(num))-1]}
	elif num < den:
		args = {"label":"upsampling %d:%d" % (num, den), "linestyle":"-", "color":"rgbm"[int(log2(den))-1]}
	else: # num == den
		args = {"label":"pass-through 1:1", "linestyle":"-.", "color":"k", "linewidth":4}
	savetxt('corrected_gaussian_noise_response_%d_%d.txt' % (num, den), resp)
	plot(filt_lens, resp, **args)

figure()
plot_response(1, 1)
for x in range(1,4):
	plot_response(1, 2**x)
	plot_response(2**x, 1)
xticks(filt_lens, fontsize=10)
grid()
xlabel('filter length')
ylabel('rate-corrected output variance')
legend(loc=4)
twiny()
xticks(filt_lens, [str(x) for x in range(len(filt_lens))])
xlabel('"quality" setting')
savefig('gaussian_noise_response.pdf')
savefig('gaussian_noise_response.png')
show()

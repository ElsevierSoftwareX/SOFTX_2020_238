#!/usr/bin/env python

import pygtk
pygtk.require('2.0')
import pygst
pygst.require('0.10')
import gobject
import gst
from gstlal import pipeutil
import sys
import os
import numpy

# for plotting waveform bank
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plot

############################################################

# single optional argument is injection file
injfile = []
if len(sys.argv) > 1:
	injfile = sys.argv[1]

# downsample
samplerate = 2048

# size of queue buffers (1 second)
queuesize = int(1e9)

############################################################
# construct the pipeline

mainloop = gobject.MainLoop()
pipeline = gst.Pipeline("burstdemo")

elems = []

# advanced LIGO noise source
elems.append(pipeutil.mkelem("lal_fakeadvligosrc",
			     {"channel-name": 'LSC-STRAIN',
			      "instrument": 'H1'
			      }))

# add injection if specified
if injfile:
	elems.append(pipeutil.mkelem("lal_simulation", {"xml-location": injfile}))

# resample at the specified sample rate
elems.append(pipeutil.mkelem("audioresample"))
elems.append(pipeutil.mkelem("capsfilter",
			     {"caps": gst.Caps("audio/x-raw-float,width=64,rate=%d" % (samplerate))
			      }))

# apply whitening
elems.append(pipeutil.mkelem("lal_whiten",
			     {"zero-pad": 0,
			      "fft-length": 8,
			      "average-samples": 32,
			      "median-samples": 9,
			     }))

elems.append(pipeutil.mkelem("progressreport"))

# sine-gaussian generator, given time vector
def sg(t,q,f0):
	dt = t - t[len(t)/2]
	tau = ((numpy.pi*f0)/q)**2
	out = numpy.exp(-(dt**2.0)*tau) * numpy.sin(f0*numpy.pi*dt)
	return out / sum(out**2)**0.5 

tvec = numpy.arange(samplerate/6)/float(samplerate)
qvec = numpy.linspace(2,40,5)

# # constant q plains
# fvec = numpy.logspace(1.6,3,50)
# qbank = numpy.zeros((len(qvec)*len(fvec), len(tvec)))
# for i,f in enumerate(fvec):
# 	for j,q in enumerate(qvec):
# 		qbank[len(qvec)*i + j,:] = sg(tvec,q,f)

# constant duration q
fvec = numpy.logspace(1.6,3,100)
qbank = numpy.zeros((len(fvec), len(tvec)))
for i,f in enumerate(fvec):
	q = 4*(f/40.0)
	qbank[i,:] = sg(tvec,q,f)

print qbank.shape

# plot.plot(qbank.T)
# plot.plot(qbank[-2,:])
# plot.savefig('qplot.pdf')
# sys.exit()

# make the filter bank
elems.append(pipeutil.mkelem("lal_firbank",
			     {"name": "Q",
			      "time-domain": False,
			      "fir-matrix": qbank,
			     }))

elems.append(pipeutil.mkelem("progressreport"))

# sum square of the snr channels
elems.append(pipeutil.mkelem("lal_sumsquares"))

elems.append(pipeutil.mkelem("queue", {"max-size-time": queuesize}))

elems.append(pipeutil.mkelem("cairovis_lineseries",
			     {"title": "Omega detection"}))
# elems.append(pipeutil.mkelem("cairovis_waterfall",
# 			     {"title": "OmegaGram",
# 			      "history": gst.SECOND,
# 			      }))

elems.append(pipeutil.mkelem("capsfilter",
			     {"caps": gst.Caps("video/x-raw-rgb,framerate=24/1,width=800,height=600")
			      }))
elems.append(pipeutil.mkelem("ximagesink",
			     {"sync": False,
			      "async": False,
 			      }))

# link the elements together
for elem in elems:
	pipeline.add(elem)
gst.element_link_many(*elems)

# dump the pipeline dot file if the dot file is specified
if 'GST_DEBUG_DUMP_DOT_DIR' in os.environ:
	gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_ALL, 'burst')

############################################################
# run the pipeline

class Handler(object):
	def __init__(self, mainloop, pipeline):
		self.mainloop = mainloop
		self.pipeline = pipeline

		bus = pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message", self.on_message)

	def on_message(self, bus, message):
		if message.type == gst.MESSAGE_EOS:
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
		elif message.type == gst.MESSAGE_ERROR:
			gerr, dbgmsg = message.parse_error()
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
			sys.exit("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg))


handler = Handler(mainloop, pipeline)

pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()

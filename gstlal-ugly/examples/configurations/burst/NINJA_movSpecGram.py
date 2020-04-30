#!/usr/bin/env python3

import optparse
import sys
import os
import numpy

# for plotting waveform bank
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plot

############################################################
# options

parser = optparse.OptionParser(
	version = "%prog ??",
	usage = "%prog [options]",
	description = "GSTLAL burt movie on NINJA data"
	)

parser.add_option("-s", "--spectrogram",
                  help="turn on spectrogram plot",
                  dest="plotSpec",
                  action="store_true",
		  default=False
                  )
parser.add_option("-l", "--lineseries",
                  help="turn on lineseries plot of sumsquare",
                  dest="plotLine",
                  action="store_true",
		  default=False
                  )

parser.add_option("-o", "--out-put",
                  help="save the output movie run into a file",
                  dest="outmov",
                  action="store_true",
		  default=False
                  )


(options, args) = parser.parse_args()


############################################################
# construct the pipeline

# import gstreamer stuff after option parsing
from gstlal import pipeutil
from gstlal.pipeutil import gobject, gst

#delete previous movie file if any
cmd = 'rm -f *.ogg'
os.system(cmd)

#framecache file

GpsStartTime=871147452
GpsStopTime=876357564

cmdcache='ligo_data_find  --observatory=H --url-type=file --type=H1_NINJA_LOWMASS_TEST2 --gps-start-time=' + str(GpsStartTime) + ' --gps-end-time=' + str(GpsStopTime)  + ' --lal-cache > H1frame.cache'
os.system(cmdcache)

# downsample
samplerate = 4096
IFO="H1"
ChanName="LDAS-STRAIN"
TYPE="_NINJA_LOWMASS_TEST2"
# size of queue buffers (1 second)
queuesize = gst.SECOND
mainloop = gobject.MainLoop()
pipeline = gst.Pipeline("NINJAmovSpecGram")


elems = []


#Read the framefiles

elems.append(pipeutil.mkelem("lal_framesrc",
			     {"blocksize":65536 ,
			      "location":"H1frame.cache",
			      "instrument":IFO,
			      "channel-name":ChanName
			    }))

elems.append(pipeutil.mkelem("audiochebband",
			     {"lower-frequency":40,
			      "upper-frequency":2500,
			      "poles":8
			     }))

elems.append(pipeutil.mkelem("audioamplify",
			     {"clipping-method":3,
			     "amplification":2e+17
			     }))

elems.append(pipeutil.mkelem("lal_adder"))
elems.append(pipeutil.mkelem("audioconvert"))
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

# constant duration q
fvec = numpy.logspace(1.6,3,100)
qbank = numpy.zeros((len(fvec), len(tvec)))
for i,f in enumerate(fvec):
	q = 4*(f/40.0)
	qbank[i,:] = sg(tvec,q,f)


# make the filter bank
elems.append(pipeutil.mkelem("lal_firbank",
			     {"name": "Q",
			      "time-domain": False,
			      "fir-matrix": qbank,
				  "block-length-factor": 2,
			     }))

elems.append(pipeutil.mkelem("progressreport"))

elems.append(pipeutil.mkelem("tee"))
tee = elems[-1]

##############################
# tee line to waterfall
if options.plotSpec:
	print >>sys.stderr, "plotting spectrogram..."
	elems.append(pipeutil.mkelem("queue", {"max-size-time": 2}))
	elems.append(pipeutil.mkelem("pow"))
	elems.append(pipeutil.mkelem("queue", {"max-size-time": 2}))
	elems.append(pipeutil.mkelem("cairovis_waterfall",
				     {"title": "Moving Spectrogram",
				      "z-scale": 0,
				      "z-autoscale":False,
				      "z-min": 0.0,
				      "z-max": 10.0,
				      "colormap": "jet",
				      "history": 1
				      }))
	elems.append(pipeutil.mkelem("capsfilter",
				     {"caps": gst.Caps("video/x-raw-rgb,framerate=24/1,width=400,height=300")
				      }))
	if options.outmov:
	 elems.append(pipeutil.mkelem("ffmpegcolorspace"))
	 elems.append(pipeutil.mkelem("theoraenc"))	
	 elems.append(pipeutil.mkelem("oggmux"))
	 elems.append(pipeutil.mkelem("filesink",
				      {"location":"MovSpecGram_"+IFO+TYPE+".ogg",
				      "append":True,
				      "sync": False,
				      "async": False,
				       }))
	else:
	 elems.append(pipeutil.mkelem("ximagesink",
				       {"sync": False,
				       "async": False,
				        }))

else:
	elems.append(pipeutil.mkelem("fakesink"))

for elem in elems:
	pipeline.add(elem)
gst.element_link_many(*elems)


elems[0].seek(1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_FLUSH,
             gst.SEEK_TYPE_SET, GpsStartTime * 1e9,
             gst.SEEK_TYPE_SET, GpsStopTime * 1e9)


##############################
# tee line to sum square of the snr channels
elems = []

elems.append(pipeutil.mkelem("lal_sumsquares"))

if options.plotLine:
	print >>sys.stderr, "plotting lineseries..."

	elems.append(pipeutil.mkelem("queue", {"max-size-time": queuesize}))
	elems.append(pipeutil.mkelem("cairovis_lineseries",
				     {"title": "Detection"}))
	elems.append(pipeutil.mkelem("capsfilter",
				     {"caps": gst.Caps("video/x-raw-rgb,framerate=24/1,width=400,height=300")
				      }))
	if options.outmov:
	 elems.append(pipeutil.mkelem("ffmpegcolorspace"))
	 elems.append(pipeutil.mkelem("theoraenc"))	
	 elems.append(pipeutil.mkelem("oggmux"))
	 elems.append(pipeutil.mkelem("filesink",
				      {"location":"Detection_"+IFO+TYPE+".ogg",
				      "append":True,
				      "sync": False,
				      "async": False,
				       }))
	else:						
	 elems.append(pipeutil.mkelem("ximagesink",
				      {"sync": False,
				       "async": False,
				       }))

else:
	elems.append(pipeutil.mkelem("fakesink"))

for elem in elems:
	pipeline.add(elem)
gst.element_link_many(tee, *elems)

############################################################

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

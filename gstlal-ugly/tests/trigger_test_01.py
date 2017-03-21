#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import numpy
import sys, os
from gstlal import pipeparts, simplehandler
import test_common

from collections import deque
import StringIO

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst


####################
# 
#      classes
#
####################   


class MultiChannelHandler(simplehandler.Handler):
	def __init__(self, *args, **kwargs):
		self.output = kwargs.pop("output")
		self.instrument = kwargs.pop("instrument")
		super(MultiChannelHandler, self).__init__(*args, **kwargs)
		self.timedeq = deque(maxlen = 10000)

	def do_on_message(self, bus, message):
		return False

	def prehandler(self,elem):
		buf = elem.emit("pull-preroll")
		del buf
		return Gst.FlowReturn.OK

	#def bufhandler(self, elem, sink_dict):
	def bufhandler(self, elem):
		buf = elem.emit("pull-sample").get_buffer()
		buftime = int(buf.pts / 1e9)
		(result, mapinfo) = buf.map(Gst.MapFlags.READ)
		assert result
		if mapinfo.data:
			data = StringIO.StringIO(mapinfo.data).getvalue()
			#channel = sink_dict[elem]
			fdata = ""
			for line in data.split('\n'):
				if len(line) > 0:
					#fdata += "%s\t" % channel + line.rstrip('\r') + "\n"
					fdata += line.rstrip('\r') + "\n"
			self.timedeq.append(buftime)
		else:
			buf.unmap(mapinfo)
			del buf
			return Gst.FlowReturn.OK

		# Save a "latest"
		self.to_trigger_file(os.path.join(self.output, "output_triggers.txt"), fdata)
		
		buf.unmap(mapinfo)
		del buf
		return Gst.FlowReturn.OK

	def to_trigger_file(self, path, data):
		#os.remove(path)
		with open(path, 'a') as f:
 			f.write(data)

#
# =============================================================================
#
#                               Pipeline Builder
#
# =============================================================================
#


def build_and_run(pipelinefunc, name, segment = None, **pipelinefunc_kwargs):
	for key, value in pipelinefunc_kwargs.items():
		print("{0} = {1}".format(key, value))
	print >>sys.stderr, "=== Running Test %s ===" % name
	mainloop = GObject.MainLoop()
	pipeline = Gst.Pipeline(name = name)
	handler = MultiChannelHandler(mainloop, pipeline, output = "/home/dmeacher/local/src/gstlal/gstlal-ugly/tests", instrument = None)
	pipeline = pipelinefunc(pipeline, name, handler, **pipelinefunc_kwargs)
	if segment is not None:
		if pipeline.set_state(Gst.State.PAUSED) == Gst.StateChangeReturn.FAILURE:
			raise RuntimeError("pipeline failed to enter PLAYING state")
		pipeline.seek(1.0, Gst.Format(Gst.Format.TIME), Gst.SeekFlags.FLUSH, Gst.SeekType.SET, segment[0].ns(), Gst.SeekType.SET, segment[1].ns())
	if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
		raise RuntimeError("pipeline failed to enter PLAYING state")
	mainloop.run()

#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#



#
# check for proper peak finding
#


def peak_test_01a(pipeline,name,handler):
	#
	# try changing these.  test should still work!
	#

	initial_channels = 2
	rate = 2048	#Hz
	width = 32
	sine_frequency = 1
	gap_frequency = 0.1	# Hz
	gap_threshold = 0.7	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	peak_window = 2048 	# samples
	wave = 0

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = initial_channels, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	head = pipeparts.mktogglecomplex(pipeline, head)
	head = pipeparts.mktaginject(pipeline, head, "instrument=H1,channel-name=LSC-STRAIN,units=strain")

	#head = test_common.gapped_complex_test_src(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, wave = wave, freq = sine_frequency, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "itac_test_01a_control.dump", tags = "instrument=H1,channel-name=LSC-STRAIN,units=strain")
	#head = tee = pipeparts.mktee(pipeline, head)
	#pipeparts.mktrigger(pipeline, head, peak_window, "test_bank.xml")

	# Does not recieve EOS, hangs
	#pipeparts.mktrigger(pipeline, head, peak_window,autocorrelation_matrix = numpy.ones((1,21), dtype=numpy.complex))

	#head = pipeparts.mkqueue(pipeline, pipeparts.mkitac(pipeline, head, peak_window, "test_bank.xml", autocorrelation_matrix = numpy.array([[0+0.j, 0+0.j, 1+1.j, 0+0.j, 0+0.j]])))
	#head = pipeparts.mkprogressreport(pipeline, head, "test")

	#
	# output the before and after
	#
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	#pipeparts.mkfakesink(pipeline, head)
	
	#a = pipeparts.mkappsink(pipeline, pipeparts.mkqueue(pipeline, head))

	head = pipeparts.mkgeneric(pipeline, head, "lal_nxydump")

	sink = pipeparts.mkappsink(pipeline, head, max_buffers = 1, sync = False)
	sink.connect("new-sample",  handler.bufhandler)
	sink.connect("new-preroll", handler.prehandler)

	#outfile = open("itac_test_01a_out.dump", "w")

	#def dump_triggers(elem, output = outfile):
	#	for row in SnglInspiralTable.from_buffer(elem.emit("pull-buffer")):
	#		print >>outfile, row.end_time + row.end_time_ns*1e-9, row.snr, row.chisq, row.chisq_dof

	#a.connect_after("new-buffer", dump_triggers)
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, tee)), "itac_test_01a_in.dump")

	#
	# done
	#

	#if "GST_DEBUG_DUMP_DOT_DIR" in os.environ:
	#	gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_ALL, "peak_test_01a")

	return pipeline

#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


#test_common.build_and_run(peak_test_01a, "peak_test_01a")
build_and_run(peak_test_01a, "peak_test_01a")


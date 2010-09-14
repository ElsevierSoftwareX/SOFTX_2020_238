#!/usr/bin/env python
"""

Unit tests for lal_skymap.

"""
__author__       = "Leo Singer <leo.singer@ligo.org>"
__copyright__    = "Copyright 2010, Leo Singer"


import unittest
from fixtures import *
from gstlal.pipeutil import *
from gstlal.lloidparts import mkelems_fast
from gstlal.pipeparts import mkappsink
import pylal.xlal.datatypes.snglinspiraltable as sngl
import random


class TestSkymap(PipelineTestFixture):

	def runTest(self):
		ifos = ("H1", "L1", "V1")
		start_time = 963704084
		stop_time = start_time + 8

		skymap = mkelems_fast(self.pipeline,
			"lal_skymap",
			{
				"bank-filename": "../examples/banks/test_bank.xml",
				"trigger-present-padding": 6 * gst.MSECOND,
				"trigger-absent-padding": 4 * gst.MSECOND
			}
		)[-1]
		for ifo in ifos:
			elems = mkelems_fast(self.pipeline,
				"audiotestsrc", {"wave": "gaussian-noise", "volume": 1, "samplesperbuffer": 4096*8},
				"capsfilter", {"caps": gst.Caps("audio/x-raw-float,channels=2,rate=4096")},
				"lal_togglecomplex",
				"taginject", {"tags": "instrument=%s" % ifo},
				"queue",
				skymap
			)
			elems[0].send_event(gst.event_new_seek(1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_NONE,
				gst.SEEK_TYPE_SET, start_time * gst.SECOND, gst.SEEK_TYPE_SET, stop_time * gst.SECOND
			))
		appsrc = mkelems_fast(self.pipeline,
			"appsrc",
			{
				"caps": gst.Caps("application/x-lal-snglinspiral,channels=%d" % len(ifos)),
				"format": "time",
			},
			"queue",
			skymap
		)[0]
		appsink = mkappsink(self.pipeline, skymap)

		self.pipeline.set_state(gst.STATE_PLAYING)

		# Uncomment to get pipeline graph
		gst.DEBUG_BIN_TO_DOT_FILE(self.pipeline, gst.DEBUG_GRAPH_SHOW_ALL, "skymap")

		sngl_inspiral_len = len(buffer(sngl.SnglInspiralTable()))
		buf = gst.buffer_new_and_alloc(sngl_inspiral_len * len(ifos))
		mid_time = (start_time + stop_time) / 2
		for i_ifo, ifo in enumerate(ifos):
			s = sngl.SnglInspiralTable()
			s.end_time = mid_time
			s.end_time_ns = long(500 * gst.MSECOND + random.random() * 2 * gst.MSECOND)
			s.sigmasq = 1.0
			s.ifo = ifo
			s.mass1 = 1.758074
			s.mass2 = 1.124539
			buf[(i_ifo*sngl_inspiral_len):(i_ifo+1)*sngl_inspiral_len] = buffer(s)
		buf.timestamp = (mid_time - 1) * gst.SECOND
		buf.duration = 2 * gst.SECOND
		buf.offset = gst.BUFFER_OFFSET_NONE
		buf.offset_end = gst.BUFFER_OFFSET_NONE

		self.assertEqual(appsrc.emit("push-buffer", buf), gst.FLOW_OK)
		buf = appsink.emit("pull-buffer")

		# TODO Wire up appsrc and appsink with test data
		#appsink.connect_after('new-buffer', self.coinc_new_buffer, None)
		
		self.mainloop.run()

if __name__ == '__main__':
	suite = unittest.main()

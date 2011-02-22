#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
# A test gstreamer pipeline to see if the IIR filter element works


import numpy
import sys
from gstlal import pipeparts
import test_common
from gstlal.elements.check_timestamps import mkchecktimestamps
from pylal import spawaveform
from gstlal import cbc_template_iir


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#


#
# is the iirbank element an identity transform when given a unit impulse?
# in and out timeseries should be identical modulo start/stop transients
#

#
# Produce IIR Coefficients & Delays
#

def iirbank_test_01a(pipeline):
        #
        # try changing these.  test should still work!
        #

        rate = 4096     # Hz
        gap_frequency = None    # Hz
        gap_threshold = 0.0     # of 1
        buffer_length = 1.0     # seconds
        test_duration = 5.0     # seconds
        wave = 10               # ticks
        volume = 1.0;
        freq = 0.01#12.0

        #
        # build pipeline
        #
        print "Before Test Src"
        head = test_common.gapped_test_src(pipeline,
                                           wave = wave,
                                           freq = freq,
					   buffer_length = buffer_length,
                                           rate = rate,
                                           test_duration = test_duration,
                                           gap_frequency = gap_frequency,
                                           gap_threshold = gap_threshold,
                                           control_dump_filename = "iirbank_test_01a_control.dump")
        print "After Test Src"
        head = tee = pipeparts.mktee(pipeline, head)

        #
        # Create IIR bank
        #

	#a1, b0, delay = cbc_template_iir.makeiirbank()
#       the = 1.0

        #ip = spawaveform.iirinnerproduct(a1, b0, delay, psd)

        M = numpy.loadtxt('test.out')
        amp = M[:,0]
        phase = M[:,1]
        a1, b0, delay = spawaveform.iir(amp, phase, 0.01, 0.2, 0.2)
        psd = numpy.ones(amp.shape[0]/2)

        #
        # Build rest of pipeline
        #

        head = pipeparts.mkiirbank(pipeline, head, a1 = a1, b0 = b0, delay = delay)

        head = outtee = pipeparts.mktee(pipeline, head)
        #outtee = tee = pipeparts.mktee(pipeline, tee)
        pipeparts.mkplaybacksink(pipeline, pipeparts.mkqueue(pipeline,outtee))

        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "iirbank_test_01a_out.dump")
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "iirbank_test_01a_in.dump")

        #
        # done
        #
        # gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_NON_DEFAULT_PARAMS, "iir_single_waveform_pipeline")

        return pipeline


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#

#name = "iirbank_test_01a"
#mainloop = gobject.MainLoop()
#pipeline = gst.Pipeline(name)
#handler = Handler(mainloop, iirbank_test_01a(pipeline))

#pipeline = gst.Pipeline()
#pipeline = iirbank_test_01a(pipeline)

#gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_NON_DEFAULT_PARAMS, "iir_single_waveform_pipeline")

#pipeline.set_state(gst.STATE_PLAYING)

test_common.build_and_run(iirbank_test_01a, "iirbank_test_01a")


#numpy.zeros((1, ), dtype = "cdouble")

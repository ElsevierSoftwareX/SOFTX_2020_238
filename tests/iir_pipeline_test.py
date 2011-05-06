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
from gstlal.pipeutil import gobject, gst

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

        rate = 4096             # Sample rate in Hz
        gap_frequency = None    # Hz
        gap_threshold = 0.0     # of 1
        buffer_length = 1.0     # seconds
        test_duration = 15.0    # seconds
        wave = 1                # 0 sine wave, 1 square wave, 8 ticks
        volume = 0.1            # volume
        freq = 0.1              # frequency

        #
        # Create IIR bank
        #

	amp, phase, f = cbc_template_iir.waveform(1.4, 1.4, 40, 1500, 4096)
	f = open("template.txt", "w")
	for n in range(len(amp)):
		print >>f, "%f, %f" % (amp[n], phase[n])
	f.close()

        a1, b0, delay = spawaveform.iir(amp, phase, 0.04, 0.9, 0.25)
	out = spawaveform.iirresponse(test_duration * rate, a1, b0, delay)
	f = open("response.txt","w")
	for n in range(len(out)):
		print >>f, "%f, %f" % (out[n].real, out[n].imag)
	f.close()
        psd = numpy.ones(amp.shape[0]/2)
	a1 =numpy.array([a1])
	b0 =numpy.array([b0])
	delay =numpy.array([delay])
	print a1.shape, b0.shape, delay.shape

        #
        # build pipeline
        #

        head = test_common.gapped_test_src(pipeline,
                                           wave = wave,
                                           freq = freq,
					   buffer_length = buffer_length,
                                           rate = rate,
                                           test_duration = test_duration,
                                           gap_frequency = gap_frequency,
                                           gap_threshold = gap_threshold,
                                           control_dump_filename = "iirbank_test_01a_control.dump")

        head = tee = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "iirbank_test_01a_in.dump")

        head = pipeparts.mkiirbank(pipeline, head, a1 = a1, b0 = b0, delay = delay)
	head = outtee = pipeparts.mktee(pipeline, head)

	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, outtee), "iirbank_test_01a_out.dump")

        pipeparts.mkplaybacksink(pipeline, pipeparts.mkqueue(pipeline, head))

        #
        # done
        #

	gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_NON_DEFAULT_PARAMS, "iir_single_waveform_pipeline_new")
	print >>sys.stderr, "finished pipeline"
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
#handler = test_common.Handler(mainloop, iirbank_test_01a(pipeline))

#pipeline = gst.Pipeline()
#pipeline = iirbank_test_01a(pipeline)



#pipeline.set_state(gst.STATE_PLAYING)

test_common.build_and_run(iirbank_test_01a, "iirbank_test_01a")

print >>sys.stderr, "exit"
#numpy.zeros((1, ), dtype = "cdouble")

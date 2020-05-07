#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import numpy
import sys, os
from gstlal import pipeparts
import test_common, gst
from gstlal.snglinspiraltable import GSTLALSnglInspiral

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


def peak_test_01a(pipeline):
	#
	# try changing these.  test should still work!
	#

	in_rate = 32	# Hz
	sine_frequency = 1
	gap_frequency = 0.1	# Hz
	gap_threshold = 0.7	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	peak_window = 16 	# samples
	wave = 0

	#
	# build pipeline
	#

	head = test_common.gapped_complex_test_src(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, wave = wave, freq = sine_frequency, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "itac_test_01a_control.dump", tags = "instrument=H1,channel-name=LSC-STRAIN,units=strain")
	head = tee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkqueue(pipeline, pipeparts.mkitac(pipeline, head, peak_window, "test_bank.xml", autocorrelation_matrix = numpy.array([[0+0.j, 0+0.j, 1+1.j, 0+0.j, 0+0.j]])))
	head = pipeparts.mkprogressreport(pipeline, head, "test")

	#
	# output the before and after
	#
	
	a = pipeparts.mkappsink(pipeline, pipeparts.mkqueue(pipeline, head))

	outfile = open("itac_test_01a_out.dump", "w")

	def dump_triggers(elem, output = outfile):
		for row in GSTLALSnglInspiral.from_buffer(elem.emit("pull-buffer")):
			print(row.end_time + row.end_time_ns*1e-9, row.snr, row.chisq, row.chisq_dof, file=outfile)

	a.connect_after("new-buffer", dump_triggers)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, tee)), "itac_test_01a_in.dump")

	#
	# done
	#

	if "GST_DEBUG_DUMP_DOT_DIR" in os.environ:
		gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_ALL, "peak_test_01a")

	return pipeline

#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


test_common.build_and_run(peak_test_01a, "peak_test_01a")


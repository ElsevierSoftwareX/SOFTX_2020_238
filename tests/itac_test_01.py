#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import numpy
import sys
from gstlal import pipeparts
import test_common
from pylal.xlal.datatypes.snglinspiraltable import from_buffer as sngl_inspirals_from_buffer

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
	gap_threshold = 0.5	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 5.0	# seconds
	peak_window = 16 	# samples
	wave = 0

	#
	# build pipeline
	#

	head = test_common.complex_test_src(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, wave = wave, freq = sine_frequency)
	head = pipeparts.mktaginject(pipeline, head, "instrument=H1,channel-name=LSC-STRAIN,units=strain")
	head = tee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkqueue(pipeline, pipeparts.mkitac(pipeline, head, peak_window, "test_bank.xml"))
	
	#
	# output the before and after
	#
	
	a = pipeparts.mkappsink(pipeline, pipeparts.mkqueue(pipeline, head))

	outfile = open("itac_test_01a_out.dump", "w")

	def dump_triggers(elem, output = outfile):
		for row in sngl_inspirals_from_buffer(elem.emit("pull-buffer")):
			print >>outfile, row.end_time + row.end_time_ns*1e-9, row.snr

	a.connect_after("new-buffer", dump_triggers)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, tee)), "itac_test_01a_in.dump")

	#
	# done
	#

	return pipeline

#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


test_common.build_and_run(peak_test_01a, "peak_test_01a")


#!/opt/local/bin/python

import os

bank_file = """
<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
<LIGO_LW>
	<Table Name="sngl_burst:table">
		<Column Type="int_4s" Name="sngl_burst:peak_time_ns"/>
		<Column Type="int_4s" Name="sngl_burst:start_time_ns"/>
		<Column Type="int_4s" Name="sngl_burst:stop_time_ns"/>
		<Column Type="ilwd:char" Name="sngl_burst:process_id"/>
		<Column Type="lstring" Name="sngl_burst:ifo"/>
		<Column Type="int_4s" Name="sngl_burst:peak_time"/>
		<Column Type="int_4s" Name="sngl_burst:start_time"/>
		<Column Type="int_4s" Name="sngl_burst:stop_time"/>
		<Column Type="real_4" Name="sngl_burst:duration"/>
		<Column Type="real_4" Name="sngl_burst:time_lag"/>
		<Column Type="real_4" Name="sngl_burst:peak_frequency"/>
		<Column Type="lstring" Name="sngl_burst:search"/>
		<Column Type="real_4" Name="sngl_burst:central_freq"/>
		<Column Type="lstring" Name="sngl_burst:channel"/>
		<Column Type="real_4" Name="sngl_burst:amplitude"/>
		<Column Type="real_4" Name="sngl_burst:snr"/>
		<Column Type="real_4" Name="sngl_burst:confidence"/>
		<Column Type="real_8" Name="sngl_burst:chisq"/>
		<Column Type="real_8" Name="sngl_burst:chisq_dof"/>
		<Column Type="real_4" Name="sngl_burst:flow"/>
		<Column Type="real_4" Name="sngl_burst:fhigh"/>
		<Column Type="real_4" Name="sngl_burst:bandwidth"/>
		<Column Type="real_4" Name="sngl_burst:tfvolume"/>
		<Column Type="real_4" Name="sngl_burst:hrss"/>
		<Column Type="ilwd:char" Name="sngl_burst:event_id"/>
		<Stream Name="sngl_burst:table" Type="Local" Delimiter=",">
			0,0,0,"process:process_id:0","H1",0,0,0,0.0625,0,24,"gstlal_excesspower",24,"awesome full of GW channel",0,0,0,0,1,20,28,8,0,0,"sngl_burst:event_id:0"
		</Stream>
	</Table>
</LIGO_LW>
"""
bank_filename = "gstlal_excesspower_bank_H1_FAKE-STRAIN_level_0.xml"
bank_fd = open( bank_filename, "w" )
print(bank_file, file=bank_fd)
bank_fd.close()

import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

from pylal.xlal.datatypes.snglburst import SnglBurst
from gstlal import pipeio
from gstlal.pipeutil import gstlal_element_register
from gstlal import pipeparts

pipeline = gst.Pipeline("pipeline")
mainloop = gobject.MainLoop()

funnel = gst.element_factory_make("funnel")
pipeline.add(funnel)

for i in range(5):

	"""
	head = gst.element_factory_make( "audiotestsrc" )
	head.set_property( "wave", 9 )
	head.set_property( "volume", 1.0 )
	pipeline.add(head)

	elem = gst.element_factory_make( "taginject" )
	elem.set_property( "tags", "instrument=\"H1\",channel-name=\"FAKE-STRAIN-%d\"" % i )
	pipeline.add(elem)
	head.link(elem)
	head = elem
	"""

	head = pipeparts.mkfakeadvLIGOsrc( pipeline, 
		instrument="H1", 
		channel_name = "FAKE-STRAIN-%d" % i 
	)
	head = pipeparts.mkprogressreport( pipeline, head, name = "data_%d" % i )

	head = pipeparts.mkbursttriggergen( pipeline, head, 
		n = int(1e5),
		bank = bank_filename 
	)
	head = pipeparts.mkprogressreport( pipeline, head, name = "triggers_%d" % i )
	head = pipeparts.mkqueue( pipeline, head )

	head.link(funnel)

head = funnel
head = pipeparts.mkprogressreport( pipeline, head, name = "triggers_all" )

head = pipeparts.mkgeneric( pipeline, head, "lal_hveto" )

pipeparts.mkfilesink( pipeline, head, filename="testing" )

os.remove( bank_filename )

#pipeparts.write_dump_dot(pipeline, "test", verbose = True)
pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()

"""
def get_triggers(elem):
	print("get triggers called")
	buffer = elem.emit("pull-buffer")
	for row in SnglBurst.from_buffer(buffer):
		print(row.channel)

appsink = gst.element_factory_make("appsink")
appsink.connect_after("new-buffer", get_triggers)
pipeline.add(appsink)
head.link(appsink)
"""

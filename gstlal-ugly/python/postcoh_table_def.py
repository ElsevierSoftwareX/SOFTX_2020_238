
from glue.ligolw import table
from glue.ligolw import ilwd
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS

PostcohInspiralID = ilwd.get_ilwdchar_class(u"postcoh", u"event_id")
# defined in postcohinspiral_table.h
class PostcohInspiralTable(table.Table):
	tableName = "postcoh:table"
	validcolumns = {
			"event_id":	"ilwd:char",
			"end_time":	"int_4s",
			"end_time_ns":	"int_4s",
			"end_time_L":	"int_4s",
			"end_time_ns_L":"int_4s",
			"end_time_H":	"int_4s",
			"end_time_ns_H":"int_4s",
			"end_time_V":	"int_4s",
			"end_time_ns_V":"int_4s",
			"snglsnr_L":	"real_4",
			"snglsnr_H":	"real_4",
			"snglsnr_V":	"real_4",
			"coaphase_L":	"real_4",
			"coaphase_H":	"real_4",
			"coaphase_V":	"real_4",
			"chisq_L":	"real_4",
			"chisq_H":	"real_4",
			"chisq_V":	"real_4",
			"is_background":"int_4s",
			"livetime":	"int_4s",
			"ifos":		"lstring",
			"pivotal_ifo":	"lstring",
			"tmplt_idx":	"int_4s",
			"pix_idx":	"int_4s",
			"cohsnr":	"real_4",
			"nullsnr":	"real_4",
			"cmbchisq":	"real_4",
			"spearman_pval":"real_4",
			"fap":		"real_4",
			"far_h":	"real_4",
			"far_l":	"real_4",
			"far_v":	"real_4",
			"far_h_1w":	"real_4",
			"far_l_1w":	"real_4",
			"far_v_1w":	"real_4",
			"far_h_1d":	"real_4",
			"far_l_1d":	"real_4",
			"far_v_1d":	"real_4",
			"far_h_2h":	"real_4",
			"far_l_2h":	"real_4",
			"far_v_2h":	"real_4",
			"far":		"real_4",
			"far_2h":	"real_4",
			"far_1d":	"real_4",
			"far_1w":	"real_4",
			"skymap_fname":	"lstring",
			"template_duration": "real_8",
			"mass1":	"real_4",
			"mass2":	"real_4",
			"mchirp":	"real_4",
			"mtotal":	"real_4",
			"spin1x":	"real_4",
			"spin1y":	"real_4",
			"spin1z":	"real_4",
			"spin2x":	"real_4",
			"spin2y":	"real_4",
			"spin2z":	"real_4",
			"eta":		"real_4",
			"ra":		"real_8",
			"dec":		"real_8",
			"deff_L":	"real_8",
			"deff_H":	"real_8",
			"deff_V":	"real_8"
	}
	constraints = "PRIMARY KEY (event_id)"
	next_id = PostcohInspiralID(1)

class PostcohInspiral(table.TableRow):
	__slots__ = PostcohInspiralTable.validcolumns.keys()

	#
	# Properties
	#

	@property
	def end(self):
		if self.end_time is None and self.end_time_ns is None:
			return None
		return LIGOTimeGPS(self.end_time, self.end_time_ns)

	@end.setter
	def end(self, gps):
		if gps is None:
			self.end_time = self.end_time_ns = None
		else:
			self.end_time, self.end_time_ns = gps.gpsSeconds, gps.gpsNanoSeconds


PostcohInspiralTable.RowType = PostcohInspiral

# ref: glue.ligolw.lsctables
# Override portions of a lsctables.LIGOLWContentHandler class
#

TableByName = {
		table.StripTableName(PostcohInspiralTable.tableName): PostcohInspiralTable
		}


def use_in(ContentHandler):
	"""
	Modify ContentHandler, a sub-class of
	glue.ligolw.LIGOLWContentHandler, to cause it to use the Table
	classes defined in this module when parsing XML documents.

	Example:

	>>> from glue.ligolw import ligolw
	>>> class MyContentHandler(ligolw.LIGOLWContentHandler):
	...	pass
	...
	>>> use_in(MyContentHandler)
	<class 'glue.ligolw.lsctables.MyContentHandler'>
	"""
	#ContentHandler = table.use_in(ContentHandler)

	def startTable(self, parent, attrs, __orig_startTable = ContentHandler.startTable):
		name = table.StripTableName(attrs[u"Name"])
		if name in TableByName:
			return TableByName[name](attrs)
		return __orig_startTable(self, parent, attrs)

	ContentHandler.startTable = startTable
	return ContentHandler


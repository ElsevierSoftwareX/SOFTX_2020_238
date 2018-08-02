
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import ilwd
from glue.ligolw import dbtables
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS
from xml.sax.xmlreader import AttributesImpl
# so they can be inserted into a database
dbtables.ligolwtypes.ToPyType["ilwd:char"] = unicode


PostcohInspiralID = ilwd.get_ilwdchar_class(u"postcoh", u"event_id")
# defined in postcohinspiral_table.h
class PostcohInspiralTable(table.Table):
	tableName = "postcoh"
	validcolumns = {
			"process_id":	"ilwd:char",
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
			"deff_V":	"real_8",
			"rank":		"real_8"
	}
	constraints = "PRIMARY KEY (event_id)"
	next_id = PostcohInspiralID(0)

class PostcohInspiral(table.Table.RowType):
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
		PostcohInspiralTable.tableName: PostcohInspiralTable
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
	# need to comment out the next clause in case there are other use_in performed
	# e.g. lsctables.use_in before this use_in
	#ContentHandler = table.use_in(ContentHandler)

	def startTable(self, parent, attrs, __orig_startTable = ContentHandler.startTable):
		name = table.Table.TableName(attrs[u"Name"])
		if name in TableByName:
			return TableByName[name](attrs)
		return __orig_startTable(self, parent, attrs)

	ContentHandler.startTable = startTable
	return ContentHandler


class PostcohInspiralDBTable(dbtables.DBTable):
	tableName = PostcohInspiralTable.tableName
	validcolumns = PostcohInspiralTable.validcolumns
	constraints = PostcohInspiralTable.constraints
	next_id = PostcohInspiralTable.next_id
	RowType = PostcohInspiralTable.RowType
	how_to_index = PostcohInspiralTable.how_to_index

DBTableByName = {
		PostcohInspiralDBTable.tableName: PostcohInspiralDBTable
		}


def DB_use_in(ContentHandler):
	"""
	Modify ContentHandler, a sub-class of
	glue.ligolw.LIGOLWContentHandler, to cause it to use the DBTable
	class defined in this module when parsing XML documents.  Instances
	of the class must provide a connection attribute.  When a document
	is parsed, the value of this attribute will be passed to the
	DBTable class' .__init__() method as each table object is created,
	and thus sets the database connection for all table objects in the
	document.

	Example:

	>>> import sqlite3
	>>> from glue.ligolw import ligolw
	>>> class MyContentHandler(ligolw.LIGOLWContentHandler):
	...	def __init__(self, *args):
	...		super(MyContentHandler, self).__init__(*args)
	...		self.connection = sqlite3.connection()
	...
	>>> use_in(MyContentHandler)

	Multiple database files can be in use at once by creating a content
	handler class for each one.
	"""

	def startTable(self, parent, attrs, __orig_startTable = ContentHandler.startTable):
		name = table.Table.TableName(attrs[u"Name"])
		if name in DBTableByName:
			return DBTableByName[name](attrs, connection = self.connection)
		return __orig_startTable(self, parent, attrs)

	ContentHandler.startTable = startTable
	return ContentHandler

"""
get_xml() will use the custom postcoh_table_def.DBTableByName to ganrantee valid column types when sqlite->xml
e.g. sqlite REAL to real_4 instead of the default real_8
original code: dbtables.get_xml()

"""

def get_xml(connection, table_names = None):
	"""
	Construct an XML document tree wrapping around the contents of the
	database.  On success the return value is a ligolw.LIGO_LW element
	containing the tables as children.  Arguments are a connection to
	to a database, and an optional list of table names to dump.  If
	table_names is not provided the set is obtained from get_table_names()
	"""
	ligo_lw = ligolw.LIGO_LW()

	if table_names is None:
		table_names = dbtables.get_table_names(connection)

	for table_name in table_names:
		# build the table document tree.  copied from
		# lsctables.New()
		try:
			cls = DBTableByName[table_name]
		except KeyError:
			cls = dbtables.DBTable
		table_elem = cls(AttributesImpl({u"Name": u"%s:table" % table_name}), connection = connection)
		for column_name, column_type in dbtables.get_column_info(connection, table_elem.Name):
			if table_elem.validcolumns is not None:
				# use the pre-defined column type
				column_type = table_elem.validcolumns[column_name]
			else:
				# guess the column type
				column_type = ligolwtypes.FromSQLiteType[column_type]
			table_elem.appendChild(table.Column(AttributesImpl({u"Name": u"%s:%s" % (table_name, column_name), u"Type": column_type})))
		table_elem._end_of_columns()
		table_elem.appendChild(table.TableStream(AttributesImpl({u"Name": u"%s:table" % table_name, u"Delimiter": table.TableStream.Delimiter.default, u"Type": table.TableStream.Type.default})))
		ligo_lw.appendChild(table_elem)
	return ligo_lw




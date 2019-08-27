#!/usr/bin/env python
#
# Copyright (C) 2019  Kipp Cannon
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


"""
Revert an XML document tree to "iwld:char" format for upload to gracedb.
This does not solve the generic problem of document conversion, it is
specifically written for the case of gstlal_inspiral's gracedb uploads.
"""


from glue.ligolw import ilwd
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from ligo.lw.lsctables import TableByName as ligo_lw_TableByName
from ligo.lw.param import Param as ligo_lw_Param
from ligo.lw.table import Column as ligo_lw_Column


#
# dictionary mapping lsctables table class to dictionary mapping column
# name to ilwd:char class.  we only consider tables that are named in both
# glue.ligolw.lsctables and ligo.lw.lsctables, assuming we won't be
# converting documents that contain tables that have been removed from the
# latter.
#

ilwdchar_tables = dict((tblname, dict((ligo_lw_Column.ColumnName(colname), None) for colname, coltype in tblcls.validcolumns.items() if coltype == u"ilwd:char")) for tblname, tblcls in lsctables.TableByName.items() if tblname in ligo_lw_TableByName and u"ilwd:char" in tblcls.validcolumns.values())

destrip_column = dict((key, dict(value)) for key, value in ilwdchar_tables.items())
for tblname, colnamemap in destrip_column.items():
	for colname in list(colnamemap):
		colnamemap[colname], = (destripped for destripped in ligo_lw_TableByName[tblname].validcolumns if ligo_lw_Column.ColumnName(destripped) == colname)

for tblname, colnamemap in ilwdchar_tables.items():
	for colname in list(colnamemap):
		destripped = destrip_column[tblname][colname]
		try:
			tblprefix = ligo_lw_Column.ColumnName.table_name(destripped)
		except ValueError:
			# columns whose names don't have a prefix are in
			# their own table
			tblprefix = tblname
		colnamemap[colname] = ilwd.get_ilwdchar_class(tblprefix, ligo_lw_Column.ColumnName(colname))


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


def do_it_to(xmldoc):
	"""
	NOTE:  this performs an in-place transcription of the contents of
	the XML document tree.  This should be assumed to be a destructive
	operation on the contents of the tree.  If you wish to hold
	references to any of the Table elements or other structures in the
	tree and wish them to remain intact so they can be used afterwards,
	make copies first
	"""
	#
	# walk the tree finding Table elements
	#

	for table in list(xmldoc.getElementsByTagName(ligolw.Table.tagName)):
		#
		# this is not the table we're looking for
		#

		if table.Name not in ilwdchar_tables:
			continue

		#
		# make a copy of the table with glue.ligolw's lsctables and
		# replace the old table with the new table in the XML tree
		#

		newtable = table.parentNode.replaceChild(lsctables.New(lsctables.TableByName[table.Name], table.columnnames), table)

		#
		# build a row transcription function for this table
		#

		if table.Name != "coinc_event_map":
			ilwdclsmap = ilwdchar_tables[table.Name]
			newrowtype = newtable.RowType
			def newrow(row, nonilwdcharattrs = tuple(colname for colname in table.columnnames if colname not in ilwdclsmap), ilwdcharattrs = tuple(colname for colname in table.columnnames if colname in ilwdclsmap)):
				kwargs = dict((attr, getattr(row, attr)) for attr in nonilwdcharattrs)
				kwargs.update((attr, ilwdclsmap[attr](getattr(row, attr))) for attr in ilwdcharattrs)
				return newrowtype(**kwargs)
		else:
			# event_id IDs obtain their table name prefix from
			# the table_name column
			newrowtype = newtable.RowType
			def newrow(row, coinc_id_ilwdcls = ilwdchar_tables["coinc_event"]["coinc_event_id"]):
				# FIXME this is probably a dumb way to do this,
				# but it shouldn't matter once we have no
				# reason to convert back to ilwdchar
				if "event_id" in ilwdchar_tables[row.table_name]:
					event_id = ilwdchar_tables[row.table_name]["event_id"](row.event_id)
				elif "simulation_id" in ilwdchar_tables[row.table_name]:
					event_id = ilwdchar_tables[row.table_name]["simulation_id"](row.event_id)
				elif "coinc_event_id" in ilwdchar_tables[row.table_name]:
					event_id = ilwdchar_tables[row.table_name]["coinc_event_id"](row.event_id)
				else:
					raise KeyError("event_id, simulation_id or coinc_event_id not in " +  ilwdchar_tables[row.table_name])
				return newrowtype(
					table_name = row.table_name,
					event_id = event_id,
					coinc_event_id = coinc_id_ilwdcls(row.coinc_event_id)
				)

		#
		# transcribe rows from the old table into the new table
		#

		newtable.extend(newrow(row) for row in table)

		#
		# dispose of the old table
		#

		table.unlink()

	#
	# walk the tree looking for Param elements containing sngl_inspiral
	# IDs and convert to ilwd:char
	#

	ilwdcls = ilwdchar_tables["sngl_inspiral"]["event_id"]
	for param in list(ligo_lw_Param.getParamsByName(xmldoc, "event_id")):
		param.Type = u"ilwd:char"
		param.pcdata = ilwdcls(param.pcdata)

	#
	# done
	#

	return xmldoc

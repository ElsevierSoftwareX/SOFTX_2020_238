#!/usr/bin/env python3

from glue.ligolw import lsctables
from glue.ligolw import utils
import sys

filename, out_filename = sys.argv[1:3]

bank_xmldoc = utils.load_filename(filename, gz=filename.endswith(".gz"))
bank_sngl_table = lsctables.table.get_table( bank_xmldoc,lsctables.SnglInspiralTable.tableName )

mass_pairs = set()
for i, child in enumerate(bank_sngl_table):
	mass_pair = (child.mass1, child.mass2)
	if mass_pair in mass_pairs:
		del bank_sngl_table[i]
	mass_pairs.add(mass_pair)

utils.write_filename(bank_xmldoc, out_filename, gz=out_filename.endswith(".gz"))

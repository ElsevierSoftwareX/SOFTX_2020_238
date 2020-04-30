#!/usr/bin/env python3
import sys
from glue.ligolw import lsctables
from glue.ligolw import utils

fname = sys.argv[1]

xmldoc = utils.load_filename(fname)
tbl = lsctables.table.get_table(xmldoc, "sim_inspiral")
start = tbl[0].geocent_end_time - 100
for row in tbl: row.geocent_end_time -= start
utils.write_filename(xmldoc, fname)

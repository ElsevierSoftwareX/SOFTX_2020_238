import sys
from glue.ligolw import lsctables
from glue.ligolw import utils

for filename in sys.argv[1:]:
	for row in lsctables.table.get_table(utils.load_filename(filename, gz = (filename or "stdin").endswith(".gz"), verbose = True), "sngl_inspiral"):
		print row.get_end(), row.snr, row.chisq

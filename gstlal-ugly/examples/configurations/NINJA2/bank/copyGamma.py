#!/usr1/bin/python

import sys
import glue
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils

lalapps_tmplt_file = sys.argv[1]
sbank_tmplt_file = sys.argv[2]

lalapps_tmplt_xmldoc=utils.load_filename(lalapps_tmplt_file, gz=lalapps_tmplt_file.endswith(".gz"))
lalapps_sngl_inspiral_table=lsctables.table.get_table(lalapps_tmplt_xmldoc, lsctables.SnglInspiralTable.tableName)

sbank_tmplt_xmldoc=utils.load_filename(sbank_tmplt_file, gz=sbank_tmplt_file.endswith(".gz"))
sbank_sngl_inspiral_table=lsctables.table.get_table(sbank_tmplt_xmldoc, lsctables.SnglInspiralTable.tableName)

Gamma0=lalapps_sngl_inspiral_table[0].Gamma0
Gamma1=lalapps_sngl_inspiral_table[0].Gamma1
Gamma2=lalapps_sngl_inspiral_table[0].Gamma2
Gamma3=lalapps_sngl_inspiral_table[0].Gamma3
Gamma4=lalapps_sngl_inspiral_table[0].Gamma4
Gamma5=lalapps_sngl_inspiral_table[0].Gamma5
Gamma6=lalapps_sngl_inspiral_table[0].Gamma6
Gamma7=lalapps_sngl_inspiral_table[0].Gamma7
Gamma8=lalapps_sngl_inspiral_table[0].Gamma8
Gamma9=lalapps_sngl_inspiral_table[0].Gamma9

for row in sbank_sngl_inspiral_table:
	row.Gamma0=Gamma0
	row.Gamma1=Gamma1
	row.Gamma2=Gamma2
	row.Gamma3=Gamma3
	row.Gamma4=Gamma4
	row.Gamma5=Gamma5
	row.Gamma6=Gamma6
	row.Gamma7=Gamma7
	row.Gamma8=Gamma8
	row.Gamma9=Gamma9

#output=filename
utils.write_filename(sbank_tmplt_xmldoc, sbank_tmplt_file)

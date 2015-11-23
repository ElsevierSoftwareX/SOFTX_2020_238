
from glue.ligolw import table

# defined in postcohinspiral_table.h
class PostcohInspiralTable(table.Table):
	tableName = "postcoh:table"
	validcolumns = {
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
			"coa_phase_L":	"real_4",
			"coa_phase_H":	"real_4",
			"coa_phase_V":	"real_4",
			"is_background":"int_4s",
			"livetime":	"int_4s",
			"ifos":		"lstring",
			"pivotal_ifo":	"lstring",
			"tmplt_idx":	"int_4s",
			"pix_idx":	"int_4s",
			"maxsnglsnr":	"real_4",
			"cohsnr":	"real_4",
			"nullsnr":	"real_4",
			"chisq":	"real_4",
			"spearman_pval":"real_4",
			"fap":		"real_4",
			"far":		"real_4",
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
			"dec":		"real_8"
	}



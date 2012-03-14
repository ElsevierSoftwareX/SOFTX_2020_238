#!/usr1/bin/python

# Script to generate followup wiki tables
# Tables are printed to stdout
# To save as file redirect output 

import sys

#import from glue needed modules to access tables
from glue.ligolw import utils
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables

#get file names
snrs = sys.argv[1]
ifo_missed = sys.argv[2:]

#load file
xmldoc=utils.load_filename(snrs)
snr_table=lsctables.table.get_table(xmldoc, 'snr')

for missedFile in ifo_missed:
	xmldoc=utils.load_filename(missedFile)
	sim_inspiral_table=lsctables.table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)

	ifo = missedFile[:2]
	# get missed injection parameters for each ifo
	#geocent_end_time=[]
	#mass1=[]
	#mass2=[]
	#mchirp=[]
	#eta=[]
	#distance=[]
	#spin1z=[]
	#spin2z=[]
	#eff_dist_h=[]
	#eff_dist_l=[]
	#eff_dist_v=[]
	#f_lower=[]
	#f_final=[]
	#snr_h1=[]
	#snr_l1=[]
	#snr_v1=[]

	print '|| # || ', ifo, ' ||  geocent_end_time || mass1 || mass2 || row.mchirp || eta || distance || spin1z || spin2z || chi || eff_dist_h || eff_dist_l || eff_dist_v || f_lower || f_final || h1_snr || l1_snr || v1_snr ||' 

	for i,row in enumerate(sim_inspiral_table):
		#geocent_end_time.append(row.geocent_end_time)
		#mass1.append(row.mass1)
		#mass2.append(row.mass2)
		#mchirp.append(row.mchirp)
		#eta.append(row.eta)
		#distance.append(row.distance)
		#spin1z.append(row.spin1z)
		#spin2z.append(row.spin2z)
		#eff_dist_h.append(row.eff_dist_h)
		#eff_dist_l.append(row.eff_dist_l)
		#eff_dist_v.append(row.eff_dist_v)
		#f_lower.append(row.f_lower)
		#f_final.append(row.f_final)
	

		# get snrs of injections at the times of missed injections
		for snr_row in snr_table:
			if snr_row.geocent_end_time == row.geocent_end_time:
				#snr_h1.append(snr_row.h1_snr)
				#snr_l1.append(snr_row.l1_snr)
				#snr_v1.append(snr_row.v1_snr)

				#print values as strings in a wiki table format
				if (ifo=='H1' and row.eff_dist_h <= 250.0) or (ifo=='L1' and row.eff_dist_l <= 250.0) or (ifo=='V1' and row.eff_dist_v <= 250.0):
					print '||<#00FFFF>', i, '||<#00FFFF>', ifo, '||<#00FFFF>', row.geocent_end_time,'||<#00FFFF>',row.mass1,'||<#00FFFF>', row.mass2, '||<#00FFFF>', row.mchirp, '||<#00FFFF>', row.eta, '||<#00FFFF>', row.distance, '||<#00FFFF>', row.spin1z, '||<#00FFFF>', row.spin2z, '||<#00FFFF>', (row.mass1 * row.spin1z)/(row.mass1 + row.mass2), '||<#00FFFF>', row.eff_dist_h, '||<#00FFFF>', row.eff_dist_l, '||<#00FFFF>', row.eff_dist_v, '||<#00FFFF>', row.f_lower, '||<#00FFFF>', row.f_final, '||<#00FFFF>', snr_row.h1_snr, '||<#00FFFF>', snr_row.l1_snr, '||<#00FFFF>', snr_row.v1_snr, '||' 
				else:
					print '||', i, '||', ifo, '||', row.geocent_end_time,'||',row.mass1,'||', row.mass2, '||', row.mchirp, '||', row.eta, '||', row.distance, '||', row.spin1z, '||', row.spin2z, '||',(row.mass1 * row.spin1z)/(row.mass1 + row.mass2), '||', row.eff_dist_h, '||', row.eff_dist_l, '||', row.eff_dist_v, '||', row.f_lower, '||', row.f_final, '||', snr_row.h1_snr, '||', snr_row.l1_snr, '||', snr_row.v1_snr, '||' 

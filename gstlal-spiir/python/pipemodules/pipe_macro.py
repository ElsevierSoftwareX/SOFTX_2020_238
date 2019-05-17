# need to be consistent with pipe_macro.h
# log10_cohsnr
xmin = 0.54
xmax = 3.0
xstep = 0.0082
xbin = 300
# log10_chisq
ymin = -0.15
ymax = 3.5
ystep = 0.0122
ybin = 300

# rank
rankmin = 2
rankmax = 32
rankstep = 0.1 # 30/299
rankbin = 300

SOURCE_TYPE_BNS = 1
ONLINE_SEG_TYPE_NAME = "postcohprocessed"
STATS_XML_ID_NAME = "gstlal_postcohspiir_stats"
DETRSP_XML_ID_NAME = "gstlal_postcoh_detrsp_map"
DETRSP_XML_PARAM_NAME_GPS_START = "gps_start"
DETRSP_XML_PARAM_NAME_GPS_STEP = "gps_step"
SNR_RATE_SUFFIX	= "lgsnr_rate"
CHISQ_RATE_SUFFIX = "lgchisq_rate"
SNR_CHISQ_RATE_SUFFIX = "lgsnr_lgchisq_rate"	
SNR_CHISQ_PDF_SUFFIX = "lgsnr_lgchisq_pdf"	
BACKGROUND_XML_RANK_NAME = "background_rank"	
BACKGROUND_XML_FEATURE_NAME = "background_feature"	
RANK_MAP_SUFFIX = "rank_map"	
RANK_RATE_SUFFIX = "rank_rate"	
RANK_PDF_SUFFIX = "rank_pdf"	
RANK_FAP_SUFFIX = "rank_fap"	
SIGNAL_XML_FEATURE_NAME = "signal_feature"
SIGNAL_XML_RANK_NAME = "signal_rank"	



# This IFO_MAP should reflect the same order of ifos in include/pipe_macro.h
IFO_MAP = ["H1", "L1", "V1"]

import itertools
import re
def get_sorted_ifo_string(ifos_string):
	ifos = re.findall('..', ifos_string)
	ifo_idx_sorted = sorted([IFO_MAP.index(ifo) for ifo in ifos])
	sorted_ifo_list = [IFO_MAP[ifo_idx] for ifo_idx in ifo_idx_sorted]
	return ''.join(sorted_ifo_list)

def get_ifo_combos(ifos):
	""" return ifo combinations given a ifo list, e.g.:['H1','L1'] """
	# first make sure ifo_list is ordered the same ways as IFO_MAP
	ifo_idx_sorted = sorted([IFO_MAP.index(ifo) for ifo in ifos])
	sorted_ifo_list = [IFO_MAP[ifo_idx] for ifo_idx in ifo_idx_sorted]
	ifo_combos = []
	for combo_len in range(2, len(sorted_ifo_list)+1):
		this_combo_list = list(itertools.combinations(sorted_ifo_list, combo_len))
		combo_str_list = ["".join(list(i_combo_list)) for i_combo_list in this_combo_list]
		ifo_combos.extend(combo_str_list)
	return ifo_combos



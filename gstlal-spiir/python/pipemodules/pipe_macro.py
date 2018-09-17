# need to be consistent with pipe_macro.h
# cohsnr
xmin = 0.54
xmax = 3.0
xstep = 0.0082
xbin = 300
# chisq
ymin = -1.2
ymax = 3.5
ystep = 0.0173
ybin = 300

# rank
rankmin = -30
rankmax = 0
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
RANK_MAP_SUFFIX = "rank_map"	
RANK_RATE_SUFFIX = "rank_rate"	
RANK_PDF_SUFFIX = "rank_pdf"	
RANK_FAP_SUFFIX = "rank_fap"	
SIGNAL_XML_FEATURE_NAME = "signal_feature"
SIGNAL_XML_RANK_NAME = "signal_rank"	



# This IFO_MAP be reflect the same order of ifos of the IFO_COMBO_MAP in background_stats_utils.c
IFO_MAP = ["H1", "L1", "V1"]

import itertools
def get_ifo_combos(ifos):
    # make sure ifo_list is ordered the same ways as the postcoh code: L1, H1, V1, xx
    ifo_idx_sorted = sorted([IFO_MAP.index(ifo) for ifo in ifos])
    ordered_ifo_list = [IFO_MAP[ifo_idx] for ifo_idx in ifo_idx_sorted]
    ifo_combos = []
    for combo_len in range(2, len(ordered_ifo_list)+1):
        this_combo_list = list(itertools.combinations(ordered_ifo_list, combo_len))
        combo_str_list = ["".join(list(i_combo_list)) for i_combo_list in this_combo_list]
        ifo_combos.extend(combo_str_list)

    return ifo_combos



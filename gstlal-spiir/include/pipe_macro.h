#ifndef __PIPE_MACRO_H__
#define __PIPE_MACRO_H__
/* FIXME: upgrade to include more detectors like KAGRA */
#define IFO_LEN 2
#define MAX_IFOS 3
typedef struct _IFOType {
	const char* name;
	int index;
} IFOType;

static const IFOType IFOMap[MAX_IFOS] = {
	{"H1", 0},
	{"L1", 1},
	{"V1", 2},
};
#define MAX_IFO_COMBOS 7 // 2^3-1
static const IFOType IFOComboMap[MAX_IFO_COMBOS] = {
	{"H1", 0},
	{"L1", 1},
	{"V1", 2},
	{"H1L1", 3},
	{"H1V1", 4},
	{"L1V1", 5},
	{"H1L1V1", 6},
};
/* function given a random ifo, output the index in the IFOComboMap list, implemented in background_stats_utils.c */
int get_icombo(char *ifos);

#ifndef MAX_ALLIFO_LEN
#define MAX_ALLIFO_LEN 14
#endif

/* definition of array for background statistics */
#define LOGSNR_CMIN	0.54 // center of the first bin
#define LOGSNR_CMAX	3.0 // center of the last bin
#define LOGSNR_NBIN	300 // step is 0.01
#define LOGCHISQ_CMIN	-1.2
#define LOGCHISQ_CMAX	4.0
#define LOGCHISQ_NBIN	300

#define LOGRANK_CMIN	-30 // 10^-30, minimum cdf, extrapolating if less than this min
#define LOGRANK_CMAX	0 //
#define	LOGRANK_NBIN	300 // FIXME: enough for accuracy

/* array names in xml files */
#define	BACKGROUND_XML_FEATURE_NAME		"background_feature"
#define	SNR_RATES_SUFFIX			"_lgsnr_rates"
#define	CHISQ_RATES_SUFFIX			"_lgchisq_rates"
#define	SNR_CHISQ_RATES_SUFFIX			"_lgsnr_lgchisq_rates"	
#define	SNR_CHISQ_PDF_SUFFIX			"_lgsnr_lgchisq_pdf"	
#define	BACKGROUND_XML_RANK_NAME		"background_rank"	
#define	RANK_MAP_SUFFIX				"_rank_map"	
#define	RANK_RATES_SUFFIX			"_rank_rates"	
#define	RANK_PDF_SUFFIX				"_rank_pdf"	
#define	RANK_FAP_SUFFIX				"_rank_fap"	

#define	ZEROLAG_XML_FEATURE_NAME		"zerolag_feature"
#define	ZEROLAG_XML_RANK_NAME		"zerolag_rank"	
#define STATS_XML_ID_NAME   "gstlal_postcohspiir_stats"
#define STATS_XML_TYPE_BACKGROUND   1
#define STATS_XML_TYPE_ZEROLAG   2
#define STATS_XML_WRITE_START   1
#define STATS_XML_WRITE_END 2
#define MAX(a,b) (a>b?a:b)
#define	PNOISE_MIN_LIMIT	-30
#define	PSIG_MIN_LIMIT	-30
#define	LR_MIN_LIMIT	-30
#endif


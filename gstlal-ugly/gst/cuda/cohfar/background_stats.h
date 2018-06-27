/*
 * Copyright (C) 2015 Qi Chu <qi.chu@uwa.edu.au>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef __BACKGROUND_STATS_H__
#define __BACKGROUND_STATS_H__

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#define IFO_LEN 2

#define LOGSNR_CMIN	0.54 // center of the first bin
#define LOGSNR_CMAX	3.0 // center of the last bin
#define LOGSNR_NBIN	300 // step is 0.01
#define LOGCHISQ_CMIN	-1.2
#define LOGCHISQ_CMAX	4.0
#define LOGCHISQ_NBIN	300

#define LOGRANK_CMIN	-30 // 10^-30, minimum cdf, extrapolating if less than this min
#define LOGRANK_CMAX	0 //
#define	LOGRANK_NBIN	300 // FIXME: enough for accuracy

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

#define NSTATS_TO_PROMPT 50 //deprecated. supposed to be used to collect last 50 seconds of background stats.
typedef struct {
	double	cmin;
	double	cmax;
	int	nbin;
	double	step;
	double	step_2;
	void	*data; // gsl_vector_long
} Bins1D;

typedef struct {
	double	cmin_x;
	double	cmax_x;
	int	nbin_x;
	double	step_x;
	double	step_x_2;
	double	cmin_y;
	double	cmax_y;
	int	nbin_y;
	double	step_y;
	double	step_y_2;
	void	*data; //gsl_matrix or gsl_matrix_long
} Bins2D;

/*
 * Ranking statistics and its distribution from background counts
 */

typedef struct {
	Bins1D	*rank_pdf;
	Bins1D	*rank_fap;
	Bins1D	*rank_rates;
	Bins2D	*rank_map; // map of the lgsnr-lgchisq value to rank value
} RankingStats;


// FIXME: extend to 3D to include null-snr
typedef struct {
	Bins1D	*lgsnr_rates; // dimension 1 rates, lgsnr
	Bins1D	*lgchisq_rates; // dimension 2 rates, lgchisq
	Bins2D	*lgsnr_lgchisq_rates; // histogram of the (lgsnr,lgchisq) from background
	Bins2D  *lgsnr_lgchisq_pdf;
} FeatureStats;


/*
 * background statistics
 */

typedef struct {
	char	*ifos;
	RankingStats *rank;
	FeatureStats *feature;
	int hist_trials;
	long nevent;
	long livetime;
} BackgroundStats;


typedef	BackgroundStats** BackgroundStatsPointer;
typedef struct {
	BackgroundStatsPointer *plist;
	int size;
	int pos;
} BackgroundStatsPointerList; 


#endif /* __BACKGROUND_STATS_H__ */

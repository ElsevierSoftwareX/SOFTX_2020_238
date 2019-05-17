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
#include <pipe_macro.h>

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
	Bins1D	*rank_rate;
	Bins2D	*rank_map; // map of the lgsnr-lgchisq value to rank value
	double mean_rankmap;
} RankingStats;


// FIXME: extend to 3D to include null-snr
typedef struct {
	Bins1D	*lgsnr_rate; // dimension 1 rates, lgsnr
	Bins1D	*lgchisq_rate; // dimension 2 rates, lgchisq
	Bins2D	*lgsnr_lgchisq_rate; // histogram of the (lgsnr,lgchisq) from background
	Bins2D  *lgsnr_lgchisq_pdf;
} FeatureStats;


/*
 * background or foreground (zerolag) statistics
 */

typedef struct {
	char	*ifos;
	RankingStats *rank;
	FeatureStats *feature;
	int hist_trials;
	long feature_nevent;
	long rank_nevent;
	long feature_livetime;
	long rank_livetime;
} TriggerStats;

typedef struct {
    TriggerStats ** multistats;
	GString *rank_xmlname;
	GString *feature_xmlname;
    int ncombo;
} TriggerStatsXML;

typedef	TriggerStats** TriggerStatsPointer;
typedef struct {
	TriggerStatsPointer *plist;
	int size;
	int pos;
} TriggerStatsPointerList; 


#endif /* __BACKGROUND_STATS_H__ */

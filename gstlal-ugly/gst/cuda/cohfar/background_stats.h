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
#define MAX_COMBOS 4

#define LOGSNR_CMIN	0.54 // center of the first bin
#define LOGSNR_CMAX	3.0 // center of the last bin
#define LOGSNR_NBIN	300 // step is 0.01
#define LOGCHISQ_CMIN	-1.2
#define LOGCHISQ_CMAX	4.0
#define LOGCHISQ_NBIN	300

#define	BACKGROUND_XML_RATES_NAME "background_rates"
#define	BACKGROUND_XML_SNR_SUFFIX "_lgsnr"
#define	BACKGROUND_XML_CHISQ_SUFFIX "_lgchisq"
#define	BACKGROUND_XML_HIST_SUFFIX "_histogram"	
#define	BACKGROUND_XML_PDF_NAME	"background_pdf"	
#define	BACKGROUND_XML_FAP_NAME	"background_fap"	
#define BACKGROUND_XML_SNR_CHISQ_SUFFIX "_lgsnr_lgchisq"


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
 * background (SNR, chisq) rates
 */

typedef struct {
	Bins1D	*lgsnr_bins;
	Bins1D	*lgchisq_bins;
	Bins2D	*hist;
} BackgroundRates;


/*
 * background statistics
 */

typedef struct {
	char	*ifos;
	BackgroundRates *rates;
	Bins2D	*pdf;
	Bins2D	*fap;
	int hist_trials;
	long nevent;
	long duration;
} BackgroundStats;

#endif /* __BACKGROUND_STATS_H__ */

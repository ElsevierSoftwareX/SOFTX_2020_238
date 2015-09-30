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

typedef struct {
	float	min;
	float	max;
	int	nbin;
	float	step;
	gsl_vector	*data;
} Bins1D;

typedef struct {
	float	x_min;
	float	x_max;
	int	x_nbin;
	float	x_step;
	float	y_min;
	float	y_max;
	int	y_nbin;
	float	y_step;
	gsl_matrix	*data;
} Bins2D;

/*
 * background (SNR, chisq) rates
 */

typedef struct {
	Bins1D	*logsnr_bins;
	Bins1D	*logchisq_bins;
} BackgroundRates;


/*
 * background statistics
 */

typedef struct {
	char	*ifos;
	BackgroundRates *rates;
	Bins2D	*pdf;
	Bins2D	*cdf;
} BackgroundStats;

#endif /* __BACKGROUND_STATS_H__ */

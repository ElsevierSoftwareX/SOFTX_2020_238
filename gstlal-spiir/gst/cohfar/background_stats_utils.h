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

#ifndef __BACKGROUND_STATS_UTILS_H__
#define __BACKGROUND_STATS_UTILS_H__

#include <glib.h>
#include "background_stats.h"

#define IFO_LEN 2
#define MAX_COMBOS 4

#define LOGSNR_CMIN	0.6 // center of the first bin
#define LOGSNR_CMAX	2.49 // center of the last bin
#define LOGSNR_NBIN	190 // step is 0.01
#define LOGCHISQ_CMIN	-0.5
#define LOGCHISQ_CMAX	2.49
#define LOGCHISQ_NBIN	300

extern char *IFO_COMBO_MAP[];

int get_icombo(char *ifos);
	
Bins1D *
bins1D_create_long(double cmin, double cmax, int nbin);

Bins2D *
bins2D_create(double x_cmin, double x_cmax, int x_nbin, double y_cmin, double y_cmax, int y_nbin);

Bins2D *
bins2D_create_long(double x_cmin, double x_cmax, int x_nbin, double y_cmin, double y_cmax, int y_nbin);

BackgroundStats **
background_stats_create(char *ifos);

int
get_idx_bins1D(double val, Bins1D *bins);

void
background_stats_rates_update(double snr, double chisq, BackgroundRates *rates);

void
background_stats_rates_add(BackgroundRates *rates1, BackgroundRates *rates2);

gboolean
background_stats_rates_to_pdf(BackgroundRates *rates, Bins2D *pdf);

void
background_stats_pdf_to_cdf(Bins2D *pdf, Bins2D *cdf);


double
background_stats_bins2D_get_val(double snr, double chisq, Bins2D *bins);

gboolean
background_stats_from_xml(BackgroundStats **stats, const int ncombo, const char *filename);

gboolean
background_stats_to_xml(BackgroundStats **stats, const int ncombo, const char *filename);

#endif /* __BACKGROUND_STATS_UTILS_H__ */


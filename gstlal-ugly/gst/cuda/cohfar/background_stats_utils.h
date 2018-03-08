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
#include <cohfar/background_stats.h>

extern char *IFO_COMBO_MAP[];

int get_icombo(char *ifos);
	
Bins1D *
bins1D_create_long(double cmin, double cmax, int nbin);

Bins2D *
bins2D_create(double cmin_x, double cmax_x, int nbin_x, double cmin_y, double cmax_y, int nbin_y);

Bins2D *
bins2D_create_long(double cmin_x, double cmax_x, int nbin_x, double cmin_y, double cmax_y, int nbin_y);

BackgroundStats **
background_stats_create(char *ifos);

int
bins1D_get_idx(double val, Bins1D *bins);

void
background_stats_feature_rates_update(double snr, double chisq, FeatureStats *feature, BackgroundStats *cur_stats);
 
void
background_stats_feature_rates_add(FeatureStats *feature1, FeatureStats *feature2, BackgroundStats *cur_stats);
 

void
background_stats_feature_rates_to_pdf(FeatureStats *feature);


double
bins2D_get_val(double snr, double chisq, Bins2D *bins);

gboolean
background_stats_from_xml(BackgroundStats **stats, const int ncombo, int *hist_trials, const char *filename);

gboolean
background_stats_to_xml(BackgroundStats **stats, const int ncombo, int hist_trials, const char *filename);

double
gen_fap_from_feature(double snr, double chisq, BackgroundStats *stats);
#endif /* __BACKGROUND_STATS_UTILS_H__ */


/*
 * Copyright (C) 2015 Qi Chu <qi.chu@ligo.org>
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


#include <cohfar/background_stats_utils.h>

#include <math.h>
#include <string.h>
#include <gsl/gsl_vector_long.h>
#include <gsl/gsl_matrix.h>

#include <cohfar/ssvkernel.h>
#include <cohfar/knn_kde.h>
#include <pipe_macro.h>
#include <postcohtable.h>

#define RANK_MIN_LIMIT 1e-100
#define EPSILON 1e-6

int scan_trigger_ifos(int icombo, PostcohInspiralTable * trigger) {
	int cur_ifo = 0, one_ifo_size = sizeof(char) * IFO_LEN;
	char final_ifos[MAX_ALLIFO_LEN];
	gboolean pass_test = TRUE;
	if ((icombo == 6 || icombo == 3 || icombo == 4)) { // H1L1, H1V1 or H1L1V1
		if (trigger->snglsnr_H > EPSILON) {
			strncpy(final_ifos + IFO_LEN*cur_ifo, "H1", one_ifo_size);
			cur_ifo++;
		}
		else
			pass_test = FALSE;
	}

	if ((icombo == 6 || icombo == 3 || icombo == 5)) { // H1L1, L1V1 or H1L1V1
		if (trigger->snglsnr_L > EPSILON) {
			strncpy(final_ifos + IFO_LEN*cur_ifo, "L1", one_ifo_size);
			cur_ifo++;
		}
		else
			pass_test = FALSE;

	}

	if ((icombo == 6 || icombo == 4 || icombo == 5)) { // H1V1, L1V1 or H1L1V1
		if (trigger->snglsnr_V > EPSILON) {
			strncpy(final_ifos + IFO_LEN*cur_ifo, "V1", one_ifo_size);
			cur_ifo++;
		}
		else
			pass_test = FALSE;

	}
	if (pass_test != TRUE) {
		strncpy(trigger->ifos, final_ifos, cur_ifo * sizeof(char) * IFO_LEN);
		trigger->ifos[IFO_LEN*cur_ifo] = '\0';
		return get_icombo(trigger->ifos);
	}else 
		return icombo;
}


int get_icombo(char *ifos) {
	int icombo = 0; 
	unsigned len_in = strlen(ifos);
	int nifo_in = (int) len_in / IFO_LEN, nifo_map, iifo, jifo;
	for (icombo=0; icombo<MAX_IFO_COMBOS; icombo++) {
		nifo_map = 0;
		if (len_in == strlen(IFOComboMap[icombo].name)) {
			for (iifo=0; iifo<nifo_in; iifo++) {
				// this allows V1H1 found as with the IFOComboMap's H1V1
				for (jifo=0; jifo<nifo_in; jifo++)
					if (strncmp(ifos+iifo*IFO_LEN, IFOComboMap[icombo].name+jifo*IFO_LEN, IFO_LEN) == 0) {
						nifo_map++;
						//printf("nifo %d, in_ifo %d, cmp ifo %d, map_ifo %d\n", nifo_in, iifo, jifo, nifo_map);
					}
			}
		}
		if (nifo_in == nifo_map)
			return IFOComboMap[icombo].index;
	}
	fprintf(stderr,"get_icombo: failed to get index for %s, strlen %u, ifos need to end with null terminator\n",ifos, len_in);
	return -1;
}

int get_ncombo(int nifo) {
	g_assert(pow(2,nifo)-1 <= MAX_IFO_COMBOS);
	return MAX_IFO_COMBOS;
}

int get_ifo_idx(char *ifo)
{
	if (g_strcmp0(ifo, "H1") == 0)
		return H_INDEX;
	if (g_strcmp0(ifo, "L1") == 0)
		return L_INDEX;
	if (g_strcmp0(ifo, "V1") == 0)
		return V_INDEX;

}

Bins1D *
bins1D_long_create(double cmin, double cmax, int nbin) 
{
  Bins1D * bins = (Bins1D *) malloc(sizeof(Bins1D));
  bins->cmin = cmin;
  bins->cmax = cmax;
  bins->nbin = nbin;
  bins->step = (cmax - cmin)/ (nbin - 1);
  bins->step_2 = bins->step / 2;
  bins->data = gsl_vector_long_alloc(nbin);
  gsl_vector_long_set_zero(bins->data);
  return bins;
}

void
bins1D_long_destroy(Bins1D* bins) 
{
  gsl_vector_long_free(bins->data);
  free(bins);
}

Bins1D *
bins1D_create(double cmin, double cmax, int nbin) 
{
  Bins1D * bins = (Bins1D *) malloc(sizeof(Bins1D));
  bins->cmin = cmin;
  bins->cmax = cmax;
  bins->nbin = nbin;
  bins->step = (cmax - cmin)/ (nbin - 1);
  bins->step_2 = bins->step / 2;
  bins->data = gsl_vector_alloc(nbin);
  gsl_vector_set_zero(bins->data);
  return bins;
}

void
bins1D_destroy(Bins1D* bins) 
{
  gsl_vector_free(bins->data);
  free(bins);
}




Bins2D *
bins2D_create(double cmin_x, double cmax_x, int nbin_x, double cmin_y, double cmax_y, int nbin_y) 
{
  Bins2D * bins = (Bins2D *) malloc(sizeof(Bins2D));
  bins->cmin_x = cmin_x;
  bins->cmax_x = cmax_x;
  bins->nbin_x = nbin_x;
  bins->step_x = (cmax_x - cmin_x) / (nbin_x - 1);
  bins->step_x_2 = bins->step_x / 2;
  bins->cmin_y = cmin_y;
  bins->cmax_y = cmax_y;
  bins->nbin_y = nbin_y;
  bins->step_y = (cmax_y - cmin_y) / (nbin_y - 1);
  bins->step_y_2 = bins->step_y / 2;
  bins->data = gsl_matrix_alloc(nbin_x, nbin_y);
  gsl_matrix_set_zero(bins->data);
  return bins;
}
void
bins2D_destroy(Bins2D* bins) 
{
  gsl_matrix_free(bins->data);
  free(bins);
}


Bins2D *
bins2D_long_create(double cmin_x, double cmax_x, int nbin_x, double cmin_y, double cmax_y, int nbin_y) 
{
  Bins2D * bins = (Bins2D *) malloc(sizeof(Bins2D));
  bins->cmin_x = cmin_x;
  bins->cmax_x = cmax_x;
  bins->nbin_x = nbin_x;
  bins->step_x = (cmax_x - cmin_x) / (nbin_x - 1);
  bins->cmin_y = cmin_y;
  bins->cmax_y = cmax_y;
  bins->nbin_y = nbin_y;
  bins->step_y = (cmax_y - cmin_y) / (nbin_y - 1);
  bins->data = gsl_matrix_long_alloc(nbin_x, nbin_y);
  gsl_matrix_long_set_zero(bins->data);
  return bins;
}
void
bins2D_long_destroy(Bins2D* bins) 
{
  gsl_matrix_long_free(bins->data);
  free(bins);
}


void
trigger_stats_reset(TriggerStats **multistats, int ncombo)
{
  int icombo;
  FeatureStats *feature;
  RankingStats *rank;
  for (icombo=0; icombo<ncombo; icombo++) {
	  feature = multistats[icombo]->feature;
	  gsl_vector_long_set_zero((gsl_vector_long *)feature->lgsnr_rate->data);
	  gsl_vector_long_set_zero((gsl_vector_long *)feature->lgchisq_rate->data);
	  gsl_matrix_long_set_zero((gsl_matrix_long *)feature->lgsnr_lgchisq_rate->data);
	  rank = multistats[icombo]->rank;
	  gsl_vector_long_set_zero((gsl_vector_long *)rank->rank_rate->data);
	  multistats[icombo]->feature_nevent = 0;
	  multistats[icombo]->rank_nevent = 0;
	  multistats[icombo]->feature_livetime = 0;
	  multistats[icombo]->rank_livetime = 0;
  }

}
void
trigger_stats_xml_reset(TriggerStatsXML *stats)
{
	trigger_stats_reset(stats->multistats, stats->ncombo);
}

FeatureStats *
feature_stats_create()
{
  FeatureStats *feature = (FeatureStats *) malloc(sizeof(FeatureStats));
  feature->lgsnr_rate = bins1D_long_create(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN);
  feature->lgchisq_rate = bins1D_long_create(LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN);
  feature->lgsnr_lgchisq_rate = bins2D_long_create(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN, LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN);
  feature->lgsnr_lgchisq_pdf = bins2D_create(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN, LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN);
  return feature;
}

void
feature_stats_destroy( FeatureStats *feature)
{
  bins1D_long_destroy(feature->lgsnr_rate);
  feature->lgsnr_rate = NULL;
  bins1D_long_destroy(feature->lgchisq_rate);
  feature->lgchisq_rate = NULL;
  bins2D_long_destroy(feature->lgsnr_lgchisq_rate);
  feature->lgsnr_lgchisq_rate = NULL;
  bins2D_destroy(feature->lgsnr_lgchisq_pdf);
  feature->lgsnr_lgchisq_pdf = NULL;
  free(feature);
}
 

RankingStats *
rank_stats_create()
{
  RankingStats *rank = (RankingStats *) malloc(sizeof(RankingStats));
  rank->rank_map = bins2D_create(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN, LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN);
  rank->rank_rate = bins1D_long_create(LOGRANK_CMIN, LOGRANK_CMAX, LOGRANK_NBIN);
  rank->rank_pdf = bins1D_create(LOGRANK_CMIN, LOGRANK_CMAX, LOGRANK_NBIN);
  rank->rank_fap = bins1D_create(LOGRANK_CMIN, LOGRANK_CMAX, LOGRANK_NBIN);
  return rank;
}
void
rank_stats_destroy(RankingStats *rank)
{
  bins1D_long_destroy(rank->rank_rate);
  rank->rank_rate = NULL;
  bins1D_destroy(rank->rank_pdf);
  rank->rank_pdf = NULL;
  bins1D_destroy(rank->rank_fap);
  rank->rank_fap = NULL;
  bins2D_destroy(rank->rank_map);
  rank->rank_map = NULL;
  free(rank);
}
 
TriggerStats **
trigger_stats_create(int ncombo)
{
  TriggerStats ** multistats = (TriggerStats **) malloc(sizeof(TriggerStats *) * ncombo);

  int icombo = 0;
  for (icombo=0; icombo<ncombo; icombo++) {
	multistats[icombo] = (TriggerStats *) malloc(sizeof(TriggerStats));
	TriggerStats *cur_stats = multistats[icombo];
	// FIXME: what if HV or LV combo
	//printf("len %s, %d\n", IFOComboMap[icombo].name, strlen(IFOComboMap[icombo].name));
	cur_stats->ifos = malloc(strlen(IFOComboMap[icombo].name) * sizeof(char));
	strncpy(cur_stats->ifos, IFOComboMap[icombo].name, strlen(IFOComboMap[icombo].name) * sizeof(char));
	// create feature
	cur_stats->feature = feature_stats_create();
	// our rank, cdf
	cur_stats->rank = rank_stats_create();
	cur_stats->feature_nevent = 0;
	cur_stats->rank_nevent = 0;
	cur_stats->feature_livetime = 0;
	cur_stats->rank_livetime = 0;
  }
  return multistats;
}

TriggerStatsXML *
trigger_stats_xml_create(char *ifos, int stats_type) {
	TriggerStatsXML * stats = (TriggerStatsXML *) malloc(sizeof(TriggerStatsXML));
	if (stats_type == STATS_XML_TYPE_BACKGROUND) {
		stats->feature_xmlname = g_string_new(BACKGROUND_XML_FEATURE_NAME);
		stats->rank_xmlname = g_string_new(BACKGROUND_XML_RANK_NAME);
	} else if (stats_type == STATS_XML_TYPE_ZEROLAG) {
		stats->feature_xmlname = g_string_new(ZEROLAG_XML_FEATURE_NAME);
		stats->rank_xmlname = g_string_new(ZEROLAG_XML_RANK_NAME);
	} else if (stats_type == STATS_XML_TYPE_SIGNAL) {
		stats->feature_xmlname = g_string_new(SIGNAL_XML_FEATURE_NAME);
		stats->rank_xmlname = g_string_new(SIGNAL_XML_RANK_NAME);
	}

	int nifo = strlen(ifos) / IFO_LEN;
	int ncombo = get_ncombo(nifo);
	stats->multistats = trigger_stats_create(ncombo);
	stats->ncombo = ncombo;
	return stats;
}



void
trigger_stats_destroy(TriggerStats **multistats, int ncombo)
{
  int icombo = 0;

  for (icombo=0; icombo<ncombo; icombo++) {
	TriggerStats *cur_stats = multistats[icombo];
	feature_stats_destroy(cur_stats->feature);
	cur_stats->feature = NULL;
	rank_stats_destroy(cur_stats->rank);
	cur_stats->rank = NULL;
	free(cur_stats->ifos);
	cur_stats->ifos = NULL;
	free(cur_stats);
	cur_stats = NULL;
  }
  free(multistats);
  multistats = NULL;
}
void
trigger_stats_xml_destroy(TriggerStatsXML * stats)
{
	g_string_free(stats->feature_xmlname, TRUE);
	g_string_free(stats->rank_xmlname, TRUE);
	trigger_stats_destroy(stats->multistats, stats->ncombo);
}

TriggerStatsPointerList *
trigger_stats_list_create(char *ifos)
{
  int nifo = 0, ncombo = 0, icombo = 0;
  nifo = strlen(ifos) / IFO_LEN;

  ncombo = get_ncombo(nifo);
  TriggerStatsPointerList * stats_list = (TriggerStatsPointerList *) malloc(sizeof(TriggerStatsPointerList));
  stats_list->size = NSTATS_TO_PROMPT;
  stats_list->pos = 0;
  stats_list->plist = (TriggerStatsPointer *) malloc(sizeof(TriggerStatsPointer)*NSTATS_TO_PROMPT);

  int ilist = 0;
  TriggerStats **stats = NULL;

  for (ilist=0; ilist<NSTATS_TO_PROMPT; ilist++) {
	stats = (TriggerStats **) malloc(sizeof(TriggerStats *) * ncombo);
	for (icombo=0; icombo<ncombo; icombo++) {
	  stats[icombo] = (TriggerStats *) malloc(sizeof(TriggerStats));
	  TriggerStats *cur_stats = stats[icombo];
	  //printf("len %s, %d\n", IFOComboMap[icombo].name, strlen(IFOComboMap[icombo].name));
	  cur_stats->ifos = malloc(strlen(IFOComboMap[icombo].name) * sizeof(char));
	  strncpy(cur_stats->ifos, IFOComboMap[icombo].name, strlen(IFOComboMap[icombo].name) * sizeof(char));
	  cur_stats->feature_nevent = 0;
	  cur_stats->rank_nevent = 0;
	  cur_stats->feature_livetime = 0;
	  cur_stats->rank_livetime = 0;
	}
	stats_list->plist[ilist] = stats;
  }
  return stats_list;
}


/*
 * background rate utils
 */

// return the index given a value
int
bins1D_get_idx_lr(double val, Bins1D *bins)
{
  if (val < -20)
	  return 0;
  int bin = (val - bins->cmin - bins->step_2) / bins->step;

  if (bin < 0)
	bin = 0;
  
  if (bin >= bins->nbin)
	bin = bins->nbin - 1;

  return bin;
}
int
bins1D_get_idx(double val, Bins1D *bins)
{
  if (val < DBL_MIN)
	  return 0;
  int bin = (log10(val) - bins->cmin - bins->step_2) / bins->step;

  if (bin < 0)
	bin = 0;
  
  if (bin >= bins->nbin)
	bin = bins->nbin - 1;

  return bin;
}

// return the lower boudnary of the bin
double
bins1D_get_low_bound(Bins1D *bins, int ibin)
{
  g_assert(ibin>= 0 && ibin < bins->nbin);
  return bins->cmin - bins->step_2 + ibin * bins->step;
}

// return the upper boudnary of the bin
double
bins1D_get_up_bound(Bins1D *bins, int ibin)
{
  g_assert(ibin>= 0 && ibin < bins->nbin);
  return bins->cmin + bins->step_2 + ibin * bins->step;
}




void
trigger_stats_feature_rate_update_all(gsl_vector *snr_vec, gsl_vector *chisq_vec, FeatureStats *feature, TriggerStats *cur_stats)
{

	int ievent, nevent = (int) snr_vec->size;
	for (ievent=0; ievent<nevent; ievent++) {
		trigger_stats_feature_rate_update(gsl_vector_get(snr_vec, ievent), gsl_vector_get(chisq_vec, ievent), feature, cur_stats);
	}
}

double
get_prob_noise_sngl(int icombo, PostcohInspiralTable *intable, TriggerStatsXML *margi_statsxml)
{
	double lgp_noise = 0;
	int missing_ifo = -1;
	double step_x = margi_statsxml->multistats[H_INDEX]->rank->rank_map->step_x, step_y = margi_statsxml->multistats[H_INDEX]->rank->rank_map->step_y;
	double binsz = step_x * step_y;

	if ((icombo == 6 || icombo == 3 || icombo == 4)) { // H1L1, H1V1 or H1L1V1
		FeatureStats *feature = margi_statsxml->multistats[H_INDEX]->feature;	
		int snr_idx = bins1D_get_idx(intable->snglsnr_H, feature->lgsnr_rate);
		int chisq_idx = bins1D_get_idx(intable->chisq_H, feature->lgchisq_rate);
		gsl_matrix *rankval_mat = margi_statsxml->multistats[H_INDEX]->rank->rank_map->data;
		lgp_noise += log10(binsz * gsl_matrix_get(rankval_mat, snr_idx, chisq_idx));
	}

	if ((icombo == 6 || icombo == 3 || icombo == 5)) { // H1L1, L1V1 or H1L1V1
		FeatureStats *feature = margi_statsxml->multistats[L_INDEX]->feature;	
		int snr_idx = bins1D_get_idx(intable->snglsnr_L, feature->lgsnr_rate);
		int chisq_idx = bins1D_get_idx(intable->chisq_L, feature->lgchisq_rate);
		gsl_matrix *rankval_mat = margi_statsxml->multistats[L_INDEX]->rank->rank_map->data;
		lgp_noise += log10(binsz * gsl_matrix_get(rankval_mat, snr_idx, chisq_idx));
	}


	if ((icombo == 6 || icombo == 4 || icombo == 5)) { // H1V1, L1V1 or H1L1V1
		FeatureStats *feature = margi_statsxml->multistats[V_INDEX]->feature;	
		int snr_idx = bins1D_get_idx(intable->snglsnr_V, feature->lgsnr_rate);
		int chisq_idx = bins1D_get_idx(intable->chisq_V, feature->lgchisq_rate);
		gsl_matrix *rankval_mat = margi_statsxml->multistats[V_INDEX]->rank->rank_map->data;
		lgp_noise += log10(binsz * gsl_matrix_get(rankval_mat, snr_idx, chisq_idx));
	}

	// to normalize different combinations of detectors
	// add the mean values of rank_maps of the non-participating IFOS
	if (icombo == 3) // H1L1
		missing_ifo = V_INDEX;

	if (icombo == 4) // H1V1
		missing_ifo = L_INDEX;

	if (icombo == 5) // L1V1
		missing_ifo = H_INDEX;

	if (missing_ifo > -1)
		lgp_noise += log10(binsz * margi_statsxml->multistats[missing_ifo]->rank->mean_rankmap);

	return lgp_noise;
}

double
get_prob_noise (double cohsnr, double cmbchisq, TriggerStats *margi_stats)
{
	FeatureStats *feature = margi_stats->feature;	
	int snr_idx = bins1D_get_idx(cohsnr, feature->lgsnr_rate);
	int chisq_idx = bins1D_get_idx(cmbchisq, feature->lgchisq_rate);
	gsl_matrix *rankval_mat = margi_stats->rank->rank_map->data;
	return log10(gsl_matrix_get(rankval_mat, snr_idx, chisq_idx));


}
double
get_prob_null(PostcohInspiralTable *intable)
{
	int nifo = strlen(intable->ifos)/IFO_LEN;
	int dof = (nifo -2) * 2;
	if (dof > 0)
		return log10(gsl_ran_chisq_pdf(intable->nullsnr * (intable->nullsnr), dof));
	else
		// to normalize different combinations of detectors
		// add mean nullsnr^2 for 3det LHV is 0.7, chisq pdf of 0.7 and dof 2 = 0.35 and log10(0.35) = -0.45
		return -0.45; 
}

double
get_prob_snrs(int icombo, PostcohInspiralTable *intable, float *sense_ratio)
{
	double lgp_snr = 0;
	float this_mean, this_snr, mismatch = 1;
	if (g_strcmp0(intable->pivotal_ifo, "H1") == 0) {
		this_snr = intable->snglsnr_H;
		if (icombo == 6 || icombo == 3) { // H1L1 or H1L1V1
			this_mean = this_snr*sense_ratio[LH_INDEX];
			lgp_snr += log10(gsl_ran_gaussian_pdf(intable->snglsnr_L- this_mean, this_mean*mismatch));
		}
		if (icombo == 6 || icombo == 4) { // H1V1 or H1L1V1
			this_mean = this_snr*sense_ratio[VH_INDEX];
			lgp_snr += log10(gsl_ran_gaussian_pdf(intable->snglsnr_V- this_mean, this_mean*mismatch));
		}
	}

	if (g_strcmp0(intable->pivotal_ifo, "L1") == 0) {
		this_snr = intable->snglsnr_L;
		if (icombo == 6 || icombo == 3) { // H1L1 or H1L1V1
			this_mean = this_snr*sense_ratio[HL_INDEX];
			lgp_snr += log10(gsl_ran_gaussian_pdf(intable->snglsnr_H- this_mean, this_mean*mismatch));
		}
		if (icombo == 6 || icombo == 5) { // L1V1 or H1L1V1
			this_mean = this_snr*sense_ratio[VL_INDEX];
			lgp_snr += log10(gsl_ran_gaussian_pdf(intable->snglsnr_V- this_mean, this_mean*mismatch));
		}
	}
	
	if (g_strcmp0(intable->pivotal_ifo, "V1" ) == 0) {
		this_snr = intable->snglsnr_V;
		if (icombo == 6 || icombo == 4) { // H1V1 or H1L1V1
			this_mean = this_snr*sense_ratio[HV_INDEX];
			lgp_snr += log10(gsl_ran_gaussian_pdf(intable->snglsnr_H- this_mean, this_mean*mismatch));
		}
		if (icombo == 6 || icombo == 5) { // L1V1 or H1L1V1
			this_mean = this_snr*sense_ratio[LV_INDEX];
			lgp_snr += log10(gsl_ran_gaussian_pdf(intable->snglsnr_L- this_mean, this_mean*mismatch));
		}
	}

	if (icombo < 6)
		// to normalize different combinations of detectors, e.g. 2det to 3det,
		// assuming the ghost third SNR is right at the mean value. +log10(gaussian.pdf(0, 1))
		return lgp_snr - 0.399; 
	else
		return lgp_snr;

}
double
calc_lr(PostcohInspiralTable *intable, TriggerStatsXML *margi_statsxml, float *sense_ratio)
{
	//double lgp_noise = get_prob_noise(intable->cohsnr, intable->cmbchisq, margi_stats);
	
	int icombo = get_icombo(intable->ifos);
	double lgp_noise = get_prob_noise_sngl(icombo, intable, margi_statsxml);
	double lgp_signal = 0;
	lgp_signal += get_prob_snrs(icombo, intable, sense_ratio);
	lgp_signal += get_prob_null(intable);
	//printf("background %d, end time %d, ifos %s, cohsnr %f, cmbchisq %f, nullsnr %f, snr_l %f, snr_h %f, snr_v %f, chisq_l %f, chisq_h %f chisq_v %f, lgp_noise %f, lgp_signal %f, lgp_null %f\n", intable->is_background, intable->end_time.gpsSeconds, intable->ifos, intable->cohsnr, intable->cmbchisq, intable->nullsnr, intable->snglsnr_L, intable->snglsnr_H, intable->snglsnr_V, intable->chisq_L, intable->chisq_H, intable->chisq_V, lgp_noise, get_prob_snrs(icombo, intable, sense_ratio), get_prob_null(intable));
	return lgp_signal-lgp_noise;
}

int
get_rank_idx(PostcohInspiralTable *intable, TriggerStatsXML *margi_statsxml, int icombo, float *sense_ratio)
{
	double lglr = calc_lr(intable, margi_statsxml, sense_ratio);
	/* update rate */
	int ibin = bins1D_get_idx_lr(lglr, margi_statsxml->multistats[icombo]->rank->rank_rate);
	//if (ibin > 100)
	//	printf("lr %e, ibin %d, cohsnr %f, cmbchisq %f, snr_l %f, snr_h %f, snr_v %f, chisq_l %f, chisq_h %f chisq_v %f \n", lr, ibin, intable->cohsnr, intable->cmbchisq, intable->snglsnr_L, intable->snglsnr_H, intable->snglsnr_V, intable->chisq_L, intable->chisq_H, intable->chisq_V);

	return ibin;
}

void
trigger_stats_rank_rate_update(int ibin, TriggerStats *cur_stats)
{

	/* update rank rate */
	gsl_vector_long *rratedata = cur_stats->rank->rank_rate->data;
	gsl_vector_long_set(rratedata, ibin, gsl_vector_long_get(rratedata, ibin) + 1);
	cur_stats->rank_nevent++;

}

void
trigger_stats_feature_rate_update(double snr, double chisq, FeatureStats *feature, TriggerStats *cur_stats)
{
	int snr_idx = bins1D_get_idx(snr, feature->lgsnr_rate);
	int chisq_idx = bins1D_get_idx(chisq, feature->lgchisq_rate);

	gsl_vector_long *snr_vec = (gsl_vector_long *)feature->lgsnr_rate->data;
	gsl_vector_long *chisq_vec = (gsl_vector_long *)feature->lgchisq_rate->data;
	gsl_matrix_long *hist_mat = (gsl_matrix_long *)feature->lgsnr_lgchisq_rate->data;

	gsl_vector_long_set(snr_vec, snr_idx, gsl_vector_long_get(snr_vec, snr_idx) + 1);
	gsl_vector_long_set(chisq_vec, chisq_idx, gsl_vector_long_get(chisq_vec, chisq_idx) + 1);
	gsl_matrix_long_set(hist_mat, snr_idx, chisq_idx, gsl_matrix_long_get(hist_mat, snr_idx, chisq_idx) + 1);
	cur_stats->feature_nevent++;
}

// add rank2 to rank1
void
trigger_stats_rank_rate_add(RankingStats *rank1, RankingStats *rank2, TriggerStats *cur_stats)
{
	gsl_vector_long_add((gsl_vector_long *)rank1->rank_rate->data, (gsl_vector_long *)rank2->rank_rate->data);
	cur_stats->rank_nevent = gsl_vector_long_sum((gsl_vector_long *)rank1->rank_rate->data);
}

void
trigger_stats_rank_livetime_add(TriggerStats **stats_out, TriggerStats **stats_in, const int icombo)
{
	stats_out[icombo]->rank_livetime += stats_in[icombo]->rank_livetime;
}

void
trigger_stats_feature_rate_add(FeatureStats *feature1, FeatureStats *feature2, TriggerStats *cur_stats)
{
	gsl_vector_long_add((gsl_vector_long *)feature1->lgsnr_rate->data, (gsl_vector_long *)feature2->lgsnr_rate->data);
	gsl_vector_long_add((gsl_vector_long *)feature1->lgchisq_rate->data, (gsl_vector_long *)feature2->lgchisq_rate->data);
	gsl_matrix_long_add((gsl_matrix_long *)feature1->lgsnr_lgchisq_rate->data, (gsl_matrix_long *)feature2->lgsnr_lgchisq_rate->data);
	cur_stats->feature_nevent = gsl_vector_long_sum((gsl_vector_long *)feature1->lgsnr_rate->data);
}

void
trigger_stats_feature_livetime_add(TriggerStats **stats_out, TriggerStats **stats_in, const int icombo)
{
	stats_out[icombo]->feature_livetime += stats_in[icombo]->feature_livetime;
}
/*
 * background pdf direnctly from rate
 */
void
trigger_stats_feature_livetime_inc(TriggerStats **stats, const int icombo)
{
	stats[icombo]->feature_livetime += 1;
}
void
trigger_stats_rank_livetime_inc(TriggerStats **stats, const int icombo)
{
	stats[icombo]->rank_livetime += 1;
}
void
trigger_stats_feature_rate_to_pdf_hist(FeatureStats *feature)
{

	gsl_vector_long *snr = feature->lgsnr_rate->data;

	long nevent = gsl_vector_long_sum(snr);
	//printf("nevent %ld\n", nevent);
	if (nevent == 0)
		return;

	Bins2D *pdf = feature->lgsnr_lgchisq_pdf;
	int nbin_x = pdf->nbin_x, nbin_y = pdf->nbin_y;
	int ibin_x, ibin_y;
	gsl_matrix *pdfdata = pdf->data;

	/*
	 * set the pdf = rate / nevent for each bin
	 */
	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) {
		for (ibin_y=0; ibin_y<nbin_y; ibin_y++) {
			//printf("hist x %d, y %d, value %ld\n", ibin_x, ibin_y, gsl_matrix_long_get(rate->hist, ibin_x, ibin_y));
			gsl_matrix_set(pdfdata, ibin_x, ibin_y, ((double)gsl_matrix_long_get((gsl_matrix_long *)feature->lgsnr_lgchisq_rate->data, ibin_x, ibin_y))/((double)nevent));
		}
	}
}



/*
 * background pdf utils, consistent with the matlab pdf code, knn
 */

void
trigger_stats_feature_rate_to_pdf(FeatureStats *feature)
{	
	gsl_vector_long *snr = feature->lgsnr_rate->data;
	long nevent = gsl_vector_long_sum(snr);

	if (nevent == 0)
		return;

	Bins2D *pdf = feature->lgsnr_lgchisq_pdf;
	gsl_vector *tin_snr = gsl_vector_alloc(pdf->nbin_x);
	gsl_vector *tin_chisq = gsl_vector_alloc(pdf->nbin_y);
	gsl_vector_linspace(pdf->cmin_x, pdf->cmax_x, pdf->nbin_x, tin_snr);
	gsl_vector_linspace(pdf->cmin_y, pdf->cmax_y, pdf->nbin_y, tin_chisq);

	knn_kde(tin_snr, tin_chisq, (gsl_matrix_long *)feature->lgsnr_lgchisq_rate->data, (gsl_matrix *)pdf->data);

	gsl_vector_free(tin_snr);
	gsl_vector_free(tin_chisq);
}


/* deprecated: ssvkernel-2d estimation is not 
 * consistent with histogram
 */

gboolean
trigger_stats_feature_rate_to_pdf_ssvkernel(FeatureStats *feature)
{

	gsl_vector_long *snr = feature->lgsnr_rate->data;
	gsl_vector_long *chisq = feature->lgchisq_rate->data;
	Bins2D *pdf = feature->lgsnr_lgchisq_pdf;

	long nevent = gsl_vector_long_sum(snr);
	if (nevent == 0)
		return TRUE;
	gsl_vector *snr_double = gsl_vector_alloc(snr->size);
	gsl_vector *chisq_double = gsl_vector_alloc(chisq->size);
	gsl_vector_long_to_double(snr, snr_double);
	gsl_vector_long_to_double(chisq, chisq_double);

	gsl_vector *tin_snr = gsl_vector_alloc(pdf->nbin_x);
	gsl_vector *tin_chisq = gsl_vector_alloc(pdf->nbin_y);
	gsl_vector_linspace(pdf->cmin_x, pdf->cmax_x, pdf->nbin_x, tin_snr);
	gsl_vector_linspace(pdf->cmin_y, pdf->cmax_y, pdf->nbin_y, tin_chisq);

	gsl_matrix *result_snr = gsl_matrix_alloc(pdf->nbin_x, pdf->nbin_x);
	gsl_matrix *result_chisq = gsl_matrix_alloc(pdf->nbin_y, pdf->nbin_y);

	ssvkernel_from_hist(snr_double, tin_snr, result_snr);
	ssvkernel_from_hist(chisq_double, tin_chisq, result_chisq);


	//two-dimensional histogram
	gsl_matrix *histogram = gsl_matrix_alloc(snr->size, chisq->size);
	gsl_matrix_long_to_double((gsl_matrix_long *)feature->lgsnr_lgchisq_rate->data, histogram);
	//gsl_matrix_hist3(snr_data, chisq_data, temp_tin_snr, temp_tin_chisq, histogram);

	//Compute the 'scale' variable in matlab code 'test.m'
	unsigned i, j;
	for(i=0;i<histogram->size1;i++){
		for(j=0;j<histogram->size2;j++){
			double temp = gsl_matrix_get(histogram,i,j);
			temp = temp/(gsl_vector_get(snr_double, i)*gsl_vector_get(chisq_double, j));
			if(isnan(temp))
				gsl_matrix_set(histogram,i,j,0);
			else
				gsl_matrix_set(histogram,i,j,temp);
		}
	}

	//compute the two-dimensional estimation
	gsl_matrix * result = pdf->data;
	gsl_matrix * temp_matrix = gsl_matrix_alloc(tin_snr->size,tin_chisq->size);
	for(i=0;i<tin_snr->size;i++){
		for(j=0;j<tin_chisq->size;j++){
			gsl_matrix_get_col(snr_double,result_snr,i);
			gsl_matrix_get_col(chisq_double,result_chisq,j);
			gsl_matrix_xmul(snr_double,chisq_double,temp_matrix);
			gsl_matrix_mul_elements(temp_matrix,histogram);
			gsl_matrix_set(result,i,j,gsl_matrix_sum(temp_matrix)/(double)nevent);
		}
	}

	// normalize pdf
	double step_x = pdf->step_x, step_y = pdf->step_y;
	double pdf_sum;
	pdf_sum = step_x * step_y * gsl_matrix_sum(result);
	gsl_matrix_scale(result, 1/pdf_sum);
	//printf("pdf sum %f\n", gsl_matrix_sum(result) * step_x * step_y);
	
	gsl_vector_free(snr_double);
	gsl_vector_free(chisq_double);
	gsl_matrix_free(histogram);
	gsl_vector_free(tin_snr);
	gsl_vector_free(tin_chisq);
	gsl_matrix_free(result_snr);
	gsl_matrix_free(result_chisq);
	gsl_matrix_free(temp_matrix);
	return TRUE;
}
// deprecated along with trigger_stats_pdf_to_fap
static double gsl_matrix_accum_pdf(gsl_matrix *pdfdata, gsl_matrix *cdfdata, double cur_cdf)
{
	int nbin_x = pdfdata->size1, nbin_y = pdfdata->size2;
	int ibin_x, ibin_y;
	double fap = 0.0;
	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) 
		for (ibin_y=0; ibin_y<nbin_y; ibin_y++) 
			if (gsl_matrix_get(cdfdata, ibin_x, ibin_y) <= cur_cdf)
				fap += gsl_matrix_get(pdfdata, ibin_x, ibin_y);

	return fap;

}
static double calc_rank_pdf_val(gsl_matrix *pdfdata, gsl_matrix *cdfdata, double rank_min, double rank_max)
{
	int nbin_x = pdfdata->size1, nbin_y = pdfdata->size2;
	int ibin_x, ibin_y;
	double pdf = 0.0, cur_cdf;
	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) 
		for (ibin_y=0; ibin_y<nbin_y; ibin_y++) {
			cur_cdf = gsl_matrix_get(cdfdata, ibin_x, ibin_y);
			if (cur_cdf <= rank_max && cur_cdf > rank_min)
				pdf += gsl_matrix_get(pdfdata, ibin_x, ibin_y);
		}

	return pdf;

}
static double calc_rank_rate_val(gsl_matrix_long *ratedata, gsl_matrix *cdfdata, double rank_min, double rank_max)
{
	int nbin_x = ratedata->size1, nbin_y = ratedata->size2;
	int ibin_x, ibin_y;
	double cur_cdf;
	long rate = 0;
	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) 
		for (ibin_y=0; ibin_y<nbin_y; ibin_y++) {
			cur_cdf = gsl_matrix_get(cdfdata, ibin_x, ibin_y);
			if (cur_cdf <= rank_max && cur_cdf > rank_min)
				rate += gsl_matrix_long_get(ratedata, ibin_x, ibin_y);
		}

	return rate;

}
/* deprecated, using the feature to rank function instead. 
 * this is acutally fap */
void
trigger_stats_pdf_to_fap(Bins2D *pdf, Bins2D *fap)
{
	int nbin_x = pdf->nbin_x, nbin_y = pdf->nbin_y;
	int ibin_x, ibin_y;
	   	double	tmp;
	gsl_matrix *pdfdata = pdf->data, *fapdata = fap->data;
	double pdf_sum = gsl_matrix_sum(pdfdata);
	// no data values, return
	if (pdf_sum < 1e-5)
		return;

	/* cdf is our rankings statistic, 
		 * NOTE, here cdf is not a real cumulative distribution function since it is multiplied by step_x*step_y  */
	gsl_matrix *cdfdata = gsl_matrix_calloc(pdfdata->size1, pdfdata->size2);

	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
			tmp = 0;
			if (ibin_y > 0)
				tmp += gsl_matrix_get(cdfdata, ibin_x, ibin_y-1);
			if (ibin_x < nbin_x-1)
				tmp += gsl_matrix_get(cdfdata, ibin_x+1, ibin_y);
			if (ibin_x < nbin_x-1 && ibin_y > 0)
				tmp -= gsl_matrix_get(cdfdata, ibin_x+1, ibin_y-1);
			tmp += gsl_matrix_get(pdfdata, ibin_x, ibin_y);
			gsl_matrix_set(cdfdata, ibin_x, ibin_y, tmp);
		}
	}
	/* get fap from cdf data */
	double cur_cdf, cur_fap;
	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) {
		for (ibin_y=0; ibin_y<nbin_y; ibin_y++) {
			cur_cdf = gsl_matrix_get(cdfdata, ibin_x, ibin_y);
			cur_fap = gsl_matrix_accum_pdf(pdfdata, cdfdata, cur_cdf);
			gsl_matrix_set(fapdata, ibin_x, ibin_y, cur_fap*pdf->step_x*pdf->step_y);
		}
	}
	/* fap could be zero, set fap=0 to fap=next smallest value */
	double second_smallest_fap = 1.0;
	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) {
		for (ibin_y=0; ibin_y<nbin_y; ibin_y++) {
			cur_fap = gsl_matrix_get(fapdata, ibin_x, ibin_y);
			if (cur_fap > 0.0 && second_smallest_fap > cur_fap) 
				second_smallest_fap = cur_fap;
		}
	}

	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) {
		for (ibin_y=0; ibin_y<nbin_y; ibin_y++) {
			cur_fap = gsl_matrix_get(fapdata, ibin_x, ibin_y);
			if (cur_fap == 0.0) 
				gsl_matrix_set(fapdata, ibin_x, ibin_y, second_smallest_fap);
		}
	}
	
	gsl_matrix_free(cdfdata);

	//printf("fap cmax %f\n", gsl_matricmax_x(fapdata));
}

static double
cx2pdf(double chisq, double dof, double r)
{
	// wiki non-central chi-square
	double prefactor = 0.5 * exp(-0.5*(chisq + r))*pow((chisq/r), dof/4-0.5);
	// printf("besselof %f\n", sqrt(r * chisq));
	return prefactor * gsl_sf_bessel_I0(sqrt(r * chisq));
}
	

static double
ncx2pdf(double chisq, double dof, double r)
{
	// wiki non-central chi-square
	double prefactor = 0.5 * exp(-0.5*(chisq + r))*pow((chisq/r), dof/4-0.5);
	// printf("besselof %f\n", sqrt(r * chisq));
	return prefactor * gsl_sf_bessel_I0(sqrt(r * chisq));
}
	
static void
signal_stats_gen_pdfmap(Bins2D *fpdf)
{
	double prob, logcohsnr, logchisq, cohsnr, chisq, sum_y, sum_all = 0;
	int nbin_x = fpdf->nbin_x, nbin_y = fpdf->nbin_y;
	int ibin_x, ibin_y;
	gsl_matrix *fpdfdata = fpdf->data;

	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		sum_y = 0; 
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
			logcohsnr = fpdf->step_x * ibin_x + fpdf->cmin_x;
			logchisq = fpdf->step_y * ibin_y + fpdf->cmin_y;
			cohsnr = pow(10, logcohsnr);
		   	chisq = pow(10, logchisq);
			// non-central chi-square pdf, degree of freedom 2, delta (r) parameter 1 + (cohsnr*0.045)^2 fitted by MDC BNS injection
			// FIXME: r parameter will be different for other type of injections ?
			// prob = ncx2pdf(chisq, 2, 1 + pow(cohsnr, 2) * 0.002025);
			// chisq/cohsnrsq = 0.001;
			//prob = ncx2pdf(chisq/pow(cohsnr, 2), 2, 1 + pow(cohsnr, 2) * 0.002025);
			prob = 0.0;

			gsl_matrix_set(fpdfdata, ibin_x, ibin_y, prob);
			sum_y += prob;
		}
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
			/* FIXME: normalize over a cohsnr */
			if (sum_y > 1e-30)
				gsl_matrix_set(fpdfdata, ibin_x, ibin_y, gsl_matrix_get(fpdfdata, ibin_x, ibin_y)/sum_y);
		}
	}
	sum_all = gsl_matrix_sum(fpdfdata);
	/* normalize pdf */
	if (sum_all > 1e-30)
		gsl_matrix_scale(fpdfdata, 1/(sum_all*fpdf->step_x*fpdf->step_y));

}
static void
signal_stats_gen_ratemap_from_pdf(FeatureStats *feature)
{
	int nevent = 100000000; // 100 million
	long cur_rate;
	int ibin_x, ibin_y;
	int nbin_x = feature->lgsnr_lgchisq_pdf->nbin_x, nbin_y = feature->lgsnr_lgchisq_pdf->nbin_y;
	gsl_matrix *fpdfdata = feature->lgsnr_lgchisq_pdf->data;
	gsl_matrix_long *fratedata = feature->lgsnr_lgchisq_rate->data;

	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
			cur_rate = (long)(int)(nevent * gsl_matrix_get(fpdfdata, ibin_x, ibin_y));
			gsl_matrix_long_set(fratedata, ibin_x, ibin_y, cur_rate);
		}
	}
}

void
signal_stats_init(TriggerStatsXML *sgstats, int source_type)
{
	int icombo, ncombo = sgstats->ncombo;
	if (source_type == SOURCE_TYPE_BNS) {
		for (icombo=0; icombo<ncombo; icombo++) {
			TriggerStats *stats = sgstats->multistats[icombo];
			signal_stats_gen_pdfmap(stats->feature->lgsnr_lgchisq_pdf);
			signal_stats_gen_ratemap_from_pdf(stats->feature);
		}
	} else {
		fprintf(stderr, "source not supported !");
	}
}

/* convert collected rate into rank rate */
void
trigger_stats_feature_to_cdf(FeatureStats *feature, RankingStats *rank)
{
	Bins2D *fpdf = feature->lgsnr_lgchisq_pdf;
	int nbin_x = fpdf->nbin_x, nbin_y = fpdf->nbin_y;
	int ibin_x, ibin_y;
   	double	tmp;
	gsl_matrix *fpdfdata = fpdf->data;
	double fpdf_sum = gsl_matrix_sum(fpdfdata);
	// no data values, return
	if (fpdf_sum < 1e-5)
		return;

	/* cdf is our rankings statistic
	 * use the enlongated-estimated pdf to get the full cdf to cover very significant region
	 * this requires that the pdf estimation should be realiable in that region. We will later
	 * use the elonged pdf to get the distribution of cdf.  */
	gsl_matrix *cdfdata = rank->rank_map->data;
	Bins2D *cdf = rank->rank_map;

	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
			tmp = 0;
			if (ibin_y > 0)
				tmp += gsl_matrix_get(cdfdata, ibin_x, ibin_y-1);
			if (ibin_x < nbin_x-1)
				tmp += gsl_matrix_get(cdfdata, ibin_x+1, ibin_y);
			if (ibin_x < nbin_x-1 && ibin_y > 0)
				tmp -= gsl_matrix_get(cdfdata, ibin_x+1, ibin_y-1);
			tmp += gsl_matrix_get(fpdfdata, ibin_x, ibin_y) * fpdf->step_x * fpdf->step_y;
			// note here we actually assign comulative prob. FIXME: overflow problem ?
			gsl_matrix_set(cdfdata, ibin_x, ibin_y, tmp);

		}
	}
	double second_smallest_cdf = 1.0, cur_cdf;
	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
			cur_cdf = gsl_matrix_get(cdfdata, ibin_x, ibin_y);
			if (cur_cdf > 0.0 && second_smallest_cdf > cur_cdf) 
				second_smallest_cdf = cur_cdf;
		}
	}

	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
		cur_cdf = gsl_matrix_get(cdfdata, ibin_x, ibin_y);
		if (cur_cdf < 1e-200) 
			gsl_matrix_set(cdfdata, ibin_x, ibin_y, second_smallest_cdf);

		}
	}

	// note the rank_min and max is based on log10 scale, need to log10 the cdf data
	// also set minimal cdf to be RANK_MIN_LIMIT
//	double cur_cdf;
//	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
//		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++){
//			/* avoid the log10(0) error, since the rank_min is -30, assign rank to be less than -30 is accurate enough*/
//			cur_cdf = gsl_matrix_get(cdfdata, ibin_x, ibin_y);
//			cur_cdf = MAX(RANK_MIN_LIMIT, cur_cdf);
//			gsl_matrix_set(cdfdata, ibin_x, ibin_y, log10(cur_cdf));
//		}
//	}
	// normalize cdf
	double step_x = cdf->step_x, step_y = cdf->step_y;
	double pdf_sum, binsz = step_x * step_y;
	pdf_sum = step_x * step_y * gsl_matrix_sum(cdfdata);
	gsl_matrix_scale(cdfdata, 1/pdf_sum);
	//printf("pdf sum %f\n", gsl_matrix_sum(result) * step_x * step_y);
	
	// calculate the mean of cdf
	tmp = 0;
	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
			cur_cdf = gsl_matrix_get(cdfdata, ibin_x, ibin_y);
			tmp += binsz * gsl_matrix_get(fpdfdata, ibin_x, ibin_y) * cur_cdf;
		}
	}
	rank->mean_rankmap = tmp;
	
}
	
/* convert collected rate into rank rate */
void
trigger_stats_rank_to_fap(RankingStats *rank)
{
	if (gsl_vector_long_sum((gsl_vector_long *)rank->rank_rate->data) < 100)
		return;
	Bins1D *rank_pdf = rank->rank_pdf;
	gsl_vector *rpdfdata = rank_pdf->data;
	gsl_vector *tin_rank = gsl_vector_alloc(rank_pdf->nbin);
	gsl_vector_linspace(rank_pdf->cmin, rank_pdf->cmax, rank_pdf->nbin, tin_rank);
	gsl_vector_long *rratedata = rank->rank_rate->data;
	knn_kde_1D(tin_rank, rratedata, rpdfdata);
	int ibin, nbin = rank_pdf->nbin;	
	double second_smallest_pdf = 1.0, cur_pdf;
	for (ibin=0; ibin<nbin; ibin++) {
			cur_pdf = gsl_vector_get(rpdfdata, ibin);
			if (cur_pdf > 0.0 && second_smallest_pdf > cur_pdf) 
				second_smallest_pdf = cur_pdf;
	}

	double pdf_sum = 0;
	for (ibin=0; ibin<nbin; ibin++) {
		cur_pdf = gsl_vector_get(rpdfdata, ibin);
		if (cur_pdf < 1e-200) 
			gsl_vector_set(rpdfdata, ibin, second_smallest_pdf);
		pdf_sum += gsl_vector_get(rpdfdata, ibin);

	}
	/* normalize pdf */
	gsl_vector_scale(rpdfdata, 1/(pdf_sum*rank_pdf->step));

	/* calculate fap from pdf */
	gsl_vector *rfapdata = rank->rank_fap->data;
	double pdf_acum = 0;
	for (ibin=nbin-1; ibin>=0; ibin--) {
		cur_pdf = gsl_vector_get(rpdfdata, ibin);
		pdf_acum += cur_pdf;
		if (pdf_acum * (rank_pdf->step) < FLT_MIN)
			gsl_vector_set(rfapdata, ibin, FLT_MIN);
		else
			gsl_vector_set(rfapdata, ibin, pdf_acum * (rank_pdf->step));
	}


	if (fabs(gsl_vector_max(rfapdata) - 1.0) > 1e-2)
		fprintf(stderr, "fap cmax %f\n", gsl_vector_max(rfapdata));
}

/* convert collected rate into rank rate */
void
trigger_stats_feature_to_rank(FeatureStats *feature, RankingStats *rank)
{
	Bins2D *fpdf = feature->lgsnr_lgchisq_pdf;
	int nbin_x = fpdf->nbin_x, nbin_y = fpdf->nbin_y;
	int ibin_x, ibin_y;
   	double	tmp;
	gsl_matrix *fpdfdata = fpdf->data;
	double fpdf_sum = gsl_matrix_sum(fpdfdata);
	// no data values, return
	if (fpdf_sum < 1e-5)
		return;

	/* cdf is our rankings statistic
	 * use the enlongated-estimated pdf to get the full cdf to cover very significant region
	 * this requires that the pdf estimation should be realiable in that region. We will later
	 * use the elonged pdf to get the distribution of cdf.  */
	gsl_matrix *cdfdata = rank->rank_map->data;

	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++) {
			tmp = 0;
			if (ibin_y > 0)
				tmp += gsl_matrix_get(cdfdata, ibin_x, ibin_y-1);
			if (ibin_x < nbin_x-1)
				tmp += gsl_matrix_get(cdfdata, ibin_x+1, ibin_y);
			if (ibin_x < nbin_x-1 && ibin_y > 0)
				tmp -= gsl_matrix_get(cdfdata, ibin_x+1, ibin_y-1);
			tmp += gsl_matrix_get(fpdfdata, ibin_x, ibin_y) * fpdf->step_x * fpdf->step_y;
			// note here we actually assign comulative prob. FIXME: overflow problem ?
			gsl_matrix_set(cdfdata, ibin_x, ibin_y, tmp);

		}
	}
	// note the rank_min and max is based on log10 scale, need to log10 the cdf data
	// also set minimal cdf to be RANK_MIN_LIMIT
	double cur_cdf;
	for (ibin_x=nbin_x-1; ibin_x>=0; ibin_x--) {
		for (ibin_y=0; ibin_y<=nbin_y-1; ibin_y++){
			/* avoid the log10(0) error, since the rank_min is -30, assign rank to be less than -30 is accurate enough*/
			cur_cdf = gsl_matrix_get(cdfdata, ibin_x, ibin_y);
			cur_cdf = MAX(RANK_MIN_LIMIT, cur_cdf);
			gsl_matrix_set(cdfdata, ibin_x, ibin_y, log10(cur_cdf));
		}
	}
	/* generate rank distribution from rate. The rank_rate will only be used for reference.
	 * We generate fap using the enlongated pdf
	 * to cover significant region */
	gsl_vector_long *rratedata = rank->rank_rate->data;
	gsl_matrix_long *fratedata = feature->lgsnr_lgchisq_rate->data;
	double cur_rank_min, cur_rank_max, cur_pdf;
	int nbin_rank = rank->rank_rate->nbin, ibin;
	gsl_vector *rpdfdata = rank->rank_pdf->data;
	long cur_rate;
	for (ibin=0; ibin<nbin_rank; ibin++) {
		// FIXME:consider non-even distribution of cdf bins
		if (ibin == 0)
			cur_rank_min = log10(RANK_MIN_LIMIT) -1;
		else
			cur_rank_min = bins1D_get_low_bound(rank->rank_pdf, ibin);
		cur_rank_max = bins1D_get_up_bound(rank->rank_pdf, ibin);
		cur_pdf = calc_rank_pdf_val(fpdfdata, cdfdata, cur_rank_min, cur_rank_max);
		cur_pdf = cur_pdf*fpdf->step_x*fpdf->step_y/rank->rank_pdf->step;
		gsl_vector_set(rpdfdata, ibin, cur_pdf);
		/* set the rate */
		cur_rate = calc_rank_rate_val(fratedata, cdfdata, cur_rank_min, cur_rank_max);
		gsl_vector_long_set(rratedata, ibin, cur_rate);
	}
	/* rank pdf could be zero, set pdf=0 to pdf=next smallest value */
	double second_smallest_pdf = 1.0;
	for (ibin_x=0; ibin_x<nbin_rank; ibin_x++) {
			cur_pdf = gsl_vector_get(rpdfdata, ibin_x);
			if (cur_pdf > 0.0 && second_smallest_pdf > cur_pdf) 
				second_smallest_pdf = cur_pdf;
	}

	double pdf_sum = 0;
	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) {
		cur_pdf = gsl_vector_get(rpdfdata, ibin_x);
		if (cur_pdf < 1e-200) 
			gsl_vector_set(rpdfdata, ibin_x, second_smallest_pdf);
		pdf_sum += gsl_vector_get(rpdfdata, ibin_x);

	}
	/* normalize pdf */
	gsl_vector_scale(rpdfdata, 1/(pdf_sum*rank->rank_pdf->step));

	/* calculate fap from pdf */
	gsl_vector *rfapdata = rank->rank_fap->data;
	double pdf_acum = 0;
	for (ibin_x=0; ibin_x<nbin_x; ibin_x++) {
		cur_pdf = gsl_vector_get(rpdfdata, ibin_x);
		pdf_acum += cur_pdf;
		if (pdf_acum * (rank->rank_pdf->step) < FLT_MIN)
			gsl_vector_set(rfapdata, ibin, FLT_MIN);
		else
			gsl_vector_set(rfapdata, ibin_x, pdf_acum * (rank->rank_pdf->step));
	}

	if (fabs(gsl_vector_max(rfapdata) - 1.0) > 1e-2)
		fprintf(stderr, "fap cmax %f\n", gsl_vector_max(rfapdata));
}

double
trigger_stats_get_val_from_map(double snr, double chisq, Bins2D *bins)
{
  double lgsnr = log10(snr), lgchisq = log10(chisq);
  int x_idx = 0, y_idx = 0;
  x_idx = MIN(MAX((lgsnr - bins->cmin_x - bins->step_x_2) / bins->step_x, 0), bins->nbin_x-1);
  y_idx = MIN(MAX((lgchisq - bins->cmin_y - bins->step_y_2) / bins->step_y, 0), bins->nbin_y-1);
  return gsl_matrix_get((gsl_matrix *)bins->data, x_idx, y_idx);
}

/*
 * background xml utils
 */
gboolean
trigger_stats_xml_from_xml(TriggerStatsXML *stats, int *hist_trials, const char *filename)
{
  /* sanity check */
  if (!g_file_test(filename, G_FILE_TEST_EXISTS)) {
    return FALSE;
  }

  int nelem = 13; // 4 for feature, 4 for rank, 5 for feature and rank nevent, feature and rank livetime, feature mean of rankmap
  int ncombo = stats->ncombo;
  int nnode = ncombo * nelem + 1, icombo; // 1 for hist_trials
  /* read rate */

  XmlNodeStruct * xns = (XmlNodeStruct *) malloc(sizeof(XmlNodeStruct) * nnode);
  XmlArray *array_lgsnr_rate = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_lgchisq_rate = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_lgsnr_lgchisq_rate = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_lgsnr_lgchisq_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_rank_map = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlParam *param_feature_nevent = (XmlParam *) malloc(sizeof(XmlParam) * ncombo);
  XmlParam *param_mean_rankmap = (XmlParam *) malloc(sizeof(XmlParam) * ncombo);
  XmlParam *param_rank_nevent = (XmlParam *) malloc(sizeof(XmlParam) * ncombo);
  XmlParam *param_feature_livetime = (XmlParam *) malloc(sizeof(XmlParam) * ncombo);
  XmlParam *param_rank_livetime = (XmlParam *) malloc(sizeof(XmlParam) * ncombo);
  XmlArray *array_rank_rate = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_rank_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_rank_fap = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);

  int pos_xns;
  for (icombo=0; icombo<ncombo; icombo++) {
	pos_xns = icombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_%s:array",  stats->feature_xmlname->str, IFOComboMap[icombo].name, SNR_RATE_SUFFIX);
	xns[pos_xns].processPtr = readArray;
	xns[pos_xns].data = &(array_lgsnr_rate[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_%s:array",  stats->feature_xmlname->str, IFOComboMap[icombo].name, CHISQ_RATE_SUFFIX);
	xns[pos_xns].processPtr = readArray;
	xns[pos_xns].data = &(array_lgchisq_rate[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_%s:array",  stats->feature_xmlname->str, IFOComboMap[icombo].name, SNR_CHISQ_RATE_SUFFIX);
	xns[pos_xns].processPtr = readArray;
	xns[pos_xns].data = &(array_lgsnr_lgchisq_rate[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_%s:array",  stats->feature_xmlname->str, IFOComboMap[icombo].name, SNR_CHISQ_PDF_SUFFIX);
	xns[pos_xns].processPtr = readArray;
	xns[pos_xns].data = &(array_lgsnr_lgchisq_pdf[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_%s:array",  stats->rank_xmlname->str, IFOComboMap[icombo].name, RANK_MAP_SUFFIX);
	xns[pos_xns].processPtr = readArray;
	xns[pos_xns].data = &(array_rank_map[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_nevent:param",  stats->feature_xmlname->str, IFOComboMap[icombo].name);
	xns[pos_xns].processPtr = readParam;
	xns[pos_xns].data = &(param_feature_nevent[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_mean_rankmap:param",  stats->feature_xmlname->str, IFOComboMap[icombo].name);
	xns[pos_xns].processPtr = readParam;
	xns[pos_xns].data = &(param_mean_rankmap[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_nevent:param",  stats->rank_xmlname->str, IFOComboMap[icombo].name);
	xns[pos_xns].processPtr = readParam;
	xns[pos_xns].data = &(param_rank_nevent[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_livetime:param",  stats->feature_xmlname->str, IFOComboMap[icombo].name);
	xns[pos_xns].processPtr = readParam;
	xns[pos_xns].data = &(param_feature_livetime[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_livetime:param",  stats->rank_xmlname->str, IFOComboMap[icombo].name);
	xns[pos_xns].processPtr = readParam;
	xns[pos_xns].data = &(param_rank_livetime[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_%s:array",  stats->rank_xmlname->str, IFOComboMap[icombo].name, RANK_RATE_SUFFIX);
	xns[pos_xns].processPtr = readArray;
	xns[pos_xns].data = &(array_rank_rate[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_%s:array",  stats->rank_xmlname->str, IFOComboMap[icombo].name, RANK_PDF_SUFFIX);
	xns[pos_xns].processPtr = readArray;
	xns[pos_xns].data = &(array_rank_pdf[icombo]);

	pos_xns += ncombo;
	sprintf((char *)xns[pos_xns].tag, "%s:%s_%s:array",  stats->rank_xmlname->str, IFOComboMap[icombo].name, RANK_FAP_SUFFIX);
	xns[pos_xns].processPtr = readArray;
	xns[pos_xns].data = &(array_rank_fap[icombo]);

  }

  XmlParam *param_hist_trials = (XmlParam *) malloc(sizeof(XmlParam) * 1);

  pos_xns = nelem * ncombo;
  GString *hist_name = g_string_new(NULL);
  g_string_printf(hist_name, "%s:hist_trials:param", stats->feature_xmlname->str);
  sprintf((char *)xns[pos_xns].tag, hist_name->str);
  xns[pos_xns].processPtr = readParam;
  xns[pos_xns].data = param_hist_trials;
  g_string_free(hist_name, TRUE);
 
  parseFile(filename, xns, nnode);

  /* load to stats */

  TriggerStats **multistats = stats->multistats;
  int nbin_x = multistats[0]->feature->lgsnr_lgchisq_pdf->nbin_x, nbin_y = multistats[0]->feature->lgsnr_lgchisq_pdf->nbin_y;
  int x_size = sizeof(double) * nbin_x, y_size = sizeof(double) * nbin_y;
  int xy_size = sizeof(double) * nbin_x * nbin_y;

  /* make sure the dimensions of the acquired array is consistent 
   * with the dimensions we can read set in the .h file
   */
  g_assert(array_lgsnr_rate[0].dim[0] == nbin_x);
  g_assert(array_lgchisq_rate[0].dim[0] == nbin_y);

  for (icombo=0; icombo<ncombo; icombo++) {
	TriggerStats *cur_stats = multistats[icombo];
	FeatureStats *feature = cur_stats->feature;
	RankingStats *rank = cur_stats->rank;
	memcpy(((gsl_vector_long *)feature->lgsnr_rate->data)->data, (long *)array_lgsnr_rate[icombo].data, x_size);
	memcpy(((gsl_vector_long *)feature->lgchisq_rate->data)->data, (long *)array_lgchisq_rate[icombo].data, y_size);
	memcpy(((gsl_matrix_long *)feature->lgsnr_lgchisq_rate->data)->data, (long *)array_lgsnr_lgchisq_rate[icombo].data, xy_size);
	memcpy(((gsl_matrix *)feature->lgsnr_lgchisq_pdf->data)->data, array_lgsnr_lgchisq_pdf[icombo].data, xy_size);

	memcpy(((gsl_matrix *)rank->rank_map->data)->data, array_rank_map[icombo].data, xy_size);
	memcpy(((gsl_vector_long *)rank->rank_rate->data)->data, (long *)array_rank_rate[icombo].data, y_size);
	memcpy(((gsl_vector *)rank->rank_pdf->data)->data, (long *)array_rank_pdf[icombo].data, y_size);
	memcpy(((gsl_vector *)rank->rank_fap->data)->data, (long *)array_rank_fap[icombo].data, y_size);
	cur_stats->feature_nevent = *((long *)param_feature_nevent[icombo].data);
	cur_stats->rank->mean_rankmap = *((double *)param_mean_rankmap[icombo].data);
	cur_stats->rank_nevent = *((long *)param_rank_nevent[icombo].data);
	cur_stats->feature_livetime = *((long *)param_feature_livetime[icombo].data);
	cur_stats->rank_livetime = *((long *)param_rank_livetime[icombo].data);
	//printf("filename %s, icombo %d, fap addr %p\n", filename, icombo, ((gsl_matrix *)cur_stats->fap->data)->data);
	//printf("icombo %d, nevent addr %p, %p\n", icombo, (param_nevent[icombo].data), (&(param_nevent[icombo]))->data);
  }
  *hist_trials = *((int *)param_hist_trials->data);

  // FIXME: some sanity check for file loading
  //printf( "load stats file\n");
  /*
   * Cleanup function for the XML library.
   */
  xmlCleanupParser();
  /*
   * this is to debug memory for regression tests
   */
  xmlMemoryDump();

  /*
   * free the allocated memory for xml reading
   */
  for (icombo=0; icombo<ncombo; icombo++) {
	free(array_lgsnr_rate[icombo].data);
	free(array_lgchisq_rate[icombo].data);
	free(array_lgsnr_lgchisq_rate[icombo].data);
	free(array_lgsnr_lgchisq_pdf[icombo].data);
	free(param_feature_nevent[icombo].data);
	free(param_mean_rankmap[icombo].data);
	free(param_rank_nevent[icombo].data);
	free(param_feature_livetime[icombo].data);
	free(param_rank_livetime[icombo].data);
	free(array_rank_map[icombo].data);
	free(array_rank_rate[icombo].data);
	free(array_rank_pdf[icombo].data);
	free(array_rank_fap[icombo].data);
  }
  free(array_lgsnr_rate);
  free(array_lgchisq_rate);
  free(array_lgsnr_lgchisq_rate);
  free(array_lgsnr_lgchisq_pdf);
  free(param_feature_nevent);
  free(param_mean_rankmap);
  free(param_rank_nevent);
  free(param_feature_livetime);
  free(param_rank_livetime);
  free(array_rank_map);
  free(array_rank_rate);
  free(array_rank_pdf);
  free(array_rank_fap);
  free(param_hist_trials->data);
  free(param_hist_trials);
  free(xns);

  return TRUE;
}

static  gboolean
write_stats_xmlheader(xmlTextWriterPtr *pwriter, const char *filename, gchar *id_name) {
  /* Create a new XmlWriter for uri, with no compression. */
  *pwriter = xmlNewTextWriterFilename(filename, 1);
  xmlTextWriterPtr writer = *pwriter;
  if (writer == NULL) {
	  printf("testXmlwriterFilename: Error creating the xml writer\n");
	  return FALSE;
  }

  int rc;
  rc = xmlTextWriterSetIndent(writer, 1);
  rc = xmlTextWriterSetIndentString(writer, BAD_CAST "\t");

  /* Start the document with the xml default for the version,
   * encoding utf-8 and the default for the standalone
   * declaration. */
  rc = xmlTextWriterStartDocument(writer, NULL, MY_ENCODING, NULL);
  if (rc < 0) {
	  printf
		  ("testXmlwriterFilename: Error at xmlTextWriterStartDocument\n");
	  return FALSE;
  }

  rc = xmlTextWriterWriteDTD(writer, BAD_CAST "LIGO_LW", NULL, BAD_CAST "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt", NULL);
  if (rc < 0) {
	  printf
		  ("testXmlwriterFilename: Error at xmlTextWriterWriteDTD\n");
	  return FALSE;
  }

  /* Start an element named "LIGO_LW". Since thist is the first
   * element, this will be the root element of the document. */
  rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
  if (rc < 0) {
	  printf
		  ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
	  return FALSE;
  }

  /* Start an element named "LIGO_LW" as child of EXAMPLE. */
  rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
  if (rc < 0) {
	  printf
		  ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
	  return FALSE;
  }

  /* Add an attribute with name "Name" and value id_name to LIGO_LW. */
  rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name",
								   BAD_CAST id_name);
  if (rc < 0) {
	  printf
		  ("testXmlwriterFilename: Error at xmlTextWriterWriteAttribute\n");
	  return FALSE;
  }
  return TRUE;
}
static void
construct_table_content(XmlTable *table, XmlHashVal *vals, GString *name, float cmin, float cmax, int nbin)
{
  table->hashContent = g_hash_table_new_full((GHashFunc)g_string_hash, (GEqualFunc)g_string_equal, NULL, NULL);

  vals[0].name = g_string_new(NULL);
  g_string_printf(vals[0].name, "%s:cmin", name->str);
  g_array_append_val(table->names, *g_string_new(vals[0].name->str));
  vals[0].type = g_string_new("real_4");
  vals[0].data = g_array_new(FALSE, FALSE, sizeof(float));
  g_array_append_val(vals[0].data, cmin);
  g_hash_table_insert(table->hashContent, vals[0].name, vals + 0);

  vals[1].name = g_string_new(NULL);
  g_string_printf(vals[1].name, "%s:cmax", name->str);
  g_array_append_val(table->names, *g_string_new(vals[1].name->str));
  vals[1].type = g_string_new("real_4");
  vals[1].data = g_array_new(FALSE, FALSE, sizeof(float));
  g_array_append_val(vals[1].data, cmax);
  g_hash_table_insert(table->hashContent, vals[1].name, vals + 1);

  vals[2].name = g_string_new(NULL);
  g_string_printf(vals[2].name, "%s:nbin", name->str);
  g_array_append_val(table->names, *g_string_new(vals[2].name->str));
  vals[2].type = g_string_new("int_4s");
  vals[2].data = g_array_new(FALSE, FALSE, sizeof(float));
  g_array_append_val(vals[2].data, nbin);
  g_hash_table_insert(table->hashContent, vals[2].name, vals + 2);
}

gboolean
trigger_stats_xml_dump(TriggerStatsXML *stats, int hist_trials, const char *filename, int write_type, xmlTextWriterPtr *pwriter)
{
  if (write_type == STATS_XML_WRITE_START || write_type == STATS_XML_WRITE_FULL) {
	  gboolean rt = write_stats_xmlheader(pwriter, filename, STATS_XML_ID_NAME);
	  if (rt == FALSE) {
		  printf("not able to write stats header\n");
	  }
  }
  printf("write %s\n", stats->rank_xmlname->str);
  xmlTextWriterPtr writer = *pwriter;
  int icombo = 0, ncombo = stats->ncombo;
  XmlArray *array_lgsnr_rate = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_lgchisq_rate = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_lgsnr_lgchisq_rate = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_lgsnr_lgchisq_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_rank_map = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_rank_rate = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_rank_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_rank_fap = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);

  TriggerStats **multistats = stats->multistats;
  int nbin_x = multistats[0]->feature->lgsnr_lgchisq_pdf->nbin_x, nbin_y = multistats[0]->feature->lgsnr_lgchisq_pdf->nbin_y;
  int x_size = sizeof(double) * nbin_x, y_size = sizeof(double) * nbin_y;
  int xy_size = sizeof(double) * nbin_x * nbin_y;

  for (icombo=0; icombo<ncombo; icombo++) {
	TriggerStats *cur_stats = multistats[icombo];
	FeatureStats *feature = cur_stats->feature;
	RankingStats *rank = cur_stats->rank;
	// assemble lgsnr_rate
	array_lgsnr_rate[icombo].ndim = 1;
	array_lgsnr_rate[icombo].dim[0] = nbin_x;
	array_lgsnr_rate[icombo].data = (long *) malloc(x_size);
	memcpy(array_lgsnr_rate[icombo].data, ((gsl_vector_long *)feature->lgsnr_rate->data)->data, x_size);
	// assemble lgchisq_rate
	array_lgchisq_rate[icombo].ndim = 1;
	array_lgchisq_rate[icombo].dim[0] = nbin_y;
	array_lgchisq_rate[icombo].data = (long *) malloc(y_size);
	memcpy(array_lgchisq_rate[icombo].data, ((gsl_vector_long *)feature->lgchisq_rate->data)->data, y_size);
	// assemble lgsnr_lgchisq_rate
	array_lgsnr_lgchisq_rate[icombo].ndim = 2;
	array_lgsnr_lgchisq_rate[icombo].dim[0] = nbin_x;
	array_lgsnr_lgchisq_rate[icombo].dim[1] = nbin_y;
	array_lgsnr_lgchisq_rate[icombo].data = (long *) malloc(xy_size);
	memcpy(array_lgsnr_lgchisq_rate[icombo].data, ((gsl_matrix_long *)feature->lgsnr_lgchisq_rate->data)->data, xy_size);
	// aseemble lgsnr_lgchisq_pdf
	array_lgsnr_lgchisq_pdf[icombo].ndim = 2;
	array_lgsnr_lgchisq_pdf[icombo].dim[0] = nbin_x;
	array_lgsnr_lgchisq_pdf[icombo].dim[1] = nbin_y;
	array_lgsnr_lgchisq_pdf[icombo].data = (double *) malloc(xy_size);
	memcpy(array_lgsnr_lgchisq_pdf[icombo].data, ((gsl_matrix *)feature->lgsnr_lgchisq_pdf->data)->data, xy_size);
	// assemble rank_map
	array_rank_map[icombo].ndim = 2;
	array_rank_map[icombo].dim[0] = nbin_x;
	array_rank_map[icombo].dim[1] = nbin_y;
	array_rank_map[icombo].data = (double *) malloc(x_size * y_size);
	memcpy(array_rank_map[icombo].data, ((gsl_matrix *)rank->rank_map->data)->data, xy_size);
	// assemble rank_rate
	array_rank_rate[icombo].ndim = 1;
	array_rank_rate[icombo].dim[0] = nbin_x;
	array_rank_rate[icombo].data = (long *) malloc(x_size);
	memcpy(array_rank_rate[icombo].data, ((gsl_vector_long *)rank->rank_rate->data)->data, x_size);
	// assemble rank_pdf
	array_rank_pdf[icombo].ndim = 1;
	array_rank_pdf[icombo].dim[0] = nbin_x;
	array_rank_pdf[icombo].data = (double *) malloc(x_size);
	memcpy(array_rank_pdf[icombo].data, ((gsl_vector *)rank->rank_pdf->data)->data, x_size);
	// assemble rank_fap
	array_rank_fap[icombo].ndim = 1;
	array_rank_fap[icombo].dim[0] = nbin_x;
	array_rank_fap[icombo].data = (double *) malloc(x_size);
	memcpy(array_rank_fap[icombo].data, ((gsl_vector *)rank->rank_fap->data)->data, x_size);
  }

  /* write a table*/

  XmlTable *rank_range_table, *feature_range_table;
  rank_range_table = (XmlTable *)malloc(sizeof(XmlTable));
  GString *name = g_string_new(NULL);
  g_string_printf(name, "%s:%s", stats->rank_xmlname->str, RANK_RATE_SUFFIX);

  rank_range_table->tableName = g_string_new(NULL);
  g_string_printf(rank_range_table->tableName, "%s:table", name->str);

  rank_range_table->delimiter = g_string_new(",");
    
  rank_range_table->names = g_array_new(FALSE, FALSE, sizeof(GString)); 

  XmlHashVal *vals = (XmlHashVal*)malloc(sizeof(XmlHashVal)*3);
  construct_table_content(rank_range_table, vals, name, LOGRANK_CMIN, LOGRANK_CMAX, LOGRANK_NBIN);
  int rt = ligoxml_write_Table(writer, rank_range_table);

  /* free memory used by constructing a XmlTable */
  for (int ival = 0; ival< 3; ival++) {
	  g_array_free(vals[ival].data, TRUE);
	  g_string_free(vals[ival].name, TRUE);
	  g_string_free(vals[ival].type, TRUE);
  }
  free(vals);
  g_string_free(name, TRUE);
  freeTable(rank_range_table);
  free(rank_range_table);
  /* end of memory free of the XmlTable */

  XmlParam param_mean_rankmap;
  param_mean_rankmap.data = (double *) malloc (sizeof(double));

  XmlParam param_feature_nevent;
  param_feature_nevent.data = (long *) malloc (sizeof(long));

  XmlParam param_rank_nevent;
  param_rank_nevent.data = (long *) malloc (sizeof(long));

  XmlParam param_feature_livetime;
  param_feature_livetime.data = (long *) malloc (sizeof(long));

  XmlParam param_rank_livetime;
  param_rank_livetime.data = (long *) malloc (sizeof(long));


  GString *array_name = g_string_new(NULL);
  GString *param_name = g_string_new(NULL);
  for (icombo=0; icombo<ncombo; icombo++) {
	// write features
	g_string_printf(array_name, "%s:%s_%s:array",  stats->feature_xmlname->str, IFOComboMap[icombo].name, SNR_RATE_SUFFIX);
	ligoxml_write_Array(writer, &(array_lgsnr_rate[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
	g_string_printf(array_name, "%s:%s_%s:array",  stats->feature_xmlname->str, IFOComboMap[icombo].name, CHISQ_RATE_SUFFIX);
	ligoxml_write_Array(writer, &(array_lgchisq_rate[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
	g_string_printf(array_name, "%s:%s_%s:array",  stats->feature_xmlname->str, IFOComboMap[icombo].name, SNR_CHISQ_RATE_SUFFIX);
	ligoxml_write_Array(writer, &(array_lgsnr_lgchisq_rate[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
	g_string_printf(array_name, "%s:%s_%s:array",  stats->feature_xmlname->str, IFOComboMap[icombo].name, SNR_CHISQ_PDF_SUFFIX);
	ligoxml_write_Array(writer, &(array_lgsnr_lgchisq_pdf[icombo]), BAD_CAST "real_8", BAD_CAST " ", BAD_CAST array_name->str);

	// write rank
	g_string_printf(array_name, "%s:%s_%s:array",  stats->rank_xmlname->str, IFOComboMap[icombo].name, RANK_MAP_SUFFIX);
	ligoxml_write_Array(writer, &(array_rank_map[icombo]), BAD_CAST "real_8", BAD_CAST " ", BAD_CAST array_name->str);
	g_string_printf(array_name, "%s:%s_%s:array",  stats->rank_xmlname->str, IFOComboMap[icombo].name, RANK_RATE_SUFFIX);
	ligoxml_write_Array(writer, &(array_rank_rate[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
	g_string_printf(array_name, "%s:%s_%s:array",  stats->rank_xmlname->str, IFOComboMap[icombo].name, RANK_PDF_SUFFIX);
	ligoxml_write_Array(writer, &(array_rank_pdf[icombo]), BAD_CAST "real_8", BAD_CAST " ", BAD_CAST array_name->str);
	g_string_printf(array_name, "%s:%s_%s:array",  stats->rank_xmlname->str, IFOComboMap[icombo].name, RANK_FAP_SUFFIX);
	ligoxml_write_Array(writer, &(array_rank_fap[icombo]), BAD_CAST "real_8", BAD_CAST " ", BAD_CAST array_name->str);

	// write nevent
	g_string_printf(param_name, "%s:%s_nevent:param",  stats->feature_xmlname->str, IFOComboMap[icombo].name);
	((long *)param_feature_nevent.data)[0] = multistats[icombo]->feature_nevent;
	ligoxml_write_Param(writer, &param_feature_nevent, BAD_CAST "int_8s", BAD_CAST param_name->str);
	g_string_printf(param_name, "%s:%s_nevent:param",  stats->rank_xmlname->str, IFOComboMap[icombo].name);
	((long *)param_rank_nevent.data)[0] = multistats[icombo]->rank_nevent;
	ligoxml_write_Param(writer, &param_rank_nevent, BAD_CAST "int_8s", BAD_CAST param_name->str);

	// write mean of rank_map
	g_string_printf(param_name, "%s:%s_mean_rankmap:param",  stats->feature_xmlname->str, IFOComboMap[icombo].name);
	((long *)param_mean_rankmap.data)[0] = multistats[icombo]->rank->mean_rankmap;
	ligoxml_write_Param(writer, &param_mean_rankmap, BAD_CAST "real_8", BAD_CAST param_name->str);

	// write livetime
	g_string_printf(param_name, "%s:%s_livetime:param",  stats->feature_xmlname->str, IFOComboMap[icombo].name);
	((long *)param_feature_livetime.data)[0] = multistats[icombo]->feature_livetime;
	ligoxml_write_Param(writer, &param_feature_livetime, BAD_CAST "int_8s", BAD_CAST param_name->str);

	g_string_printf(param_name, "%s:%s_livetime:param",  stats->rank_xmlname->str, IFOComboMap[icombo].name);
	((long *)param_rank_livetime.data)[0] = multistats[icombo]->rank_livetime;
	ligoxml_write_Param(writer, &param_rank_livetime, BAD_CAST "int_8s", BAD_CAST param_name->str);
  }

  XmlParam param_hist_trials;
  param_hist_trials.data = (int *) malloc (sizeof(int));
  g_string_printf(param_name, "%s:hist_trials:param", stats->feature_xmlname->str);
  *((int *)param_hist_trials.data) = hist_trials;
  ligoxml_write_Param(writer, &param_hist_trials, BAD_CAST "int_4s", BAD_CAST param_name->str);

  g_string_free(param_name, TRUE);
  g_string_free(array_name, TRUE);

  if (write_type == STATS_XML_WRITE_END || write_type == STATS_XML_WRITE_FULL) {
	/* Since we do not want to
	 * write any other elements, we simply call xmlTextWriterEndDocument,
	 * which will do all the work. */
	int rc = xmlTextWriterEndDocument(writer);
	if (rc < 0) {
		printf
			("testXmlwriterFilename: Error at xmlTextWriterEndDocument\n");
		return FALSE;
	}
  
	xmlFreeTextWriter(writer);
  }
  /*
   * free all alocated memory
   */
  free(param_feature_nevent.data);
  free(param_mean_rankmap.data);
  free(param_rank_nevent.data);
  free(param_feature_livetime.data);
  free(param_rank_livetime.data);
  free(param_hist_trials.data);
  for (icombo=ncombo-1; icombo>=0; icombo--) {
	freeArray(array_lgsnr_rate + icombo);
	freeArray(array_lgchisq_rate + icombo);
	freeArray(array_lgsnr_lgchisq_rate + icombo);
	freeArray(array_lgsnr_lgchisq_pdf + icombo);
	freeArray(array_rank_map + icombo);
	freeArray(array_rank_rate + icombo);
	freeArray(array_rank_pdf + icombo);
	freeArray(array_rank_fap + icombo);
  }
  free(array_lgsnr_rate);
  free(array_lgchisq_rate);
  free(array_lgsnr_lgchisq_rate);
  free(array_lgsnr_lgchisq_pdf);
  free(array_rank_map);
  free(array_rank_rate);
  free(array_rank_pdf);
  free(array_rank_fap);

  /* rename the file, prevent write/ read of the same file problem.
   * rename will wait for file count of the filename to be 0.  */
  return TRUE;
}

void
trigger_stats_pdf_from_data(gsl_vector *data_dim1, gsl_vector *data_dim2, Bins1D *lgsnr_rate, Bins1D *lgchisq_rate, Bins2D *pdf)
{

	//tin_dim1 and tin_dim2 contains points at which estimations are computed
	size_t num_bin1 = LOGSNR_NBIN, num_bin2 = LOGCHISQ_NBIN; //each bin's size
	gsl_vector * tin_dim1 =  gsl_vector_alloc(num_bin1);
	gsl_vector * tin_dim2 =  gsl_vector_alloc(num_bin2);
	//bin of each dimension
#if 0
		double tin_dim1_cmax = gsl_vector_max(data_dim1) + 0.5;  // linspace in power (i.e. logspace)
	double tin_dim1_cmin = gsl_vector_cmin(data_dim1) - 0.5;
	double tin_dim2_cmax = gsl_vector_max(data_dim2) + 0.5;
	double tin_dim2_cmin = gsl_vector_cmin(data_dim2) - 0.5;
	
	gsl_vector_linspace(tin_dim1_cmin, tin_dim1_cmax, num_bin1, tin_dim1);
	gsl_vector_linspace(tin_dim2_cmin, tin_dim2_cmax, num_bin2, tin_dim2);
#endif

	gsl_vector_linspace(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN, tin_dim1);
	gsl_vector_linspace(LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN, tin_dim2);
	gsl_vector * y_hist_result_dim1 = gsl_vector_alloc(num_bin1);
	gsl_vector * y_hist_result_dim2 = gsl_vector_alloc(num_bin2);
	//histogram of each dimension
	gsl_matrix * result_dim1  = gsl_matrix_alloc(tin_dim1->size,tin_dim1->size);
	gsl_matrix * result_dim2  = gsl_matrix_alloc(tin_dim2->size,tin_dim2->size);

	//Compute temporary result of each dimension, equal to the 'y1' and 'y2' in matlab code 'test.m';
	ssvkernel(data_dim1,tin_dim1,y_hist_result_dim1,result_dim1);
	printf("snr data %d, completed\n", data_dim1->size);
	ssvkernel(data_dim2,tin_dim2,y_hist_result_dim2,result_dim2);
	printf("chisq data %d, completed\n", data_dim2->size);

	gsl_vector_double_to_long(y_hist_result_dim1, (gsl_vector_long *)lgsnr_rate->data);
	gsl_vector_double_to_long(y_hist_result_dim2, (gsl_vector_long *)lgchisq_rate->data);
	//two-dimensional histogram
	gsl_vector * temp_tin_dim1 =  gsl_vector_alloc(num_bin1);
	gsl_vector * temp_tin_dim2 =  gsl_vector_alloc(num_bin2);
	gsl_vector_memcpy(temp_tin_dim1,tin_dim1);
	gsl_vector_add_constant(temp_tin_dim1,-gsl_vector_mindiff(tin_dim1)/2);
	gsl_vector_memcpy(temp_tin_dim2,tin_dim2);
	gsl_vector_add_constant(temp_tin_dim2,-gsl_vector_mindiff(tin_dim2)/2);
	gsl_matrix * histogram = gsl_matrix_alloc(tin_dim1->size,tin_dim2->size);
	gsl_matrix_hist3(data_dim1,data_dim2,temp_tin_dim1,temp_tin_dim2,histogram);
	printf("histogram estimation done\n");

	//Compute the 'scale' variable in matlab code 'test.m'
	unsigned i, j;
	for(i=0;i<histogram->size1;i++){
		for(j=0;j<histogram->size2;j++){
			double temp = gsl_matrix_get(histogram,i,j);
			temp = temp/(gsl_vector_get(y_hist_result_dim1,i)*gsl_vector_get(y_hist_result_dim2,j));
			if(isnan(temp))
				gsl_matrix_set(histogram,i,j,0);
			else
				gsl_matrix_set(histogram,i,j,temp);
		}
	}

	//compute the two-dimensional estimation
	gsl_matrix * result = pdf->data;//final result
	gsl_matrix * temp_matrix = gsl_matrix_alloc(tin_dim1->size,tin_dim2->size);
	for(i=0;i<tin_dim1->size;i++){
		for(j=0;j<tin_dim2->size;j++){
			gsl_matrix_get_col(y_hist_result_dim1,result_dim1,i);
			gsl_matrix_get_col(y_hist_result_dim2,result_dim2,j);
			gsl_matrix_xmul(y_hist_result_dim1,y_hist_result_dim2,temp_matrix);
			gsl_matrix_mul_elements(temp_matrix,histogram);
			gsl_matrix_set(result,i,j,gsl_matrix_sum(temp_matrix)/(double)data_dim1->size);
		}
	}

}

double
get_cdf_from_feature(double snr, double chisq, TriggerStats *stats)
{
	RankingStats *rank = stats->rank;
	double step_x = rank->rank_map->step_x, step_y = rank->rank_map->step_y;

	if (fabs(snr) < EPSILON || fabs(chisq) < EPSILON)
		return 0.0;
	return trigger_stats_get_val_from_map(snr, chisq, rank->rank_map) * step_x * step_y;
	/* the bins1D_get_idx will compute log10(x) first and then find index, so need to 10^rank_val for this function */
}


float
get_fap_from_feature(double snr, double chisq, TriggerStats *stats)
{
	RankingStats *rank = stats->rank;
	if (fabs(snr) < EPSILON || fabs(chisq) < EPSILON)
		return 0.0;
	double rank_val = trigger_stats_get_val_from_map(snr, chisq, rank->rank_map);
	/* the bins1D_get_idx will compute log10(x) first and then find index, so need to 10^rank_val for this function */
	int rank_idx = bins1D_get_idx(pow(10, rank_val), rank->rank_pdf);
	double fap = gsl_vector_get(rank->rank_fap->data, rank_idx);
	if (fap < FLT_MIN && fap > 0)
		fap = FLT_MIN;
	return fap;
}

void set_sense_ratio(gchar *value, int *nifo, char **pifos, float *sense_ratio)
{
	gchar **isense, **ifo_sense = g_strsplit(value, ",", -1);
	int sense[MAX_NIFO];
	*nifo = 0;
	GString *tmp_ifos = g_string_new(NULL);
	for (isense = ifo_sense; *isense; isense++) {
		gchar** this_ifo_split = g_strsplit(*isense, ":", -1);
		g_string_append_printf(tmp_ifos, *this_ifo_split);
		int idx = atoi(*(this_ifo_split + 1));
		sense[get_ifo_idx(*this_ifo_split)] = atoi(*(this_ifo_split + 1));
		//printf("ifo %d: sense %d\n", get_ifo_idx(*this_ifo_split), sense[get_ifo_idx(*this_ifo_split)] );
		*nifo = *nifo +1;
		g_strfreev(this_ifo_split);
	}
	*pifos = g_strdup(tmp_ifos->str);
	g_strfreev(ifo_sense);
	g_string_free(tmp_ifos, TRUE);
	if (sense[H_INDEX] > 0) {
		sense_ratio[LH_INDEX] = (float)sense[L_INDEX]/ sense[H_INDEX];
		sense_ratio[VH_INDEX] = (float)sense[V_INDEX]/ sense[H_INDEX];
	}	
	if (sense[L_INDEX] > 0) {
		sense_ratio[HL_INDEX] = (float)sense[H_INDEX]/ sense[L_INDEX];
		sense_ratio[VL_INDEX] = (float)sense[V_INDEX]/ sense[L_INDEX];
	}	
	if (sense[V_INDEX] > 0) {
		sense_ratio[LV_INDEX] = (float)sense[L_INDEX]/ sense[V_INDEX];
		sense_ratio[HV_INDEX] = (float)sense[H_INDEX]/ sense[V_INDEX];
	}	
	//printf("sense ratio : %f, %f, %f, %f\n", sense_ratio[0], sense_ratio[1], sense_ratio[2], sense_ratio[3]);
}



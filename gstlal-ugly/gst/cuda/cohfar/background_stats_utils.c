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

#include <LIGOLw_xmllib/LIGOLwHeader.h>
#include <cohfar/ssvkernel.h>
#include <cohfar/knn_kde.h>

char *IFO_COMBO_MAP[] = {"L1", "H1", "V1", "H1L1", "H1V1", "L1V1", "H1L1V1"};

int get_icombo(char *ifos) {
	int icombo = 0; 
	unsigned len_in = strlen(ifos);
	int nifo_in = (int) len_in / IFO_LEN, nifo_map, iifo, jifo;
	for (icombo=0; icombo<MAX_COMBOS; icombo++) {
		nifo_map = 0;
		if (len_in == strlen(IFO_COMBO_MAP[icombo])) {
			for (iifo=0; iifo<nifo_in; iifo++) {
				for (jifo=0; jifo<nifo_in; jifo++)
					if (strncmp(ifos+iifo*IFO_LEN, IFO_COMBO_MAP[icombo]+jifo*IFO_LEN, IFO_LEN) == 0)
					nifo_map++;
			}
		}
		if (nifo_in == nifo_map)
			return icombo;
	}

	return -1;
}

int get_ncombo(int nifo) {
	// FIXME: hard-coded
	return 7;
  
}

Bins1D *
bins1D_create_long(double cmin, double cmax, int nbin) 
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

Bins2D *
bins2D_create(double x_cmin, double x_cmax, int x_nbin, double y_cmin, double y_cmax, int y_nbin) 
{
  Bins2D * bins = (Bins2D *) malloc(sizeof(Bins2D));
  bins->x_cmin = x_cmin;
  bins->x_cmax = x_cmax;
  bins->x_nbin = x_nbin;
  bins->x_step = (x_cmax - x_cmin) / (x_nbin - 1);
  bins->x_step_2 = bins->x_step / 2;
  bins->y_cmin = y_cmin;
  bins->y_cmax = y_cmax;
  bins->y_nbin = y_nbin;
  bins->y_step = (y_cmax - y_cmin) / (y_nbin - 1);
  bins->y_step_2 = bins->y_step / 2;
  bins->data = gsl_matrix_alloc(x_nbin, y_nbin);
  gsl_matrix_set_zero(bins->data);
  return bins;
}

Bins2D *
bins2D_create_long(double x_cmin, double x_cmax, int x_nbin, double y_cmin, double y_cmax, int y_nbin) 
{
  Bins2D * bins = (Bins2D *) malloc(sizeof(Bins2D));
  bins->x_cmin = x_cmin;
  bins->x_cmax = x_cmax;
  bins->x_nbin = x_nbin;
  bins->x_step = (x_cmax - x_cmin) / (x_nbin - 1);
  bins->y_cmin = y_cmin;
  bins->y_cmax = y_cmax;
  bins->y_nbin = y_nbin;
  bins->y_step = (y_cmax - y_cmin) / (y_nbin - 1);
  bins->data = gsl_matrix_long_alloc(x_nbin, y_nbin);
  gsl_matrix_long_set_zero(bins->data);
  return bins;
}
void
background_stats_reset(BackgroundStats **stats, int ncombo)
{
  int icombo;
  BackgroundRates *rates;
  for (icombo=0; icombo<ncombo; icombo++) {
	  rates = stats[icombo]->rates;
	  gsl_vector_long_set_zero((gsl_vector_long *)rates->lgsnr_bins->data);
	  gsl_vector_long_set_zero((gsl_vector_long *)rates->lgchisq_bins->data);
	  gsl_matrix_long_set_zero((gsl_matrix_long *)rates->hist->data);
  }

}

BackgroundStats **
background_stats_create(char *ifos)
{
  int nifo = 0, ncombo = 0, icombo = 0;
  nifo = strlen(ifos) / IFO_LEN;

  ncombo = get_ncombo(nifo);
  BackgroundStats ** stats = (BackgroundStats **) malloc(sizeof(BackgroundStats *) * ncombo);

  for (icombo=0; icombo<ncombo; icombo++) {
    stats[icombo] = (BackgroundStats *) malloc(sizeof(BackgroundStats));
    BackgroundStats *cur_stats = stats[icombo];
    //printf("len %s, %d\n", IFO_COMBO_MAP[icombo], strlen(IFO_COMBO_MAP[icombo]));
    cur_stats->ifos = malloc(strlen(IFO_COMBO_MAP[icombo]) * sizeof(char));
    strncpy(cur_stats->ifos, IFO_COMBO_MAP[icombo], strlen(IFO_COMBO_MAP[icombo]) * sizeof(char));
    cur_stats->rates = (BackgroundRates *) malloc(sizeof(BackgroundRates));
    BackgroundRates *rates = cur_stats->rates;
    rates->lgsnr_bins = bins1D_create_long(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN);
    rates->lgchisq_bins = bins1D_create_long(LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN);
    rates->hist = bins2D_create_long(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN, LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN);
    cur_stats->pdf = bins2D_create(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN, LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN);
    cur_stats->cdf = bins2D_create(LOGSNR_CMIN, LOGSNR_CMAX, LOGSNR_NBIN, LOGCHISQ_CMIN, LOGCHISQ_CMAX, LOGCHISQ_NBIN);
  }
  return stats;
}

/*
 * background rates utils
 */
int
get_idx_bins1D(double val, Bins1D *bins)
{
  double lgval = log10(val); // double

  if (lgval < bins->cmin) 
    return 0;
  
  if (lgval > bins->cmax) 
    return bins->nbin - 1;

  return (int) ((lgval - bins->cmin - bins->step_2) / bins->step);
}

void
background_stats_rates_update_all(gsl_vector *snr_vec, gsl_vector *chisq_vec, BackgroundRates *rates)
{

	int ievent, nevent = (int) snr_vec->size;
	for (ievent=0; ievent<nevent; ievent++) {
		background_stats_rates_update(gsl_vector_get(snr_vec, ievent), gsl_vector_get(chisq_vec, ievent), rates);
	}
}

void
background_stats_rates_update(double snr, double chisq, BackgroundRates *rates)
{
	int snr_idx = get_idx_bins1D(snr, rates->lgsnr_bins);
	int chisq_idx = get_idx_bins1D(chisq, rates->lgchisq_bins);

	gsl_vector_long *snr_vec = (gsl_vector_long *)rates->lgsnr_bins->data;
	gsl_vector_long *chisq_vec = (gsl_vector_long *)rates->lgchisq_bins->data;
	gsl_matrix_long *hist_mat = (gsl_matrix_long *)rates->hist->data;

	gsl_vector_long_set(snr_vec, snr_idx, gsl_vector_long_get(snr_vec, snr_idx) + 1);
	gsl_vector_long_set(chisq_vec, chisq_idx, gsl_vector_long_get(chisq_vec, chisq_idx) + 1);
	gsl_matrix_long_set(hist_mat, snr_idx, chisq_idx, gsl_matrix_long_get(hist_mat, snr_idx, chisq_idx) + 1);
}

void
background_stats_rates_add(BackgroundRates *rates1, BackgroundRates *rates2)
{
	gsl_vector_long_add((gsl_vector_long *)rates1->lgsnr_bins->data, (gsl_vector_long *)rates2->lgsnr_bins->data);
	gsl_vector_long_add((gsl_vector_long *)rates1->lgchisq_bins->data, (gsl_vector_long *)rates2->lgchisq_bins->data);
	gsl_matrix_long_add((gsl_matrix_long *)rates1->hist->data, (gsl_matrix_long *)rates2->hist->data);
}

/*
 * background cdf utils
 */

gboolean
background_stats_rates_to_pdf(BackgroundRates *rates, Bins2D *pdf)
{

	gsl_vector_long *snr = rates->lgsnr_bins->data;
	gsl_vector_long *chisq = rates->lgchisq_bins->data;

	long nevent = gsl_vector_long_sum(snr);
	if (nevent == 0)
		return TRUE;
	gsl_vector *snr_double = gsl_vector_alloc(snr->size);
	gsl_vector *chisq_double = gsl_vector_alloc(chisq->size);
	gsl_vector_long_to_double(snr, snr_double);
	gsl_vector_long_to_double(chisq, chisq_double);

	gsl_vector *tin_snr = gsl_vector_alloc(pdf->x_nbin);
	gsl_vector *tin_chisq = gsl_vector_alloc(pdf->y_nbin);
	gsl_vector_linspace(pdf->x_cmin, pdf->x_cmax, pdf->x_nbin, tin_snr);
	gsl_vector_linspace(pdf->y_cmin, pdf->y_cmax, pdf->y_nbin, tin_chisq);

	knn_kde(tin_snr, tin_chisq, (gsl_matrix_long *)rates->hist->data, (gsl_matrix *)pdf->data);

}


/* deprecated: ssvkernel-2d estimation is not 
 * consistent with histogram
 */

gboolean
background_stats_rates_to_pdf_ssvkernel(BackgroundRates *rates, Bins2D *pdf)
{

	gsl_vector_long *snr = rates->lgsnr_bins->data;
	gsl_vector_long *chisq = rates->lgchisq_bins->data;

	long nevent = gsl_vector_long_sum(snr);
	if (nevent == 0)
		return TRUE;
	gsl_vector *snr_double = gsl_vector_alloc(snr->size);
	gsl_vector *chisq_double = gsl_vector_alloc(chisq->size);
	gsl_vector_long_to_double(snr, snr_double);
	gsl_vector_long_to_double(chisq, chisq_double);

	gsl_vector *tin_snr = gsl_vector_alloc(pdf->x_nbin);
	gsl_vector *tin_chisq = gsl_vector_alloc(pdf->y_nbin);
	gsl_vector_linspace(pdf->x_cmin, pdf->x_cmax, pdf->x_nbin, tin_snr);
	gsl_vector_linspace(pdf->y_cmin, pdf->y_cmax, pdf->y_nbin, tin_chisq);

	gsl_matrix *result_snr = gsl_matrix_alloc(pdf->x_nbin, pdf->x_nbin);
	gsl_matrix *result_chisq = gsl_matrix_alloc(pdf->y_nbin, pdf->y_nbin);

	ssvkernel_from_hist(snr_double, tin_snr, result_snr);
	ssvkernel_from_hist(chisq_double, tin_chisq, result_chisq);


	//two-dimensional histogram
	gsl_matrix *histogram = gsl_matrix_alloc(snr->size, chisq->size);
	gsl_matrix_long_to_double((gsl_matrix_long *)rates->hist->data, histogram);
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
	double x_step = pdf->x_step, y_step = pdf->y_step;
	double pdf_sum;
       	pdf_sum = x_step * y_step * gsl_matrix_sum(result);
	gsl_matrix_scale(result, 1/pdf_sum);
	//printf("pdf sum %f\n", gsl_matrix_sum(result) * x_step * y_step);
	
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

void
background_stats_pdf_to_cdf(Bins2D *pdf, Bins2D *cdf)
{
	int x_nbin = pdf->x_nbin, y_nbin = pdf->y_nbin;
	int ix, iy;
       	double	tmp;
	gsl_matrix *pdfdata = pdf->data, *cdfdata = cdf->data;

	for (ix=x_nbin-1; ix>=0; ix--) {
		for (iy=0; iy<=y_nbin-1; iy++) {
			tmp = 0;
			if (iy > 0)
				tmp += gsl_matrix_get(cdfdata, ix, iy-1);
			if (ix < x_nbin-1)
				tmp += gsl_matrix_get(cdfdata, ix+1, iy);
			if (ix < x_nbin-1 && iy > 0)
				tmp -= gsl_matrix_get(cdfdata, ix+1, iy-1);
			tmp += gsl_matrix_get(pdfdata, ix, iy) * pdf->x_step * pdf->y_step;
			gsl_matrix_set(cdfdata, ix, iy, tmp);
		}
	}
	//printf("cdf cmax %f\n", gsl_matrix_cmax(cdfdata));
}

double
background_stats_bins2D_get_val(double snr, double chisq, Bins2D *bins)
{
  double lgsnr = log10(snr), lgchisq = log10(chisq);
  int x_idx = 0, y_idx = 0;
  x_idx = MIN(MAX((lgsnr - bins->x_cmin - bins->x_step_2) / bins->x_step, 0), bins->x_nbin-1);
  y_idx = MIN(MAX((lgchisq - bins->y_cmin - bins->y_step_2) / bins->y_step, 0), bins->y_nbin-1);
  return gsl_matrix_get(bins->data, x_idx, y_idx);
}

/*
 * background xml utils
 */
gboolean
background_stats_from_xml(BackgroundStats **stats, const int ncombo, const char *filename)
{
  int nnode = ncombo * 5, icombo; // 3 for rates, 1 for pdf, 1 for cdf
  /* read rates */

  XmlNodeStruct * xns = (XmlNodeStruct *) malloc(sizeof(XmlNodeStruct) * nnode);
  XmlArray *array_lgsnr_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_lgchisq_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_hist = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_cdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);

  for (icombo=0; icombo<ncombo; icombo++) {
    sprintf((char *)xns[icombo].tag, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_SUFFIX);
    xns[icombo].processPtr = readArray;
    xns[icombo].data = &(array_lgsnr_bins[icombo]);

    sprintf((char *)xns[icombo+ncombo].tag, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_CHISQ_SUFFIX);
    xns[icombo+ncombo].processPtr = readArray;
    xns[icombo+ncombo].data = &(array_lgchisq_bins[icombo]);

    sprintf((char *)xns[icombo+2*ncombo].tag, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_HIST_SUFFIX);
    xns[icombo+2*ncombo].processPtr = readArray;
    xns[icombo+2*ncombo].data = &(array_hist[icombo]);

    sprintf((char *)xns[icombo+3*ncombo].tag, "%s:%s%s:array",  BACKGROUND_XML_PDF_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_CHISQ_SUFFIX);
    xns[icombo+3*ncombo].processPtr = readArray;
    xns[icombo+3*ncombo].data = &(array_pdf[icombo]);

    sprintf((char *)xns[icombo+4*ncombo].tag, "%s:%s%s:array",  BACKGROUND_XML_CDF_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_CHISQ_SUFFIX);
    xns[icombo+4*ncombo].processPtr = readArray;
    xns[icombo+4*ncombo].data = &(array_cdf[icombo]);
  }

  printf("before parse stats file\n");
  parseFile(filename, xns, nnode);

  // FIXME: need sanity check that number of rows and columns are the same
  // with the struct of BackgroundStats
  /* load to stats */

  printf( "after parse stats file\n" );
  int x_nbin = stats[0]->pdf->x_nbin, y_nbin = stats[0]->pdf->y_nbin;
  int x_size = sizeof(double) * x_nbin, y_size = sizeof(double) * y_nbin;
  int xy_size = sizeof(double) * x_nbin * y_nbin;

  /* make sure the dimensions of the acquired array is consistent 
   * with the dimensions we can read set in the .h file
   */
  g_assert(array_lgsnr_bins[0].dim[0] == x_nbin);
  g_assert(array_lgchisq_bins[0].dim[0] == y_nbin);

  for (icombo=0; icombo<ncombo; icombo++) {
    BackgroundStats *cur_stats = stats[icombo];
    BackgroundRates *rates = cur_stats->rates;
    memcpy(((gsl_vector_long *)rates->lgsnr_bins->data)->data, (long *)array_lgsnr_bins[icombo].data, x_size);
    memcpy(((gsl_vector_long *)rates->lgchisq_bins->data)->data, (long *)array_lgchisq_bins[icombo].data, y_size);
    memcpy(((gsl_matrix_long *)rates->hist->data)->data, (long *)array_hist[icombo].data, xy_size);
    memcpy(((gsl_matrix *)cur_stats->pdf->data)->data, array_pdf[icombo].data, xy_size);
    memcpy(((gsl_matrix *)cur_stats->cdf->data)->data, array_cdf[icombo].data, xy_size);
  }

  for (icombo=0; icombo<ncombo; icombo++) {
    freeArray(array_lgsnr_bins + icombo);
    freeArray(array_lgchisq_bins + icombo);
    freeArray(array_hist + icombo);
    freeArray(array_pdf + icombo);
    freeArray(array_cdf + icombo);
  }

  free(xns);
  xmlCleanupParser();
  xmlMemoryDump();
  return TRUE;
}


gboolean
background_stats_to_xml(BackgroundStats **stats, const int ncombo, const char *filename)
{
  gchar *tmp_filename = g_strdup_printf("%s_next", filename);
  int icombo = 0;
  XmlParam param_range;
  param_range.data = (double *) malloc(sizeof(double) * 2);
  XmlArray *array_lgsnr_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_lgchisq_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_hist = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_cdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);

  int x_nbin = stats[0]->pdf->x_nbin, y_nbin = stats[0]->pdf->y_nbin;
  int x_size = sizeof(double) * x_nbin, y_size = sizeof(double) * y_nbin;
  int xy_size = sizeof(double) * x_nbin * y_nbin;

  for (icombo=0; icombo<ncombo; icombo++) {
    BackgroundStats *cur_stats = stats[icombo];
    BackgroundRates *rates = cur_stats->rates;
    array_lgsnr_bins[icombo].ndim = 1;
    array_lgsnr_bins[icombo].dim[0] = x_nbin;
    array_lgsnr_bins[icombo].data = (long *) malloc(x_size);
    memcpy(array_lgsnr_bins[icombo].data, ((gsl_vector_long *)rates->lgsnr_bins->data)->data, x_size);
    array_lgchisq_bins[icombo].ndim = 1;
    array_lgchisq_bins[icombo].dim[0] = y_nbin;
    array_lgchisq_bins[icombo].data = (long *) malloc(y_size);
    memcpy(array_lgchisq_bins[icombo].data, ((gsl_vector_long *)rates->lgchisq_bins->data)->data, y_size);
    array_hist[icombo].ndim = 2;
    array_hist[icombo].dim[0] = x_nbin;
    array_hist[icombo].dim[1] = y_nbin;
    array_hist[icombo].data = (long *) malloc(xy_size);
    memcpy(array_hist[icombo].data, ((gsl_matrix_long *)rates->hist->data)->data, xy_size);
    array_pdf[icombo].ndim = 2;
    array_pdf[icombo].dim[0] = x_nbin;
    array_pdf[icombo].dim[1] = y_nbin;
    array_pdf[icombo].data = (double *) malloc(xy_size);
    memcpy(array_pdf[icombo].data, ((gsl_matrix *)cur_stats->pdf->data)->data, xy_size);
    array_cdf[icombo].ndim = 2;
    array_cdf[icombo].dim[0] = x_nbin;
    array_cdf[icombo].dim[1] = y_nbin;
    array_cdf[icombo].data = (double *) malloc(x_size * y_size);
    memcpy(array_cdf[icombo].data, ((gsl_matrix *)cur_stats->cdf->data)->data, xy_size);
  
  }


  int rc;
  xmlTextWriterPtr writer;

  /* Create a new XmlWriter for uri, with no compression. */
  writer = xmlNewTextWriterFilename(tmp_filename, 1);
  if (writer == NULL) {
      printf("testXmlwriterFilename: Error creating the xml writer\n");
      return FALSE;
  }

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

  /* Add an attribute with name "Name" and value "gstlal_spiir_cohfar" to LIGO_LW. */
  rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name",
                                   BAD_CAST "gstlal_spiir_cohfar");
  if (rc < 0) {
      printf
          ("testXmlwriterFilename: Error at xmlTextWriterWriteAttribute\n");
      return FALSE;
  }

  GString *param_name = g_string_new(NULL);

  g_string_printf(param_name, "%s:%s_range:param",  BACKGROUND_XML_RATES_NAME, BACKGROUND_XML_SNR_SUFFIX);
  ((double *)param_range.data)[0] = LOGSNR_CMIN;
  ((double *)param_range.data)[1] = LOGSNR_CMAX;
  ligoxml_write_Param(writer, &param_range, BAD_CAST "real_8", BAD_CAST "DOUBLE");

  g_string_printf(param_name, "%s:%s_range:param",  BACKGROUND_XML_RATES_NAME, BACKGROUND_XML_CHISQ_SUFFIX);
  ((double *)param_range.data)[0] = LOGCHISQ_CMIN;
  ((double *)param_range.data)[1] = LOGCHISQ_CMAX;
  ligoxml_write_Param(writer, &param_range, BAD_CAST "real_8", BAD_CAST "DOUBLE");

  g_string_free(param_name, TRUE);

  GString *array_name = g_string_new(NULL);
  for (icombo=0; icombo<ncombo; icombo++) {
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_SUFFIX);
    ligoxml_write_Array(writer, &(array_lgsnr_bins[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_CHISQ_SUFFIX);
    ligoxml_write_Array(writer, &(array_lgchisq_bins[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_HIST_SUFFIX);
    ligoxml_write_Array(writer, &(array_hist[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_PDF_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_CHISQ_SUFFIX);
    ligoxml_write_Array(writer, &(array_pdf[icombo]), BAD_CAST "real_8", BAD_CAST " ", BAD_CAST array_name->str);
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_CDF_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_CHISQ_SUFFIX);
    ligoxml_write_Array(writer, &(array_cdf[icombo]), BAD_CAST "real_8", BAD_CAST " ", BAD_CAST array_name->str);
  }
  g_string_free(array_name, TRUE);

  /* Since we do not want to
   * write any other elements, we simply call xmlTextWriterEndDocument,
   * which will do all the work. */
  rc = xmlTextWriterEndDocument(writer);
  if (rc < 0) {
      printf
          ("testXmlwriterFilename: Error at xmlTextWriterEndDocument\n");
      return FALSE;
  }

  xmlFreeTextWriter(writer);
  free(param_range.data);
  for (icombo=0; icombo<ncombo; icombo++) {
    freeArray(array_lgsnr_bins + icombo);
    freeArray(array_lgchisq_bins + icombo);
    freeArray(array_hist + icombo);
    freeArray(array_pdf + icombo);
    freeArray(array_cdf + icombo);
  }
  /* rename the file, prevent write/ read of the same file problem.
   * rename will wait for file count of the filename to be 0.  */
  printf("rename from %s\n", tmp_filename);
  g_rename(tmp_filename, filename);
  g_free(tmp_filename);
  return TRUE;
}

void
background_stats_pdf_from_data(gsl_vector *data_dim1, gsl_vector *data_dim2, Bins1D *lgsnr_bins, Bins1D *lgchisq_bins, Bins2D *pdf)
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

	gsl_vector_double_to_long(y_hist_result_dim1, (gsl_vector_long *)lgsnr_bins->data);
	gsl_vector_double_to_long(y_hist_result_dim2, (gsl_vector_long *)lgchisq_bins->data);
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

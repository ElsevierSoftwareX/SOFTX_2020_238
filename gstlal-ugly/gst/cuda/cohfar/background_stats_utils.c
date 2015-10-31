#include "background_stats_utils.h"

#include <math.h>
#include <string.h>
#include <gsl/gsl_vector_long.h>
#include <gsl/gsl_matrix.h>

#include "../LIGOLw_xmllib/LIGOLwHeader.h"
#include "ssvkernel.h"
#include "background_stats_xml.h"

char *IFO_COMBO_MAP[] = {"H1L1", "H1V1", "L1V1", "H1L1V1"};

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

Bins1D *
bins1D_create_long(float min, float max, int nbin) 
{
  Bins1D * bins = (Bins1D *) malloc(sizeof(Bins1D));
  bins->min = min;
  bins->max = max;
  bins->nbin = nbin;
  bins->step = (max - min)/ nbin;
  bins->data = gsl_vector_long_alloc(nbin);
  gsl_vector_long_set_zero(bins->data);
  return bins;
}

Bins2D *
bins2D_create(float x_min, float x_max, int x_nbin, float y_min, float y_max, int y_nbin) 
{
  Bins2D * bins = (Bins2D *) malloc(sizeof(Bins2D));
  bins->x_min = x_min;
  bins->x_max = x_max;
  bins->x_nbin = x_nbin;
  bins->x_step = (x_max - x_min) / x_nbin;
  bins->y_min = y_min;
  bins->y_max = y_max;
  bins->y_nbin = y_nbin;
  bins->y_step = (y_max - y_min) / y_nbin;
  bins->data = gsl_matrix_alloc(x_nbin, y_nbin);
  gsl_matrix_set_zero(bins->data);
  return bins;
}

Bins2D *
bins2D_create_long(float x_min, float x_max, int x_nbin, float y_min, float y_max, int y_nbin) 
{
  Bins2D * bins = (Bins2D *) malloc(sizeof(Bins2D));
  bins->x_min = x_min;
  bins->x_max = x_max;
  bins->x_nbin = x_nbin;
  bins->x_step = (x_max - x_min) / x_nbin;
  bins->y_min = y_min;
  bins->y_max = y_max;
  bins->y_nbin = y_nbin;
  bins->y_step = (y_max - y_min) / y_nbin;
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
	  gsl_vector_long_set_zero((gsl_vector_long *)rates->logsnr_bins->data);
	  gsl_vector_long_set_zero((gsl_vector_long *)rates->logchisq_bins->data);
	  gsl_matrix_long_set_zero((gsl_matrix_long *)rates->hist->data);
  }

}

BackgroundStats **
background_stats_create(char *ifos)
{
  int nifo = 0, ncombo = 0, icombo = 0;
  nifo = strlen(ifos) / IFO_LEN;
  ncombo = pow(2, nifo) - 1 - nifo;
  BackgroundStats ** stats = (BackgroundStats **) malloc(sizeof(BackgroundStats *) * ncombo);

  for (icombo=0; icombo<ncombo; icombo++) {
    stats[icombo] = (BackgroundStats *) malloc(sizeof(BackgroundStats));
    BackgroundStats *cur_stats = stats[icombo];
    //printf("len %s, %d\n", IFO_COMBO_MAP[icombo], strlen(IFO_COMBO_MAP[icombo]));
    cur_stats->ifos = malloc(strlen(IFO_COMBO_MAP[icombo]) * sizeof(char));
    strncpy(cur_stats->ifos, IFO_COMBO_MAP[icombo], strlen(IFO_COMBO_MAP[icombo]) * sizeof(char));
    cur_stats->rates = (BackgroundRates *) malloc(sizeof(BackgroundRates));
    BackgroundRates *rates = cur_stats->rates;
    rates->logsnr_bins = bins1D_create_long(LOGSNR_MIN, LOGSNR_MAX, LOGSNR_NBIN);
    rates->logchisq_bins = bins1D_create_long(LOGCHISQ_MIN, LOGCHISQ_MAX, LOGCHISQ_NBIN);
    rates->hist = bins2D_create_long(LOGSNR_MIN, LOGSNR_MAX, LOGSNR_NBIN, LOGCHISQ_MIN, LOGCHISQ_MAX, LOGCHISQ_NBIN);
    cur_stats->pdf = bins2D_create(LOGSNR_MIN, LOGSNR_MAX, LOGSNR_NBIN, LOGCHISQ_MIN, LOGCHISQ_MAX, LOGCHISQ_NBIN);
    cur_stats->cdf = bins2D_create(LOGSNR_MIN, LOGSNR_MAX, LOGSNR_NBIN, LOGCHISQ_MIN, LOGCHISQ_MAX, LOGCHISQ_NBIN);
  }
  return stats;
}

/*
 * background rates utils
 */
int
get_idx_bins1D(float val, Bins1D *bins)
{
  float logval = log10f(val); // float

  if (logval < bins->min) 
    return 0;
  
  if (logval > bins->max) 
    return bins->nbin - 1;

  return (int) ((logval - bins->min) / bins->step);
}

void
background_stats_rates_update(float snr, float chisq, BackgroundRates *rates)
{
	int snr_idx = get_idx_bins1D(snr, rates->logsnr_bins);
	int chisq_idx = get_idx_bins1D(chisq, rates->logchisq_bins);

	gsl_vector_long *snr_vec = (gsl_vector_long *)rates->logsnr_bins->data;
	gsl_vector_long *chisq_vec = (gsl_vector_long *)rates->logchisq_bins->data;
	gsl_matrix_long *hist_mat = (gsl_matrix_long *)rates->hist->data;

	gsl_vector_long_set(snr_vec, snr_idx, gsl_vector_long_get(snr_vec, snr_idx) + 1);
	gsl_vector_long_set(chisq_vec, chisq_idx, gsl_vector_long_get(chisq_vec, chisq_idx) + 1);
	gsl_matrix_long_set(hist_mat, snr_idx, chisq_idx, gsl_matrix_long_get(hist_mat, snr_idx, chisq_idx) + 1);
}

void
background_stats_rates_add(BackgroundRates *rates1, BackgroundRates *rates2)
{
	gsl_vector_long_add((gsl_vector_long *)rates1->logsnr_bins->data, (gsl_vector_long *)rates2->logsnr_bins->data);
	gsl_vector_long_add((gsl_vector_long *)rates1->logchisq_bins->data, (gsl_vector_long *)rates2->logchisq_bins->data);
	gsl_matrix_long_add((gsl_matrix_long *)rates1->hist->data, (gsl_matrix_long *)rates2->hist->data);
}

/*
 * background cdf utils
 */

gboolean
background_stats_rates_to_pdf(BackgroundRates *rates, Bins2D *pdf)
{

	gsl_vector_long *snr = rates->logsnr_bins->data;
	gsl_vector_long *chisq = rates->logchisq_bins->data;

	long nevent = gsl_vector_long_sum(snr);
	gsl_vector *snr_double = gsl_vector_alloc(snr->size);
	gsl_vector *chisq_double = gsl_vector_alloc(chisq->size);
	gsl_vector_long_to_double(snr, snr_double);
	gsl_vector_long_to_double(chisq, chisq_double);

	gsl_vector *tin_snr = gsl_vector_alloc(pdf->x_nbin);
	gsl_vector *tin_chisq = gsl_vector_alloc(pdf->y_nbin);
	gsl_vector_linspace(pdf->x_min, pdf->x_max, pdf->x_nbin, tin_snr);
	gsl_vector_linspace(pdf->y_min, pdf->y_max, pdf->y_nbin, tin_chisq);

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
	float x_step = pdf->x_step, y_step = pdf->y_step;
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
	//printf("cdf max %f\n", gsl_matrix_max(cdfdata));
}

double
background_stats_bins2D_get_val(float snr, float chisq, Bins2D *bins)
{
  float logsnr = log10f(snr), logchisq = log10f(chisq);
  int x_idx = 0, y_idx = 0;
  x_idx = MIN(MAX((logsnr - bins->x_min) / bins->x_step, 0), bins->x_nbin-1);
  y_idx = MIN(MAX((logchisq - bins->y_min) / bins->y_step, 0), bins->y_nbin-1);
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
  XmlArray *array_logsnr_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_logchisq_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_hist = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_cdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);

  for (icombo=0; icombo<ncombo; icombo++) {
    sprintf((char *)xns[icombo].tag, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_SUFFIX);
    xns[icombo].processPtr = readArray;
    xns[icombo].data = &(array_logsnr_bins[icombo]);

    sprintf((char *)xns[icombo+ncombo].tag, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_CHISQ_SUFFIX);
    xns[icombo+ncombo].processPtr = readArray;
    xns[icombo+ncombo].data = &(array_logchisq_bins[icombo]);

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

  printf("before parse\n");
  parseFile(filename, xns, nnode);

  // FIXME: need sanity check that number of rows and columns are the same
  // with the struct of BackgroundStats
  /* load to stats */

  int x_nbin = stats[0]->pdf->x_nbin, y_nbin = stats[0]->pdf->y_nbin;
  int x_size = sizeof(double) * x_nbin, y_size = sizeof(double) * y_nbin;
  int xy_size = sizeof(double) * x_nbin * y_nbin;

  for (icombo=0; icombo<ncombo; icombo++) {
    BackgroundStats *cur_stats = stats[icombo];
    BackgroundRates *rates = cur_stats->rates;
    memcpy(((gsl_vector_long *)rates->logsnr_bins->data)->data, (long *)array_logsnr_bins[icombo].data, x_size);
    memcpy(((gsl_vector_long *)rates->logchisq_bins->data)->data, (long *)array_logchisq_bins[icombo].data, y_size);
    memcpy(((gsl_matrix_long *)rates->hist->data)->data, (long *)array_hist[icombo].data, xy_size);
    memcpy(((gsl_matrix *)cur_stats->pdf->data)->data, array_pdf[icombo].data, xy_size);
    memcpy(((gsl_matrix *)cur_stats->cdf->data)->data, array_cdf[icombo].data, xy_size);
  }
  printf("done memcpy\n");
  for (icombo=0; icombo<ncombo; icombo++) {
    freeArray(array_logsnr_bins + icombo);
    freeArray(array_logchisq_bins + icombo);
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
  param_range.data = (float *) malloc(sizeof(float) * 2);
  XmlArray *array_logsnr_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_logchisq_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_hist = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_cdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);

  int x_nbin = stats[0]->pdf->x_nbin, y_nbin = stats[0]->pdf->y_nbin;
  int x_size = sizeof(double) * x_nbin, y_size = sizeof(double) * y_nbin;
  int xy_size = sizeof(double) * x_nbin * y_nbin;

  for (icombo=0; icombo<ncombo; icombo++) {
    BackgroundStats *cur_stats = stats[icombo];
    BackgroundRates *rates = cur_stats->rates;
    array_logsnr_bins[icombo].ndim = 1;
    array_logsnr_bins[icombo].dim[0] = x_nbin;
    array_logsnr_bins[icombo].data = (long *) malloc(x_size);
    memcpy(array_logsnr_bins[icombo].data, ((gsl_vector_long *)rates->logsnr_bins->data)->data, x_size);
    array_logchisq_bins[icombo].ndim = 1;
    array_logchisq_bins[icombo].dim[0] = y_nbin;
    array_logchisq_bins[icombo].data = (long *) malloc(y_size);
    memcpy(array_logchisq_bins[icombo].data, ((gsl_vector_long *)rates->logchisq_bins->data)->data, y_size);
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
  ((float *)param_range.data)[0] = LOGSNR_MIN;
  ((float *)param_range.data)[1] = LOGSNR_MAX;
  ligoxml_write_Param(writer, &param_range, BAD_CAST "real_4", BAD_CAST "FLOAT");

  g_string_printf(param_name, "%s:%s_range:param",  BACKGROUND_XML_RATES_NAME, BACKGROUND_XML_CHISQ_SUFFIX);
  ((float *)param_range.data)[0] = LOGCHISQ_MIN;
  ((float *)param_range.data)[1] = LOGCHISQ_MAX;
  ligoxml_write_Param(writer, &param_range, BAD_CAST "real_4", BAD_CAST "FLOAT");

  g_string_free(param_name, TRUE);

  GString *array_name = g_string_new(NULL);
  for (icombo=0; icombo<ncombo; icombo++) {
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_SUFFIX);
    ligoxml_write_Array(writer, &(array_logsnr_bins[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_CHISQ_SUFFIX);
    ligoxml_write_Array(writer, &(array_logchisq_bins[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
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
    freeArray(array_logsnr_bins + icombo);
    freeArray(array_logchisq_bins + icombo);
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
background_stats_pdf_from_data(gsl_vector *data_dim1, gsl_vector *data_dim2, Bins2D *pdf)
{

	//tin_dim1 and tin_dim2 contains points at which estimations are computed
	size_t num_bin1 = LOGSNR_NBIN, num_bin2 = LOGCHISQ_NBIN; //each bin's size
	gsl_vector * tin_dim1 =  gsl_vector_alloc(num_bin1);
	gsl_vector * tin_dim2 =  gsl_vector_alloc(num_bin2);
    //bin of each dimension
#if 0
    	double tin_dim1_max = gsl_vector_max(data_dim1) + 0.5;  // linspace in power (i.e. logspace)
	double tin_dim1_min = gsl_vector_min(data_dim1) - 0.5;
	double tin_dim2_max = gsl_vector_max(data_dim2) + 0.5;
	double tin_dim2_min = gsl_vector_min(data_dim2) - 0.5;
	
	gsl_vector_linspace(tin_dim1_min, tin_dim1_max, num_bin1, tin_dim1);
	gsl_vector_linspace(tin_dim2_min, tin_dim2_max, num_bin2, tin_dim2);
#endif

	gsl_vector_linspace(LOGSNR_MIN, LOGSNR_MAX, LOGSNR_NBIN, tin_dim1);
	gsl_vector_linspace(LOGCHISQ_MIN, LOGCHISQ_MAX, LOGCHISQ_NBIN, tin_dim2);
	gsl_vector * y_hist_result_dim1 = gsl_vector_alloc(num_bin1);
	gsl_vector * y_hist_result_dim2 = gsl_vector_alloc(num_bin2);
    //histogram of each dimension
	gsl_matrix * result_dim1  = gsl_matrix_alloc(tin_dim1->size,tin_dim1->size);
	gsl_matrix * result_dim2  = gsl_matrix_alloc(tin_dim2->size,tin_dim2->size);

	//Compute temporary result of each dimension, equal to the 'y1' and 'y2' in matlab code 'test.m';
	ssvkernel(data_dim2,tin_dim2,y_hist_result_dim2,result_dim2);
	ssvkernel(data_dim1,tin_dim1,y_hist_result_dim1,result_dim1);

	//two-dimensional histogram
	gsl_vector * temp_tin_dim1 =  gsl_vector_alloc(num_bin1);
	gsl_vector * temp_tin_dim2 =  gsl_vector_alloc(num_bin2);
	gsl_vector_memcpy(temp_tin_dim1,tin_dim1);
	gsl_vector_add_constant(temp_tin_dim1,-gsl_vector_mindiff(tin_dim1)/2);
	gsl_vector_memcpy(temp_tin_dim2,tin_dim2);
	gsl_vector_add_constant(temp_tin_dim2,-gsl_vector_mindiff(tin_dim2)/2);
	gsl_matrix * histogram = gsl_matrix_alloc(tin_dim1->size,tin_dim2->size);
	gsl_matrix_hist3(data_dim1,data_dim2,temp_tin_dim1,temp_tin_dim2,histogram);

	//Compute the 'scale' variable in matlab code 'test.m'
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

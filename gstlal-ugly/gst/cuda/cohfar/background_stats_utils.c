#include "background_stats_utils.h"

#define MIN(a, b) a>b?b:a
#define MAX(a, b) a>b?a:b

Bins1D *
bins1D_create(float min, float max, int nbin) 
{
  Bins1D * bins = (Bins1D *) malloc(sizeof(Bins1D));
  bins->min = min;
  bins->max = max;
  bins->nbin = nbin;
  bins->step = (max - min)/ nbin;
  bins->data = gsl_vector_alloc(nbin);
  gsl_vector_set_zero(bins->data);
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

BackgroundStats **
background_stats_create(char *ifos)
{
  int iifo, jifo, nifo = 0, ncombo = 0, icombo = 0;
  nifo = strlen(ifos) / IFO_LEN;
  ncombo = power(2, nifo) - 1 - nifo;
  BackgroundStats ** stats = (BackgroundStats **) malloc(sizeof(BackgroundStats *) * ncombo);

  for (icombo=0; icombo<ncombo; icombo++) {
    stats[icombo] = (BackgroundStats *) malloc(sizeof(BackgroundStats));
    BackgroundStats *cur_stats = stats[icombo];
    strncpy(cur_stats->ifos, IFO_COMBO_MAP[icombo], strlen(IFO_COMBO_MAP[icombo]) * sizeof(char));
    cur_stats->rates = (BackgroundRates *) malloc(sizeof(BackgroundRates));
    BackgroundRates *rates = cur_stats->rates;
    rates->logsnr_bins = bins1D_create(LOGSNR_MIN, LOGSNR_MAX, LOGSNR_NBIN);
    rates->logchisq_bins = bins1D_create(LOGCHISQ_MIN, LOGCHISQ_MAX, LOGCHISQ_NBIN);
    cur_stats->pdf = bins2D_create(LOGSNR_MIN, LOGSNR_MAX, LOGSNR_NBIN, LOGCHISQ_MIN, LOGCHISQ_MAX, LOGCHISQ_NBIN);
    cur_stats->cdf = bins2D_create(LOGSNR_MIN, LOGSNR_MAX, LOGSNR_NBIN, LOGCHISQ_MIN, LOGCHISQ_MAX, LOGCHISQ_NBIN);
  }
  return stats;
}

/*
 * background rates utils
 */
gboolean
add_background_val_to_rates(float val, Bins1D *bins)
{
  float logval = log10f(val); // float
  if (logval < bins->min) {
    gsl_vector_set(bins->data, 0, gsl_vector_get(bins->data, 0) + 1;)
    return TRUE;
  }
  if (logval > bins->max) {
    gsl_vector_set(bins->data, bins->nbin-1, gsl_vector_get(bins->data, bins->nbin-1) + 1);
    return TRUE;
  }

  int idx = (logval - bins->min) / bins->step;
  gsl_vector_set(bins->data, idx, gsl_vector_get(bins->data, idx) + 1);
  return TURE;
}

/*
 * background cdf utils
 */
double
background_stats_get_cdf(float snr, float chisq, Bins2D *bins)
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
  int nnode = ncombo * 4, icombo; // 2 for rates, 1 for pdf, 1 for cdf
  /* read rates */

  XmlNodeStruct * xns = (XmlNodeStruct *) malloc(sizeof(XmlNodeStruct) * ncombo);
  XmlArray *array_logsnr_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_logchisq_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_cdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);

  for (icombo=0; icombo<ncombo; icombo++) {
    sprintf((char *)xns[icombo].tag, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_SUFFIX);
    xns[icombo].processPtr = readArray;
    xns[icombo].data = &(array_logsnr_bins[icombo]);

    sprintf((char *)xns[icombo+ncombo].tag, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_CHISQ_SUFFIX);
    xns[icombo].processPtr = readArray;
    xns[icombo].data = &(array_logchisq_bins[icombo]);

    sprintf((char *)xns[icombo+2*ncombo].tag, "%s:%s%s:array",  BACKGROUND_XML_PDF_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_CHISQ_SUFFIX);
    xns[icombo].processPtr = readArray;
    xns[icombo].data = &(array_pdf[icombo]);

    sprintf((char *)xns[icombo+3*ncombo].tag, "%s:%s%s:array",  BACKGROUND_XML_CDF_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_CHISQ_SUFFIX);
    xns[icombo].processPtr = readArray;
    xns[icombo].data = &(array_cdf[icombo]);
  }

  parseFile(fname, xns, nnode);

  // FIXME: need sanity check that number of rows and columns are the same
  // with the struct of BackgroundStats
  /* load to stats */

  int x_nbin = stats[0]->pdf->x_nbin, y_nbin = stats[0]->pdf->y_nbin;
  int x_size = sizeof(double) * x_nbin, y_size = sizeof(double) * y_nbin;
  for (iicombo=0; icombo<ncombo; icombo++) {
    BackgroundStats *cur_stats = stats[icombo];
    BackgroundRates *rates = cur_stats->rates;
    memcpy(rates->logsnr_bins->data->data, array_logsnr_bins[icombo].data, x_size);
    memcpy(rates->logchisq_bins->data->data, array_logchisq_bins[icombo].data, y_size);
    memcpy(cur_stats->pdf->data->data, array_pdf[icombo].data, x_size * y_size);
    memcpy(cur_stats->cdf->data->data, array_cdf[icombo].data, x_size * y_size);
  }
  for (iicombo=0; icombo<ncombo; icombo++) {
    freeArray(array_logsnr_bins + icombo);
    freeArray(array_logchisq_bins + icombo);
    freeArray(array_pdf + icombo);
    freeArray(array_cdf + icombo);
  }

  free(xns);
  xmlCleanupParser();
  xmlMemoryDump();
}


gboolean
background_stats_to_xml(BackgroundStats **stats, const int ncombo, const char *filename)
{
  int icombo = 0;
  XmlArray *array_logsnr_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_logchisq_bins = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_pdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);
  XmlArray *array_cdf = (XmlArray *) malloc(sizeof(XmlArray) * ncombo);

  int x_nbin = stats[0]->pdf->x_nbin, y_nbin = stats[0]->pdf->y_nbin;
  int x_size = sizeof(double) * x_nbin, y_size = sizeof(double) * y_nbin;

  for (icombo=0; icombo<ncombo; icombo++) {
    BackgroundStats *cur_stats = stats[icombo];
    BackgroundRates *rates = cur_stats->rates;
    array_logsnr_bins[icombo].ndim = 1;
    array_logsnr_bins[icombo].dim[0] = x_nbin;
    array_logsnr_bins[icombo].data = (long *) malloc(x_size);
    memcpy(array_logsnr_bins[icombo].data, rates->logsnr_bins->data->data, x_size);
    array_logchisq_bins[icombo].ndim = 1;
    array_logchisq_bins[icombo].dim[0] = y_nbin;
    array_logchisq_bins[icombo].data = (long *) malloc(y_size);
    memcpy(array_logchisq_bins[icombo].data, rates->logchisq_bins->data->data, y_size);
    array_pdf[icombo].ndim = 2;
    array_pdf[icombo].dim[0] = x_nbin;
    array_pdf[icombo].dim[1] = y_nbin;
    array_pdf[icombo].data = (double *) malloc(x_size * y_size);
    memcpy(array_pdf[icombo].data, rates->pdf->data->data, x_size * y_size);
    array_cdf[icombo].ndim = 2;
    array_cdf[icombo].dim[0] = x_nbin;
    array_cdf[icombo].dim[1] = y_nbin;
    array_cdf[icombo].data = (double *) malloc(x_size * y_size);
    memcpy(array_cdf[icombo].data, rates->cdf->data->data, x_size * y_size);
  
  }


  int rc;
  xmlTextWriterPtr writer;

  /* Create a new XmlWriter for uri, with no compression. */
  writer = xmlNewTextWriterFilename(uri, 0);
  if (writer == NULL) {
      printf("testXmlwriterFilename: Error creating the xml writer\n");
      return;
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
      return;
  }

  rc = xmlTextWriterWriteDTD(writer, BAD_CAST "LIGO_LW", NULL, BAD_CAST "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt", NULL);
  if (rc < 0) {
      printf
          ("testXmlwriterFilename: Error at xmlTextWriterWriteDTD\n");
      return;
  }

  /* Start an element named "LIGO_LW". Since thist is the first
   * element, this will be the root element of the document. */
  rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
  if (rc < 0) {
      printf
          ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
      return;
  }

  /* Start an element named "LIGO_LW" as child of EXAMPLE. */
  rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
  if (rc < 0) {
      printf
          ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
      return;
  }

  /* Add an attribute with name "Name" and value "gstlal_spiir_cohfar" to LIGO_LW. */
  rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name",
                                   BAD_CAST "gstlal_spiir_cohfar");
  if (rc < 0) {
      printf
          ("testXmlwriterFilename: Error at xmlTextWriterWriteAttribute\n");
      return;
  }

  GString *array_name = g_string_new(NULL);
  for (icombo=0; icombo<ncombo; icombo++) {
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_SUFFIX);
    ligoxml_write_Array(writer, &(array_logsnr_bins[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_RATES_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_CHISQ_SUFFIX);
    ligoxml_write_Array(writer, &(array_logchisq_bins[icombo]), BAD_CAST "int_8s", BAD_CAST " ", BAD_CAST array_name->str);
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_PDF_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_CHISQ_SUFFIX);
    ligoxml_write_Array(writer, &(array_pdf[icombo]), BAD_CAST "real_8", BAD_CAST " ", BAD_CAST array_name->str);
    g_string_printf(array_name, "%s:%s%s:array",  BACKGROUND_XML_CDF_NAME, IFO_COMBO_MAP[icombo], BACKGROUND_XML_SNR_CHISQ_SUFFIX);
    ligoxml_write_Array(writer, &(array_cdf[icombo]), BAD_CAST "real_8", BAD_CAST " ", BAD_CAST array_name->str);
  }
  g_string_free(array_name);

  /* Since we do not want to
   * write any other elements, we simply call xmlTextWriterEndDocument,
   * which will do all the work. */
  rc = xmlTextWriterEndDocument(writer);
  if (rc < 0) {
      printf
          ("testXmlwriterFilename: Error at xmlTextWriterEndDocument\n");
      return;
  }

  xmlFreeTextWriter(writer);
  for (iicombo=0; icombo<ncombo; icombo++) {
    freeArray(array_logsnr_bins + icombo);
    freeArray(array_logchisq_bins + icombo);
    freeArray(array_pdf + icombo);
    freeArray(array_cdf + icombo);
  }
}

/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2009 Mireia Crispin Ortuzar <mcrispin@caltech.edu>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * SECTION:element-lal_autochisq
 *
 * FIXME:Describe lal_autochisq here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! lal_autochisq ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

/*
 * stuff from the C library
 */


#include <math.h>
#include <stdint.h>
#include <string.h>
#include <complex.h>


/*
 *  stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include "gstlal.h"
#include "gstlal_whiten.h"
#include "low_latency_inspiral_functions.h"
#include "gstlal_autochisq.h"

/*
 *  stuff from LAL
 */

#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/Date.h>
#include <lal/Sequence.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/LALNoiseModels.h>
#include <lal/Units.h>
#include <lal/LALComplex.h>
#include <lal/Window.h>
#include <lal/FindChirp.h>
#include <lal/AVFactories.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/FindChirpTD.h>
#include <lal/FindChirp.h>
#include <lal/LALError.h>
#include <lal/LALStdio.h>
#include <lal/TimeFreqFFT.h>
#include <lal/RealFFT.h>
#include <lal/LALInspiral.h>




/*
 *  * stuff from GSL
 *   */

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>



GST_DEBUG_CATEGORY_STATIC (gst_lalautochisq_debug);
#define GST_CAT_DEFAULT gst_lalautochisq_debug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum property
{
  PROP_0,
  PROP_SILENT,
  ARG_TEMPLATE_BANK,
  ARG_REFERENCE_PSD
};


/****** Parameters ********/

#define DEFAULT_T_START 0
#define DEFAULT_T_END 29 
#define DEFAULT_SNR_LENGTH 2048 /* samples */
#define TEMPLATE_SAMPLE_RATE 4096       /* Hertz */
#define TOLERANCE 0.99
#define TEMPLATE_DURATION 128 /*seconds*/

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("audio/x-raw-float, rate=(int)[1, MAX], channels=(int)[1, MAX], endianness=(int)1234, width=(int)64")
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("audio/x-raw-float, rate=(int)[1, MAX], channels=(int)[1, MAX], endianness=(int)1234, width=(int)64")
    );


GST_BOILERPLATE (Gstlalautochisq, gst_lalautochisq, GstBaseTransform,
    GST_TYPE_BASE_TRANSFORM);

static void gst_lalautochisq_set_property (GObject * object, enum property prop_id, 
            const GValue * value, GParamSpec * pspec);
static void gst_lalautochisq_get_property (GObject * object, enum property prop_id, 
            GValue * value, GParamSpec * pspec);

static gboolean get_unit_size (GstBaseTransform * trans, GstCaps * caps, guint * size);
static gboolean set_caps (GstBaseTransform * trans, GstCaps * incaps, GstCaps * outcaps);
//static GstCaps * transform_caps (GstBaseTransform * base_transform,
//    GstPadDirection direction, GstCaps *caps);
static gboolean start (GstBaseTransform *trans);
static GstFlowReturn transform (GstBaseTransform * trans, GstBuffer * inbuf, GstBuffer * outbuf);
static int generate_templates(Gstlalautochisq *element);

/* GObject vmethod implementations */

static void
gst_lalautochisq_base_init (gpointer gclass)
{
  fprintf(stderr, "open base init\n");
  GstElementClass *element_class = GST_ELEMENT_CLASS (gclass);
  GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS (gclass);

  gst_element_class_set_details_simple(element_class,
    "lalautochisq",
    "Filter/Audio",
    "Computes the chisquared time series from a filter's autocorrelation",
    "Mireia Crispin Ortuzar <mcrispin@caltech.edu>");

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_factory));

  transform_class->get_unit_size = get_unit_size;
  transform_class->set_caps = set_caps;
  //transform_class->transform_caps = transform_caps;
  transform_class->transform = transform;
  transform_class->start = start;
  fprintf(stderr, "close base init\n");
}

/* initialize the lal_autochisq's class */
static void
gst_lalautochisq_class_init (GstlalautochisqClass * klass)
{
  fprintf(stderr, "open class init\n");
  GObjectClass *gobject_class;
  GstBaseTransformClass *base_transform_class;

  gobject_class = (GObjectClass *) klass;
  base_transform_class = (GstBaseTransformClass *) klass;
  
  gobject_class->set_property = gst_lalautochisq_set_property;
  gobject_class->get_property = gst_lalautochisq_get_property;
  
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));
  
  g_object_class_install_property (gobject_class, ARG_TEMPLATE_BANK, g_param_spec_string("template-bank", "XML Template Bank", "Name of LIGO Light Weight XML file containing inspiral template bank", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, ARG_REFERENCE_PSD, g_param_spec_string("reference-psd", "Reference PSD", "Name of file from which to read a reference PSD", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  fprintf(stderr, "close class init\n");
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_lalautochisq_init (Gstlalautochisq * filter,
    GstlalautochisqClass * gclass)
{
  fprintf(stderr, "open init\n");
  filter->silent = FALSE;
  
  filter->reference_psd_filename = NULL;
  filter->template_bank_filename = NULL;
  filter->rate = 0;
  filter->channels = 0;
  filter->t_start =  0;
  filter->t_end = 29;
  filter->t_total_duration = 29;

  filter->A = NULL;
  fprintf(stderr, "close init\n");
}

static void
gst_lalautochisq_set_property (GObject * object, enum property prop_id,
    const GValue * value, GParamSpec * pspec)
{
  fprintf(stderr, "set property\n");
  Gstlalautochisq *filter = GST_LAL_AUTOCHISQ (object);
  fprintf(stderr, "inside set property\n");
  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case ARG_TEMPLATE_BANK:
      free(filter->template_bank_filename);
      filter->template_bank_filename = g_value_dup_string(value);
      break;
    case ARG_REFERENCE_PSD:
      free(filter->reference_psd_filename);
      filter->reference_psd_filename = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  fprintf(stderr, "close set property\n");
  }
}


static void
gst_lalautochisq_get_property (GObject * object, enum property prop_id,
    GValue * value, GParamSpec * pspec)
{
  fprintf(stderr, "get property\n");
  Gstlalautochisq *filter = GST_LAL_AUTOCHISQ (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case ARG_TEMPLATE_BANK:
      g_value_set_string(value, filter->template_bank_filename);
      break;
    case ARG_REFERENCE_PSD:
      g_value_set_string(value, filter->reference_psd_filename);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* GstElement vmethod implementations */

static int generate_bank(
		     gsl_matrix **U,
		     gsl_vector **chifacs,
		     gsl_matrix **A,
		     const char *xml_bank_filename,
		     const char *reference_psd_filename,
		     int base_sample_rate,
		     int down_samp_fac,
		     double t_start,
		     double t_end,
		     double t_total_duration,
		     double tolerance,
		     int verbose)
{

  fprintf(stderr, "entering generate_bank\n");

  InspiralTemplate *bankRef = NULL, *bankRow, *bankHead=NULL;
  
  fprintf(stderr, "before numtemps, address of the pointer %p\n", &bankHead);
  fprintf(stderr, "before numtemps, address of the xml file %p\n", &xml_bank_filename);
  
  int numtemps = InspiralTmpltBankFromLIGOLw( &bankHead, xml_bank_filename, -1, -1);
  fprintf(stderr, "after numtemps\n");
  double minChirpMass;

  fprintf(stderr, "declaring stuff \n");
  double jreference=0;
  double tshift=1;
  size_t j;
  size_t numsamps = round((t_end - t_start) * base_sample_rate / down_samp_fac);
  size_t autocorr_numsamps = 50;
  size_t full_numsamps = base_sample_rate*TEMPLATE_DURATION;
  COMPLEX16TimeSeries *template_out;
  COMPLEX16TimeSeries *convolution=NULL;
  COMPLEX16TimeSeries *autocorrelation=NULL;
  COMPLEX16TimeSeries *short_autocorr=NULL;
  COMPLEX16FrequencySeries *fft_template;
  COMPLEX16FrequencySeries *fft_template_full;
  COMPLEX16FrequencySeries *fft_template_full_reference=NULL;
  COMPLEX16FrequencySeries *somethingelse=NULL;
  COMPLEX16TimeSeries *template_reference=NULL;
  COMPLEX16FrequencySeries *template_product=NULL;

  REAL8FrequencySeries *psd;
  REAL8FFTPlan *fwdplan;
  COMPLEX16FFTPlan *revplan;

  fprintf(stderr, "I'M INSIDE THE GENERATE BANK FUNCTION!\n");

  if (numtemps <= 0) {
    fprintf(stderr, "FAILED generate_ban() numtemps <= 0\n");
    exit(1);
    }

  if (verbose) fprintf(stderr, "read %d template\n", numtemps);
  
  //fprintf(stderr, "t_end %e, t_start %e, numsamps %u\n", t_end, t_start, numsamps);

  /* There are twice as many waveforms as templates  */

  *U = gsl_matrix_calloc(numsamps, 2 * numtemps);
  *A = gsl_matrix_calloc(autocorr_numsamps, numtemps);
  *chifacs = gsl_vector_calloc(numtemps);
  
  g_mutex_lock(gstlal_fftw_lock);
  fwdplan = XLALCreateForwardREAL8FFTPlan(full_numsamps, 0);
  if (!fwdplan)
    {
    fprintf(stderr, "FAILED Generating the forward plan failed\n");
    exit(1);
    }
  revplan = XLALCreateReverseCOMPLEX16FFTPlan(full_numsamps, 0);
  if (!revplan)
    {
    fprintf(stderr, "FAILED Generating the reverse plan failed\n");
    exit(1);
    }
  g_mutex_unlock(gstlal_fftw_lock);

 /* Create workspace vectors */
  template_out = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  convolution = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  autocorrelation = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  short_autocorr = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, autocorr_numsamps);
  template_reference = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  fft_template = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 1.0 / TEMPLATE_DURATION, &lalDimensionlessUnit, full_numsamps / 2 + 1);
  fft_template_full = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 1.0 / TEMPLATE_DURATION, &lalDimensionlessUnit, full_numsamps);
  fft_template_full_reference = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 1.0 / TEMPLATE_DURATION, &lalDimensionlessUnit, full_numsamps);
  template_product = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 1.0 / TEMPLATE_DURATION, &lalDimensionlessUnit, full_numsamps);


  /* Get the reference psd */
  psd = gstlal_get_reference_psd(reference_psd_filename, template_out->f0, 1.0/TEMPLATE_DURATION, fft_template->data->length);
  
  if (!template_out || !convolution || !template_reference || !fft_template || !fft_template_full || !fft_template_full_reference || !template_product || !psd){
     fprintf(stderr, "FAILED Allocating template or reading psd failed\n");
     exit(1);
     }

  if (verbose) fprintf(stderr, "template_out->data->length %d fft_template->data->length %d fft_template_full->data->length %d \n",template_out->data->length, fft_template->data->length, fft_template_full->data->length);

  /* "fix" the templates in the bank */
  for(bankRow = bankHead; bankRow; bankRow = bankRow->next)
  {
  bankRow->fFinal = 0.95 * (base_sample_rate / 2.0 - 1); // 95% of Nyquest
  bankRow->fLower = 25.0;
  bankRow->tSampling = base_sample_rate;
  bankRow->fCutoff = bankRow->fFinal;
  bankRow->order = LAL_PNORDER_TWO;
  bankRow->signalAmplitude = 1.0;
  }

  REAL8 minMass, mineta;
  LALPNOrder minorder;

  if(bankHead) {
    minChirpMass = bankHead->chirpMass;
    minMass = bankHead->mass1+bankHead->mass2;
    mineta = bankHead->eta;
    minorder = bankHead->order;
    bankRef= bankHead;

    for(bankRow = bankHead, j=0; bankRow; bankRow = bankRow->next, j++)
      if(bankRow->chirpMass < minChirpMass)
        {
	minChirpMass = bankRow->chirpMass;
	minMass = bankRow->mass1+bankRow->mass2;
	mineta = bankRow->eta;
	minorder = bankRow->order;
	bankRef = bankRow;
	jreference = j;
	}
    }

  if(create_template_from_sngl_inspiral(bankRef, *U, *A, *chifacs, base_sample_rate, down_samp_fac, t_end, t_total_duration, autocorr_numsamps, 0, jreference, template_reference, convolution, autocorrelation, short_autocorr, template_product, fft_template, fft_template_full_reference, somethingelse, fwdplan, revplan, psd) < 0)
    {
    exit(1);
    }
  if(verbose)
    {
    fprintf(stderr, "Reference Template: M_chirp=%e\n", bankRef->chirpMass);
    }

  /* creates the templates */
  for (bankRow = bankHead, j=0; bankRow; bankRow = bankRow->next, j++)
  {
    if(j!=jreference)
    {
     if(create_template_from_sngl_inspiral(bankRef, *U, *A, *chifacs, base_sample_rate, down_samp_fac, t_end, t_total_duration, autocorr_numsamps, tshift, j, template_out, convolution, autocorrelation, short_autocorr, template_product, fft_template, fft_template_full, fft_template_full_reference, fwdplan, revplan, psd) < 0)
       {
       exit(1);
       }
     if(verbose)
       {
       fprintf(stderr, "template: M_chirp=%e\n", bankRow->chirpMass);
       }
     }
   }
  
  /* Destroy!!! */

  /* Destroy plans */
  g_mutex_lock(gstlal_fftw_lock);
  XLALDestroyREAL8FFTPlan(fwdplan);
  XLALDestroyCOMPLEX16FFTPlan(revplan);
  g_mutex_unlock(gstlal_fftw_lock);

  /* Destroy time/freq series */
  XLALDestroyCOMPLEX16FrequencySeries(fft_template);
  XLALDestroyCOMPLEX16FrequencySeries(fft_template_full);
  XLALDestroyCOMPLEX16FrequencySeries(fft_template_full_reference);
  XLALDestroyCOMPLEX16FrequencySeries(template_product);
  XLALDestroyCOMPLEX16TimeSeries(template_out);
  XLALDestroyCOMPLEX16TimeSeries(convolution);
  XLALDestroyCOMPLEX16TimeSeries(autocorrelation);
  XLALDestroyCOMPLEX16TimeSeries(short_autocorr);
  XLALDestroyCOMPLEX16TimeSeries(template_reference);

  /* free the template list */
  while(bankHead)
      {
      InspiralTemplate *next = bankHead->next;
      XLALFree(bankHead);
      bankHead = next;
      }

  return 0;
}


static int generate_templates(Gstlalautochisq *element)
{
  int verbose = 1;
  gsl_matrix *U;
  gsl_vector *chifacs;

  /*fprintf(stderr, "ADJUSTING TIME\n");
  if(element->t_start > element->t_total_duration)    // Adjusting the time
  	element->t_start = element->t_total_duration;
  if(element->t_end < element->t_start)
  	element->t_end = element->t_start;
  else if(element->t_end > element->t_total_duration)
  	element->t_end = element->t_total_duration;*/

  /*generate bank*/

  fprintf(stderr, "ABOUT TO GENERATE BANK\n");

  generate_bank(&U, &chifacs, &element->A, element->template_bank_filename, element->reference_psd_filename, TEMPLATE_SAMPLE_RATE, TEMPLATE_SAMPLE_RATE / element->rate, element->t_start, element->t_end, element->t_total_duration, TOLERANCE, verbose);

  gsl_matrix_free(U);
  gsl_vector_free(chifacs);

  fprintf(stderr, "BANK GENERATED!\n"); 
 /* FIXME: check for discontinuity? */

  return 0;

} 


/* return the number of samples in the autocorrelation functions */
static int autocorrelation_samples(const Gstlalautochisq *element)
{
  return element->A->size1;
}


/* get_unit_size()
 */

static gboolean
get_unit_size (GstBaseTransform * trans, GstCaps * caps, guint * size)
{
  fprintf(stderr, "unit size\n");
  GstStructure *str;
  gint channels;
 
  str = gst_caps_get_structure(caps, 0);
  if(!gst_structure_get_int(str, "channels", &channels)) {
      g_print("No channels available!!\n");
      return FALSE;
  }

  *size = sizeof(double) * channels;

  return TRUE;
}


/* set_caps function
 */
 
static gboolean
set_caps (GstBaseTransform * trans, GstCaps * incaps, GstCaps * outcaps)
{
  Gstlalautochisq *element = GST_LAL_AUTOCHISQ(trans);
  GstStructure *str;
  gint rate;
  gint channels;

  fprintf(stderr, "set caps\n"); 
  str = gst_caps_get_structure(incaps, 0);
  if(!gst_structure_get_int(str, "channels", &channels)) {
      g_print("No channels available!!\n");
      return FALSE;
  }
  if(!gst_structure_get_int(str, "rate", &rate)) {
      g_print("No channels available!!\n");
      return FALSE;
  }
 
  /* FIXME:  should check that channels/2 matches the number of templates */
  element->channels = channels / 2;
  element->rate = rate;

  return TRUE;
}

/* transform_caps function
 * */



/* Given the pad in this direction and the given caps, what caps are allowed on the other pad in this element? */


/*static GstCaps * transform_caps (GstBaseTransform * base_transform, GstPadDirection direction, GstCaps * caps)
{
  int i;
  GstStructure *structure;
  GValue new_value = { 0 };
  const GValue *value;
  
  caps = gst_caps_copy (caps);

  for (i=0; i<gst_caps_get_size(caps); i++) // We won't need the loop if caps are fixed caps
  {
  	structure = gst_caps_get_structure (caps, i);
	
	//Transform the width
	value = gst_structure_get_value (structure, "width");
	transform_value (&new_value, value, direction);
	gst_structure_set_value (structure, "width", &new_value);
	g_value_unset (&new_value);

	//Transform the height
	value = gst_structure_get_value (structure, "height");
	transofrm_value (&new_value, value, direction);
	gst_structure_set_value (structure, "height", &new_value);
	g_value_unset (&new_value);

  }

  return caps;
}*/

static gboolean start (GstBaseTransform *trans)
{
  fprintf(stderr, "start\n");
  Gstlalautochisq *element = GST_LAL_AUTOCHISQ(trans);
  element->adapter = gst_adapter_new(); // initialize the adapter?? 

  return TRUE;
}

/* chain function (transform)
 * this function does the actual processing
 */

static GstFlowReturn transform (GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
  complex double *indata;
  gint sample, channel;
  int insamples, outsamples;
  Gstlalautochisq *element = GST_LAL_AUTOCHISQ(trans);
 
  fprintf(stderr, "ABOUT TO START \n");

  /* Autocorrelation matrix
   * */

  if(!element->A) {
    GstBuffer *statebuf;

    generate_templates(element);
    fprintf(stderr, "AUTOCORRELATION MATRIX DONE!\n");

    statebuf = gst_buffer_new_and_alloc((autocorrelation_samples(element)-1)/2 * element->channels * sizeof(complex double));
    memset(GST_BUFFER_DATA(statebuf), 0, GST_BUFFER_SIZE(statebuf));
    gst_adapter_push(element->adapter, statebuf);
  }


  /* * 
   * Adapter + chi squared test
   * */

  gst_adapter_push(element->adapter, inbuf);
 
  fprintf(stderr, "number of channels %i\n", element->channels);
  /* Sizes of the buffers */
  insamples = gst_adapter_available(element->adapter) / (sizeof(complex double)*element->channels); // size of the incoming buffer
  outsamples = insamples - (autocorrelation_samples(element) - 1); // size of the outgoing buffer

  /* Checks if there are enough samples to do the chi-squared test */
  if(outsamples<=0)
  {
  	fprintf(stderr, "Could not calcualate chisq\n");
	GST_BUFFER_SIZE(outbuf)=0;
	return GST_FLOW_OK;
  }

  /* Takes the required number of samples out of the adapter */
  indata = (complex double *)gst_adapter_peek(element->adapter, insamples * element->channels * sizeof(*indata));
  
  /* Chi-squared test */
  for (sample=0; sample < outsamples; sample++)
    { 
    /* Pointers to the right sample of the buffers */
    complex double *insample = &indata[element->channels * sample];
    complex double *outsample = &((complex double *)GST_BUFFER_DATA(outbuf))[element->channels * sample];
    for (channel=0; channel < element->channels; channel++) 
       {
       complex double snr = insample[(autocorrelation_samples(element)-1)/2*element->channels + channel];
       double chisq = 0;
       double norm = 0;
       int i;

       for (i=0; i < autocorrelation_samples(element); i++)
	  {
          complex double snrprev = insample[i*element->channels + channel];
	  
	  /* Chi-Squared */
	  chisq += pow(gsl_matrix_get(element->A, i, channel)*snr-snrprev,2);
	  
	  /* Normalization */
	  norm += 1 - pow(gsl_matrix_get(element->A, i, channel), 2);
	  }
       
       /* Populates the outgoing buffer */
       outsample[channel]=chisq/norm;
       }
    }
  /* FIXME: need to set buffer metadata correctly */
  GST_BUFFER_SIZE(outbuf) = outsamples * element->channels * sizeof(complex double);
  GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + outsamples;
 
  return GST_FLOW_OK;
}




/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
lal_autochisq_init (GstPlugin * lal_autochisq)
{
   fprintf(stderr, "STARTING...\n");

   /* debug category for fltering log messages
   *
   * exchange the string 'Template lal_autochisq' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_lalautochisq_debug, "lal_autochisq",
      0, "Template lal_autochisq");


  return gst_element_register (lal_autochisq, "lal_autochisq", GST_RANK_NONE,
      GST_TYPE_LAL_AUTOCHISQ);
}


/* gstreamer looks for this structure to register lal_autochisqs
 *
 * exchange the string 'Template lal_autochisq' with your lal_autochisq description
 */
/*GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "lal_autochisq",
    "Template lal_autochisq",
    lal_autochisq_init,
    VERSION,
    "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
)*/

/*
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
  ARG_TEMPLATE_BANK = 1,
  ARG_REFERENCE_PSD
};


/****** Parameters ********/

#define DEFAULT_T_START 0
#define DEFAULT_T_END 29 
#define DEFAULT_SNR_LENGTH 2048 /* samples */
#define TEMPLATE_SAMPLE_RATE 4096       /* Hertz */
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
static gboolean stop (GstBaseTransform *trans);
static GstFlowReturn transform (GstBaseTransform * trans, GstBuffer * inbuf, GstBuffer * outbuf);
static int generate_templates(Gstlalautochisq *element);

/* GObject vmethod implementations */

static void
gst_lalautochisq_base_init (gpointer gclass)
{
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
  transform_class->stop = stop;
}

/* initialize the lal_autochisq's class */
static void
gst_lalautochisq_class_init (GstlalautochisqClass * klass)
{
  GObjectClass *gobject_class;
  GstBaseTransformClass *base_transform_class;

  gobject_class = (GObjectClass *) klass;
  base_transform_class = (GstBaseTransformClass *) klass;

  gobject_class->set_property = gst_lalautochisq_set_property;
  gobject_class->get_property = gst_lalautochisq_get_property;

  g_object_class_install_property (gobject_class, ARG_TEMPLATE_BANK, g_param_spec_string("template-bank", "XML Template Bank", "Name of LIGO Light Weight XML file containing inspiral template bank", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, ARG_REFERENCE_PSD, g_param_spec_string("reference-psd", "Reference PSD", "Name of file from which to read a reference PSD", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
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
  filter->reference_psd_filename = NULL;
  filter->template_bank_filename = NULL;
  filter->rate = 0;
  filter->channels = 0;
  filter->t_start =  0;
  filter->t_end = 29;
  filter->t_total_duration = 29;
  filter->adapter = NULL;
  filter->adapter_is_empty = TRUE;
  filter->A = NULL;
  filter->norm = NULL;
  filter->t0 = GST_CLOCK_TIME_NONE;
  filter->offset0 = GST_BUFFER_OFFSET_NONE;
  filter->next_in_offset = GST_BUFFER_OFFSET_NONE;
  filter->next_out_offset = GST_BUFFER_OFFSET_NONE;
  gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter),TRUE);
}

static void
gst_lalautochisq_set_property (GObject * object, enum property prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstlalautochisq *filter = GST_LAL_AUTOCHISQ (object);
  switch (prop_id) {
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
  }
}


static void
gst_lalautochisq_get_property (GObject * object, enum property prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstlalautochisq *filter = GST_LAL_AUTOCHISQ (object);

  switch (prop_id) {
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


/* return the number of samples in the autocorrelation functions */
static int autocorrelation_samples(const Gstlalautochisq *element)
{
  return element->A->size1;
}

static int generate_templates(Gstlalautochisq *element)
{
  int verbose = 1;
  gsl_matrix *U;
  gsl_vector *chifacs;
  int channel;

  generate_bank(&U, &chifacs, &element->A, element->template_bank_filename, element->reference_psd_filename, TEMPLATE_SAMPLE_RATE, TEMPLATE_SAMPLE_RATE / element->rate, element->t_start, element->t_end, element->t_total_duration, verbose);
  gsl_matrix_free(U);
  gsl_vector_free(chifacs);

  element->norm = gsl_vector_alloc(element->channels);
  for (channel=0; channel < element->channels; channel++) 
     {
     double norm = 0;
     int i;

     for (i=0; i < autocorrelation_samples(element); i++)
       norm += 1 - pow(gsl_matrix_get(element->A, i, channel), 2);

     gsl_vector_set(element->norm, channel, norm);
     }

  return 0;
} 


/* get_unit_size()
 */

static gboolean
get_unit_size (GstBaseTransform * trans, GstCaps * caps, guint * size)
{
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

static gboolean start (GstBaseTransform *trans)
{
  Gstlalautochisq *element = GST_LAL_AUTOCHISQ(trans);
  element->adapter = gst_adapter_new();
  return TRUE;
}

static gboolean stop (GstBaseTransform *trans)
{
  Gstlalautochisq *element = GST_LAL_AUTOCHISQ(trans);
  g_object_unref(element->adapter);
  element->adapter = NULL;

  if(element->A) {
    gsl_matrix_free(element->A);
    element->A = NULL;
  }
  if(element->norm) {
    gsl_vector_free(element->norm);
    element->norm = NULL;
  }
  return TRUE;
}

static int push_zeros(Gstlalautochisq *element, int samples)
{
  GstBuffer *zerobuf = gst_buffer_new_and_alloc(samples * element->channels * sizeof(complex double));
  if(!zerobuf)
    return -1;
  memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
  gst_adapter_push(element->adapter, zerobuf);
  return 0;
}

static void set_metadata (Gstlalautochisq *element, GstBuffer *buf, guint64 outsamples)
{
  GST_BUFFER_SIZE(buf) = outsamples * element->channels * sizeof(complex double);
  GST_BUFFER_OFFSET(buf) = element->next_out_offset;
  element->next_out_offset += outsamples;
  GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
  GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
  GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
}

static GstFlowReturn chisquared (GstBuffer *outbuf, Gstlalautochisq *element)
{
  /* Takes the required number of samples out of the adapter */
  int insamples = gst_adapter_available(element->adapter) / (sizeof(complex double)*element->channels);
  int outsamples = insamples - (autocorrelation_samples(element) - 1); // size of the outgoing buffer
  complex double *indata = (complex double *)gst_adapter_peek(element->adapter, insamples * element->channels * sizeof(*indata));
  int sample, channel; 

  if(outsamples <= 0)
    return GST_BASE_TRANSFORM_FLOW_DROPPED;

  /* Chi-squared test */
  for (sample=0; sample < outsamples; sample++)
    { 
    /* Pointers to the right sample of the buffers */
    complex double *insample = &indata[element->channels * sample];
    complex double *outsample = &((complex double *)GST_BUFFER_DATA(outbuf))[element->channels * sample];
    for (channel=0; channel < element->channels; channel++) 
       {
       complex double snr = insample[(autocorrelation_samples(element)-1)/2*element->channels + channel];
       complex double invsnrphase = cexp(-I*carg(snr));
       double chisq = 0;
       int i;

       for (i=0; i < autocorrelation_samples(element); i++)
	  {
          complex double snrprev = insample[i*element->channels + channel];

	  /* Chi-Squared */
	  chisq += pow(creal((gsl_matrix_get(element->A, i, channel)*snr-snrprev)*invsnrphase),2);
	  }

       /* Populates the outgoing buffer */
       outsample[channel]=chisq/gsl_vector_get(element->norm, channel);
       }
    }
  gst_adapter_flush(element->adapter, outsamples * element->channels * sizeof(*indata));
  set_metadata(element, outbuf, outsamples);

  return GST_FLOW_OK;
}

/*
 * set gaps
 * this function is called only if there is a gap
 */

static GstFlowReturn set_gaps (Gstlalautochisq *element, GstBuffer *outbuf, int length_gap_in)
{
  int length_gap_out = length_gap_in - (autocorrelation_samples(element) - 1);

  if(length_gap_out <= 0) {
    /* Push length_gap_in zeroes in the adapter */
    push_zeros(element, length_gap_in);

    /* Run chisquared test */
    return chisquared(outbuf, element);
  } else {
    GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(element));
    GstBuffer *buf;
    GstFlowReturn result;
    int available_samples;

    /* Push length_A-1 zeroes into adapter  */
    push_zeros(element, autocorrelation_samples(element) - 1);

    /* Run chisquared test to finish off adapter's contents and push
     * downstream */
    available_samples = gst_adapter_available(element->adapter) / (sizeof(complex double)*element->channels);
    result = gst_pad_alloc_buffer(srcpad, element->next_out_offset, (available_samples - (autocorrelation_samples(element) - 1)) * element->channels * sizeof(complex double), GST_PAD_CAPS(srcpad), &buf);
    if(result!=GST_FLOW_OK) {
      return result;
    }
    result = chisquared(buf, element);
    if(result!=GST_FLOW_OK) {
      return result;
    }
    element->adapter_is_empty = TRUE;
    result = gst_pad_push(srcpad, buf);
    if(result!=GST_FLOW_OK) {
      return result;
    }

    /* make outbuf a gap of length length_gap_out */
    GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
    GST_BUFFER_SIZE(outbuf) = length_gap_out * element->channels * sizeof(complex double);
    memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
    set_metadata(element, outbuf, length_gap_out);
  }

  return GST_FLOW_OK;
}

 /*
  * Chain (transform) function
  */

static GstFlowReturn transform (GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
  Gstlalautochisq *element = GST_LAL_AUTOCHISQ(trans);

  /*
   * Autocorrelation matrix
   */

  if(!element->A) {
    generate_templates(element);

    /* Pushes zeroes into the adapter */
    push_zeros(element, (autocorrelation_samples(element)-1)/2);
    element->adapter_is_empty = TRUE;
  }

  /* FIXME: check for discontinuity? */

  /*
   * timestamp and offset book-keeping.
   *
   * FIXME:  this should be done in an event handler
   */

  if(!GST_CLOCK_TIME_IS_VALID(element->t0)) {
    element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
    element->offset0 = GST_BUFFER_OFFSET(inbuf);
    element->next_in_offset = element->offset0;
    element->next_out_offset = element->offset0;
  }
  element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

  /*
   * gap logic
   */

  if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
    gst_buffer_ref(inbuf);	/* don't let the adapter free it */
    gst_adapter_push(element->adapter, inbuf);
    element->adapter_is_empty = FALSE;
    return chisquared(outbuf, element);
  } else if(element->adapter_is_empty) {
    /* base transform has given us an output buffer that is the same size
     * as the input buffer, which is the size we need now.  make it a gap
     * and send downstream. */
    GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
    memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
    set_metadata(element, outbuf, GST_BUFFER_SIZE(outbuf) / (sizeof(complex double)*element->channels));
  } else {
    return set_gaps (element, outbuf, GST_BUFFER_SIZE(inbuf) / (sizeof(complex double) * element->channels));
  }

  return GST_FLOW_OK;
}

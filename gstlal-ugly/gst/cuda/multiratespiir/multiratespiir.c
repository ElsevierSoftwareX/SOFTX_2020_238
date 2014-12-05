/* GStreamer
 * Copyright (C) Qi Chu,
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more deroll-offss.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * SECTION:element-MultirateSPIIR
 *
 * gst-launch -v 
 */

/* TODO:
 *  - no update of SpiirState at run time. should support streaming format 
 *  changes such as width/ rate/ quality change at run time. Should 
 *  support IIR bank changes at run time.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/base/gstadapter.h>
#include <gstlal/gstlal.h>


#include "multiratespiir.h"
#include "multiratespiir_utils.h"
#include "spiir_state_macro.h"
#include "spiir_state_utils.h"

#include <cuda_runtime.h>

#define GST_CAT_DEFAULT cuda_multirate_spiir_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cuda_multirate_spiir", 0, "cuda_multirate_spiir element");
}


GST_BOILERPLATE_FULL(
	CudaMultirateSPIIR,
	cuda_multirate_spiir,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM,
	additional_initializations
);

enum
{
  PROP_0,
  PROP_IIR_BANK,
  PROP_BANK_ID,
  PROP_GAP_HANDLE
};

//FIXME: not support width=64 yet
static GstStaticPadTemplate cuda_multirate_spiir_sink_template =
GST_STATIC_PAD_TEMPLATE (
		"sink",
		GST_PAD_SINK, 
		GST_PAD_ALWAYS, 
		GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 32"
		));

static GstStaticPadTemplate cuda_multirate_spiir_src_template =
GST_STATIC_PAD_TEMPLATE (
		"src",
		GST_PAD_SRC, 
		GST_PAD_ALWAYS, 
		GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 32"
		));

static void cuda_multirate_spiir_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void cuda_multirate_spiir_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */
static gboolean cuda_multirate_spiir_get_unit_size (GstBaseTransform * base,
    GstCaps * caps, guint * size);
static GstCaps *cuda_multirate_spiir_transform_caps (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps);
static gboolean cuda_multirate_spiir_set_caps (GstBaseTransform * base,
    GstCaps * incaps, GstCaps * outcaps);
static GstFlowReturn cuda_multirate_spiir_transform (GstBaseTransform * base,
    GstBuffer * inbuf, GstBuffer * outbuf);
static gboolean cuda_multirate_spiir_transform_size (GstBaseTransform * base,
   GstPadDirection direction, GstCaps * caps, guint size, GstCaps * othercaps,
    guint * othersize);
static gboolean cuda_multirate_spiir_event (GstBaseTransform * base,
    GstEvent * event);
static gboolean cuda_multirate_spiir_start (GstBaseTransform * base);
static gboolean cuda_multirate_spiir_stop (GstBaseTransform * base);
// FIXME: query
//static gboolean cuda_multirate_spiir_query (GstPad * pad, GstQuery * query);
//static const GstQueryType *cuda_multirate_spiir_query_type (GstPad * pad);


static void
cuda_multirate_spiir_base_init (gpointer g_class)
{
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (g_class);

  gst_element_class_set_details_simple (gstelement_class, "Multirate SPIIR",
      "multi level downsample + spiir + upsample", "single rate data stream -> multi template SNR streams",
      "Qi Chu <qi.chu@ligo.org>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&cuda_multirate_spiir_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&cuda_multirate_spiir_sink_template));

  GST_BASE_TRANSFORM_CLASS (g_class)->start =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_start);
  GST_BASE_TRANSFORM_CLASS (g_class)->stop =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_stop);
  GST_BASE_TRANSFORM_CLASS (g_class)->get_unit_size =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_get_unit_size);
  GST_BASE_TRANSFORM_CLASS (g_class)->transform_caps =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_transform_caps);
  GST_BASE_TRANSFORM_CLASS (g_class)->set_caps =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_set_caps);
  GST_BASE_TRANSFORM_CLASS (g_class)->transform =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_transform);
  GST_BASE_TRANSFORM_CLASS (g_class)->transform_size =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_transform_size);
  GST_BASE_TRANSFORM_CLASS (g_class)->event =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_event);

}

static void
cuda_multirate_spiir_class_init (CudaMultirateSPIIRClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;

  gobject_class->set_property = GST_DEBUG_FUNCPTR (cuda_multirate_spiir_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (cuda_multirate_spiir_get_property);

  g_object_class_install_property (gobject_class, PROP_IIR_BANK,
 			g_param_spec_value_array(
				"spiir-bank",
				"IIR bank feedback coefficients",
				"A parallel bank of first order IIR filter feedback coefficients.The format is : num_depths, num_templates in highest depth, num_filters in highest depth, a1 for the first template, a1 for the second template, ..., b0, ..., delay, ..., num_templates in second highest depth, ...",
				g_param_spec_double(
					"coefficient",
					"Coefficient",
					"Feedback coefficient",
					-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			)
		  );

  g_object_class_install_property (gobject_class, PROP_BANK_ID,
				g_param_spec_int(
					"bank-id",
					"Bank ID",
					"bank ID",
					0, 1000, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				)
			);

  g_object_class_install_property (gobject_class, PROP_GAP_HANDLE,
				g_param_spec_int(
					"gap-handle",
					"gap handling",
					"restart after gap (1), or gap is treated as 0 (0)",
					0, 1, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				)
			);

}

static void
cuda_multirate_spiir_init (CudaMultirateSPIIR * element,
    CudaMultirateSPIIRClass * klass)
{
//  GstBaseTransform *trans = GST_BASE_TRANSFORM (element);
  element->iir_bank_lock = g_mutex_new();
  element->iir_bank_available = g_cond_new();
  element->bank = NULL;
  element->bank_len = 0;
  element->num_depths = 0;
  element->outchannels = 0;


//  gst_base_transform_set_gap_aware (trans, TRUE);
//  gst_pad_set_query_function (trans->srcpad, cuda_multirate_spiir_query);
// gst_pad_set_query_type_function (trans->srcpad,
//      cuda_multirate_spiir_query_type);
}

/* vmethods */
static gboolean
cuda_multirate_spiir_start (GstBaseTransform * base)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR (base);

  element->adapter = gst_adapter_new();
  element->need_discont = TRUE;
  element->num_gap_samples = 0;
  element->need_tail_drain = FALSE;
  element->t0 = GST_CLOCK_TIME_NONE;
  element->offset0 = GST_BUFFER_OFFSET_NONE;
  element->next_in_offset = GST_BUFFER_OFFSET_NONE;
  element->samples_in = 0;
  element->samples_out = 0;
  element->spstate = NULL;
  element->spstate_initialised = FALSE;
  element->num_exe_samples = 4096; // assumes the rate=4096Hz
  element->num_head_cover_samples = 13120; // assumes the rate=4096Hz, down quality = 9
  element->num_tail_cover_samples = 13104; // assumes the rate=4096Hz

  return TRUE;
}

static gboolean
cuda_multirate_spiir_stop (GstBaseTransform * base)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR (base);

  g_mutex_free (element->iir_bank_lock);
  g_cond_free (element->iir_bank_available);

  if (element->spstate) {
    spiir_state_destroy (element->spstate, element->num_depths);
  }

  g_object_unref (element->adapter);
  element->adapter = NULL;

  return TRUE;
}

static gboolean
cuda_multirate_spiir_get_unit_size (GstBaseTransform * base, GstCaps * caps,
    guint * size)
{
  gint width, channels;
  GstStructure *structure;
  gboolean ret;

  g_return_val_if_fail (size != NULL, FALSE);

  /* this works for both float and int */
  structure = gst_caps_get_structure (caps, 0);
  ret = gst_structure_get_int (structure, "width", &width);
  ret &= gst_structure_get_int (structure, "channels", &channels);

  if (G_UNLIKELY (!ret))
    return FALSE;

  *size = (width / 8) * channels;
  GST_DEBUG_OBJECT (base, "get unit size of caps %d", *size);

  return TRUE;
}

static GstCaps *
cuda_multirate_spiir_transform_caps (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR(base);

  GstCaps *othercaps;

  othercaps = gst_caps_copy(caps);

  switch(direction) {
  case GST_PAD_SRC:
    /*
     * sink caps is the same with src caps, except it only has 1 channel
     */

    gst_structure_set(gst_caps_get_structure(othercaps, 0), "channels", G_TYPE_INT, 1, NULL);
    GST_LOG("setting channels to 1\n");
    break;

  case GST_PAD_SINK:
    /*
     * src caps is the same with sink caps, except it only has number of channels that equals to the number of templates
     */
    g_mutex_lock (element->iir_bank_lock);
    if (!element->bank) 	
      g_cond_wait (element->iir_bank_available, element->iir_bank_lock);

    gst_structure_set(gst_caps_get_structure(othercaps, 0), "channels", G_TYPE_INT, cuda_multirate_spiir_get_outchannels(element), NULL);
  
    g_mutex_unlock (element->iir_bank_lock);
    break;
	  
  case GST_PAD_UNKNOWN:
    GST_ELEMENT_ERROR(base, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
    gst_caps_unref(othercaps);
    return GST_CAPS_NONE;
  }

  return othercaps;
}


// Note: sizes calculated here are uplimit sizes, not necessarily the true sizes. 

static gboolean
cuda_multirate_spiir_transform_size (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps, guint size, GstCaps * othercaps,
    guint * othersize)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR(base);
  gboolean ret = TRUE;

  guint unit_size, other_unit_size;
  GST_LOG_OBJECT (base, "asked to transform size %d in direction %s",
      size, direction == GST_PAD_SINK ? "SINK" : "SRC");

  if (!cuda_multirate_spiir_get_unit_size (base, caps, &unit_size))
	  return FALSE;

  if (!cuda_multirate_spiir_get_unit_size (base, othercaps, &other_unit_size))
	  return FALSE;

    
  if (direction == GST_PAD_SINK) {
    /* 
     * asked to convert size of an incoming buffer. The output size 
     * is the uplimit size.
     */
//    g_assert(element->bank_initialised == TRUE);
    GST_LOG_OBJECT (base, "available samples  %d", cuda_multirate_spiir_get_available_samples (element));
    *othersize = (size / unit_size + cuda_multirate_spiir_get_available_samples (element)) * other_unit_size;
  } else {
    /* asked to convert size of an outgoing buffer. 
     */
//    g_assert(element->bank_initialised == TRUE);
    *othersize = (size / unit_size) * other_unit_size;
  }

  GST_LOG_OBJECT (base, "transformed size %d to %d", size,
      *othersize);

  return ret;
}

static gboolean
cuda_multirate_spiir_set_caps (GstBaseTransform * base, GstCaps * incaps,
    GstCaps * outcaps)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR (base);
  GstStructure *s;
  gint rate;
  gint channels;
  gint width;
  gboolean success = TRUE;

  GST_LOG ("incaps %" GST_PTR_FORMAT ", outcaps %"
      GST_PTR_FORMAT, incaps, outcaps);

  s = gst_caps_get_structure(outcaps, 0);
  success &= gst_structure_get_int(s, "channels", &channels);
  success &= gst_structure_get_int(s, "width", &width);
  success &= gst_structure_get_int(s, "rate", &rate);

  if (!success) 
    GST_ERROR_OBJECT(element, "unable to parse and/or accept caps %" GST_PTR_FORMAT, outcaps);
  if (element->bank!=NULL && (channels != (gint) cuda_multirate_spiir_get_outchannels(element)))
    /* impossible to happen */
    GST_ERROR_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, cuda_multirate_spiir_get_outchannels(element), outcaps);
  if (width != (gint) element->width) {
    if (element->spstate_initialised)
      /*
       * FIXME :do not support width change at run time
       */
      GST_ERROR_OBJECT(element, "width != %d in %" GST_PTR_FORMAT, element->width, outcaps);
    else
      element->width = width; }
  if (rate != (gint) element->rate) {
    if (element->spstate_initialised)
      /*
       * FIXME: do not support rate change at run time
       */
      GST_ERROR_OBJECT(element, "rate != %d in %" GST_PTR_FORMAT, element->rate, outcaps);
    else
      element->rate = rate; }

  /* transform_caps already done, num_depths already set */
  if (!element->spstate_initialised) {
    cuda_multirate_spiir_init_cover_samples(&element->num_head_cover_samples, &element->num_tail_cover_samples, rate, element->num_depths, DOWN_FILT_LEN*2, UP_FILT_LEN);
    cuda_multirate_spiir_update_exe_samples(&element->num_exe_samples, element->num_head_cover_samples);
    GST_DEBUG_OBJECT (element, "number of cover samples set to (%d, %d), number of exe samples set to %d", element->num_head_cover_samples, element->num_head_cover_samples, element->num_exe_samples);
  }
  return success;
}

/* c downsample2x */
#if 0
static void
downsample2x(ResamplerState *state, float *in, const gint num_inchunk, float *out, gint *out_processed)
{
  float *pos_mem;
  pos_mem = state->mem;
  gint filt_offs = state->filt_len - 1;
  gint j;
  for (j = 0; j < num_inchunk; ++j)
    pos_mem[j + filt_offs] = in[j];

  /*
   * FIXME: not filter yet
   */
  *out_processed = num_inchunk/2;
  for (j = 0; j < *out_processed; ++j)
    out[j] = in[j];
}
#endif

static GstFlowReturn
cuda_multirate_spiir_assemble_gap_buffer (CudaMultirateSPIIR *element, gint len, GstBuffer *gapbuf)
{
  gint outsize = len * element->outchannels * element->width/8;
  GST_BUFFER_SIZE (gapbuf) = outsize;

  /* time */
  if (GST_CLOCK_TIME_IS_VALID (element->t0)) {
    GST_BUFFER_TIMESTAMP (gapbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out, GST_SECOND,
        element->rate);
    GST_BUFFER_DURATION (gapbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out + len,
        GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP (gapbuf);
  } else {
    GST_BUFFER_TIMESTAMP (gapbuf) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_DURATION (gapbuf) = GST_CLOCK_TIME_NONE;
  }
  /* offset */
  if (element->offset0 != GST_BUFFER_OFFSET_NONE) {
    GST_BUFFER_OFFSET (gapbuf) = element->offset0 + element->samples_out;
    GST_BUFFER_OFFSET_END (gapbuf) = GST_BUFFER_OFFSET (gapbuf) + len;
  } else {
    GST_BUFFER_OFFSET (gapbuf) = GST_BUFFER_OFFSET_NONE;
    GST_BUFFER_OFFSET_END (gapbuf) = GST_BUFFER_OFFSET_NONE;
  }
 
  if (element->need_discont) {
    GST_BUFFER_FLAG_SET (gapbuf, GST_BUFFER_FLAG_DISCONT);
    element->need_discont = FALSE;
  }

  GST_BUFFER_FLAG_SET (gapbuf, GST_BUFFER_FLAG_GAP);
  
  /* move along */
  element->samples_out += len;
  element->samples_in += len;
      

  GST_LOG_OBJECT (element,
      "Assembled gap buffer of %u bytes with timestamp %" GST_TIME_FORMAT
      " duration %" GST_TIME_FORMAT " offset %" G_GUINT64_FORMAT " offset_end %"
      G_GUINT64_FORMAT, GST_BUFFER_SIZE (gapbuf),
      GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (gapbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (gapbuf)), GST_BUFFER_OFFSET (gapbuf),
      GST_BUFFER_OFFSET_END (gapbuf));

  if (outsize == 0) {
    GST_DEBUG_OBJECT (element, "buffer dropped");
    return GST_BASE_TRANSFORM_FLOW_DROPPED;
  }

  return GST_FLOW_OK;
}
			
static GstFlowReturn
cuda_multirate_spiir_push_gap (CudaMultirateSPIIR *element, gint gap_len)
{
  GstBuffer *gapbuf;
  guint outsize = gap_len * sizeof(float) * element->outchannels;
  GstFlowReturn res =
    gst_pad_alloc_buffer_and_set_caps (GST_BASE_TRANSFORM_SRC_PAD (element),
    GST_BUFFER_OFFSET_NONE, outsize,
    GST_PAD_CAPS (GST_BASE_TRANSFORM_SRC_PAD (element)), &gapbuf);

  // FIXME: no sanity check
  res = cuda_multirate_spiir_assemble_gap_buffer (element, gap_len, gapbuf);

  res = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (element), gapbuf);
 
  if (G_UNLIKELY (res != GST_FLOW_OK))
    GST_WARNING_OBJECT (element, "Failed to push gap: %s",
        gst_flow_get_name (res));
  return res;
}



static GstFlowReturn
cuda_multirate_spiir_push_drain (CudaMultirateSPIIR *element, gint in_len)
{
  gint num_in_multidown, num_out_multidown, 
       num_out_spiirup = 0, last_num_out_spiirup = 0, old_in_len = in_len;

  num_in_multidown = MIN (old_in_len, element->num_exe_samples);

  gint outsize = 0, out_len = 0, tmp_out_len = 0, upfilt_len;
  float * in_multidown, *tmp_out;
  tmp_out_len = element->spstate[0]->upstate->mem_len;
  upfilt_len = element->spstate[0]->upstate->filt_len;
  tmp_out = (float *)malloc(element->outchannels * tmp_out_len * sizeof(float));

  gint i, j;
  GstBuffer *outbuf;
  GstFlowReturn res;
  float *outdata;

  /* To restore the buffer timestamp, out length must be equal to in length */
  //out_len = spiir_state_get_outlen (element->spstate, in_len, element->num_depths);
  if (element->num_exe_samples == element->rate)
    out_len = in_len;
  else
    out_len = in_len - element->num_tail_cover_samples;


  outsize = out_len * sizeof(float) * element->outchannels;


  res =
    gst_pad_alloc_buffer_and_set_caps (GST_BASE_TRANSFORM_SRC_PAD (element),
    GST_BUFFER_OFFSET_NONE, outsize,
    GST_PAD_CAPS (GST_BASE_TRANSFORM_SRC_PAD (element)), &outbuf);

  memset(GST_BUFFER_DATA(outbuf), 0, outsize);

  if (G_UNLIKELY (res != GST_FLOW_OK)) {
    GST_WARNING_OBJECT (element, "failed allocating buffer of %d bytes",
        outsize);
    return res;
  }

  outdata = (float *) GST_BUFFER_DATA(outbuf);
   
  while (num_in_multidown > 0) {
    
    g_assert (gst_adapter_available (element->adapter) >= num_in_multidown * sizeof(float));
    in_multidown = (float *) gst_adapter_peek (element->adapter, num_in_multidown * sizeof(float));

    num_out_multidown = multi_downsample (element->spstate, in_multidown, (gint) num_in_multidown, element->num_depths, element->stream);
    num_out_spiirup = spiirup (element->spstate, num_out_multidown, element->num_depths, tmp_out, element->stream);
    cudaStreamSynchronize(element->stream);


    for (i=0; i<num_out_spiirup; i++)
      for (j=0; j<element->outchannels; j++)
	      outdata[element->outchannels * (i + last_num_out_spiirup) + j] = tmp_out[tmp_out_len * j + i + upfilt_len - 1];

 
     /* move along */
    gst_adapter_flush (element->adapter, num_in_multidown * sizeof(float));
    in_len -= num_in_multidown;
    /* after the first filtering, update the exe_samples to the rate */
    cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->rate);
    num_in_multidown = MIN (in_len, element->num_exe_samples);
    last_num_out_spiirup += num_out_spiirup;
 }

    g_assert(last_num_out_spiirup <= out_len);
    free(tmp_out);

    /* time */
    if (GST_CLOCK_TIME_IS_VALID (element->t0)) {
      GST_BUFFER_TIMESTAMP (outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out, GST_SECOND,
        element->rate);
      GST_BUFFER_DURATION (outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out + out_len,
        GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP (outbuf);
    } else {
      GST_BUFFER_TIMESTAMP (outbuf) = GST_CLOCK_TIME_NONE;
      GST_BUFFER_DURATION (outbuf) = GST_CLOCK_TIME_NONE;
    }
    /* offset */
    if (element->offset0 != GST_BUFFER_OFFSET_NONE) {
      GST_BUFFER_OFFSET (outbuf) = element->offset0 + element->samples_out;
      GST_BUFFER_OFFSET_END (outbuf) = GST_BUFFER_OFFSET (outbuf) + out_len;
    } else {
    GST_BUFFER_OFFSET (outbuf) = GST_BUFFER_OFFSET_NONE;
    GST_BUFFER_OFFSET_END (outbuf) = GST_BUFFER_OFFSET_NONE;
    }
 
    if (element->need_discont) {
      GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_DISCONT);
      element->need_discont = FALSE;
    }

    element->samples_out += out_len;
    element->samples_in += old_in_len;


    GST_BUFFER_SIZE (outbuf) =
      outsize;

    GST_LOG_OBJECT (element,
      "Push_drain: Converted to buffer of %" G_GUINT32_FORMAT
      " samples (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
      GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
      G_GUINT64_FORMAT, out_len, GST_BUFFER_SIZE (outbuf),
      GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (outbuf)),
      GST_BUFFER_OFFSET (outbuf), GST_BUFFER_OFFSET_END (outbuf));

    if (outsize == 0) {
      GST_DEBUG_OBJECT (element, "buffer dropped");
      gst_object_unref(outbuf);
      return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

  res = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (element), outbuf);

  if (G_UNLIKELY (res != GST_FLOW_OK))
    GST_WARNING_OBJECT (element, "Failed to push drain: %s",
        gst_flow_get_name (res));
  return res;


    return GST_FLOW_OK;

}



static gint 
cuda_multirate_spiir_process (CudaMultirateSPIIR *element, gint in_len, GstBuffer *outbuf)
{
  gint num_exe_samples, num_in_multidown, num_out_multidown, 
       num_out_spiirup, last_num_out_spiirup = 0, old_in_len = in_len;

  num_exe_samples = element->num_exe_samples;
  num_in_multidown = MIN (old_in_len, num_exe_samples);

  gint outsize = 0, out_len = 0, tmp_out_len = 0, upfilt_len;
  float * in_multidown, *tmp_out;
  upfilt_len = element->spstate[0]->upstate->filt_len;
  tmp_out_len = element->spstate[0]->upstate->mem_len;
  tmp_out = (float *)malloc(element->outchannels * tmp_out_len * sizeof(float));

  gint i, j;
  float *outdata;

  if (element->num_exe_samples == element->rate)
    out_len = in_len;
  else
    out_len = in_len - element->num_tail_cover_samples;

  outsize = out_len * sizeof(float) * element->outchannels;

  GST_DEBUG_OBJECT (element, "tmp_out_len %d, out len predicted %d", tmp_out_len, out_len);
  outdata = (float *) GST_BUFFER_DATA(outbuf);
   
  while (num_in_multidown > 0) {
    
    g_assert (gst_adapter_available (element->adapter) >= num_in_multidown * sizeof(float));
    in_multidown = (float *) gst_adapter_peek (element->adapter, num_in_multidown * sizeof(float));

    num_out_multidown = multi_downsample (element->spstate, in_multidown, (gint) num_in_multidown, element->num_depths, element->stream);
    num_out_spiirup = spiirup (element->spstate, num_out_multidown, element->num_depths, tmp_out, element->stream);


    /* reshape to the outbuf data */
    for (i=0; i<num_out_spiirup; i++)
      for (j=0; j<element->outchannels; j++)
	      outdata[element->outchannels * (i + last_num_out_spiirup) + j] = tmp_out[tmp_out_len * j + i + upfilt_len - 1];

    GST_DEBUG_OBJECT (element, "done cpy data to BUFFER");
 
   /* move along */
    gst_adapter_flush (element->adapter, num_in_multidown * sizeof(float));
    in_len -= num_in_multidown;
    num_in_multidown = MIN (in_len, num_exe_samples);
    last_num_out_spiirup += num_out_spiirup;
 }

    g_assert(last_num_out_spiirup == out_len);
    free(tmp_out);


    /* time */
    if (GST_CLOCK_TIME_IS_VALID (element->t0)) {
      GST_BUFFER_TIMESTAMP (outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out, GST_SECOND,
        element->rate);
      GST_BUFFER_DURATION (outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out + out_len,
        GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP (outbuf);
    } else {
      GST_BUFFER_TIMESTAMP (outbuf) = GST_CLOCK_TIME_NONE;
      GST_BUFFER_DURATION (outbuf) = GST_CLOCK_TIME_NONE;
    }
    /* offset */
    if (element->offset0 != GST_BUFFER_OFFSET_NONE) {
      GST_BUFFER_OFFSET (outbuf) = element->offset0 + element->samples_out;
      GST_BUFFER_OFFSET_END (outbuf) = GST_BUFFER_OFFSET (outbuf) + out_len;
    } else {
    GST_BUFFER_OFFSET (outbuf) = GST_BUFFER_OFFSET_NONE;
    GST_BUFFER_OFFSET_END (outbuf) = GST_BUFFER_OFFSET_NONE;
    }
 
    if (element->need_discont) {
      GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_DISCONT);
      element->need_discont = FALSE;
    }

    element->samples_out += out_len; 
    element->samples_in += old_in_len;


    GST_BUFFER_SIZE (outbuf) =
      outsize;

    GST_LOG_OBJECT (element,
      "Converted to buffer of %" G_GUINT32_FORMAT
      " samples (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
      GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
      G_GUINT64_FORMAT, out_len, GST_BUFFER_SIZE (outbuf),
      GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (outbuf)),
      GST_BUFFER_OFFSET (outbuf), GST_BUFFER_OFFSET_END (outbuf));

    if (outsize == 0) {
      GST_DEBUG_OBJECT (element, "buffer dropped");
      return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    /* after the first filtering, update the exe_samples to the rate */
    cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->rate);



    return GST_FLOW_OK;
}


/*
 * construct a buffer of zeros and push into adapter
 */


static void adapter_push_zeros(CudaMultirateSPIIR *element, unsigned samples)
{
	GstBuffer *zerobuf = gst_buffer_new_and_alloc(samples * (element->width / 8)); 
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
	}
	memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
	gst_adapter_push(element->adapter, zerobuf);
}



static GstFlowReturn
cuda_multirate_spiir_transform (GstBaseTransform * base, GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  /*
   * output buffer is generated in cuda_multirate_spiir_process function.
   */

  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR (base);
  GstFlowReturn res;

  gulong size;
  size = GST_BUFFER_SIZE (inbuf);

  GST_LOG_OBJECT (element, "transforming %s+%s buffer of %ld bytes, ts %"
      GST_TIME_FORMAT ", duration %" GST_TIME_FORMAT ", offset %"
      G_GINT64_FORMAT ", offset_end %" G_GINT64_FORMAT,
      GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "GAP" : "NONGAP",
      GST_BUFFER_IS_DISCONT(inbuf) ? "DISCONT" : "CONT",
      size, GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (inbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (inbuf)),
      GST_BUFFER_OFFSET (inbuf), GST_BUFFER_OFFSET_END (inbuf));

  /*
   * initialise spiir state
   */


  if (!element->spstate_initialised) {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    element->deviceID = (element->bank_id + 1) % deviceCount ;
    cudaSetDevice(element->deviceID);
    cudaStreamCreate(&element->stream);

    element->spstate = spiir_state_create (element->bank, element->bank_len,
		    element->num_head_cover_samples, element->num_exe_samples,
		    element->width, element->rate, element->stream);

    if (!element->spstate) {
      GST_ERROR_OBJECT(element, "spsate could not be initialised");
      return GST_FLOW_ERROR;
    }

    element->spstate_initialised = TRUE;
  }

  /* check for timestamp discontinuities;  reset if needed, and set
   * flag to resync timestamp and offset counters and send event
   * downstream */

  if (G_UNLIKELY (GST_BUFFER_IS_DISCONT (inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
    GST_DEBUG_OBJECT (element, "reset spstate");
    spiir_state_reset (element->spstate, element->num_depths, element->stream);
    gst_adapter_clear (element->adapter);
    element->need_discont = TRUE;

    /*
     * (re)sync timestamp and offset book-keeping. Set t0 and offset0 to be the timestamp and offset of the inbuf.
     */

    element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
    element->offset0 = GST_BUFFER_OFFSET(inbuf);
    element->num_gap_samples = 0;
    element->need_tail_drain = FALSE;
    element->samples_in = 0;
    element->samples_out = 0;
    cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->num_head_cover_samples);
    
  }

  element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

  /* 0-length buffers are produced to inform downstreams for current timestamp  */
  if (size == 0) {
	 GST_BUFFER_TIMESTAMP (outbuf) = GST_BUFFER_TIMESTAMP (inbuf);
	 GST_BUFFER_OFFSET (outbuf) = GST_BUFFER_OFFSET (inbuf);
	 GST_BUFFER_OFFSET_END (outbuf) = GST_BUFFER_OFFSET (inbuf);
	 GST_BUFFER_DURATION (outbuf) = 0;
	 GST_BUFFER_SIZE (outbuf) = GST_BUFFER_SIZE (inbuf); 
	 return GST_FLOW_OK;
  }

  gint in_samples, num_exe_samples, num_head_cover_samples, num_tail_cover_samples;
  in_samples = GST_BUFFER_SIZE (inbuf) / (element->width / 8);
  num_exe_samples = element->num_exe_samples;
  num_head_cover_samples = element->num_head_cover_samples;
  num_tail_cover_samples = element->num_tail_cover_samples;
  guint64 history_gap_samples, gap_buffer_len;
  gint num_zeros, adapter_len, num_filt_samples;


  switch (element->gap_handle) {

    case 1: // restart after gap
  

  /* 
   * gap handling cuda_multirate_spiir_get_available_samples (element)
   */

  if (GST_BUFFER_FLAG_IS_SET (inbuf, GST_BUFFER_FLAG_GAP)) {
    history_gap_samples = element->num_gap_samples;
    element->num_gap_samples += in_samples;

    /*
     * if receiving GAPs from the beginning, assemble same length GAPs
     */
    if (!element->need_tail_drain) {

        /*
         * one gap buffer
         */
        gap_buffer_len = in_samples;
	res = cuda_multirate_spiir_assemble_gap_buffer (element, gap_buffer_len, outbuf);
 
	if (res != GST_FLOW_OK)
          return res;
	else 
	  return GST_FLOW_OK;
    }

    /*
     * history is already cover the roll-offs, 
     * produce the gap buffer
     */
    if (history_gap_samples >= (guint64) num_tail_cover_samples) {
        /* 
	 * no process, gap buffer in place
	 */
	gap_buffer_len = in_samples;
	res = cuda_multirate_spiir_assemble_gap_buffer (element, gap_buffer_len, outbuf);

	if (res != GST_FLOW_OK)
          return res;
    }

    /*
     * if receiving GAPs from some time later :
     * history number of gaps is not enough to cover the 
     * total roll-offs of all the resamplers, check if current
     * number of gap samples will cover the roll-offs
     */
    if (history_gap_samples < (guint64) num_tail_cover_samples) {
      /* 
       * if current number of gap samples more than we can 
       * cover the roll-offs offset, process the buffer; 
       * otherwise absorb the inbuf
       */
      if (element->num_gap_samples >= (guint64) num_tail_cover_samples) {
        /*
         * one buffer to cover the roll-offs
         */
        num_zeros = num_tail_cover_samples - history_gap_samples;
        adapter_push_zeros (element, num_zeros);
	adapter_len = cuda_multirate_spiir_get_available_samples(element);
        res = cuda_multirate_spiir_push_drain (element, adapter_len);
   	if (res != GST_FLOW_OK)
          return res;

        /*
         * one gap buffer
         */
        gap_buffer_len = in_samples - num_zeros;
	res = cuda_multirate_spiir_assemble_gap_buffer (element, gap_buffer_len, outbuf);
	if (res != GST_FLOW_OK)
          return res;

      } else {
        /* 
	 * if could not cover the roll-offs,
	 * absorb the buffer 
	 */ 
        num_zeros = in_samples;
        adapter_push_zeros (element, num_zeros);
        GST_INFO_OBJECT(element, "inbuf absorbed %d zero samples", num_zeros);
        return GST_BASE_TRANSFORM_FLOW_DROPPED;
      }
    } 
  }


  /* 
   * inbuf is not gap 
   */

  if (!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
    /*
     * history is gap, and gap samples has already cover the roll-offs,
     * reset spiir state
     * if history gap is smaller than a tail cover, continue processing.
     */
    if (element->num_gap_samples >= (guint64) num_tail_cover_samples) {
      if (element->need_tail_drain) {
        adapter_len = cuda_multirate_spiir_get_available_samples(element);
        cuda_multirate_spiir_push_gap(element, element->num_tail_cover_samples + adapter_len);
        gst_adapter_clear (element->adapter);

      }
      spiir_state_reset (element->spstate, element->num_depths, element->stream);
      cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->num_head_cover_samples);
      num_exe_samples = element->num_exe_samples;
    }

    element->num_gap_samples = 0;
    element->need_tail_drain = TRUE;
    adapter_len = cuda_multirate_spiir_get_available_samples(element);
    /*
     * here merely speed consideration: if samples ready to be processed are less than num_exe_samples,
     * wait until there are over num_exe_samples
     */
    if (in_samples < num_exe_samples - adapter_len) {
      /* absorb the buffer */
      gst_buffer_ref(inbuf);	/* don't let the adapter free it */
      gst_adapter_push (element->adapter, inbuf);
      GST_INFO_OBJECT(element, "inbuf absorbed %d samples", in_samples);
      return GST_BASE_TRANSFORM_FLOW_DROPPED;

    } else {
      /*
       * filter
       */
      gst_buffer_ref(inbuf);	/* don't let the adapter free it */
      gst_adapter_push (element->adapter, inbuf);
      /* 
       * to speed up, number of samples to be filtered is times of num_exe_samples
       */
      adapter_len = cuda_multirate_spiir_get_available_samples(element);
      if (element->num_exe_samples == element->rate)
        num_filt_samples = gst_util_uint64_scale_int (adapter_len, 1, num_exe_samples) * num_exe_samples;
      else
        num_filt_samples = num_exe_samples;
      res = cuda_multirate_spiir_process(element, num_filt_samples, outbuf);
      if (res != GST_FLOW_OK)
        return res;
    }
  }
  break;

  case 0: // gap is treated as 0; 

    if (GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) 
        adapter_push_zeros (element, in_samples);
    else {
      gst_buffer_ref(inbuf);	/* don't let the adapter free it */
      gst_adapter_push (element->adapter, inbuf);
    }
     /* 
     * to speed up, number of samples to be filtered is times of num_exe_samples
     */
    adapter_len = cuda_multirate_spiir_get_available_samples(element);
    if (adapter_len >= element->num_exe_samples) {
      res = cuda_multirate_spiir_process(element, element->num_exe_samples, outbuf);
      if (res != GST_FLOW_OK)
        return res;
    } else {
        GST_INFO_OBJECT(element, "inbuf absorbed %d zero samples", num_zeros);
        return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }
    break;

  default:

    GST_ERROR_OBJECT(element, "gap handling not supported");
    break;
  }

  return GST_FLOW_OK;

}


static gboolean
cuda_multirate_spiir_event (GstBaseTransform * base, GstEvent * event)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR (base);

  switch (GST_EVENT_TYPE (event)) {
#if 0
    case GST_EVENT_FLUSH_STOP:
      cuda_multirate_spiir_reset_spstate (element);
      if (element->state)
        element->funcs->skip_zeros (element->state);
      element->num_gap_samples = 0;
      element->need_tail_drain = FALSE;
      element->t0 = GST_CLOCK_TIME_NONE;
      element->in_offset0 = GST_BUFFER_OFFSET_NONE;
      element->out_offset0 = GST_BUFFER_OFFSET_NONE;
      element->samples_in = 0;
      element->samples_out = 0;
      element->need_discont = TRUE;
      break;

#endif
    case GST_EVENT_NEWSEGMENT:
      
      GST_DEBUG_OBJECT(element, "EVENT NEWSEGMENT");
      if (element->need_tail_drain) {
        GST_DEBUG_OBJECT(element, "NEWSEGMENT, clear tails.");
	if (element->num_gap_samples >= element->num_tail_cover_samples) {
		cuda_multirate_spiir_push_gap(element, element->num_tail_cover_samples);
	} else {

        adapter_push_zeros (element, element->num_tail_cover_samples);
	int adapter_len = cuda_multirate_spiir_get_available_samples(element);
        cuda_multirate_spiir_push_drain (element, adapter_len);
	}

        spiir_state_reset (element->spstate, element->num_depths, element->stream);
      }
      element->num_gap_samples = 0;
      element->need_tail_drain = FALSE;
      element->t0 = GST_CLOCK_TIME_NONE;
      element->offset0 = GST_BUFFER_OFFSET_NONE;
      element->next_in_offset = GST_BUFFER_OFFSET_NONE;
      element->samples_in = 0;
      element->samples_out = 0;
      element->need_discont = TRUE;
      cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->num_head_cover_samples);

      break;

    case GST_EVENT_EOS:
 
      GST_DEBUG_OBJECT(element, "EVENT EOS");
      if (element->need_tail_drain) {
	if (element->num_gap_samples >= element->num_tail_cover_samples) {
          GST_DEBUG_OBJECT(element, "EOS, clear tails by pushing gap, num gap samples %" G_GUINT64_FORMAT, element->num_gap_samples);
  	  cuda_multirate_spiir_push_gap(element, element->num_tail_cover_samples);
	} else {

          GST_DEBUG_OBJECT(element, "EOS, clear tails by pushing drain");
          adapter_push_zeros (element, element->num_tail_cover_samples);
 	  int adapter_len = cuda_multirate_spiir_get_available_samples(element);
          cuda_multirate_spiir_push_drain (element, adapter_len);
	}

        spiir_state_reset (element->spstate, element->num_depths, element->stream);
      }

      break;
    default:
      break;
  }

  return parent_class->event (base, event);
}


static void
cuda_multirate_spiir_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
    {
  CudaMultirateSPIIR *element;

  element = CUDA_MULTIRATE_SPIIR (object);
  gboolean success = TRUE;

  GST_OBJECT_LOCK (element);
  switch (prop_id) {

    case PROP_IIR_BANK:
      g_mutex_lock(element->iir_bank_lock);
      if (element->bank)
	      free(element->bank);

      GST_LOG_OBJECT (element, "obtaining bank");
      element->bank = gstlal_doubles_from_g_value_array (g_value_get_boxed(value), element->bank, &element->bank_len);
      if (!element->bank)
         GST_ERROR_OBJECT (element, "bank could not be initialized");

      success = cuda_multirate_spiir_parse_bank(element->bank, &element->num_depths, &element->outchannels);
      if (!success)
         GST_ERROR_OBJECT (element, "bank could not be parsed");
      /*
       * signal ready of the bank
       */
      g_cond_broadcast(element->iir_bank_available);
      GST_DEBUG_OBJECT (element, "spiir bank available, bank length %d, number of depths %d, outchannels %d", element->bank_len, element->num_depths, element->outchannels);

      g_mutex_unlock(element->iir_bank_lock);
      break;

    case PROP_BANK_ID:
      element->bank_id = g_value_get_int(value);
      break;

    case PROP_GAP_HANDLE:
      element->gap_handle = g_value_get_int(value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (element);
}

static void
cuda_multirate_spiir_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  CudaMultirateSPIIR *element;

  element = CUDA_MULTIRATE_SPIIR (object);
  GST_OBJECT_LOCK (element);

  switch (prop_id) {
    case PROP_IIR_BANK:
      g_mutex_lock(element->iir_bank_lock);
      if (element->bank)
	      g_value_take_boxed(value, gstlal_g_value_array_from_doubles(element->bank, element->bank_len));
      g_mutex_unlock(element->iir_bank_lock);
      break;
    case PROP_BANK_ID:
      g_value_set_int (value, element->bank_id);
      break;

    case PROP_GAP_HANDLE:
      g_value_set_int (value, element->gap_handle);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  GST_OBJECT_UNLOCK (element);
}


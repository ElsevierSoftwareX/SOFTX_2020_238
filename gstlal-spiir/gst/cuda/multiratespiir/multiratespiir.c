/* 
 * Copyright (C) 2014 Qi Chu
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
#include <gstlal/gstaudioadapter.h>
#include <gstlal/gstlal.h>

#include <multiratespiir/multiratespiir.h>
#include <multiratespiir/multiratespiir_utils.h>
#include <multiratespiir/multiratespiir_kernel.h>

#include <cuda_runtime.h>
#include <cuda_debug.h>

#define GST_CAT_DEFAULT cuda_multirate_spiir_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

#define ACCELERATE_MULTIRATE_SPIIR_MEMORY_COPY

/* Obsolete in gstreamer-1.0 */
#if 0
static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cuda_multirate_spiir", 0, "cuda_multirate_spiir element");
}
#endif

G_DEFINE_TYPE_WITH_CODE(
    CudaMultirateSPIIR,
    cuda_multirate_spiir,
    GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cuda_multirate_spiir", 0, "cuda_multirate_spiir element")
    );

enum
{
  PROP_0,
  PROP_IIRBANK_FNAME,
  PROP_GAP_HANDLE,
  PROP_STREAM_ID
};

//FIXME: not support width=64 yet
static GstStaticPadTemplate cuda_multirate_spiir_sink_template =
GST_STATIC_PAD_TEMPLATE (
		"sink",
		GST_PAD_SINK, 
		GST_PAD_ALWAYS, 
		GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) 1, " \
		"format = (string) {" GST_AUDIO_NE(F32) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
		)
);

static GstStaticPadTemplate cuda_multirate_spiir_src_template =
GST_STATIC_PAD_TEMPLATE (
		"src",
		GST_PAD_SRC, 
		GST_PAD_ALWAYS, 
		GST_STATIC_CAPS(
		  GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(F32) "}") ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"

		)
);

static void cuda_multirate_spiir_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void cuda_multirate_spiir_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */
static gboolean cuda_multirate_spiir_get_unit_size (GstBaseTransform * base,
    GstCaps * caps, gsize *size);
static GstCaps *cuda_multirate_spiir_transform_caps (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps);
static gboolean cuda_multirate_spiir_set_caps (GstBaseTransform * base,
    GstCaps * incaps, GstCaps * outcaps);
static GstFlowReturn cuda_multirate_spiir_transform (GstBaseTransform * base,
    GstBuffer * inbuf, GstBuffer * outbuf);
static gboolean cuda_multirate_spiir_transform_size (GstBaseTransform * base,
   GstPadDirection direction, GstCaps * caps, gsize size, GstCaps * othercaps,
    gsize * othersize);
static gboolean cuda_multirate_spiir_event (GstBaseTransform * base,
    GstEvent * event);
static gboolean cuda_multirate_spiir_start (GstBaseTransform * base);
static gboolean cuda_multirate_spiir_stop (GstBaseTransform * base);
// FIXME: query
//static gboolean cuda_multirate_spiir_query (GstPad * pad, GstQuery * query);
//static const GstQueryType *cuda_multirate_spiir_query_type (GstPad * pad);


/*
 * Obsolete, moved to class init
 */

#if 0
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
#endif


/*
 * the number of samples available in the adapter
 */
int DEBUG_LINE = __LINE__;
static gsize get_available_samples(CudaMultirateSPIIR *element)
{
	gsize size;

	g_object_get(element->adapter, "size", &size, NULL);
	GST_DEBUG("Debug multiratespiir original get_available_samples %"G_GSIZE_FORMAT " from line %d", size, DEBUG_LINE);
	//GST_DEBUG("Debug multiratespiir original long d %ld", (long) size);
	GST_DEBUG("Debug multiratespiir original u %u", size);

	return size;
}


static void
cuda_multirate_spiir_class_init (CudaMultirateSPIIRClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;

  gobject_class->set_property = GST_DEBUG_FUNCPTR (cuda_multirate_spiir_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (cuda_multirate_spiir_get_property);

  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);

  gst_element_class_set_details_simple (gstelement_class, "Multirate SPIIR",
      "multi level downsample + spiir + upsample", "single rate data stream -> multi template SNR streams",
      "Qi Chu <qi.chu@ligo.org>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&cuda_multirate_spiir_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&cuda_multirate_spiir_sink_template));

  GST_BASE_TRANSFORM_CLASS (klass)->start =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_start);
  GST_BASE_TRANSFORM_CLASS (klass)->stop =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_stop);
  GST_BASE_TRANSFORM_CLASS (klass)->get_unit_size =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_get_unit_size);
  GST_BASE_TRANSFORM_CLASS (klass)->transform_caps =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_transform_caps);
  GST_BASE_TRANSFORM_CLASS (klass)->set_caps =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_set_caps);
  GST_BASE_TRANSFORM_CLASS (klass)->transform =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_transform);
  GST_BASE_TRANSFORM_CLASS (klass)->transform_size =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_transform_size);
  GST_BASE_TRANSFORM_CLASS (klass)->sink_event =
      GST_DEBUG_FUNCPTR (cuda_multirate_spiir_event);


  g_object_class_install_property (gobject_class, PROP_IIRBANK_FNAME,
 			g_param_spec_string(
				"bank-fname",
				"The file of IIR bank feedback coefficients",
				"A parallel bank of first order IIR filter feedback coefficients.",
				NULL,
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

  g_object_class_install_property (gobject_class, PROP_STREAM_ID,
				g_param_spec_int(
					"stream-id",
					"id for cuda stream",
					"id for cuda stream",
					0, G_MAXINT, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				)
			);

}

static void cuda_multirate_spiir_init(CudaMultirateSPIIR * element)
{
//  GstBaseTransform *trans = GST_BASE_TRANSFORM (element);
  g_mutex_init(&element->iir_bank_lock);
  g_mutex_init(&element->iir_bank_available);
  element->bank_fname = NULL;
  element->num_depths = 0;
  element->outchannels = 0;
  element->spstate = NULL;
  element->spstate_initialised = FALSE;
  element->num_exe_samples = 4096; // assumes the rate=4096Hz
  element->num_head_cover_samples = 13120; // assumes the rate=4096Hz, down quality = 9
  element->num_tail_cover_samples = 13104; // assumes the rate=4096Hz

//  gst_base_transform_set_gap_aware (trans, TRUE);
//  gst_pad_set_query_function (trans->srcpad, cuda_multirate_spiir_query);
// gst_pad_set_query_type_function (trans->srcpad,
//      cuda_multirate_spiir_query_type);

  // for ACCELERATE_MULTIRATE_SPIIR_MEMORY_COPY
  element->h_snglsnr_buffer = NULL;
  element->len_snglsnr_buffer = 0;
}

/* vmethods */
static gboolean
cuda_multirate_spiir_start (GstBaseTransform * base)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR (base);

  element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);

  element->need_discont = TRUE;
  element->num_gap_samples = 0;
  element->need_tail_drain = FALSE;
  element->t0 = GST_CLOCK_TIME_NONE;
  element->offset0 = GST_BUFFER_OFFSET_NONE;
  element->next_in_offset = GST_BUFFER_OFFSET_NONE;
  element->samples_in = 0;
  element->samples_out = 0;
  return TRUE;
}

static gboolean
cuda_multirate_spiir_stop (GstBaseTransform * base)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR (base);

  g_mutex_clear(&element->iir_bank_lock);
  g_cond_clear(&element->iir_bank_available);

  if (element->spstate) {
    spiir_state_destroy (element->spstate, element->num_depths);
  }

  g_object_unref (element->adapter);
  element->adapter = NULL;

  return TRUE;
}

static gboolean
cuda_multirate_spiir_get_unit_size (GstBaseTransform * base, GstCaps * caps, gsize *size)
{
  /* FIXME: shinkee: Hard coded width for now */
  gint width = 32;
  gint channels;
  GstStructure *structure;
  gboolean ret;

  g_return_val_if_fail (size != NULL, FALSE);

  /* this works for both float and int */
  structure = gst_caps_get_structure (caps, 0);
  //ret = gst_structure_get_int (structure, "width", &width);
  ret = gst_structure_get_int (structure, "channels", &channels);

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

  GST_LOG("Debug transform caps multiratespiir %" GST_PTR_FORMAT, element);
  GST_LOG("Debug transform base %" GST_PTR_FORMAT, base);
  switch(direction) {
  case GST_PAD_SRC:
    /*
     * sink caps is the same with src caps, except it only has 1 channel
     */

    GST_LOG("Debug multirate caps %d %" GST_PTR_FORMAT, __LINE__, othercaps);
    gst_structure_set(gst_caps_get_structure(othercaps, 0), "channels", G_TYPE_INT, 1, NULL);
    GST_LOG("setting channels to 1\n");
    break;

  case GST_PAD_SINK:
    /*
     * src caps is the same with sink caps, except it only has number of channels that equals to the number of templates
     */
    //if (!g_mutex_trylock(element->iir_bank_lock))
      //printf("lock by another thread");
    g_mutex_lock(&element->iir_bank_lock);
    if(!element->spstate) 
	    g_cond_wait(&element->iir_bank_available, &element->iir_bank_lock);

  GST_LOG("Debug multirate caps %d %" GST_PTR_FORMAT, __LINE__, othercaps);
    gst_structure_set(gst_caps_get_structure(othercaps, 0), "channels", G_TYPE_INT, cuda_multirate_spiir_get_outchannels(element), NULL);
    g_mutex_unlock(&element->iir_bank_lock);
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
    GstPadDirection direction, GstCaps * caps, gsize size, 
    GstCaps *othercaps, gsize *othersize)
{
  CudaMultirateSPIIR *element = CUDA_MULTIRATE_SPIIR(base);
  gboolean ret = TRUE;

  gsize unit_size, other_unit_size;
  GST_LOG_OBJECT (base, "Debug multiratespiir asked to transform size %d in direction %s",
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
    DEBUG_LINE = __LINE__;
    GST_LOG_OBJECT (base, "Debug multiratespiir available samples %ld", get_available_samples (element));
    //*othersize = (size / unit_size + cuda_multirate_spiir_get_available_samples (element)) * other_unit_size;
    DEBUG_LINE = __LINE__;
    *othersize = (size / unit_size + get_available_samples (element)) * other_unit_size;
    GST_DEBUG( "Debug seg fault" );
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
  //gint width;
  gboolean success = TRUE;

  GST_LOG_OBJECT (element, "incaps %" GST_PTR_FORMAT ", outcaps %"
      GST_PTR_FORMAT, incaps, outcaps);

  s = gst_caps_get_structure(outcaps, 0);
  success &= gst_structure_get_int(s, "channels", &channels);
  //success &= gst_structure_get_int(s, "width", &width);
  success &= gst_structure_get_int(s, "rate", &rate);

  g_mutex_lock(&element->iir_bank_lock);
  if(!element->spstate) 
	  g_cond_wait(&element->iir_bank_available, &element->iir_bank_lock);

  if (!success) {
    GST_ERROR_OBJECT(element, "unable to parse and/or accept caps %" GST_PTR_FORMAT, outcaps);
  }

  if (channels != (gint) cuda_multirate_spiir_get_outchannels(element)) {
    /* impossible to happen */
    GST_ERROR_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, cuda_multirate_spiir_get_outchannels(element), outcaps);
      success = FALSE;
  }
#if 0
  if (width != (gint) element->width) {
      /*
       * FIXME :do not support width change at run time
       */
      GST_ERROR_OBJECT(element, "width != %d in %" GST_PTR_FORMAT, element->width, outcaps);
      success = FALSE;
  }
#endif

  if (rate != (gint) element->rate) {
      /*
       * FIXME: do not support rate change at run time
       */
      GST_ERROR_OBJECT(element, "rate != %d in %" GST_PTR_FORMAT, element->rate, outcaps);
      success = FALSE;
  }

  /* transform_caps already done, num_depths already set */

  g_mutex_unlock(&element->iir_bank_lock);
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
  gst_buffer_set_size(gapbuf, outsize);

  /* time */
  if (GST_CLOCK_TIME_IS_VALID (element->t0)) {
    GST_BUFFER_PTS(gapbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out, GST_SECOND,
        element->rate);
    GST_BUFFER_DURATION (gapbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out + len,
        GST_SECOND, element->rate) - GST_BUFFER_PTS(gapbuf);
  } else {
    GST_BUFFER_PTS(gapbuf) = 0;
    GST_BUFFER_DURATION (gapbuf) = 0;
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
      G_GUINT64_FORMAT, (unsigned int)gst_buffer_get_size(gapbuf),
      GST_TIME_ARGS (GST_BUFFER_PTS(gapbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (gapbuf)), GST_BUFFER_OFFSET (gapbuf),
      GST_BUFFER_OFFSET_END (gapbuf));

  if (outsize == 0) {
    GST_DEBUG_OBJECT (element, "buffer dropped");
    return GST_BASE_TRANSFORM_FLOW_DROPPED;
  }

  return GST_FLOW_OK;
}
			
static int 
cuda_multirate_spiir_push_gap (CudaMultirateSPIIR *element, gint gap_len)
{
  GstBuffer *gapbuf;

  if(gap_len)
  {
    gsize outsize = gap_len * sizeof(float) * element->outchannels;
  /*  gst_pad_alloc_buffer_and_set_caps (GST_BASE_TRANSFORM_SRC_PAD (element),
    GST_BUFFER_OFFSET_NONE, outsize,
    GST_PAD_CAPS (GST_BASE_TRANSFORM_SRC_PAD (element)), &gapbuf);
*/
    gapbuf = gst_buffer_new();
    if(!gapbuf)
    {
      GST_ERROR_OBJECT(element, "failure allocating gap buffer");
      return -1;
    }
    
    GST_BUFFER_OFFSET(gapbuf) = gst_audioadapter_expected_offset(element->adapter);

    // FIXME: no sanity check
    cuda_multirate_spiir_assemble_gap_buffer (element, gap_len, gapbuf);

      GST_DEBUG("Debug adapter push %" GST_PTR_FORMAT, gapbuf);
    gst_audioadapter_push(element->adapter, gapbuf);
  }
 
  return 0;
}



static GstFlowReturn
cuda_multirate_spiir_push_drain (CudaMultirateSPIIR *element, gint in_len)
{
  gint num_in_multidown, num_out_multidown, 
       num_out_spiirup = 0, last_num_out_spiirup = 0, old_in_len = in_len;

  num_in_multidown = MIN (old_in_len, element->num_exe_samples);

  gint outsize = 0, out_len = 0,  upfilt_len;
  float * in_multidown, *pos_out;
  upfilt_len = element->spstate[0]->upstate->filt_len;
#if 0
  gint tmp_out_len = 0;
  float *tmp_out;
  tmp_out_len = element->spstate[0]->upstate->mem_len;
  tmp_out = (float *)malloc(element->outchannels * tmp_out_len * sizeof(float));
#endif

  gint i, j;
  GstBuffer *outbuf;
  GstMapInfo mapinfo;
  GstFlowReturn res;
  float *outdata;

  /* To restore the buffer timestamp, out length must be equal to in length */
  //out_len = spiir_state_get_outlen (element->spstate, in_len, element->num_depths);
  if (element->num_exe_samples == element->rate)
    out_len = in_len;
  else
    out_len = in_len - element->num_tail_cover_samples;


  GST_LOG("Debug spiir push drain %" GST_PTR_FORMAT, element);
  outsize = out_len * sizeof(float) * element->outchannels;

  outbuf = gst_buffer_new_allocate(NULL, outsize, NULL);
  /*res =
    gst_pad_alloc_buffer_and_set_caps (GST_BASE_TRANSFORM_SRC_PAD (element),
    GST_BUFFER_OFFSET_NONE, outsize,
    GST_PAD_CAPS (GST_BASE_TRANSFORM_SRC_PAD (element)), &outbuf);
*/
  gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
  memset(mapinfo.data, 0, outsize);

  if (G_UNLIKELY (res != GST_FLOW_OK)) {
    GST_WARNING_OBJECT (element, "failed allocating buffer of %d bytes",
        outsize);
    return GST_FLOW_ERROR;
  }

  outdata = (float *)mapinfo.data;
#if 0  
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
#endif

  while (num_in_multidown > 0) {
    
    DEBUG_LINE = __LINE__;
    GST_DEBUG("Debug samples %d", num_in_multidown);
    g_assert (get_available_samples(element) >= num_in_multidown);
    in_multidown = g_malloc(num_in_multidown * sizeof(float));
    gst_audioadapter_copy_samples(element->adapter, in_multidown, num_in_multidown, NULL, NULL);
    //in_multidown = (float *) gst_adapter_peek (element->adapter, num_in_multidown * sizeof(float));

    num_out_multidown = multi_downsample (element->spstate, in_multidown, (gint) num_in_multidown, element->num_depths, element->stream);
    pos_out = outdata + last_num_out_spiirup * (element->outchannels);
    num_out_spiirup = spiirup (element->spstate, num_out_multidown, element->num_depths, pos_out, element->stream);
    //num_out_spiirup = spiirup (element->spstate, num_out_multidown, element->num_depths, tmp_out, element->stream);

    gst_buffer_unmap(outbuf, &mapinfo);

#if 0
    /* reshape is deprecated because it cost hugh cpu usage */
    /* reshape to the outbuf data */
    for (i=0; i<num_out_spiirup; i++)
      for (j=0; j<element->outchannels; j++)
	      outdata[element->outchannels * (i + last_num_out_spiirup) + j] = tmp_out[tmp_out_len * j + i + upfilt_len - 1];

    //memcpy(pos_out, tmp_out, sizeof(float) * num_out_spiirup * (element->outchannels));
    //free(tmp_out);
#endif


    GST_DEBUG_OBJECT (element, "done cpy data to BUFFER");
 
   /* move along */
    gst_audioadapter_flush_samples(element->adapter, num_in_multidown);
    in_len -= num_in_multidown;
    /* after the first filtering, update the exe_samples to the rate */
    cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->rate);
    num_in_multidown = MIN (in_len, element->num_exe_samples);
    last_num_out_spiirup += num_out_spiirup;
 }

    g_assert(last_num_out_spiirup <= out_len);



    /* time */
    if (GST_CLOCK_TIME_IS_VALID (element->t0)) {
      GST_BUFFER_PTS(outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out, GST_SECOND,
        element->rate);
      GST_BUFFER_DURATION (outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out + out_len,
        GST_SECOND, element->rate) - GST_BUFFER_PTS(outbuf);
    } else {
      GST_BUFFER_PTS(outbuf) = 0;
      GST_BUFFER_DURATION (outbuf) = 0;
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


    gst_buffer_set_size(outbuf, outsize);

    GST_LOG_OBJECT (element,
      "Debug Push_drain: Converted to buffer of %" G_GUINT32_FORMAT
      " samples (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
      GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
      G_GUINT64_FORMAT, out_len, (unsigned int)gst_buffer_get_size(outbuf),
      GST_TIME_ARGS (GST_BUFFER_PTS(outbuf)),
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

    g_free(in_multidown);
    return GST_FLOW_OK;

}



static gint 
cuda_multirate_spiir_process (CudaMultirateSPIIR *element, gint in_len, GstBuffer *outbuf)
{
  gint num_exe_samples, num_in_multidown, num_out_multidown, 
       num_out_spiirup, last_num_out_spiirup = 0, old_in_len = in_len;

  num_exe_samples = element->num_exe_samples;
  num_in_multidown = MIN (old_in_len, num_exe_samples);

  gint outsize = 0, out_len = 0, upfilt_len;
  float *in_multidown;
  upfilt_len = element->spstate[0]->upstate->filt_len;
  //int tmp_out_len = element->spstate[0]->upstate->mem_len;
  //float *tmp_out = (float *)malloc(element->outchannels * tmp_out_len * sizeof(float));

  gint i, j;
  float *outdata, *pos_out;

  GstMapInfo mapinfo;

  GST_DEBUG("Debug line no: %d", __LINE__ );
  if (element->num_exe_samples == element->rate)
    out_len = in_len;
  else
    out_len = in_len - element->num_tail_cover_samples;

  outsize = out_len * sizeof(float) * element->outchannels;

  GST_DEBUG_OBJECT (element, "out len predicted %d", out_len);
  GST_DEBUG_OBJECT (element, "in_len %d, num_exe_samples %d", in_len, num_exe_samples);


#ifdef ACCELERATE_MULTIRATE_SPIIR_MEMORY_COPY
  // to accelerate gpu memory copy, first gpu->cpu(pinned memory)->cpu(gstbuffer)
  // remember copy from h_snglsnr_buffer to gstbuffer
  // should update this part of code after porting to 1.0
  GST_DEBUG("Debug line no: %d", __LINE__ );
  g_assert(element->len_snglsnr_buffer > 0 || (element->len_snglsnr_buffer == 0 && element->h_snglsnr_buffer == NULL));
  if (outsize > element->len_snglsnr_buffer) {
    if (element->h_snglsnr_buffer != NULL){
      cudaFreeHost(element->h_snglsnr_buffer);
    } 
    GST_DEBUG("Debug line no: %d cudaMalloc", __LINE__ );
    cudaMallocHost((void**)&element->h_snglsnr_buffer, outsize);
    element->len_snglsnr_buffer = outsize;
  }
  outdata = (float*) element->h_snglsnr_buffer;
#else

  GST_DEBUG("Debug line no: %d", __LINE__ );
  gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
  outdata = (float *)mapinfo.data;
#endif
   
  while (num_in_multidown > 0) {
    
    DEBUG_LINE = __LINE__;
  GST_DEBUG("Debug line no: %d", __LINE__ );
    GST_DEBUG("Debug samples element %d", get_available_samples(element));
    GST_DEBUG("Debug samples num_in_multidown %d", num_in_multidown);
    g_assert ((int) (get_available_samples(element)) >= (int)num_in_multidown);
    in_multidown = g_malloc(num_in_multidown * sizeof(float));
    gst_audioadapter_copy_samples(element->adapter, in_multidown, num_in_multidown, NULL, NULL);
  GST_DEBUG("Debug line no: %d", __LINE__ );
    //in_multidown = (float *) gst_adapter_peek (element->adapter, num_in_multidown * sizeof(float));

    num_out_multidown = multi_downsample (element->spstate, in_multidown, (gint) num_in_multidown, element->num_depths, element->stream);
    pos_out = outdata + last_num_out_spiirup * (element->outchannels);
    num_out_spiirup = spiirup (element->spstate, num_out_multidown, element->num_depths, pos_out, element->stream);
    //num_out_spiirup = spiirup (element->spstate, num_out_multidown, element->num_depths, tmp_out, element->stream);


    GST_DEBUG_OBJECT (element, "Debug done cpy data to BUFFER %d", num_in_multidown);
 
   /* move along */
    gst_audioadapter_flush_samples(element->adapter, num_in_multidown);
    in_len -= num_in_multidown;
    num_in_multidown = MIN (in_len, num_exe_samples);
    last_num_out_spiirup += num_out_spiirup;
 }

  g_assert(last_num_out_spiirup == out_len);
  GST_DEBUG("Debug line no: %d", __LINE__ );

#ifdef ACCELERATE_MULTIRATE_SPIIR_MEMORY_COPY
  /*
    gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
  GST_DEBUG("Debug line no: %d map size: %d", __LINE__, mapinfo.size );
    memcpy((void*)&mapinfo.data, outdata, outsize);
  GST_DEBUG("Debug line no: %d", __LINE__ );
    gst_buffer_unmap(outbuf, &mapinfo);
    */
  GST_DEBUG("Debug line no: %d outsize: %d", __LINE__, outsize );
  gst_buffer_fill(outbuf, 0, outdata, outsize);
#endif


  GST_DEBUG("Debug line no: %d", __LINE__ );
    /* time */
    if (GST_CLOCK_TIME_IS_VALID (element->t0)) {
      GST_BUFFER_PTS(outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out, GST_SECOND,
        element->rate);
      GST_BUFFER_DURATION (outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out + out_len,
        GST_SECOND, element->rate) - GST_BUFFER_PTS(outbuf);
    } else {
      GST_BUFFER_PTS(outbuf) = 0;
      GST_BUFFER_DURATION (outbuf) = 0;
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


  GST_DEBUG("Debug line no: %d", __LINE__ );
    gst_buffer_set_size(outbuf, outsize);

    GST_LOG_OBJECT (element,
      "Converted to buffer of %" G_GUINT32_FORMAT
      " samples (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
      GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
      G_GUINT64_FORMAT, out_len, (unsigned int)gst_buffer_get_size(outbuf),
      GST_TIME_ARGS (GST_BUFFER_PTS(outbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (outbuf)),
      GST_BUFFER_OFFSET (outbuf), GST_BUFFER_OFFSET_END (outbuf));

    if (outsize == 0) {
      GST_DEBUG_OBJECT (element, "buffer dropped");
      return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    /* after the first filtering, update the exe_samples to the rate */
    cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->rate);


    g_free(in_multidown);
    return GST_FLOW_OK;
}


/*
 * construct a buffer of zeros and push into adapter
 */


static void adapter_push_zeros(CudaMultirateSPIIR *element, unsigned samples)
{
  GstBuffer *zerobuf = gst_buffer_new(); 
  if(!zerobuf) {
    GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");

    GST_BUFFER_FLAG_SET(zerobuf, GST_BUFFER_FLAG_GAP);
    GST_BUFFER_PTS(zerobuf) = gst_audioadapter_expected_timestamp(element->adapter);
    
    if(!GST_BUFFER_PTS_IS_VALID(zerobuf))
      GST_BUFFER_PTS(zerobuf) = 0;
    GST_BUFFER_DURATION(zerobuf) = gst_util_uint64_scale_int_round(samples, GST_SECOND, GST_AUDIO_INFO_RATE(&element->audio_info));
    GST_BUFFER_OFFSET(zerobuf) = gst_audioadapter_expected_offset(element->adapter);
    
    if(!GST_BUFFER_OFFSET_IS_VALID(zerobuf))
      GST_BUFFER_OFFSET(zerobuf) = 0;
    
    GST_BUFFER_OFFSET_END(zerobuf) = GST_BUFFER_OFFSET(zerobuf) + samples;

    GST_DEBUG("Debug line no: %d %" GST_PTR_FORMAT, __LINE__, zerobuf);
    gst_audioadapter_push(element->adapter, zerobuf);
  }
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

  gsize size;
  size = gst_buffer_get_size(inbuf);

  GST_LOG_OBJECT (element, "multiratespiir transforming %s+%s buffer of %ld bytes, ts %"
      GST_TIME_FORMAT ", duration %" GST_TIME_FORMAT ", offset %"
      G_GINT64_FORMAT ", offset_end %" G_GINT64_FORMAT,
      GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "GAP" : "NONGAP",
      GST_BUFFER_IS_DISCONT(inbuf) ? "DISCONT" : "CONT",
      size, GST_TIME_ARGS (GST_BUFFER_PTS(inbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (inbuf)),
      GST_BUFFER_OFFSET (inbuf), GST_BUFFER_OFFSET_END (inbuf));

  /*
   * set device context
   */

  g_mutex_lock(&element->iir_bank_lock);
  if(!element->spstate_initialised) {
	  g_cond_wait(&element->iir_bank_available, &element->iir_bank_lock);
  }
  g_mutex_unlock(&element->iir_bank_lock);

  CUDA_CHECK(cudaSetDevice(element->deviceID));
  /* check for timestamp discontinuities;  reset if needed, and set
   * flag to resync timestamp and offset counters and send event
   * downstream */

  if (G_UNLIKELY (GST_BUFFER_IS_DISCONT (inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
    GST_DEBUG_OBJECT (element, "reset spstate");
    spiir_state_reset (element->spstate, element->num_depths, element->stream);
    /* FIXME: need to push_drain of data in the adapter ? if upstream never produces discontinous data, no need to push_drain. */
    gst_audioadapter_clear (element->adapter);
    
    element->need_discont = TRUE;

    /*
     * (re)sync timestamp and offset book-keeping. Set t0 and offset0 to be the timestamp and offset of the inbuf.
     */

    element->t0 = GST_BUFFER_PTS(inbuf);
    element->offset0 = GST_BUFFER_OFFSET(inbuf);
    element->num_gap_samples = 0;
    element->need_tail_drain = FALSE;
    element->samples_in = 0;
    element->samples_out = 0;
    if (element->num_head_cover_samples > 0)
      cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->num_head_cover_samples);
    else
      cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->rate);

  }

  element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

  /* 0-length buffers are produced to inform downstreams for current timestamp  */
  if (size == 0) {
    /* time */
    if (GST_CLOCK_TIME_IS_VALID (element->t0)) {
      GST_BUFFER_PTS(outbuf) = element->t0 +
        gst_util_uint64_scale_int_round (element->samples_out, GST_SECOND,
        element->rate);
    } else {
      GST_BUFFER_PTS(outbuf) = GST_CLOCK_TIME_NONE;
    }
    /* offset */
    if (element->offset0 != GST_BUFFER_OFFSET_NONE) {
      GST_BUFFER_OFFSET (outbuf) = element->offset0 + element->samples_out;
      GST_BUFFER_OFFSET_END (outbuf) = GST_BUFFER_OFFSET (outbuf);
    } else {
    GST_BUFFER_OFFSET (outbuf) = GST_BUFFER_OFFSET_NONE;
    GST_BUFFER_OFFSET_END (outbuf) = GST_BUFFER_OFFSET_NONE;
    }
 
   GST_BUFFER_DURATION (outbuf) = 0;
   gst_buffer_set_size(outbuf, gst_buffer_get_size(inbuf));
   return GST_FLOW_OK;
  }

  gint in_samples, num_exe_samples, num_head_cover_samples, num_tail_cover_samples;
  in_samples = gst_buffer_get_size(inbuf) / (element->width / 8);
  num_exe_samples = element->num_exe_samples;
  num_head_cover_samples = element->num_head_cover_samples;
  num_tail_cover_samples = element->num_tail_cover_samples;
  gsize history_gap_samples, gap_buffer_len, adapter_len;
  gint num_zeros, num_filt_samples;


  DEBUG_LINE = __LINE__;
  GST_DEBUG("Debug get avaiable samples transform %d supposed size=%d", get_available_samples(element), size );
  GST_DEBUG("Debug gap_handle %d", element->gap_handle); 

  switch (element->gap_handle) {


    /* FIXME: case 1 may cause some bugs, have not tested it for a long time */
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
    if (history_gap_samples >= (gsize) num_tail_cover_samples) {
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
    if (history_gap_samples < (gsize) num_tail_cover_samples) {
      /* 
       * if current number of gap samples more than we can 
       * cover the roll-offs offset, process the buffer; 
       * otherwise absorb the inbuf
       */
      if (element->num_gap_samples >= (gsize) num_tail_cover_samples) {
        /*
         * one buffer to cover the roll-offs
         */
        num_zeros = num_tail_cover_samples - history_gap_samples;
        adapter_push_zeros (element, num_zeros);
	//adapter_len = cuda_multirate_spiir_get_available_samples(element);
    DEBUG_LINE = __LINE__;
	adapter_len = get_available_samples(element);
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
    if (element->num_gap_samples >= (gsize) num_tail_cover_samples) {
      if (element->need_tail_drain) {
        //adapter_len = cuda_multirate_spiir_get_available_samples(element);
    DEBUG_LINE = __LINE__;
        adapter_len = get_available_samples(element);
        cuda_multirate_spiir_push_gap(element, element->num_tail_cover_samples + adapter_len);
        gst_audioadapter_clear (element->adapter);

      }
      spiir_state_reset (element->spstate, element->num_depths, element->stream);
      cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->num_head_cover_samples);
      num_exe_samples = element->num_exe_samples;
    }

    element->num_gap_samples = 0;
    element->need_tail_drain = TRUE;
    DEBUG_LINE = __LINE__;
    adapter_len = get_available_samples(element);
    //adapter_len = cuda_multirate_spiir_get_available_samples(element);
    /*
     * here merely speed consideration: if samples ready to be processed are less than num_exe_samples,
     * wait until there are over num_exe_samples
     */
    if (in_samples < num_exe_samples - adapter_len) {
      /* absorb the buffer */
      gst_buffer_ref(inbuf);	/* don't let the adapter free it */
      gst_audioadapter_push (element->adapter, inbuf);
      GST_INFO_OBJECT(element, "inbuf absorbed %d samples", in_samples);
      return GST_BASE_TRANSFORM_FLOW_DROPPED;

    } else {
      /*
       * filter
       */
      gst_buffer_ref(inbuf);	/* don't let the adapter free it */
      GST_DEBUG("Debug adapter push %d %" GST_PTR_FORMAT, __LINE__, inbuf);
      gst_audioadapter_push (element->adapter, inbuf);
      /* 
       * to speed up, number of samples to be filtered is times of num_exe_samples
       */
    DEBUG_LINE = __LINE__;
      adapter_len = get_available_samples(element);
      if (element->num_exe_samples == element->rate)
        num_filt_samples = gst_util_uint64_scale_int (adapter_len, 1, num_exe_samples) * num_exe_samples;
      else
        num_filt_samples = num_exe_samples;
      GST_DEBUG("Debug line no: %d num_filt_samples %d", __LINE__, num_filt_samples );
      res = cuda_multirate_spiir_process(element, num_filt_samples, outbuf);
      if (res != GST_FLOW_OK)
        return res;
    }
  }
  break;

  case 0: // gap is treated as 0; 

    if (GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
      GST_DEBUG("Debug line no: %d pushing gap samples %d", __LINE__, in_samples);
        adapter_push_zeros (element, in_samples);
    }
    else {
      gst_buffer_ref(inbuf);	/* don't let the adapter free it */
      GST_DEBUG("Debug line no: %d push %" GST_PTR_FORMAT, __LINE__, inbuf);
      gst_audioadapter_push (element->adapter, inbuf);
    }
     /* 
     * to speed up, number of samples to be filtered is times of num_exe_samples
     */
    DEBUG_LINE = __LINE__;
    adapter_len = get_available_samples(element);
    GST_DEBUG("Debug adapter_len u %u", adapter_len);
    GST_DEBUG("Debug adapter_len d %d num exe samples %d", adapter_len, element->num_exe_samples);
    g_assert(element->num_exe_samples > 0);
    if ( (int) adapter_len >= (int) element->num_exe_samples) {
      if (element->num_depths > 1)
        element->need_tail_drain = TRUE;
      
      GST_DEBUG("Debug line no: %d num exe samples: %d buffer size: %d", __LINE__, element->num_exe_samples, gst_buffer_get_size(outbuf) );

      res = cuda_multirate_spiir_process(element, element->num_exe_samples, outbuf);

      GST_DEBUG("Debug line no: %d", __LINE__ );
      if (res != GST_FLOW_OK)
        return res;
    } else {
      GST_DEBUG("Debug line no: %d", __LINE__ );
        GST_INFO_OBJECT(element, "inbuf absorbed %d samples", in_samples);
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

  GST_DEBUG_OBJECT(event, "Debug event multiratespiir %" GST_PTR_FORMAT, event);

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
    case GST_EVENT_SEGMENT:
      
    GST_DEBUG_OBJECT(element, "EVENT SEGMENT");
    /* implicit assumption: spstate has been inited */
    if (element->need_tail_drain && element->num_tail_cover_samples > 0) {
	CUDA_CHECK(cudaSetDevice(element->deviceID));
        GST_DEBUG_OBJECT(element, "SEGMENT, clear tails.");
	if (element->num_gap_samples >= element->num_tail_cover_samples) {
		cuda_multirate_spiir_push_gap(element, element->num_tail_cover_samples);
	} else {
        adapter_push_zeros (element, element->num_tail_cover_samples);
	//int adapter_len = cuda_multirate_spiir_get_available_samples(element);
    DEBUG_LINE = __LINE__;
	int adapter_len = get_available_samples(element);
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
      g_mutex_lock(&element->iir_bank_lock);
      if(!element->spstate)
      	g_cond_wait(&element->iir_bank_available, &element->iir_bank_lock);
      if (element->num_head_cover_samples > 0)
        cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->num_head_cover_samples);
      else
        cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->rate);
      g_mutex_unlock(&element->iir_bank_lock);

      break;

    case GST_EVENT_EOS:
 
      GST_DEBUG_OBJECT(element, "EVENT EOS");
      if (element->need_tail_drain) {
	CUDA_CHECK(cudaSetDevice(element->deviceID));
	if (element->num_gap_samples >= element->num_tail_cover_samples) {
          GST_DEBUG_OBJECT(element, "EOS, clear tails by pushing gap, num gap samples %" G_GUINT64_FORMAT, element->num_gap_samples);
  	  cuda_multirate_spiir_push_gap(element, element->num_tail_cover_samples);
	} else {

          GST_DEBUG_OBJECT(element, "EOS, clear tails by pushing drain");
          adapter_push_zeros (element, element->num_tail_cover_samples);
    DEBUG_LINE = __LINE__;
 	  int adapter_len = get_available_samples(element);
          cuda_multirate_spiir_push_drain (element, adapter_len);
	}

        //spiir_state_reset (element->spstate, element->num_depths, element->stream);
      }

      break;
    default:
      break;
  }

  return GST_BASE_TRANSFORM_CLASS(cuda_multirate_spiir_parent_class)->sink_event (base, event);
}


static void
cuda_multirate_spiir_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
    {
  CudaMultirateSPIIR *element;

  element = CUDA_MULTIRATE_SPIIR (object);
  //gboolean success = TRUE;

  GST_OBJECT_LOCK (element);
  switch (prop_id) {

    case PROP_IIRBANK_FNAME:
      g_mutex_lock(&element->iir_bank_lock);

      GST_LOG_OBJECT (element, "obtaining bank, stream id is %ld", (long)element->stream);
      element->bank_fname = g_value_dup_string(value);
      /* bank_id is deprecated, get the stream id directly from prop 
       * must make sure stream_id has already loaded */
      //cuda_multirate_spiir_read_bank_id(element->bank_fname, &element->bank_id);
 
      int deviceCount;
      cudaGetDeviceCount(&deviceCount);
      element->deviceID = (element->stream_id) % deviceCount ;
      printf("device for spiir %s %d\n", element->bank_fname, element->deviceID);
      CUDA_CHECK(cudaSetDevice(element->deviceID));
      // cudaStreamCreateWithFlags(&element->stream, cudaStreamNonBlocking);
      cudaStreamCreate(&element->stream);

      cuda_multirate_spiir_read_ndepth_and_rate(element->bank_fname, &element->num_depths, &element->rate);

      cuda_multirate_spiir_init_cover_samples(&element->num_head_cover_samples, &element->num_tail_cover_samples, element->rate, element->num_depths, DOWN_FILT_LEN*2, UP_FILT_LEN);

      /* we consider the num_exe_samples equals to rate unless it is at the first or last buffer */
      cuda_multirate_spiir_update_exe_samples (&element->num_exe_samples, element->rate);

      element->spstate = spiir_state_create (element->bank_fname, element->num_depths, element->rate,
		    element->num_head_cover_samples, element->num_exe_samples,
		    element->stream);


      GST_DEBUG_OBJECT (element, "number of cover samples set to (%d, %d), number of exe samples set to %d", element->num_head_cover_samples, element->num_tail_cover_samples, element->num_exe_samples);

      if (!element->spstate) {
        GST_ERROR_OBJECT(element, "spsate could not be initialised");
      }

      element->spstate_initialised = TRUE;

      /*
       * signal ready of the bank
       */
      element->outchannels = element->spstate[0]->num_templates * 2;
      element->width = 32; //FIXME: only can process float data
      GST_DEBUG_OBJECT (element, "spiir bank available, number of depths %d, outchannels %d", element->num_depths, element->outchannels);
      g_cond_broadcast(&element->iir_bank_available);
      g_mutex_unlock(&element->iir_bank_lock);

      break;

    case PROP_GAP_HANDLE:
      element->gap_handle = g_value_get_int(value);
      break;

    case PROP_STREAM_ID:
      element->stream_id = g_value_get_int(value);
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
    case PROP_IIRBANK_FNAME:
      g_mutex_lock(&element->iir_bank_lock);
      g_value_set_string(value, element->bank_fname);
      g_mutex_unlock(&element->iir_bank_lock);
      break;

    case PROP_GAP_HANDLE:
      g_value_set_int (value, element->gap_handle);
      break;

    case PROP_STREAM_ID:
      g_value_set_int (value, element->stream_id);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  GST_OBJECT_UNLOCK (element);
}


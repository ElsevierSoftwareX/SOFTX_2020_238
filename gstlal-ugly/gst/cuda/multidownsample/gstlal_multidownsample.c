/*
 * multi downsample written in one element
 *
 * Copyright (C) 2014 Qi Chu
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ========================================================================
 *
 *				  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from standard library
 */

#include <math.h>
#include <string.h>

/*
 * stuff from glib/gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_tags.h>
#include <multidownsample/gstlal_multidownsample.h>

#define GET_NEW_RATE(rate, depth) (rate/(int)(pow(2,depth)))
#define DEFAULT_DEPTH 8
/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_multi_downsample_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gstlal_multi_downsample", 0, "multi downsample element");
}


GST_BOILERPLATE_FULL(GstlalMultiDownsample, gstlal_multi_downsample, GstElement, GST_TYPE_ELEMENT, additional_initializations);



/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */



/*
 * Pad templates.
 */


static GstStaticPadTemplate gstlal_multi_downsample_src_template = GST_STATIC_PAD_TEMPLATE(
	"src_%dHz",
	GST_PAD_SRC,
	GST_PAD_SOMETIMES,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) {32, 64}; " \
	)
);

static GstStaticPadTemplate gstlal_multi_downsample_sink_template = GST_STATIC_PAD_TEMPLATE(
	"sink",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) {32, 64}; " \
	)
);

/*
 * properites
 */

enum property{
	ARG_DEPTH = 8
};

static void set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
	GstlalMultiDownsample *element = GSTLAL_MULTI_DOWNSAMPLE(object);

	GST_OBJECT_LOCK(element);

	switch(prop_id){

	case ARG_DEPTH:
		element->depth = g_value_get_int(value);
		break;
	
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
	GstlalMultiDownsample *element = GSTLAL_MULTI_DOWNSAMPLE(object);

	GST_OBJECT_LOCK(element);

	switch(prop_id){
	case ARG_DEPTH:
		g_value_set_int(value, element->depth);
		break;
	
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}




/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */

gint get_buffer_rate(GstBuffer *buf)
{
	GstStructure *str;
	gint rate;
	str = gst_caps_get_structure(GST_BUFFER_CAPS(buf), 0);
	gst_structure_get_int(str, "rate", &rate);
	return rate;
}
	


typedef struct _PadState{
	guint64 next_offset;
	GstClockTime next_timestamp;
} PadState;

static void src_pad_linked_handler(GstPad *pad, GstPad *peer, gpointer data)
{
	PadState *pad_state = g_new0(PadState, 1);

	pad_state->next_timestamp = GST_CLOCK_TIME_NONE;
	pad_state->next_offset = 0;

	g_assert(gst_pad_get_element_private(pad) == NULL);
	gst_pad_set_element_private(pad, pad_state);
}


/*
 * unlinked event handler.  free pad's state
 */


static void src_pad_unlinked_handler(GstPad *pad, GstPad *peer, gpointer data)
{
	struct pad_state *pad_state = (struct pad_state *) gst_pad_get_element_private(pad);

	gst_pad_set_element_private(pad, NULL);
	g_free(pad_state);
}


static GstPad *add_src_pad_setcaps(GstlalMultiDownsample *element, const guint8 current_depth, GstBuffer *inbuf)
{
	/* parse caps of inbuf */
	GstCaps *incaps = GST_BUFFER_CAPS(inbuf);
	GstStructure *instr;
	gint rate, width, new_rate;
	gboolean success = TRUE;
	
	instr = gst_caps_get_structure(incaps, 0);
	success &= gst_structure_get_int(instr, "rate", &rate);
	success &= gst_structure_get_int(instr, "width", &width);

	if(!success)
		GST_ERROR_OBJECT(element, "unable to parse inbuf");

	/* 
	 * set caps of the src pad according to the caps of inbuf,
	 * and add src pad 
	 */

	GstPad *srcpad;
	gchar *name;
	GstPadTemplate *template;

	template = gst_static_pad_template_get(&gstlal_multi_downsample_src_template);

	name = g_strdup_printf("src_%dHz", GET_NEW_RATE(rate,current_depth));
	srcpad = gst_pad_new_from_template(template, name);
	
	gst_object_unref(template);

	GstStructure *src_str = gst_caps_get_structure(gst_pad_get_caps(srcpad), 0);
	new_rate = GET_NEW_RATE(rate, current_depth);
	gst_structure_set(src_str, "rate", G_TYPE_INT, new_rate, NULL);
	gst_structure_set(src_str, "width", G_TYPE_INT, width, NULL);

	/*
	 * connect signal handlers
	 */

	g_signal_connect(srcpad, "linked", (GCallback) src_pad_linked_handler, NULL);
	g_signal_connect(srcpad, "unlinked", (GCallback) src_pad_unlinked_handler, NULL);
	
	gst_pad_set_active(GST_PAD(srcpad), TRUE);
	gst_object_ref(srcpad);
	gst_element_add_pad(GST_ELEMENT(element), GST_PAD(srcpad));

	return GST_PAD(srcpad);

}

/*
 * get src pads, creat it if not existed
 */
 
static GstPad *get_src_pad(GstlalMultiDownsample *element, const guint8 current_depth, GstBuffer *inbuf)
{
	GstPad *srcpad;
	gchar *name;

	/* construct the name for the sink pad */

	name = g_strdup_printf("src_%dHz", GET_NEW_RATE(element->inrate,current_depth));

	srcpad = gst_element_get_static_pad(GST_ELEMENT(element), name);
	if(!srcpad){
		srcpad = add_src_pad_setcaps(element, current_depth, inbuf);
	} else{
		GST_ERROR_OBJECT(element, "failure to create src pad");
	}

	return srcpad;

}

static void set_metadata(GstPad *srcpad, GstBuffer *inbuf, GstBuffer *outbuf)
{
		PadState *pad_state = (PadState *) gst_pad_get_element_private(srcpad);
		if(GST_BUFFER_TIMESTAMP_IS_VALID(inbuf)){
			GST_BUFFER_TIMESTAMP(outbuf) = GST_BUFFER_TIMESTAMP(inbuf);
			GST_BUFFER_DURATION(outbuf) = 0 ;
			GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET_END(outbuf) = pad_state->next_offset;
		}else{
			GST_WARNING_OBJECT(outbuf, "unable to set outbuf");
		}
}

/*

 * chain(), create src pads if not existed, 
 * perform downsampler, and push buffers accordingly
 */

static GstFlowReturn chain(GstPad *pad, GstBuffer*inbuf)
{
	GstlalMultiDownsample *element = GSTLAL_MULTI_DOWNSAMPLE(gst_pad_get_parent(pad));

	gint insamples, outsamples;
	GstFlowReturn result = GST_FLOW_OK;
//	insamples = GST_BUFFER_SIZE(inbuf) / (element->inrate / 8);

	GST_DEBUG_OBJECT(element, "received %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(inbuf));
	element->inrate = get_buffer_rate(inbuf);

	GstPad *srcpad;
	gint i;

	for(i = 0; i < element->depth; i++)
	{
		srcpad = get_src_pad(element, i, inbuf);
		
		/* if not linked, continue */
		if(!gst_pad_is_linked(srcpad)){
			GST_LOG_OBJECT(srcpad, "skipping: not linked");
			gst_object_unref(srcpad);
			srcpad = NULL;
			continue;
		}

		/* allocate buffer */

		GstBuffer *outbuf;
		gint srcsize = GST_BUFFER_SIZE(inbuf)/(int)(pow(2, i)); 
		outbuf = gst_buffer_new_and_alloc(srcsize);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));

		/* set buffer caps */
		gst_buffer_set_caps(outbuf, GST_PAD_CAPS(srcpad));
		g_assert(GST_BUFFER_CAPS(outbuf) != NULL);

		/* set timestamp and duration */
		set_metadata(srcpad, inbuf, outbuf);
		/* if current_depth > 0, downsample, and push buffer */
	//	if(i > 0){
			// downsample(inbuf_data, output);
			/* construct outbuf */
			gst_pad_push(srcpad, outbuf);
//		}			
				
	}
	return result;
}

/*
 * base_init()
 */


static void gstlal_multi_downsample_base_init(gpointer klass)
{
}


/*
 * class_init()
 */


static void gstlal_multi_downsample_class_init(GstlalMultiDownsampleClass *kclass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(kclass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(kclass);

	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&gstlal_multi_downsample_src_template));	

	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&gstlal_multi_downsample_sink_template));	

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	g_object_class_install_property(
		gobject_class,
		ARG_DEPTH,
		g_param_spec_int(
			"downsample depth",
			"Downsample depth",
			"Downsample to multiple rates by a series of factors that are power of 2. The minimum output rate is determined by the depth.",
			0, 10, DEFAULT_DEPTH,
			G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS
		)
	);
	gst_element_class_set_details_simple(element_class, "Multi Downsample", "downsample to multiple rates", "multi downsample", "qi.chu <qi.chu@ligo.org");
}

/*
 * instance_init()
 */

static void gstlal_multi_downsample_init(GstlalMultiDownsample *element, GstlalMultiDownsampleClass *kclass)
{
	/* configure sink pad */
	GstPadTemplate *template;
	GstPad *sinkpad;
	template = gst_static_pad_template_get(&gstlal_multi_downsample_sink_template);
	sinkpad = gst_pad_new_from_template(template, "sink");
	gst_object_unref(template);

	gst_pad_set_chain_function(sinkpad, GST_DEBUG_FUNCPTR(chain));
//	gst_pad_set_event_function(element->sinkpad, GST_DEBUG_FUNCPTR(sink_event));

	gst_element_add_pad(GST_ELEMENT(element), sinkpad);

}



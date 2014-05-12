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

/*
 * stuff from glib/gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_tags.h>

#define GET_NEW_RATE(rate, depth) (rate/int(pow(2,depth)))
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

static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GstlalMultiDownsample *element = GSTLAL_MULTIDOWNSAMPLE(object);

	GST_OBJECT_LOCK(elment);

	switch((enum property) id){
	case ARG_DEPTH:
		element->depth = g_value_get_int(value);
		break;
	
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GstlalMultiDownsample *element = GSTLAL_MULTIDOWNSAMPLE(object);

	GST_OBJECT_LOCK(elment);

	switch((enum property) id){
	case ARG_DEPTH:
		g_value_set_int(value, element->depth);
		break;
	
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
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


typedef struct _PadState{
	guint64 next_offset;
	GstClockTime next_timestamp;
} PadState;

static void src_pad_linked_handler(GstPad *pad, GstPad *peer, gpointer data)
{
	PadState *pad_state = g_new0(struct pad_state, 1);

	pad_state->next_timestamp = GST_CLOCK_TIME_NONE;
	pad_state->next_out_offset = 0;

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
	gint rate, width;
	gboolean success = TRUE;
	
	str = gst_caps_get_structure(incaps, 0);
	success &= gst_structure_get_int(str, "rate", &rate);
	success &= gst_structure_get_int(str, "width", &width);

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
	success &= gst_stucture_set(src_str, "rate", G_TYPE_INT, GET_NEW_RATE(rate, current_depth, NULL);
	success &= gst_stucture_set(src_str, "width", G_TYPE_INT, width, NULL);
	if(!success)
		GST_ERROR_OBJECT(element, "unable to set src caps");

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

	/* construct the name for the sink pad */

	name = GET_SINK_NAME(element->inrate, current_depth);

	srcpad = gst_element_get_static_pad(GST_ELEMENT(element), name);
	if(!srcpad){
		srcpad = add_src_pad_setcaps(element, current_depth, inbuf);
	} else{
		GST_ERROR_OBJECT(element, "failure to create src pad");
	}

	return srcpad;

}

static void set_metadata(GstPad *srcpad, GstBuffer *inbuf, GstBuffer *outbuf)
{GstlalMultiDownsample *element
		PadState *pad_state = (PadState *) gst_pad_get_element_private(srcpad);
		if(GST_BUFFER_TIMESTAMP_IS_VALID(inbuf)){
			GST_BUFFER_TIMESTAMP(outbuf) = GST_BUFFER_TIMESTAMP(inbuf);
			GST_BUFFER_DURATION(outbuf) = 0 
			GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET_END(outbuf) = pad_state->next_out_offset;
		}else{
			GST_WARNING_OBJECT(outbuf, "unable to set outbuf");
		}
}
static GstCaps *src_get_caps(GstCaps* incaps, gint depth)
{
	GstStructure *str;
	gint rate;
	gboolean success = TRUE;
	
	str = gst_caps_get_structure(incaps, 0);
	success =&= gst_structure_get_int(str, "rate", &rate);

	if(success){
		GstCaps *new_caps;
		new_caps = gst_caps_copy(incaps);
		gst_caps_set_value(new_caps, "rate", GET_NEW_RATE(rate, depth));
	} else{
		GST_ERROR_OBJECT(incaps, "unable to parse caps");
	}
	return new_caps;
}
/* 
 * chain(), create src pads if not existed, 
 * perform downsampler, and push buffers accordingly
 */

static GstFlowReturn chain(GstPad *pad, GstBuffer*inbuf)
{
	GstlalMultiDownsample *element = GSTLAL_MULTIDOWSAMPLE(gst_pad_get_parent(pad));

	gint insamples, outsamples;
	GstFlowReturn result = GST_FLOW_OK;
//	insamples = GST_BUFFER_SIZE(inbuf) / (element->inrate / 8);

	GST_DEBUG_OBJECT(element, "received %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(inbuf));


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
		gint srcsize = GST_BUFFER_SIZE(inbuf)/int(pow(2, i)); 
		outbuf = gst_buffer_new_and_alloc(srcsize);
		memcpy(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));

		/* set buffer caps */
		gst_buffer_set_caps(outbuf, GST_PAD_CAPS(srcpad);
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
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&gstlal_multi_downsample_src_template));	

	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&gstlal_multi_downsample_sink_template));	

	gst_object_class_install_property(
		gobject_class,
		ARG_DEPTH,
		g_param_spec_int(
			"downsample depth",
			"Downsample depth",
			"Downsample to multiply rates by a series of factors that are power of 2. The minimum output rate is determined by the depth.",
			DEFAULT_DEPTH,
			(GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
		)
	);

}

/*
 * instance_init()
 */

static void gstlal_multi_downsample_init(GstlalMultiDownsample *element, GstlalMultiDownsampleClass *kclass)
{
	/* configure sink pad */
	GstPadTemplate *template;
	template = gst_static_pad_template_get(&gstlal_multi_downsample_sink_template);
	element->sinkpad = gst_pad_new_from_template(template, "sink");
	gst_object_unref(template);

	gst_pad_set_chain_function(element->sinkpad, GST_DEBUG_FUNCPTR(chain));
	gst_pad_set_event_function(element->sinkpad, GST_DEBUG_FUNCPTR(sink_event));

	gst_element_add_pad(GST_ELEMENT(element), element->sinkpad);

}



/*
 * Copyright (C) 2009 Stephen Privitera <sprivite@ligo.caltech.edu>,
 * Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>
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
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal.h>
#include <gstlal_delay.h>

/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], "\
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32,64}; " \
		"audio/x-raw-int, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {8,16,32,64}, " \
		"signed = (boolean) {true,false}; " \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64,128}" \
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32,64};"
		"audio/x-raw-int, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {8,16,32,64}, " \
		"signed = (boolean) {true,false}; " \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64,128}"
	)
);


GST_BOILERPLATE(
	GSTLALDelay,
	gstlal_delay,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


enum property {
	ARG_DELAY = 1
};

#define DEFAULT_DELAY 0

/*
 * get_unit_size() stores the size (in bytes) of a single sample
 * from a single channel in the buffer.
 * The "width" of a buffer is equal to the total number of channels
 * times the number of bits per channel.
 */
static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint width;
	gint channels;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}
	if(!gst_structure_get_int(str, "width", &width)) {
		GST_DEBUG_OBJECT(trans, "unable to parse width from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = width / 8 * channels;
	return TRUE;
}


/*
 * When the caps on an element's pads a finally set, this function is called.
 * We use this opportunity to record the chosen sampling rate and the unit
 * size.
 */
static gboolean set_caps(GstBaseTransform *trans,
			 GstCaps *incaps,
			 GstCaps *outcaps)
{
	GSTLALDelay *element = GSTLAL_DELAY(trans);

	/* sampling rate of this channel */
	gst_structure_get_int(gst_caps_get_structure(incaps, 0), "rate", &element->rate);

	/* size of unit sample */
	get_unit_size(trans,incaps,&element->unit_size);

	return TRUE;
}




/*
 * When an input buffer is received, prepare_output_buffer is called.
 * This function allows you to map an output buffer to a given
 * input buffer.  In this case, we use this function to set the
 * size of the output buffer.
 */
static GstFlowReturn prepare_output_buffer(GstBaseTransform *trans,
					   GstBuffer *inbuf,
					   gint size,
					   GstCaps *caps,
					   GstBuffer **outbuf)
{
	/* cast BaseTransform to GSTLALDelay */
	GSTLALDelay *element = GSTLAL_DELAY(trans);
	GstFlowReturn result;

	/* delay params  */
	guint delaysize = (guint) element->delay*element->unit_size;
	guint insize = (guint) size;

	if ( insize <= delaysize )
	   /* ignore this buffer */
	{
		*outbuf = NULL;
		result = GST_FLOW_OK;
	}
	else if ( 0 < delaysize )
	   /* pass part of this buffer */
	{
		*outbuf = gst_buffer_new_and_alloc(insize-delaysize);
		result = GST_FLOW_OK;
	}
	else
	{
		*outbuf = gst_buffer_ref(inbuf);
		result = GST_FLOW_OK;
	}

	return result;
}

/*
 * The transform size function is required to make sure that the prepare buffer
 * function gives the right output size.  Since this element doesn't convert
 * unit sizes, this is pretty easy
 */


gboolean transform_size(GstBaseTransform *trans,
			GstPadDirection direction,
			GstCaps *caps,
			guint size,
			GstCaps *othercaps,
			guint *othersize)
{
	/* cast BaseTransform to GSTLALDelay */
	GSTLALDelay *element = GSTLAL_DELAY(trans);

	/* delay params  */
	guint delaysize = (guint) element->delay*element->unit_size;
	guint insize = (guint) size;

	if (direction == GST_PAD_SINK)
	{
		if (insize <= delaysize)
			*othersize = 0;
		else if ( 0 < delaysize )
			*othersize = insize - delaysize;
		else
			*othersize = size;
		return TRUE;
	}

	if (direction == GST_PAD_SRC)
	{
		/* FIXME I have know idea what to do here */
		*othersize = size;
		return TRUE;
	}
	
	/* if we have made it this far we don't know what to do */
	return FALSE;
}


/*
 * The transform function actually does the heavy lifting on buffers.
 * Given an input buffer and an output buffer (the latter of which is
 * set in prepare_output_buffer), determine what data actually gets put
 * into the output buffer.
 */
static GstFlowReturn transform( GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALDelay *element = GSTLAL_DELAY(trans);
	GstFlowReturn result;
	guint delaysize = element->unit_size*element->delay;
	guint64 delaytime;

	if ( GST_BUFFER_SIZE(inbuf) <= delaysize )
	/* drop entire buffer */
	{
		element->delay -= GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
		result = GST_BASE_TRANSFORM_FLOW_DROPPED;
	}
	else if ( 0 < element->delay )
	/* drop part of buffer, pass the rest */
	{
		guint outsize = GST_BUFFER_SIZE(outbuf);
		guint insize = GST_BUFFER_SIZE(inbuf);
		guint8 *indata = GST_BUFFER_DATA(inbuf);
		guint8 *outdata = GST_BUFFER_DATA(outbuf);

		memcpy((void *) outdata, (const void *) (indata+insize-outsize), outsize);

		/* how much time to skip */

		delaytime = gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(inbuf),
			element->delay, GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf));

		/* set output buffer metadata */
		GST_BUFFER_TIMESTAMP(outbuf) = GST_BUFFER_TIMESTAMP(inbuf) + delaytime;
		GST_BUFFER_DURATION(outbuf) = GST_BUFFER_DURATION(inbuf) - delaytime;
		GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET(inbuf) + element->delay;
		GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_END(inbuf);
		GST_BUFFER_SIZE(outbuf) = GST_BUFFER_SIZE(inbuf) - delaysize;
		GST_BUFFER_FLAG_SET(outbuf,GST_BUFFER_FLAG_DISCONT);

		/* never come back */
		element->delay = 0;

		result = GST_FLOW_OK;
	}
	else
	/* pass entire buffer */
	{
		result = GST_FLOW_OK;
	}
/* FIXME REMOVE DEBUG ONLY */
{
const double *x;
unsigned nans = 0;
for(x = GST_BUFFER_DATA(inbuf); x < GST_BUFFER_DATA(inbuf) + GST_BUFFER_SIZE(inbuf); x++) if(isnan(*x)) nans++;
if(nans) fprintf(stderr, "full buf %s: input %" GST_BUFFER_BOUNDARIES_FORMAT " has %u nans\n", GST_ELEMENT_NAME(element), GST_BUFFER_BOUNDARIES_ARGS(inbuf), nans);
}
{
const double *x;
unsigned nans = 0;
for(x = GST_BUFFER_DATA(outbuf); x < GST_BUFFER_DATA(outbuf) + GST_BUFFER_SIZE(outbuf); x++) if(isnan(*x)) nans++;
if(nans) fprintf(stderr, "full buf %s: output %" GST_BUFFER_BOUNDARIES_FORMAT " has %u nans\n", GST_ELEMENT_NAME(element), GST_BUFFER_BOUNDARIES_ARGS(outbuf), nans);
}

	return result;
}



/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * set_property()
 */
static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALDelay *element = GSTLAL_DELAY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id)
	{
	case ARG_DELAY:
		element->delay = g_value_get_uint64(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */
static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALDelay *element = GSTLAL_DELAY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id)
	{
	case ARG_DELAY:
	  g_value_set_uint64(value, element->delay);
	  break;

	default:
	  G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
	  break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * base_init()
 */
static void
gstlal_delay_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"Drops beginning of a stream",
		"Filter/Audio",
		"Drops beginning of a stream",
		"Stephen Privitera <sprivite@ligo.caltech.edu>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR(prepare_output_buffer);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
}


/*
 * class_init()
 */
static void gstlal_delay_class_init(GSTLALDelayClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_DELAY,
		g_param_spec_uint64(
			"delay",
			"Time delay",
			"Amount of data (in samples) to ignore at front of stream.",
			0, G_MAXUINT64, DEFAULT_DELAY,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}


/*
 * init() -- equivalent to python's __init__()
 */
static void gstlal_delay_init(GSTLALDelay *filter, GSTLALDelayClass *kclass)
{
	filter->delay = DEFAULT_DELAY;
}

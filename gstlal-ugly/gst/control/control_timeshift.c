/*
 * Copyright (C) 2015	Qi Chu	<qi.chu@uwa.edu.au>
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
#include <math.h>
#include <string.h>
/*
 *  stuff from gobject/gstreamer
*/


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>


/*
 * stuff from here
 */
#include <control_timeshift.h>
#include <time.h>
/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */

/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT control_timeshift_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "control_timeshift", 0, "control_timeshift element");
}

GST_BOILERPLATE_FULL(
	ControlTimeshift,
	control_timeshift,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM,
	additional_initializations
);

enum property {
	PROP_0,
	PROP_SHIFT
};

static void control_timeshift_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void control_timeshift_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */
static GstFlowReturn control_timeshift_prepare_output_buffer (GstBaseTransform * trans,
    GstBuffer * inbuf, gint size, GstCaps *caps, GstBuffer **outbuf);
static void control_timeshift_dispose (GObject *object);

/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * set_caps()
 */


static gboolean control_timeshift_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	ControlTimeshift *element = CONTROL_TIMESHIFT(trans);
	GstStructure *s;
	gint rate;
	gboolean success = TRUE;

	s = gst_caps_get_structure(incaps, 0);
	success &= gst_structure_get_int(s, "rate", &rate);

	if(success)
		element->rate = rate;
	else
		GST_WARNING_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);

	return success;
}




/*
 * prepare_output_buffer()
 */


static GstFlowReturn control_timeshift_prepare_output_buffer(GstBaseTransform *trans, GstBuffer *inbuf, gint size, GstCaps *caps, GstBuffer **outbuf)
{
	ControlTimeshift *element = CONTROL_TIMESHIFT(trans);
	GstFlowReturn result = GST_FLOW_OK;

	gst_buffer_ref(inbuf);
	*outbuf = inbuf;

	*outbuf = gst_buffer_make_metadata_writable(*outbuf);

	if(!*outbuf) {
			GST_ERROR_OBJECT(element, "failure creating sub-buffer");
			return GST_FLOW_ERROR;
	}


    GST_LOG_OBJECT (element,
      "Input buffer of timestamp %" GST_TIME_FORMAT 
      ", offset %" G_GUINT64_FORMAT 
      ", offset_end %" G_GUINT64_FORMAT, 
      GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (*outbuf)),
      GST_BUFFER_OFFSET (*outbuf), GST_BUFFER_OFFSET_END (*outbuf));



	GstClockTime t_cur = GST_BUFFER_TIMESTAMP(*outbuf);
	guint64 offset = GST_BUFFER_OFFSET(*outbuf);
       	guint64 offset_end = GST_BUFFER_OFFSET_END(*outbuf);

	GST_BUFFER_TIMESTAMP(*outbuf) = t_cur + 1000000000L * element->shift;
	GST_BUFFER_OFFSET(*outbuf) = offset + (int)(element->rate * element->shift);
	GST_BUFFER_OFFSET_END(*outbuf) = offset_end + (int)(element->rate * element->shift);

    GST_LOG_OBJECT (element,
      "Converted to buffer of timestamp %" GST_TIME_FORMAT 
      ", offset %" G_GUINT64_FORMAT 
      ", offset_end %" G_GUINT64_FORMAT, 
      GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (*outbuf)),
      GST_BUFFER_OFFSET (*outbuf), GST_BUFFER_OFFSET_END (*outbuf));




	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */

/* handle events (search) */
static gboolean
control_timeshift_event (GstBaseTransform * base, GstEvent * event)
{
  ControlTimeshift *element = CONTROL_TIMESHIFT(base);

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
//      if (fflush (sink->file))
//        goto flush_failed;

    GST_LOG_OBJECT(element, "EVENT EOS. Finish assign fap");
      break;
    default:
      break;
  }

  return TRUE;
}



/*
 * set_property()
 */


static void control_timeshift_set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	ControlTimeshift *element = CONTROL_TIMESHIFT(object);

	GST_OBJECT_LOCK(element);
	switch(prop_id) {

		case PROP_SHIFT:
			element->shift = g_value_get_float(value);
			break;

	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void control_timeshift_get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	ControlTimeshift *element = CONTROL_TIMESHIFT(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {

		case PROP_SHIFT:
			g_value_set_float(value, element->shift);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
			break;
	}
	GST_OBJECT_UNLOCK(element);
}


/*
 * dispose()
 */


static void control_timeshift_dispose(GObject *object)
{
	G_OBJECT_CLASS(parent_class)->dispose(object);
}


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
        GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64}"
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
		"width = (int) {32, 64}"
	)
);


/*
 * base_init()
 */


static void control_timeshift_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"time shift the stream",
		"delay stream",
		"Delay stream.\n",
		"Qi Chu <qi.chu at ligo dot org>"
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&src_factory)
	
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&sink_factory)
	);

	transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR(control_timeshift_prepare_output_buffer);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(control_timeshift_set_caps);
	transform_class->event = GST_DEBUG_FUNCPTR(control_timeshift_event);

}


/*
 * class_init()
 */


static void control_timeshift_class_init(ControlTimeshiftClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
;
	gobject_class->set_property = GST_DEBUG_FUNCPTR(control_timeshift_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(control_timeshift_get_property);

	g_object_class_install_property(
		gobject_class,
		PROP_SHIFT,
		g_param_spec_float(
			"shift",
			"time shift",
			"(0) no shift; (N) delay by N seconds. ",
			-G_MAXFLOAT, G_MAXFLOAT, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}
/*
 * init()
 */


static void control_timeshift_init(ControlTimeshift *element, ControlTimeshiftClass *kclass)
{
	GST_BASE_TRANSFORM(element)->always_in_place = TRUE;
}

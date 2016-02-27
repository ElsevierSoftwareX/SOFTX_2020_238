/*
 * Fix broken discontinuity flags
 *
 * Copyright (C) 2009--2013  Kipp Cannon
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


/**
 * SECTION:gstlal_nofakedisconts
 * @short_description:  Fix broken GST_BUFFER_FLAG_DISCONT flags.
 *
 * The GStreamer base class #GstBaseTransform requires one (or more) output
 * buffers to be produced by subclasses for every input buffer received.
 * If the subclass fails to produce an output buffer this is remembered
 * by #GstBaseTransform which then sets the #GST_BUFFER_FLAG_DISCONT flag
 * on the next output buffer, regardless of whether or not the buffer is,
 * infact, discontinuous with any previous buffer.  This is annoying, and
 * since other elements often respond to the discontinuity by resetting
 * themselves it creates problems.  This bug aflicts several stock elements
 * such as the audioresampler.  To work around the problem, this element is
 * available.  This element monitors the data stream, watching timestamps
 * and offsets, and sets or clears the discontinuity flag on each buffer
 * based on exactly whether it is discontinuous with the previous buffer.
 *
 * Reviewed:  3d2cf9ea32085a2dd4854cb71b1cbaaf5547fc57 2014-08-12 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 */


/*
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <gstlal_nofakedisconts.h>


/*
 * parameters
 */


#define DEFAULT_SILENT FALSE


/*
 * ========================================================================
 *
 *                                 Properties
 *
 * ========================================================================
 */


enum property {
	ARG_SILENT = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALNoFakeDisconts *element = GSTLAL_NOFAKEDISCONTS(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SILENT:
		element->silent = g_value_get_boolean(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALNoFakeDisconts *element = GSTLAL_NOFAKEDISCONTS(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SILENT:
		g_value_set_boolean(value, element->silent);
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
 *                                    Pads
 *
 * ============================================================================
 */


/*
 * getcaps()
 */


//static GstCaps *getcaps(GstPad * pad)
//{
//    GSTLALNoFakeDisconts *element = GSTLAL_NOFAKEDISCONTS(gst_pad_get_parent(pad));
//    GstPad *otherpad = pad == element->srcpad ? element->sinkpad : element->srcpad;
//    GstCaps *peercaps, *caps;
//
//    /*
//     * get our own allowed caps.  use the fixed caps function to avoid
//     * recursing back into this function.
//     */
//
//    /* FIXME- AEP 02242016
//     * Replacing this function, likely to break. */
//    //caps = gst_pad_get_fixed_caps_func(pad);
//    caps = gst_caps_make_writable(pad);
//
//    /*
//     * get the allowed caps from the downstream peer if the peer has
//     * caps, intersect without our own.
//     */
//
//    /* FIXME- AEP 02252016
//     * Reference manual is abiguous on this function. */
//    peercaps = gst_caps_make_writable(otherpad);
//    if(peercaps) {
//        GstCaps *result = gst_caps_intersect(peercaps, caps);
//        gst_caps_unref(peercaps);
//        gst_caps_unref(caps);
//        caps = result;
//    }
//
//    /*
//     * done
//     */
//
//    gst_object_unref(element);
//    return caps;
//}
static GstCaps *drop_sink_getcaps (GstPad * pad, GstCaps * filter)
{
    GSTLALNoFakeDisconts *nofakedisconts;
    GstCaps *result, *peercaps, *current_caps, *filter_caps;
    nofakedisconts = GSTLAL_NOFAKEDISCONTS(GST_PAD_PARENT (pad));
    
    /* take filter */
    filter_caps = filter ? gst_caps_ref(filter) : NULL;
    
    /*
     * If the filter caps are empty (but not NULL), there is nothing we can
     * do, there will be no intersection
     */
    if (filter_caps && gst_caps_is_empty (filter_caps)) {
        GST_WARNING_OBJECT (pad, "Empty filter caps");
        return filter_caps;
    }
    
    /* get the downstream possible caps */
    peercaps = gst_pad_peer_query_caps(nofakedisconts->srcpad, filter_caps);
    
    /* get the allowed caps on this sinkpad */
    current_caps = gst_pad_get_pad_template_caps(pad);
    if (!current_caps)
        current_caps = gst_caps_new_any();
    
    if (peercaps) {
        /* if the peer has caps, intersect */
        GST_DEBUG_OBJECT(nofakedisconts, "intersecting peer and our caps");
        result = gst_caps_intersect_full(peercaps, current_caps, GST_CAPS_INTERSECT_FIRST);
        /* neither peercaps nor current_caps are needed any more */
        gst_caps_unref(peercaps);
        gst_caps_unref(current_caps);
    }
    else {
        /* the peer has no caps (or there is no peer), just use the allowed caps
         * of this sinkpad. */
        /* restrict with filter-caps if any */
        if (filter_caps) {
            GST_DEBUG_OBJECT(nofakedisconts, "no peer caps, using filtered caps");
            result = gst_caps_intersect_full(filter_caps, current_caps, GST_CAPS_INTERSECT_FIRST);
            /* current_caps are not needed any more */
            gst_caps_unref(current_caps);
        }
        else {
            GST_DEBUG_OBJECT(nofakedisconts, "no peer caps, using our caps");
            result = current_caps;
        }
    }
    
    result = gst_caps_make_writable (result);
    
    if (filter_caps)
        gst_caps_unref (filter_caps);
    
    GST_LOG_OBJECT (nofakedisconts, "getting caps on pad %p,%s to %" GST_PTR_FORMAT, pad, GST_PAD_NAME(pad), result);
    
    return result;
}


/*
 * acceptcaps()
 */


static gboolean drop_sink_setcaps (GSTLALNoFakeDisconts *nofakedisconts, GstPad *pad, GstCaps *caps)
{
    
    GstStructure *structure;
    gint rate, width, channels;
    gboolean success = TRUE;
    
    /*
     * parse caps
     */
    
    structure = gst_caps_get_structure(caps, 0);
    success &= gst_structure_get_int(structure, "rate", &rate);
    success &= gst_structure_get_int(structure, "width", &width);
    success &= gst_structure_get_int(structure, "channels", &channels);
    
    /*
     * try setting caps on downstream element
     */
    
    if(success)
        success = gst_pad_set_caps(nofakedisconts->srcpad, caps);
    
    /*
     * update the element metadata
     */
    
    if(success) {
        nofakedisconts->rate = rate;
        nofakedisconts->unit_size = width / 8 * channels;
    } else
        GST_ERROR_OBJECT(nofakedisconts, "unable to parse and/or accept caps %" GST_PTR_FORMAT, caps);
    
    /*
     * done
     */
    
    return success;
    
}


//static gboolean acceptcaps(GstPad *pad, GstCaps *caps)
//{
//	GSTLALNoFakeDisconts *element = GSTLAL_NOFAKEDISCONTS(gst_pad_get_parent(pad));
//	GstPad *otherpad = pad == element->srcpad ? element->sinkpad : element->srcpad;
//	gboolean success;
//
//	/*
//	 * ask downstream peer
//	 */
//
//	//success = gst_pad_peer_accept_caps(otherpad, caps);
//	success = gst_pad_peer_query_accept_caps(otherpad, caps);
//
//	/*
//	 * done
//	 */
//
//	gst_object_unref(element);
//	return success;
//}



/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
	GSTLALNoFakeDisconts *element = GSTLAL_NOFAKEDISCONTS(gst_pad_get_parent(pad));
	GstFlowReturn result;

	if(element->next_offset != GST_BUFFER_OFFSET_NONE) {
		gboolean is_discont = GST_BUFFER_IS_DISCONT(buf);

		if(GST_BUFFER_OFFSET(buf) != element->next_offset || GST_BUFFER_TIMESTAMP(buf) != element->next_timestamp) {
			if(!is_discont) {
				//buf = gst_buffer_make_metadata_writable(buf);
				buf = gst_buffer_make_writable(buf);
				GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
				if(!element->silent)
					fprintf(stderr, "%s: set missing discontinuity flag at %" GST_TIME_SECONDS_FORMAT "\n", gst_element_get_name(element), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(buf)));
			}
		} else {
			if(is_discont) {
				//buf = gst_buffer_make_metadata_writable(buf);
				buf = gst_buffer_make_writable(buf);
				GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_DISCONT);
				if(!element->silent)
					fprintf(stderr, "%s: cleared improper discontinuity flag at %" GST_TIME_SECONDS_FORMAT "\n", gst_element_get_name(element), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(buf)));
			}
		}
	}

	element->next_offset = GST_BUFFER_OFFSET_END(buf);
	element->next_timestamp = GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf);

	result = gst_pad_push(element->srcpad, buf);

	return result;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GSTLALNoFakeDisconts *element = GSTLAL_NOFAKEDISCONTS(object);

	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void class_init(gpointer klass, gpointer class_data)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Discontinuity flag fix",
		"Filter",
		"Fix incorrectly-set discontinuity flags",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"ANY"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"ANY"
			)
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_SILENT,
		g_param_spec_boolean(
			"silent",
			"Silent",
			"Don't print a message when alterning the flags in a buffer.",
			DEFAULT_SILENT,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}

/* AEP -02252016 adding gbooleans */
//drop_src_query
static gboolean drop_src_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
	gboolean res = FALSE;

	switch (GST_QUERY_TYPE (query))
	{
		default:
			res = gst_pad_query_default (pad, parent, query);
			break;
	}
	return res;
}


//drop_src_event
static gboolean drop_src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALNoFakeDisconts *nofakedisconts;
	gboolean result = TRUE;
	nofakedisconts = GSTLAL_NOFAKEDISCONTS(parent);
	GST_DEBUG_OBJECT (pad, "Got %s event on src pad", GST_EVENT_TYPE_NAME(event));

	switch (GST_EVENT_TYPE (event))
	{
		default:
			/* just forward the rest for now */
			GST_DEBUG_OBJECT(nofakedisconts, "forward unhandled event: %s", GST_EVENT_TYPE_NAME (event));
			gst_pad_event_default(pad, parent, event);
			break;
	}

	return result;
}

//drop_sink_query
static gboolean drop_sink_query(GstPad *pad, GstObject *parent, GstQuery * query)
{
	gboolean res = TRUE;
	GstCaps *filter, *caps;

	switch (GST_QUERY_TYPE (query))
	{
		case GST_QUERY_CAPS:
			gst_query_parse_caps (query, &filter);
			caps = drop_sink_getcaps (pad, filter);
			gst_query_set_caps_result (query, caps);
			gst_caps_unref (caps);
			break;
		default:
			break;
	}

	 if (G_LIKELY (query))
		return gst_pad_query_default (pad, parent, query);
	 else
		 return res;
  return res;
}

//drop_sink_event
static gboolean drop_sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALNoFakeDisconts *nofakedisconts = GSTLAL_NOFAKEDISCONTS(parent);
	gboolean res = TRUE;
        GstCaps *caps;

	GST_DEBUG_OBJECT(pad, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch (GST_EVENT_TYPE (event))
	{
		case GST_EVENT_CAPS:
			gst_event_parse_caps(event, &caps);
			res = drop_sink_setcaps(nofakedisconts, pad, caps);
			gst_event_unref(event);
	                event = NULL;
		default:
			break;
	}
	if (G_LIKELY (event))
		return gst_pad_event_default(pad, parent, event);
	else
		return res;
}

/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */

static void instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALNoFakeDisconts *element = GSTLAL_NOFAKEDISCONTS(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	//gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps));
	//gst_pad_set_acceptcaps_function(pad, GST_DEBUG_FUNCPTR(acceptcaps));
	gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(drop_sink_query));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(drop_sink_event));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	//gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps));
	//gst_pad_set_acceptcaps_function(pad, GST_DEBUG_FUNCPTR(acceptcaps));
	gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR (drop_src_query));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR (drop_src_event));
	element->srcpad = pad;

	/* internal data */
	element->next_offset = GST_BUFFER_OFFSET_NONE;
	element->next_timestamp = GST_CLOCK_TIME_NONE;
	element->silent = DEFAULT_SILENT;
}


/*
 * gstlal_nofakedisconts_get_type().
 */


GType gstlal_nofakedisconts_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALNoFakeDiscontsClass),
			.class_init = class_init,
			.instance_size = sizeof(GSTLALNoFakeDisconts),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALNoFakeDisconts", &info, 0);
	}

	return type;
}

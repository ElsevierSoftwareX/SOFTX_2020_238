/*
 * NDS2-based frame src element
 *
 * Copyright (C) 2008  Leo Singer
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
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <math.h>
#include <stdint.h>
#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_nds2_framesrc.h>


/*
 * Parent class.
 */


static GstBaseSrcClass *parent_class = NULL;


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


static const char* DEFAULT_HOST = "marble.ligo-wa.caltech.edu";
static const int DEFAULT_PORT = 31200;
static const char* DEFAULT_REQUESTED_CHANNEL_NAME = "H1:DMT-STRAIN";


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


// Disconnect from NDS2 server and free daq and daq_channel resources.
static void disconnect_and_free_daq(GSTLALNDS2FrameSrc* element)
{
    if (element->daq)
    {
        {
            int retval = daq_disconnect(element->daq);
            if (retval)
                GST_ERROR_OBJECT(element, "daq_disconnect: %d", retval);
        }
        free(element->daq);
        element->daq = NULL;
    }
    
    if (element->daq_channel)
    {
        free(element->daq_channel);
        element->daq_channel = NULL;
    }
}


// Connect to NDS2 server.
static int connect_daq(GSTLALNDS2FrameSrc* element)
{
    if (!element->daq)
    {
        daq_t* daq = malloc(sizeof(daq_t));
        if (!daq)
        {
            GST_ERROR_OBJECT(element, "malloc");
            return FALSE;
        }
        
        int retval = daq_connect(daq, element->host, element->port, nds_v2);
        if (retval)
        {
            free(daq);
            GST_ERROR_OBJECT(element, "daq_connect: error %d", retval);
            return FALSE;
        }
        
        element->daq = daq;
    }
    return TRUE;
}



/*
 * Set channel by looking up channelname on NDS2 channel list.
 */

static int set_channel_for_channelname(GSTLALNDS2FrameSrc *element)
{
    daq_channel_t* channels = calloc(MAX_CHANNELS, sizeof(daq_channel_t));
    if (!channels)
    {
        GST_ERROR_OBJECT(element, "malloc");
        return FALSE;
    }
    
    int nchannels_received;
    int retval = daq_recv_channel_list(element->daq, channels, MAX_CHANNELS, &nchannels_received, 0, cOnline);
    if (retval)
    {
        free(channels);
        GST_ERROR_OBJECT(element, "daq_recv_channels: error %d", retval);
        return FALSE;
    }
    
    daq_channel_t* channel;
    for (channel = channels; channel < &channels[nchannels_received]; channel++)
        if (!strcmp(channel->name, element->requested_channel_name))
        {
            daq_channel_t* found_channel = malloc(sizeof(daq_channel_t));
            if (!found_channel)
            {
                free(channels);
                GST_ERROR_OBJECT(element, "malloc");
                return FALSE;
            }
            
            *found_channel = *channel;
            free(channels);
            element->daq_channel = found_channel;
            return TRUE;
        }
    
    GST_ERROR_OBJECT(element, "channel not found: %s", element->requested_channel_name);
    return FALSE;
}


/*
 * Convert a generic time series to the matching GstCaps.  Returns the
 * output of gst_caps_new_simple() --- ref()ed caps that need to be
 * unref()ed when done.
 */


static GstCaps *caps_for_channel(GSTLALNDS2FrameSrc* element)
{
	GstCaps *caps = NULL;

	switch(element->daq_channel->data_type) {
	case _16bit_integer:
		caps = gst_caps_new_simple(
			"audio/x-raw-int",
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 16,
			"depth", G_TYPE_INT, 16,
			"signed", G_TYPE_BOOLEAN, TRUE,
			NULL
		);
		break;

	case _32bit_integer:
		caps = gst_caps_new_simple(
			"audio/x-raw-int",
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 32,
			"depth", G_TYPE_INT, 32,
			"signed", G_TYPE_BOOLEAN, TRUE,
			NULL
		);
		break;

	case _64bit_integer:
		caps = gst_caps_new_simple(
			"audio/x-raw-int",
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 64,
			"depth", G_TYPE_INT, 64,
			"signed", G_TYPE_BOOLEAN, TRUE,
			NULL
		);
		break;

	case _32bit_float:
		caps = gst_caps_new_simple(
			"audio/x-raw-float",
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 32,
			NULL
		);
		break;

	case _64bit_double:
		caps = gst_caps_new_simple(
			"audio/x-raw-float",
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 64,
			NULL
		);
		break;
    
    // TODO: there is one more NDS2 daq datatype: _32bit_complex.  Should this
    // be a two-channel audio/x-raw-float?

	default:
		GST_ERROR_OBJECT(element, "unsupported NDS2 data_type: %d", element->daq_channel->data_type);
		break;
	}
    
    if (caps)
        gst_caps_set_simple(caps,
            "rate", G_TYPE_INT, (int)(element->daq_channel->rate),
            NULL);    
    
    return caps;
}



/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
    ARG_SRC_HOST = 1,
	ARG_SRC_REQUESTED_CHANNEL_NAME
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
    
	case ARG_SRC_HOST:
		g_free(element->host);
		element->host = g_value_dup_string(value);
		break;

	case ARG_SRC_REQUESTED_CHANNEL_NAME:
		g_free(element->requested_channel_name);
		element->requested_channel_name = g_value_dup_string(value);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SRC_HOST:
		g_value_set_string(value, element->host);
		break;
    
	case ARG_SRC_REQUESTED_CHANNEL_NAME:
		g_value_set_string(value, element->requested_channel_name);
		break;

	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                        GstBaseSrc Method Overrides
 *
 * ============================================================================
 */


/*
 * start()
 */

static gboolean start(GstBaseSrc *object)
{
	GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(object);
    
    // Connect to NDS2 server.
    if (!connect_daq(element))
        return FALSE;
    
    // Select channel using requested_channel_name property.
    if (!set_channel_for_channelname(element))
    {
        disconnect_and_free_daq(element);
        return FALSE;
    }
    
    // Set channel.
    if (!set_channel_for_channelname(element))
    {
        disconnect_and_free_daq(element);
        return FALSE;
    }
    
    // Request channel.
    {
        int retval = daq_request_channel_from_chanlist(element->daq, element->daq_channel);
        if (retval)
        {
            disconnect_and_free_daq(element);
            GST_ERROR_OBJECT(element, "daq_request_channel_from_chanlist: error %d", retval);
            return FALSE;
        }
    }
    
    // Request online data.
    {
        int retval = daq_request_data(element->daq, 0, 0, 1);
        if (retval)
        {
            disconnect_and_free_daq(element);
            GST_ERROR_OBJECT(element, "daq_request_online: error %d", retval);
            return FALSE;
        }
    }
    
    {
        GstCaps* caps = caps_for_channel(element);
        if(!caps) {
            disconnect_and_free_daq(element);
            GST_ERROR_OBJECT(element, "unable to construct caps");
            return FALSE;
        }
        
        /*
         * Try setting the caps on the source pad.
         */

        if(!gst_pad_set_caps(GST_BASE_SRC_PAD(object), caps)) {
            gst_caps_unref(caps);
            disconnect_and_free_daq(element);
            GST_ERROR_OBJECT(element, "unable to set caps %" GST_PTR_FORMAT " on %s", caps, GST_PAD_NAME(GST_BASE_SRC_PAD(object)));
            return FALSE;
        }
        gst_caps_unref(caps);

        /*
         * Transmit the tag list.
         */
        
        GstTagList* taglist = gst_tag_list_new_full(
            GSTLAL_TAG_CHANNEL_NAME, element->daq_channel->name,
            NULL);
        if (!taglist)
        {
            disconnect_and_free_daq(element);
            GST_ERROR_OBJECT(element, "unable to create taglist");
            return FALSE;
        }
        
        if (!gst_pad_push_event(GST_BASE_SRC_PAD(object), gst_event_new_tag(taglist)))
        {
            gst_tag_list_free(taglist);
            disconnect_and_free_daq(element);
            GST_ERROR_OBJECT(element, "unable to push taglist %" GST_PTR_FORMAT " on %s", taglist, GST_PAD_NAME(GST_BASE_SRC_PAD(object)));
            return FALSE;
        }

        return TRUE;
    }
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
	GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(object);
    
    disconnect_and_free_daq(element);
    
	return TRUE;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
    GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(basesrc);
    
    {
        int retval = daq_recv_next(element->daq);
        if (retval)
        {
            GST_ERROR_OBJECT(element, "daq_recv_next: error %d", retval);
            return GST_FLOW_ERROR;
        }
    }
    
    int data_length = daq_get_data_length(element->daq, element->daq_channel->name);
    
    {
    GstFlowReturn result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, data_length, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
    if (result != GST_FLOW_OK)
        return result;
    }
    
    if (!daq_get_channel_data(element->daq, element->daq_channel->name, (char*)GST_BUFFER_DATA(*buffer)))
    {
        GST_ERROR_OBJECT(element, "daq_get_channel_data: error");
        return GST_FLOW_ERROR;
    }
    
    return GST_FLOW_OK;
}


/*
 * is_seekable()
 */


static gboolean is_seekable(GstBaseSrc *object)
{
	return FALSE;
}


/*
 * do_seek()
 */


// Cannot seek with an online source
#if 0
static gboolean do_seek(GstBaseSrc *basesrc, GstSegment *segment)
{
	GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(basesrc);
	LIGOTimeGPS epoch;

	/*
	 * Parse the segment
	 */

	if((GstClockTime) segment->start == GST_CLOCK_TIME_NONE) {
		GST_ERROR_OBJECT(element, "seek failed:  start time is required");
		return FALSE;
	}
	XLALINT8NSToGPS(&epoch, segment->start);

	/*
	 * Try doing the seek
	 */

	if(XLALFrSeek(element->stream, &epoch)) {
		GST_ERROR_OBJECT(element, "XLALFrSeek() to %d.%09u s failed: %s", epoch.gpsSeconds, epoch.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
		/* FIXME:  can't return error or gstreamer stops working (!?) */
		/*return FALSE;*/
		GST_FIXME_OBJECT(element, "ignoring previous XLALFrSeek() failure (FIXME)");
	}

	/*
	 * Done
	 */

	basesrc->offset = 0;
	return TRUE;
}
#endif


/*
 * query: what is this for?
 */



/*
static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
    GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(basesrc);

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_FORMATS:
		gst_query_set_formats(query, 5, GST_FORMAT_DEFAULT, GST_FORMAT_BYTES, GST_FORMAT_TIME, GST_FORMAT_BUFFERS, GST_FORMAT_PERCENT);
		break;

	case GST_QUERY_CONVERT: {
		GstFormat src_format, dest_format;
		gint64 src_value, dest_value;
		guint64 timestamp;

		gst_query_parse_convert(query, &src_format, &src_value, &dest_format, &dest_value);

		switch(src_format) {
		case GST_FORMAT_DEFAULT:
		case GST_FORMAT_TIME:
			timestamp = src_value;
			break;

		case GST_FORMAT_BYTES:
			timestamp = basesrc->segment.start + gst_util_uint64_scale_int_round(src_value, GST_SECOND, element->width / 8 * element->rate);
			break;

		case GST_FORMAT_BUFFERS:
			timestamp = basesrc->segment.start + gst_util_uint64_scale_int_round(src_value, gst_base_src_get_blocksize(basesrc) * GST_SECOND, element->width / 8 * element->rate);
			break;

		case GST_FORMAT_PERCENT:
			timestamp = basesrc->segment.start + gst_util_uint64_scale_int_round(basesrc->segment.stop - basesrc->segment.start, src_value, 100);
			break;

		default:
			g_assert_not_reached();
			return FALSE;
		}
		switch(dest_format) {
		case GST_FORMAT_DEFAULT:
		case GST_FORMAT_TIME:
			dest_value = timestamp;
			break;

		case GST_FORMAT_BYTES:
			dest_value = gst_util_uint64_scale_int_round(timestamp - basesrc->segment.start, element->width / 8 * element->rate, GST_SECOND);
			break;

		case GST_FORMAT_BUFFERS:
			dest_value = gst_util_uint64_scale_int_round(timestamp - basesrc->segment.start, element->width / 8 * element->rate, gst_base_src_get_blocksize(basesrc) * GST_SECOND);
			break;

		case GST_FORMAT_PERCENT:
			dest_value = gst_util_uint64_scale_int_round(timestamp - basesrc->segment.start, 100, basesrc->segment.stop - basesrc->segment.start);
			break;

		default:
			g_assert_not_reached();
			return FALSE;
		}

		gst_query_set_convert(query, src_format, src_value, dest_format, dest_value);

		break;
	}

	default:
		return parent_class->query(basesrc, query);
	}

	return TRUE;
}
*/




/*
 * check_get_range()
 */


static gboolean check_get_range(GstBaseSrc *basesrc)
{
	return TRUE;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(object);

	g_free(element->host);
    element->host = NULL;
    g_free(element->requested_channel_name);
    element->requested_channel_name = NULL;
    disconnect_and_free_daq(element);

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static const GstElementDetails plugin_details = {
		"NDS2 Frame Source",
		"Source",
		"NDS2-based frame src element",
		"Leo Singer <leo.singer@ligo.org>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) [1, 2], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64}; " \
				"audio/x-raw-int, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {16, 32, 64}, " \
				"depth = (int) {16, 32, 64}, " \
				"signed = (boolean) true"
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);
	GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_BASE_SRC);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_SRC_HOST,
		g_param_spec_string(
			"nds2-host",
			"Host",
			"NDS2 remote host domain name or IP address",
			DEFAULT_HOST,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_SRC_REQUESTED_CHANNEL_NAME,
		g_param_spec_string(
			"requested-channel-name",
			"Requested channel name",
			"Name of the desired NDS2 channel.",
			DEFAULT_REQUESTED_CHANNEL_NAME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	/*
	 * GstBaseSrc method overrides
	 */

	gstbasesrc_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesrc_class->stop = GST_DEBUG_FUNCPTR(stop);
	gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
	gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR(is_seekable);
	//gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
	//gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);
	gstbasesrc_class->check_get_range = GST_DEBUG_FUNCPTR(check_get_range);
    
    // Start up NDS2
    daq_startup();
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(object);
	GSTLALNDS2FrameSrc *element = GSTLAL_NDS2_FRAMESRC(object);

	gst_pad_use_fixed_caps(GST_BASE_SRC_PAD(basesrc));

    element->host = g_malloc(strlen(DEFAULT_HOST)+1);
    strcpy(element->host, DEFAULT_HOST);
    
    element->port = DEFAULT_PORT;
    
	element->requested_channel_name = g_malloc(strlen(DEFAULT_REQUESTED_CHANNEL_NAME));
    strcpy(element->requested_channel_name, DEFAULT_REQUESTED_CHANNEL_NAME);
    
    element->daq = NULL;
    element->daq_channel = NULL;

	gst_base_src_set_format(GST_BASE_SRC(object), GST_FORMAT_TIME);
    gst_base_src_set_live(GST_BASE_SRC(object), TRUE);
}


/*
 * gstlal_framesrc_get_type().
 */


GType gstlal_nds2_framesrc_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALNDS2FrameSrcClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALNDS2FrameSrc),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_BASE_SRC, "lal_nds2_framesrc", &info, 0);
	}

	return type;
}

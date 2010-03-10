/*
 * NDS-based src element
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
#include <gstlal_ndssrc.h>
#include <daqc_internal.h>
#include <daqc_response.h>


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


// Can J. Zweizig add this to NDS library?
static const char* daq_strerror(int errornum)
{
    switch (errornum)
    {
        case DAQD_OK:
            return "OK";
        case DAQD_ERROR:
            return "unspecified error";
        case DAQD_NOT_CONFIGURED:
            return "not configured";
        case DAQD_INVALID_IP_ADDRESS:
            return "invalid IP address";
        case DAQD_INVALID_CHANNEL_NAME:
            return "invalid channel name";
        case DAQD_SOCKET:
            return "socket";
        case DAQD_SETSOCKOPT:
            return "setsockopt";
        case DAQD_CONNECT:
            return "connect";
        case DAQD_BUSY:
            return "busy";
        case DAQD_MALLOC:
            return "malloc";
        case DAQD_WRITE:
            return "write";
        case DAQD_VERSION_MISMATCH:
            return "version mismatch";
        case DAQD_NO_SUCH_NET_WRITER:
            return "no such net writer";
        case DAQD_NOT_FOUND:
            return "not found";
        case DAQD_GETPEERNAME:
            return "getpeername";
        case DAQD_DUP:
            return "dup";
        case DAQD_INVALID_CHANNEL_DATA_RATE:
            return "invalid channel data rate";
        case DAQD_SHUTDOWN:
            return "shutdown";
        case DAQD_NO_TRENDER:
            return "no trender";
        case DAQD_NO_MAIN:
            return "no main";
        case DAQD_NO_OFFLINE:
            return "no offline";
        case DAQD_THREAD_CREATE:
            return "thread create";
        case DAQD_TOO_MANY_CHANNELS:
            return "too many channels";
        case DAQD_COMMAND_SYNTAX:
            return "command syntax";
        case DAQD_SASL:
            return "sasl";
        case DAQD_NOT_SUPPORTED:
            return "not supported";
        default:
            return "unknown error";
    }
}


#define DAQ_GST_ERROR_OBJECT(element, msg, errnum) GST_ERROR_OBJECT((element), "%s: error %d: %s", (msg), (errnum), daq_strerror(errnum))


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */



// Disconnect from NDS server and free daq and daq_channel resources.
static void disconnect_and_free_daq(GSTLALNDSSrc* element)
{
    if (element->daq)
    {
        {
            GST_INFO_OBJECT(element, "daq_disconnect");
            int retval = daq_disconnect(element->daq);
            if (retval)
                DAQ_GST_ERROR_OBJECT(element, "daq_disconnect", retval);
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


// Connect to NDS server.
static int connect_daq(GSTLALNDSSrc* element)
{
    if (!element->daq)
    {
        daq_t* daq = malloc(sizeof(daq_t));
        if (!daq)
        {
            GST_ERROR_OBJECT(element, "malloc");
            return FALSE;
        }

        GST_INFO_OBJECT(element, "daq_connect: connecting to %s:%d", element->host, element->port);
        int retval = daq_connect(daq, element->host, element->port, nds_v2);
        if (retval)
        {
            free(daq);
            DAQ_GST_ERROR_OBJECT(element, "daq_connect", retval);
            return FALSE;
        }

        element->daq = daq;
    }
    return TRUE;
}



/*
 * Set channel by looking up channelname on NDS channel list.
 */

static int set_channel_for_channelname(GSTLALNDSSrc *element)
{
    daq_channel_t* channels = calloc(MAX_CHANNELS, sizeof(daq_channel_t));
    if (!channels)
    {
        GST_ERROR_OBJECT(element, "malloc");
        return FALSE;
    }

    int nchannels_received;

    GST_INFO_OBJECT(element, "daq_recv_channel_list");
    int retval = daq_recv_channel_list(element->daq, channels, MAX_CHANNELS, &nchannels_received, 0, cUnknown);
    if (retval)
    {
        free(channels);
        DAQ_GST_ERROR_OBJECT(element, "daq_recv_channels", retval);
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


static GstCaps *caps_for_channel(GSTLALNDSSrc* element)
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

    // TODO: there is one more NDS daq datatype: _32bit_complex.  Should this
    // be a two-channel audio/x-raw-float?

	default:
		GST_ERROR_OBJECT(element, "unsupported NDS data_type: %d", element->daq_channel->data_type);
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
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

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
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

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
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

    // Connect to NDS server.
    if (!connect_daq(element))
        return FALSE;

    // Select channel using requested_channel_name property.
    if (!set_channel_for_channelname(element))
    {
        disconnect_and_free_daq(element);
        return FALSE;
    }

    // Request channel.
    {
        GST_INFO_OBJECT(element, "daq_request_channel_from_chanlist");
        int retval = daq_request_channel_from_chanlist(element->daq, element->daq_channel);
        if (retval)
        {
            disconnect_and_free_daq(element);
            DAQ_GST_ERROR_OBJECT(element, "daq_request_channel_from_chanlist", retval);
            return FALSE;
        }
    }

    // Request data.
    {
        GST_INFO_OBJECT(element, "daq_request_data");
        int retval = 0;
        if (element->daq_channel->type == cOnline)
        {
            gst_base_src_set_live(object, TRUE);
            retval = daq_request_data(element->daq, 0, 0, 1);
        } else {
            gst_base_src_set_live(object, FALSE);
            retval = daq_request_data(element->daq, 
                gst_util_uint64_scale_int(object->segment.start, 1, GST_SECOND),
                gst_util_uint64_scale_int_ceil(object->segment.stop, 1, GST_SECOND),
                1);
            retval = daq_request_data(element->daq, 952146958, 952146958+60, 60);
        }
        
        if (retval)
        {
            disconnect_and_free_daq(element);
            DAQ_GST_ERROR_OBJECT(element, "daq_request_data", retval);
            return FALSE;
        }
        element->buffer_waiting = TRUE;
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
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

    disconnect_and_free_daq(element);

	return TRUE;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
    GSTLALNDSSrc *element = GSTLAL_NDSSRC(basesrc);

    if (!element->buffer_waiting) {
        GST_INFO_OBJECT(element, "daq_recv_next");
        int retval = daq_recv_next(element->daq);
        if (retval)
        {
            DAQ_GST_ERROR_OBJECT(element, "daq_recv_next", retval);
            return GST_FLOW_ERROR;
        }
    }

    int bytes_per_sample;
    int data_length;
    int rate;
    GST_INFO_OBJECT(element, "daq_get_channel_status");
    {
        chan_req_t* chan_req = daq_get_channel_status(element->daq, element->daq_channel->name);
        if (!chan_req)
        {
            GST_ERROR_OBJECT(element, "daq_get_channel_status: error");
            return GST_FLOW_ERROR;
        }
        bytes_per_sample = data_type_size(chan_req->data_type);
        data_length = chan_req->status;
        rate = chan_req->rate;
        if (chan_req->rate != (double)rate)
        {
            GST_ERROR_OBJECT(element, "non-integer sample rate not supported (%f != %d)", chan_req->rate, rate);
            return GST_FLOW_ERROR;
        }
        if (data_length < 0)
        {
            DAQ_GST_ERROR_OBJECT(element, "daq channel status", -data_length);
            return GST_FLOW_ERROR;
        }
        if (data_length % bytes_per_sample != 0)
        {
            GST_ERROR_OBJECT(element, "daq buffer length is not multiple of data type length");
            return GST_FLOW_ERROR;
        }
    }

    {
        GstFlowReturn result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, data_length, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
        if (result != GST_FLOW_OK)
            return result;
    }

    GST_INFO_OBJECT(element, "daq_get_channel_data");
    if (!daq_get_channel_data(element->daq, element->daq_channel->name, (char*)GST_BUFFER_DATA(*buffer)))
    {
        gst_buffer_unref(*buffer);
        *buffer = NULL;
        GST_ERROR_OBJECT(element, "daq_get_channel_data: error");
        return GST_FLOW_ERROR;
    }

    // TODO: Ask John Zweizig how to get timestamp and duration of block; this
    // struct is part of an obsolete interface according to Doxygen documentation
    guint64 nsamples = data_length / bytes_per_sample;
    basesrc->offset += nsamples;
    GST_BUFFER_OFFSET_END(*buffer) = basesrc->offset;
    GST_BUFFER_TIMESTAMP(*buffer) = GST_SECOND * element->daq->tb->gps + element->daq->tb->gpsn;
    GST_BUFFER_DURATION(*buffer) = GST_SECOND * nsamples / rate;

    element->buffer_waiting = FALSE;
    return GST_FLOW_OK;
}


/*
 * is_seekable()
 */


static gboolean is_seekable(GstBaseSrc *basesrc)
{
    return FALSE;
    /*
    GSTLALNDSSrc *element = GSTLAL_NDSSRC(basesrc);
    
    // If no NDS connection exists, assume that the channel is not seekable.
    if (element->daq)
    {
        chan_req_t* chan_req = daq_get_channel_status(element->daq, element->daq_channel->name);
        if (chan_req)
        {
            if (chan_req->type == cOnline)
            {
                return TRUE;
            }
        } else {
            GST_ERROR_OBJECT(element, "daq_get_channel_status: error");
        }
    } else {
        GST_WARNING_OBJECT(element, "assuming channel is not seekable because NDS is not connected");
    }
    
    return FALSE;
     */
}



/*
 * do_seek()
 */


static gboolean do_seek(GstBaseSrc *basesrc, GstSegment* segment)
{
    /*
    GSTLALNDSSrc *element = GSTLAL_NDSSRC(basesrc);
    
    if (element->daq)
    {
        chan_req_t* chan_req = daq_get_channel_status(element->daq, element->daq_channel->name);
        if (chan_req)
        {
            if (chan_req->type == cOnline)
            {
                GST_ERROR_OBJECT(element, "channel is 'online', seek not allowed");
                return FALSE;
            }
        } else {
            GST_ERROR_OBJECT(element, "daq_get_channel_status: error");
            return FALSE;
        }
        
        int retval = daq_request_data(element->daq, 
                                  gst_util_uint64_scale_int(segment->start, 1, GST_SECOND),
                                  gst_util_uint64_scale_int_ceil(segment->stop, 1, GST_SECOND),
                                  1);
        if (retval)
        {
            DAQ_GST_ERROR_OBJECT(element, "daq_request_data", retval);
            return FALSE;
        }
        element->buffer_waiting = TRUE;
    }
    */
    return TRUE;
}



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
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

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
		"NDS Source",
		"Source",
		"NDS-based src element",
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
			"host",
			"Host",
			"NDS1 or NDS2 remote host name or IP address",
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
			"Name of the desired NDS channel.",
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
    gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
	gstbasesrc_class->check_get_range = GST_DEBUG_FUNCPTR(check_get_range);

    // Start up NDS
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
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

	gst_pad_use_fixed_caps(GST_BASE_SRC_PAD(basesrc));

    element->host = g_malloc(strlen(DEFAULT_HOST)+1);
    strcpy(element->host, DEFAULT_HOST);

    element->port = DEFAULT_PORT;

	element->requested_channel_name = g_malloc(strlen(DEFAULT_REQUESTED_CHANNEL_NAME));
    strcpy(element->requested_channel_name, DEFAULT_REQUESTED_CHANNEL_NAME);

    element->daq = NULL;
    element->daq_channel = NULL;
    element->buffer_waiting = FALSE;

	gst_base_src_set_format(GST_BASE_SRC(object), GST_FORMAT_TIME);
}


/*
 * gstlal_framesrc_get_type().
 */


GType gstlal_ndssrc_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALNDSSrcClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALNDSSrc),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_BASE_SRC, "ndssrc", &info, 0);
	}

	return type;
}

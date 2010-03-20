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


#define GSTLAL_TYPE_NDSSRC_CHANTYPE (gstlal_ndssrc_chantype_get_type())
static GType
gstlal_ndssrc_chantype_get_type (void)
{
    static GType chantype_type = 0;
    static const GEnumValue chantype_values[] = {
        {cUnknown, "Unknown", "unknown"},
        {cOnline, "Online", "online"},
        {cRaw, "Raw", "raw"},
        {cRDS, "Reduced", "reduced"},
        {cSTrend, "Second trend", "second-trend"},
        {cMTrend, "Minute trend", "minute-trend"},
        {cTestPoint, "Test point", "test-point"},
        {0, NULL, NULL},
    };

    if (G_UNLIKELY (chantype_type == 0)) {
        chantype_type = g_enum_register_static ("GSTLALNDSSrcChanType",
                                                chantype_values);
    }
    return chantype_type;
}


#define GSTLAL_TYPE_NDSSRC_NDS_VERSION (gstlal_ndssrc_nds_version_get_type())
static GType
gstlal_ndssrc_nds_version_get_type (void)
{
    static GType nds_version_type = 0;
    static const GEnumValue nds_version_values[] = {
        {nds_v1, "NDS Version 1", "v1"},
        {nds_v2, "NDS Version 2", "v2"},
        {0, NULL, NULL},
    };

    if (G_UNLIKELY (nds_version_type == 0)) {
        nds_version_type = g_enum_register_static ("GSTLALNDSSrcNDSVersionType",
                                                nds_version_values);
    }
    return nds_version_type;
}


static const char* DEFAULT_HOST = "marble.ligo-wa.caltech.edu";
static const int DEFAULT_PORT = 31200;
static const enum nds_version DEFAULT_VERSION = nds_v2;
static const char* DEFAULT_CHANNEL_NAME = "H1:DMT-STRAIN";


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


/*
 * Convert a generic time series to the matching GstCaps.  Returns the
 * output of gst_caps_new_simple() --- ref()ed caps that need to be
 * unref()ed when done.
 */


static GstCaps *caps_for_channel(GSTLALNDSSrc* element)
{
	GstCaps *caps = NULL;

	switch(element->daq->chan_req_list->data_type) {
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
		GST_ERROR_OBJECT(element, "unsupported NDS data_type: %d", element->daq->chan_req_list->data_type);
		break;
	}

    if (caps)
        gst_caps_set_simple(caps,
            "rate", G_TYPE_INT, (int)(element->daq->chan_req_list->rate),
            NULL);

    return caps;
}



static gboolean push_new_caps(GSTLALNDSSrc* element)
{
    GstBaseSrc* object = GST_BASE_SRC(element);

    GstCaps* caps = caps_for_channel(element);
    if(!caps) {
        GST_ERROR_OBJECT(element, "unable to construct caps");
        return FALSE;
    }

    /*
     * Try setting the caps on the source pad.
     */

    if(!gst_pad_set_caps(GST_BASE_SRC_PAD(object), caps)) {
        gst_caps_unref(caps);
        GST_ERROR_OBJECT(element, "unable to set caps %" GST_PTR_FORMAT " on %s", caps, GST_PAD_NAME(GST_BASE_SRC_PAD(object)));
        return FALSE;
    }
    gst_caps_unref(caps);

    /*
     * Transmit the tag list.
     */

    GstTagList* taglist = gst_tag_list_new_full(
                                                GSTLAL_TAG_CHANNEL_NAME, element->daq->chan_req_list->name,
                                                NULL);
    if (!taglist)
    {
        GST_ERROR_OBJECT(element, "unable to create taglist");
        return FALSE;
    }

    if (!gst_pad_push_event(GST_BASE_SRC_PAD(object), gst_event_new_tag(taglist)))
    {
        gst_tag_list_free(taglist);
        GST_ERROR_OBJECT(element, "unable to push taglist %" GST_PTR_FORMAT " on %s", taglist, GST_PAD_NAME(GST_BASE_SRC_PAD(object)));
        return FALSE;
    }

    return TRUE;
}



static gboolean ensure_availableChannels(GSTLALNDSSrc* element)
{
    if (!element->daq)
        return FALSE;

    if (element->availableChannels)
        return TRUE;
    else {
        daq_channel_t* channels = calloc(MAX_CHANNELS, sizeof(daq_channel_t));
        if (!channels)
            GST_ERROR_OBJECT(element, "out of memory");
        else {
            int nchannels_received;
            GST_INFO_OBJECT(element, "daq_recv_channel_list");
            int retval = daq_recv_channel_list(element->daq, channels, MAX_CHANNELS, &nchannels_received, 0, element->channelType);
            if (retval)
                DAQ_GST_ERROR_OBJECT(element, "daq_recv_channel_list", retval);
            else {
                if (nchannels_received > 0)
                {
                    daq_channel_t* new_channels = realloc(channels, sizeof(daq_channel_t)*MAX_CHANNELS);
                    if (!new_channels)
                        GST_ERROR_OBJECT(element, "out of memory");
                    else {
                        element->availableChannels = new_channels;
                        element->countAvailableChannels = nchannels_received;
                        return TRUE;
                    }
                }
            }
            free(channels);
        }
    }
    return FALSE;
}



/*
 * Set channel by looking up channelname on NDS channel list.
 */

static gboolean ensure_channelSelected(GSTLALNDSSrc *element)
{
    if (element->daq->num_chan_request > 0)
        return TRUE;
    else {
        if (!ensure_availableChannels(element))
            GST_ERROR_OBJECT(element, "failed to retrieve channel list");
        else {
            daq_channel_t* channel;
            for (channel = element->availableChannels; channel < &element->availableChannels[element->countAvailableChannels]; channel++)
            {
                if (!strcmp(element->channelName, channel->name))
                {
                    GST_INFO_OBJECT(element, "daq_request_channel_from_chanlist");
                    int retval = daq_request_channel_from_chanlist(element->daq, channel);
                    if (retval)
                        DAQ_GST_ERROR_OBJECT(element, "daq_request_channel_from_chanlist", retval);
                    else {
                        if (!push_new_caps(element))
                            GST_ERROR_OBJECT(element, "failed to push new caps");
                        else {
                            element->needs_seek = TRUE;
                            return TRUE;
                        }
                        GST_INFO_OBJECT(element, "daq_clear_channel_list");
                        retval = daq_clear_channel_list(element->daq);
                        if (retval)
                            DAQ_GST_ERROR_OBJECT(element, "daq_clear_channel_list", retval);
                    }
                    break;
                }
            }
            GST_ERROR_OBJECT(element, "channel not found: %s", element->channelName);
        }
    }
    return FALSE;
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
	ARG_SRC_PORT,
    ARG_SRC_VERSION,
    ARG_SRC_CHANNEL_NAME,
    ARG_SRC_CHANNEL_TYPE,
    ARG_SRC_AVAILABLE_CHANNEL_NAMES
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

	GST_OBJECT_LOCK(element);

    if (id == ARG_SRC_HOST)
    {
		g_free(element->host);
		element->host = g_value_dup_string(value);
    }
    if (id == ARG_SRC_PORT)
    {
        element->port = g_value_get_int(value);
    }
    if (id == ARG_SRC_VERSION)
    {
        element->version = g_value_get_enum(value);
    }
    if (id == ARG_SRC_CHANNEL_NAME || id == ARG_SRC_CHANNEL_TYPE)
    {
        if (element->daq)
        {
            GST_INFO_OBJECT(element, "daq_clear_channel_list");
            int retval = daq_clear_channel_list(element->daq);
            if (retval)
                DAQ_GST_ERROR_OBJECT(element, "daq_clear_channel_list", retval);
        }
    }
    if (id == ARG_SRC_CHANNEL_NAME)
    {
		g_free(element->channelName);
		element->channelName = g_value_dup_string(value);
    }
    if (id == ARG_SRC_CHANNEL_TYPE)
    {
        if (element->availableChannels)
        {
            free(element->availableChannels);
            element->availableChannels = NULL;
            element->countAvailableChannels = 0;
        }
        element->channelType = g_value_get_enum(value);
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

    case ARG_SRC_PORT:
        g_value_set_int(value, element->port);
        break;

    case ARG_SRC_VERSION:
        g_value_set_enum(value, element->version);
        break;

	case ARG_SRC_CHANNEL_NAME:
		g_value_set_string(value, element->channelName);
		break;

    case ARG_SRC_CHANNEL_TYPE:
        g_value_set_enum(value, element->channelType);
        break;

    case ARG_SRC_AVAILABLE_CHANNEL_NAMES:
        {
            int nchannels = 0;
            if (ensure_availableChannels(element))
                nchannels = element->countAvailableChannels;

            // TODO: implement proper error checking here
            char** channel_names = calloc(nchannels+1, sizeof(char*));
            int i;
            for (i = 0; i < nchannels; i ++)
                channel_names[i] = strdup(element->availableChannels[i].name);
            channel_names[i] = NULL;
            g_value_set_boxed(value, channel_names);
        }
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

    daq_t* daq = malloc(sizeof(daq_t));
    if (!daq)
        GST_ERROR_OBJECT(element, "out of memory");
    else {
        GST_INFO_OBJECT(element, "daq_connect");
        int retval = daq_connect(daq, element->host, element->port, nds_v2);
        if (retval)
            DAQ_GST_ERROR_OBJECT(element, "daq_connect", retval);
        else {
            element->daq = daq;
            element->needs_seek = TRUE;
            return TRUE;
        }
        free(daq);
    }
    return FALSE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

    if (element->daq)
    {
        GST_INFO_OBJECT(element, "daq_disconnect");
        int retval = daq_disconnect(element->daq);
        if (retval)
            DAQ_GST_ERROR_OBJECT(element, "daq_disconnect", retval);
        free(element->daq);
        element->daq = NULL;
    }

    if (element->availableChannels)
    {
        free(element->availableChannels);
        element->availableChannels = NULL;
        element->countAvailableChannels = 0;
    }

	return TRUE;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
    GSTLALNDSSrc *element = GSTLAL_NDSSRC(basesrc);

    if (!ensure_channelSelected(element))
    {
        GST_ERROR_OBJECT(element, "failed to select channel");
        return TRUE;
    }

    int retval;
    if (element->needs_seek)
    {
        if (element->daq->chan_req_list->type == cOnline)
        {
            //gst_base_src_set_live(object, TRUE);
            GST_INFO_OBJECT(element, "daq_request_data (online)");
            retval = daq_request_data(element->daq, 0, 0, 1);
        } else {
            //gst_base_src_set_live(object, FALSE);
            gint64 start_time = gst_util_uint64_scale_int(basesrc->segment.start, 1, GST_SECOND);
            gint64 stop_time = gst_util_uint64_scale_int_ceil(basesrc->segment.stop, 1, GST_SECOND);
            if (start_time < 600000000)
                start_time = 600000000;
            if (stop_time > 9999999999)
                stop_time = 9999999999;
            GST_INFO_OBJECT(element, "daq_request_data (offline): [%lld, %lld)", start_time, stop_time);
            retval = daq_request_data(element->daq, start_time, stop_time, 16);
        }

        if (retval)
        {
            DAQ_GST_ERROR_OBJECT(element, "daq_request_data", retval);
            return FALSE;
        }

        element->needs_seek = FALSE;
    } else {
        GST_INFO_OBJECT(element, "daq_recv_next");
        retval = -daq_recv_next(element->daq);
        if (retval < 0)
        {
            DAQ_GST_ERROR_OBJECT(element, "daq_recv_next", retval);
            return FALSE;
        }
    }

    int bytes_per_sample = data_type_size(element->daq->chan_req_list->data_type);
    int data_length = element->daq->chan_req_list->status;
    int rate = element->daq->chan_req_list->rate;
    guint64 nsamples = data_length / bytes_per_sample;
    GST_INFO_OBJECT(element, "received segment [%u, %llu)", element->daq->tb->gps, element->daq->tb->gps + nsamples / rate);

    if (element->daq->chan_req_list->rate != (double)rate)
    {
        GST_ERROR_OBJECT(element, "non-integer sample rate not supported (%f != %d)", element->daq->chan_req_list->rate, rate);
        return GST_FLOW_ERROR;
    }
    if (data_length % bytes_per_sample != 0)
    {
        GST_ERROR_OBJECT(element, "daq buffer length is not multiple of data type length");
        return GST_FLOW_ERROR;
    }

    {
        GstFlowReturn result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, data_length, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
        if (result != GST_FLOW_OK)
            return result;
    }

    memcpy((char*)GST_BUFFER_DATA(*buffer), element->daq->tb->data + element->daq->chan_req_list->offset, data_length);

    // TODO: Ask John Zweizig how to get timestamp and duration of block; this
    // struct is part of an obsolete interface according to Doxygen documentation
    basesrc->offset += nsamples;
    GST_BUFFER_OFFSET_END(*buffer) = basesrc->offset;
    GST_BUFFER_TIMESTAMP(*buffer) = GST_SECOND * element->daq->tb->gps + element->daq->tb->gpsn;
    GST_BUFFER_DURATION(*buffer) = GST_SECOND * nsamples / rate;

    return GST_FLOW_OK;
}


/*
 * is_seekable()
 */


static gboolean is_seekable(GstBaseSrc *basesrc)
{
    return TRUE;
}



/*
 * do_seek()
 */


static gboolean do_seek(GstBaseSrc *basesrc, GstSegment* segment)
{
    GSTLALNDSSrc *element = GSTLAL_NDSSRC(basesrc);
    element->needs_seek = TRUE;
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
    g_free(element->channelName);
    element->channelName = NULL;

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
        ARG_SRC_PORT,
        g_param_spec_int(
            "port",
            "Port",
            "NDS1 or NDS2 remote host port",
            1,
            65535,
            DEFAULT_PORT,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
        )
    );
    g_object_class_install_property(
        gobject_class,
        ARG_SRC_VERSION,
        g_param_spec_enum(
            "nds-version",
            "NDS version",
            "NDS protocol version",
            GSTLAL_TYPE_NDSSRC_NDS_VERSION,
            DEFAULT_VERSION,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
        )
    );
	g_object_class_install_property(
		gobject_class,
		ARG_SRC_CHANNEL_NAME,
		g_param_spec_string(
			"channel-name",
			"Channel name",
			"Name of the desired NDS channel.",
			DEFAULT_CHANNEL_NAME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
    g_object_class_install_property(
        gobject_class,
        ARG_SRC_CHANNEL_TYPE,
        g_param_spec_enum(
            "channel-type",
            "Channel type",
            "Type of the desired NDS channel.",
            GSTLAL_TYPE_NDSSRC_CHANTYPE,
            cUnknown,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
        )
    );
    g_object_class_install_property(
        gobject_class,
        ARG_SRC_AVAILABLE_CHANNEL_NAMES,
        g_param_spec_boxed(
            "available-channel-names",
            "Available channel names",
            "Array of all currently available channel names of the currently selected channel type.",
            G_TYPE_STRV,
            G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
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
    GST_INFO_OBJECT(gobject_class, "daq_startup");
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
    element->version = DEFAULT_VERSION;

	element->channelName = g_malloc(strlen(DEFAULT_CHANNEL_NAME));
    strcpy(element->channelName, DEFAULT_CHANNEL_NAME);

    element->channelType = cUnknown;
    element->availableChannels = NULL;
    element->countAvailableChannels = 0;

    element->daq = NULL;
    element->needs_seek = TRUE;

	gst_base_src_set_format(GST_BASE_SRC(object), GST_FORMAT_TIME);
    gst_base_src_set_live(GST_BASE_SRC(object), TRUE);
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

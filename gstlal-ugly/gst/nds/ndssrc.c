/*
 * NDS-based src element
 *
 * Copyright (C) 2009  Leo Singer
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
 *								  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstpushsrc.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_tags.h>
#include <ndssrc.h>
#include <daqc_response.h>


#define GST_CAT_DEFAULT gstlal_ndssrc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


/*
 * ========================================================================
 *
 *								 Parameters
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
		{cSTrend, "Second trend", "s-trend"},
		{cMTrend, "Minute trend", "m-trend"},
		{cTestPoint, "Test point", "test-pt"},
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
		{nds_try, "Automatic", "auto"},
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


static const int DEFAULT_PORT = 31200;
static const enum nds_version DEFAULT_VERSION = nds_try;


#define DAQ_GST_ERROR_OBJECT(element, msg, errnum) GST_ERROR_OBJECT((element), "%s: error %d: %s", (msg), (errnum), daq_strerror(errnum))


/*
 * ========================================================================
 *
 *                                                            GstURIHandler
 *
 * ========================================================================
 */


#define URI_SCHEME "nds"


static GstURIType uri_get_type(GType type)
{
	return GST_URI_SRC;
}


static const gchar *const *uri_get_protocols(GType type)
{
	static const gchar *protocols[] = {
		URI_SCHEME,
		URI_SCHEME "1",
		URI_SCHEME "2",
		NULL
	};

	return protocols;
}


static gchar *uri_get_uri(GstURIHandler *handler)
{
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(handler);
	GString *uri = g_string_new(URI_SCHEME);

	if(element->version != nds_try)
		g_string_append_printf(uri, "%d", element->version);
	if(element->port != DEFAULT_PORT)
		g_string_append_printf(uri, "://%s:%d/", element->host, element->port);
	else
		g_string_append_printf(uri, "://%s/", element->host);
	g_string_append_uri_escaped(uri, element->channelName, NULL, FALSE);
	if(element->channelType != cUnknown)
		g_string_append_printf(uri, ",%s", g_enum_get_value(g_type_class_peek_static(GSTLAL_TYPE_NDSSRC_CHANTYPE), element->channelType)->value_nick);

	return g_string_free(uri, FALSE);
}


static gboolean uri_set_uri(GstURIHandler *handler, const gchar *uri, GError **err)
{
	/* FIXME:  report errors via err argument */
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(handler);
	gchar *scheme = g_uri_parse_scheme(uri);
	gint version;
	gchar *host = NULL;
	gint port;
	gchar *channel = NULL;
	enum chantype ctype;
	gboolean success = TRUE;

	success = !strncmp(scheme, URI_SCHEME, strlen(URI_SCHEME));
	if(success) {
		GMatchInfo *match_info = NULL;
		success &= g_regex_match(GSTLAL_NDSSRC_GET_CLASS(element)->regex, uri, 0, &match_info);
		if(success) {
			gchar *tmp;
			tmp = g_match_info_fetch(match_info, 1);
			if(tmp && tmp[0])
				version = atoi(tmp);
			else
				version = nds_try;
			g_free(tmp);
			host = g_match_info_fetch(match_info, 2);
			success &= host != NULL;
			tmp = g_match_info_fetch(match_info, 3);
			if(tmp && tmp[0])
				port = atoi(tmp);
			else
				port = DEFAULT_PORT;
			g_free(tmp);
			channel = g_match_info_fetch(match_info, 4);
			success &= channel != NULL;
			tmp = g_match_info_fetch(match_info, 5);
			if(tmp && tmp[0]) {
				GEnumValue *val = g_enum_get_value_by_nick(g_type_class_peek_static(GSTLAL_TYPE_NDSSRC_CHANTYPE), tmp);
				if(val)
					ctype = val->value;
				else
					success = FALSE;
			} else
				ctype = cUnknown;
			g_free(tmp);
		}
		g_match_info_free(match_info);
	}
	if(success) {
		GST_INFO_OBJECT(element, "setting properties from URI:  nds-version=%d, host=\"%s\", port=%d, channel-name=\"%s\", channel-type=%d", version, host, port, channel, ctype);
		g_object_set(G_OBJECT(element), "nds-version", version, "host", host, "port", port, "channel-name", channel, "channel-type", ctype, NULL);
	} else
		GST_ERROR_OBJECT(element, "failed to parse URI \"%s\"", uri);

	g_free(scheme);
	g_free(host);
	g_free(channel);
	return success;
}


static void uri_handler_init(gpointer g_iface, gpointer iface_data)
{
	GstURIHandlerInterface *iface = (GstURIHandlerInterface *) g_iface;

	iface->get_uri = GST_DEBUG_FUNCPTR(uri_get_uri);
	iface->set_uri = GST_DEBUG_FUNCPTR(uri_set_uri);
	iface->get_type = GST_DEBUG_FUNCPTR(uri_get_type);
	iface->get_protocols = GST_DEBUG_FUNCPTR(uri_get_protocols);
}


/*
 * ========================================================================
 *
 *							 Utility Functions
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
	GstCaps *caps = gst_caps_new_simple(
		"audio/x-raw",
		"rate", G_TYPE_INT, (int)(element->daq->chan_req_list->rate),
		"channels", G_TYPE_INT, 1,
		"layout", G_TYPE_STRING, "interleaved",
		NULL
	);

	if(!caps)
		goto done;

	switch(element->daq->chan_req_list->data_type) {
	case _16bit_integer:
		gst_caps_set_simple(caps,
			"format", G_TYPE_STRING, GST_AUDIO_NE(S16),
			NULL
		);
		break;

	case _32bit_integer:
		gst_caps_set_simple(caps,
			"format", G_TYPE_STRING, GST_AUDIO_NE(S32),
			NULL
		);
		break;

	case _64bit_integer:
		gst_caps_set_simple(caps,
			"format", G_TYPE_STRING, GST_AUDIO_NE(S64),
			NULL
		);
		break;

	case _32bit_float:
		gst_caps_set_simple(caps,
			"format", G_TYPE_STRING, GST_AUDIO_NE(F32),
			NULL
		);
		break;

	case _64bit_double:
		gst_caps_set_simple(caps,
			"format", G_TYPE_STRING, GST_AUDIO_NE(F64),
			NULL
		);
		break;

	case _32bit_complex:
		gst_caps_set_simple(caps,
			"format", G_TYPE_STRING, GST_AUDIO_NE(Z64),
			NULL
		);
		break;

	default:
		GST_ERROR_OBJECT(element, "unsupported NDS data_type: %d", element->daq->chan_req_list->data_type);
		gst_caps_unref(caps);
		caps = NULL;
		break;
	}

done:
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

	GstTagList* taglist;
	{
		char* full_channel_name = strdup(element->daq->chan_req_list->name);
		char* channel_name = strchr(full_channel_name, ':');
		if (channel_name)
		{
			*(channel_name++) = '\0';
			taglist = gst_tag_list_new(
				GSTLAL_TAG_CHANNEL_NAME, channel_name,
				GSTLAL_TAG_INSTRUMENT, full_channel_name,
				NULL);
		} else {
			taglist = gst_tag_list_new(
				GSTLAL_TAG_CHANNEL_NAME, full_channel_name,
				NULL);
		}

		free(full_channel_name);
	}

	if (!taglist)
	{
		GST_ERROR_OBJECT(element, "unable to create taglist");
		return FALSE;
	}

	if (!gst_pad_push_event(GST_BASE_SRC_PAD(object), gst_event_new_tag(taglist)))
	{
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
		int nchannels_received;
		int retval;

		GST_INFO_OBJECT(element, "daq_recv_channel_list");
		retval = daq_recv_channel_list(element->daq, NULL, 0, &nchannels_received, 0, element->channelType);

		if (retval)
			DAQ_GST_ERROR_OBJECT(element, "daq_recv_channel_list", retval);
		else {
			daq_channel_t* channels = calloc(nchannels_received, sizeof(daq_channel_t));
			if (!channels)
				GST_ERROR_OBJECT(element, "out of memory");
			else {
				GST_INFO_OBJECT(element, "daq_recv_channel_list");
				int old_nchannels_received = nchannels_received;
				int retval = daq_recv_channel_list(element->daq, channels, old_nchannels_received, &nchannels_received, 0, element->channelType);
				if (retval)
					DAQ_GST_ERROR_OBJECT(element, "daq_recv_channel_list", retval);
				else if (old_nchannels_received != nchannels_received)
					GST_ERROR_OBJECT(element, "daq_recv_channel_list reported %d channels available, but then returned %d", old_nchannels_received, nchannels_received);
				else {
					element->availableChannels = channels;
					element->countAvailableChannels = nchannels_received;
					return TRUE;
				}
				free(channels);
			}
		}
	}
	return FALSE;
}



/*
 * Set channel by looking up channelname on NDS channel list.
 */

static gboolean ensure_channelSelected(GSTLALNDSSrc *element)
{
	if (!element->channelName)
	{
		GST_ERROR_OBJECT(element, "required property `channel-name' not specified");
		return FALSE;
	}

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
 *								 Properties
 *
 * ============================================================================
 */


enum property {
	PROP_SRC_HOST = 1,
	PROP_SRC_PORT,
	PROP_SRC_VERSION,
	PROP_SRC_CHANNEL_NAME,
	PROP_SRC_CHANNEL_TYPE,
	PROP_SRC_AVAILABLE_CHANNEL_NAMES
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);

	GST_OBJECT_LOCK(element);

	if (id == PROP_SRC_HOST)
	{
		g_free(element->host);
		element->host = g_value_dup_string(value);
	}
	if (id == PROP_SRC_PORT)
	{
		element->port = g_value_get_int(value);
	}
	if (id == PROP_SRC_VERSION)
	{
		element->version = g_value_get_enum(value);
	}
	if (id == PROP_SRC_CHANNEL_NAME || id == PROP_SRC_CHANNEL_TYPE)
	{
		if (element->daq)
		{
			GST_INFO_OBJECT(element, "daq_clear_channel_list");
			int retval = daq_clear_channel_list(element->daq);
			if (retval)
				DAQ_GST_ERROR_OBJECT(element, "daq_clear_channel_list", retval);
		}
	}
	if (id == PROP_SRC_CHANNEL_NAME)
	{
		g_free(element->channelName);
		element->channelName = g_value_dup_string(value);
	}
	if (id == PROP_SRC_CHANNEL_TYPE)
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
	case PROP_SRC_HOST:
		g_value_set_string(value, element->host);
		break;

	case PROP_SRC_PORT:
		g_value_set_int(value, element->port);
		break;

	case PROP_SRC_VERSION:
		g_value_set_enum(value, element->version);
		break;

	case PROP_SRC_CHANNEL_NAME:
		g_value_set_string(value, element->channelName);
		break;

	case PROP_SRC_CHANNEL_TYPE:
		g_value_set_enum(value, element->channelType);
		break;

	case PROP_SRC_AVAILABLE_CHANNEL_NAMES:
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
 *						GstBaseSrc Method Overrides
 *
 * ============================================================================
 */


/*
 * start()
 */

static gboolean start(GstBaseSrc *object)
{
	GSTLALNDSSrc *element = GSTLAL_NDSSRC(object);
	gboolean result = TRUE;

	if (!element->host)
	{
		GST_ERROR_OBJECT(element, "required property `host' not specified");
		result = FALSE;
		goto done;
	}

	daq_t* daq = malloc(sizeof(daq_t));
	if (!daq) {
		GST_ERROR_OBJECT(element, "out of memory");
		result = FALSE;
		goto done;
	}

	GST_INFO_OBJECT(element, "daq_connect(daq_t*, \"%s\", %d, %d)", element->host, element->port, element->version);
	int retval = daq_connect(daq, element->host, element->port, element->version);
	if (retval) {
		DAQ_GST_ERROR_OBJECT(element, "daq_connect", retval);
		free(daq);
		result = FALSE;
		goto done;
	}

	element->daq = daq;
	element->needs_seek = TRUE;

done:
	gst_base_src_start_complete(object, result ? GST_FLOW_OK : GST_FLOW_ERROR);
	return result;
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
	GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS(G_OBJECT_GET_CLASS(basesrc));
	GstMapInfo mapinfo;

	if (!ensure_channelSelected(element))
	{
		GST_ERROR_OBJECT(element, "failed to select channel");
		return GST_FLOW_ERROR;
	}

	guint rate = element->daq->chan_req_list->rate;
	int bytes_per_sample = data_type_size(element->daq->chan_req_list->data_type);

	if (element->daq->chan_req_list->rate != (double)rate)
	{
		GST_ERROR_OBJECT(element, "non-integer sample rate not supported (%f != %u)", element->daq->chan_req_list->rate, rate);
		return GST_FLOW_ERROR;
	}

	int retval;
	gboolean should_push_newsegment = FALSE;
	if (element->needs_seek)
	{
		guint blocksize = gst_base_src_get_blocksize(basesrc);
		int stride_seconds;

		if (blocksize == G_MAXUINT)
			stride_seconds = 1;
		else
		{
			gulong bytes_per_sec = rate * bytes_per_sample;
			if (blocksize % bytes_per_sec != 0)
			{
				GST_ERROR_OBJECT(element, "property `blocksize' must correspond to an integer number of seconds");
				return GST_FLOW_ERROR;
			}

			stride_seconds = blocksize / bytes_per_sec;
		}

		if (element->channelType == cOnline)
		{
			GST_INFO_OBJECT(element, "daq_request_data(daq_t*, 0, 0, %d)", stride_seconds);
			retval = daq_request_data(element->daq, 0, 0, stride_seconds);
		} else {
			gint64 start_time = gst_util_uint64_scale_int(basesrc->segment.start, 1, GST_SECOND);
			gint64 stop_time = gst_util_uint64_scale_int_ceil(basesrc->segment.stop, 1, GST_SECOND);
			if (start_time < 600000000)
				start_time = 600000000;
			if (stop_time > 9999999999)
				stop_time = 9999999999;
			GST_INFO_OBJECT(element, "daq_request_data(daq_t*, %" G_GINT64_FORMAT ", %" G_GINT64_FORMAT ", %d)", start_time, stop_time, stride_seconds);
			retval = daq_request_data(element->daq, start_time, stop_time, stride_seconds);
		}

		if (retval)
		{
			DAQ_GST_ERROR_OBJECT(element, "daq_request_data", retval);
			return GST_FLOW_ERROR;
		}

		element->needs_seek = FALSE;
		should_push_newsegment = TRUE;
	}

	GST_INFO_OBJECT(element, "daq_recv_next");
	retval = -daq_recv_next(element->daq);
	if (retval < 0)
	{
		DAQ_GST_ERROR_OBJECT(element, "daq_recv_next", -retval);
		return GST_FLOW_ERROR;
	}

	int data_length = element->daq->chan_req_list->status;
	guint64 nsamples = data_length / bytes_per_sample;
	GST_INFO_OBJECT(element, "received segment [%" G_GUINT32_FORMAT ", %" G_GUINT64_FORMAT ")", daq_get_block_gps(element->daq), daq_get_block_gps(element->daq) + nsamples / rate);

	if (data_length % bytes_per_sample != 0)
	{
		GST_ERROR_OBJECT(element, "daq buffer length is not multiple of data type length");
		return GST_FLOW_ERROR;
	}

	{
		GstFlowReturn result = basesrc_class->alloc(basesrc, offset, data_length, buffer);
		if (result != GST_FLOW_OK)
			return result;
	}

	gst_buffer_map(*buffer, &mapinfo, GST_MAP_WRITE);
	memcpy(mapinfo.data, daq_get_block_data(element->daq) + element->daq->chan_req_list->offset, data_length);
	gst_buffer_unmap(*buffer, &mapinfo);

	// TODO: Ask John Zweizig how to get timestamp and duration of block; this
	// struct is part of an obsolete interface according to Doxygen documentation
	GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + nsamples;
	GST_BUFFER_PTS(*buffer) = GST_SECOND * daq_get_block_gps(element->daq) + daq_get_block_gpsn(element->daq);
	GST_BUFFER_DURATION(*buffer) = GST_SECOND * nsamples / rate;

	if (should_push_newsegment)
	{
		if (!gst_base_src_new_seamless_segment(basesrc,
			GST_BUFFER_PTS(*buffer),
			basesrc->segment.stop,
			GST_BUFFER_PTS(*buffer)))
		{
			GST_ERROR_OBJECT(element, "failed to create new segment");
		}
	}


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
 * ============================================================================
 *
 *								Type Support
 *
 * ============================================================================
 */


static void additional_initializations(GType type)
{
	static const GInterfaceInfo uri_handler_info = {
		uri_handler_init,
		NULL,
		NULL
	};
	g_type_add_interface_static(type, GST_TYPE_URI_HANDLER, &uri_handler_info);
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "ndssrc", 0, "ndssrc element");
}


G_DEFINE_TYPE_WITH_CODE(
	GSTLALNDSSrc,
	gstlal_ndssrc,
	GST_TYPE_BASE_SRC,
	additional_initializations(g_define_type_id)
);


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

	G_OBJECT_CLASS(gstlal_ndssrc_parent_class)->finalize(object);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void gstlal_ndssrc_class_init(GSTLALNDSSrcClass *gstlal_ndssrc_class)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(gstlal_ndssrc_class);
	GstElementClass *element_class = GST_ELEMENT_CLASS(gstlal_ndssrc_class);
	GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(gstlal_ndssrc_class);

	gstlal_ndssrc_parent_class = g_type_class_ref(GST_TYPE_BASE_SRC);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_set_details_simple(
		element_class,
		"NDS Source",
		"Source",
		"NDS-based src element",
		"Leo Singer <leo.singer@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) 1, " \
				"format = (string) {" GST_AUDIO_NE(S16) ", " GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(S64) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) "}, " \
				"layout = (string) interleaved"
			)
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_SRC_HOST,
		g_param_spec_string(
			"host",
			"Host",
			"NDS1 or NDS2 remote host name or IP address",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_SRC_PORT,
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
		PROP_SRC_VERSION,
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
		PROP_SRC_CHANNEL_NAME,
		g_param_spec_string(
			"channel-name",
			"Channel name",
			"Name of the desired NDS channel.",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_SRC_CHANNEL_TYPE,
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
		PROP_SRC_AVAILABLE_CHANNEL_NAMES,
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

	/*
	 * matches patterns like
	 *
	 * nds://nds.ligo.caltech.edu/L1:LSC-STRAIN
	 * nds1://nds.ligo.caltech.edu:31200/L1:LSC-STRAIN,reduced
	 * ...
	 */

	gstlal_ndssrc_class->regex = g_regex_new("^" URI_SCHEME "(\\d?)://([^:/]+)(?::(\\d+)|)/([^,]+)(?:,([\\w-]+)|)$", 0, 0, NULL);

	// Start up NDS
	GST_INFO_OBJECT(gobject_class, "daq_startup");
	daq_startup();
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void gstlal_ndssrc_init(GSTLALNDSSrc *element)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(element);

	gst_pad_use_fixed_caps(GST_BASE_SRC_PAD(basesrc));

	element->host = NULL;
	element->port = DEFAULT_PORT;
	element->version = DEFAULT_VERSION;
	element->channelName = NULL;
	element->channelType = cUnknown;
	element->availableChannels = NULL;
	element->countAvailableChannels = 0;
	element->daq = NULL;
	element->needs_seek = TRUE;
	gst_base_src_set_blocksize(basesrc, G_MAXUINT);

	gst_base_src_set_format(basesrc, GST_FORMAT_TIME);
}


/*
 * plugin entry point
 */


static gboolean plugin_init(GstPlugin *plugin)
{
	if (!gst_element_register(plugin, "ndssrc", GST_RANK_NONE, GSTLAL_NDSSRC_TYPE))
		return FALSE;

	/*
	 * Tell GStreamer about the custom tags.
	 */

	gstlal_register_tags();

	return TRUE;
}


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nds, "LIGO Network Data Server (NDS) v1/v2 elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

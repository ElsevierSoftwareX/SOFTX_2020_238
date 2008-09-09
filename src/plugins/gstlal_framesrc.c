/*
 * LAL cache-based .gwf frame file src element
 *
 * Copyright (C) 2008  Kipp C. Cannon
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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


#include <stdint.h>
#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>


/*
 * stuff from LAL
 */


#include <lal/Date.h>
#include <lal/LALDatatypes.h>
#include <lal/FrameStream.h>
#include <lal/LALFrameIO.h>
#include <lal/Units.h>
#include <lal/TimeSeries.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_framesrc.h>


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define DEFAULT_BUFFER_DURATION 16


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


/*
 * Prefix a channel name with the instrument name.  I.e., turn "H1" and
 * "LSC-STRAIN" into "H1:LSC-STRAIN".  If either instrument or channel_name
 * is NULL, then the corresponding part of the result is left blank and the
 * colon is omited.  The return value is NULL on failure or a
 * newly-allocated string.   The calling code should free() the string when
 * finished with it.
 */


static char *build_full_channel_name(const char *instrument, const char *channel_name)
{
	char *full_channel_name;
	int len = 2;	/* for ":" and null terminator */

	if(instrument)
		len += strlen(instrument);
	if(channel_name)
		len += strlen(channel_name);

	full_channel_name = malloc(len * sizeof(*full_channel_name));
	if(!full_channel_name)
		return NULL;

	snprintf(full_channel_name, len, instrument && channel_name ? "%s:%s" : "%s%s", instrument ? instrument : "", channel_name ? channel_name : "");

	return full_channel_name;
}


/*
 * Type-agnostic time series deallocator
 */


static void DestroyTimeSeries(void *series, LALTYPECODE type)
{
	switch(type) {
	case LAL_I4_TYPE_CODE:
		XLALDestroyINT4TimeSeries(series);
		break;
	case LAL_S_TYPE_CODE:
		XLALDestroyREAL4TimeSeries(series);
		break;
	case LAL_D_TYPE_CODE:
		XLALDestroyREAL8TimeSeries(series);
		break;
	default:
		/* should never get here */
		break;
	}
}


/*
 * Convert a generic time series to the matching GstCaps.  Returns the
 * output of gst_caps_new_simple() --- ref()ed caps that need to be
 * unref()ed when done.
 */


static GstCaps *series_to_caps(const char *instrument, const char *channel_name, void *series, LALTYPECODE type)
{
	switch(type) {
	case LAL_I4_TYPE_CODE:
		return gst_caps_new_simple(
			"audio/x-raw-int",
			"rate", G_TYPE_INT, (int) (1.0 / ((INT4TimeSeries *) series)->deltaT + 0.5),
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 32,
			"depth", G_TYPE_INT, 32,
			"signed", G_TYPE_BOOLEAN, TRUE,
			"instrument", G_TYPE_STRING, instrument,
			"channel_name", G_TYPE_STRING, channel_name,
			NULL
		);

	case LAL_S_TYPE_CODE:
		return gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", G_TYPE_INT, (int) (1.0 / ((REAL4TimeSeries *) series)->deltaT + 0.5),
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 32,
			"instrument", G_TYPE_STRING, instrument,
			"channel_name", G_TYPE_STRING, channel_name,
			NULL
		);

	case LAL_D_TYPE_CODE:
		return gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", G_TYPE_INT, (int) (1.0 / ((REAL8TimeSeries *) series)->deltaT + 0.5),
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 64,
			"instrument", G_TYPE_STRING, instrument,
			"channel_name", G_TYPE_STRING, channel_name,
			NULL
		);
	
	default:
		return NULL;
	}
}


/*
 * Retrieve a chunk of data from the series buffer, loading more data if
 * needed.
 */


static void *read_series(GSTLALFrameSrc *element, long start_sample, long length)
{
	double deltaT;
	long input_length;
	LIGOTimeGPS buffer_start_time;
	long buffer_start_sample;
	long buffer_length;

	/* retrieve the bounds of the current buffer and the input domain
	 * */
	switch(element->series_type) {
	case LAL_I4_TYPE_CODE:
		deltaT = ((INT4TimeSeries *) element->series_buffer)->deltaT;
		buffer_start_time = ((INT4TimeSeries *) element->series_buffer)->epoch;
		buffer_length = ((INT4TimeSeries *) element->series_buffer)->data->length;
		break;

	case LAL_S_TYPE_CODE:
		deltaT = ((REAL4TimeSeries *) element->series_buffer)->deltaT;
		buffer_start_time = ((REAL4TimeSeries *) element->series_buffer)->epoch;
		buffer_length = ((REAL4TimeSeries *) element->series_buffer)->data->length;
		break;

	case LAL_D_TYPE_CODE:
		deltaT = ((REAL8TimeSeries *) element->series_buffer)->deltaT;
		buffer_start_time = ((REAL8TimeSeries *) element->series_buffer)->epoch;
		buffer_length = ((REAL8TimeSeries *) element->series_buffer)->data->length;
		break;

	default:
		GST_ERROR("unsupported data type (LALTYPECODE=%d)", element->series_type);
		return NULL;
	}
	buffer_start_sample = (long) (XLALGPSDiff(&buffer_start_time, &element->start_time) / deltaT + 0.5);
	input_length = (long) (XLALGPSDiff(&element->stop_time, &element->start_time) / deltaT + 0.5);

	/* clip the requested interval to the input domain */
	if(start_sample + length < 0 || start_sample >= input_length) {
		GST_ERROR("requested interval lies outside input domain");
		return NULL;
	}
	if(start_sample < 0) {
		length += start_sample;
		start_sample = 0;
	}
	if(start_sample + length > input_length)
		length = input_length - start_sample;

	/* does the requested data start in the buffer? */
	if(start_sample < buffer_start_sample || start_sample >= buffer_start_sample + buffer_length) {
		/* the requested data does not start in the buffer --> read
		 * a new buffer */

		DestroyTimeSeries(element->series_buffer, element->series_type);
		element->series_buffer = NULL;

		/* compute the bounds of the new buffer, using the
		 * requested start time as the buffer's start time */
		buffer_start_sample = start_sample;
		buffer_length = (long) (element->series_buffer_duration / deltaT + 0.5);
		if(buffer_start_sample + buffer_length > input_length)
			buffer_length = input_length - buffer_start_sample;
		buffer_start_time = element->start_time;
		XLALGPSAdd(&buffer_start_time, buffer_start_sample * deltaT);

		/* load the buffer */
		switch(element->series_type) {
		case LAL_I4_TYPE_CODE:
			element->series_buffer = XLALFrReadINT4TimeSeries(element->stream, element->full_channel_name, &buffer_start_time, buffer_length * deltaT, 0);
			break;

		case LAL_S_TYPE_CODE:
			element->series_buffer = XLALFrReadREAL4TimeSeries(element->stream, element->full_channel_name, &buffer_start_time, buffer_length * deltaT, 0);
			break;

		case LAL_D_TYPE_CODE:
			element->series_buffer = XLALFrReadREAL8TimeSeries(element->stream, element->full_channel_name, &buffer_start_time, buffer_length * deltaT, 0);
			break;

		default:
			/* impossible, would've been caught above */
			return NULL;
		}

		if(!element->series_buffer) {
			GST_ERROR("XLALFrRead*TimeSeries() failed");
			return NULL;
		}
	}

	/* clip the requested length against the top of the buffer */
	if(start_sample + length > buffer_start_sample + buffer_length)
		length = buffer_start_sample + buffer_length - start_sample;

	/* extract the requested interval */
	switch(element->series_type) {
	case LAL_I4_TYPE_CODE:
		return XLALCutINT4TimeSeries(element->series_buffer, start_sample - buffer_start_sample, length);

	case LAL_S_TYPE_CODE:
		return XLALCutREAL4TimeSeries(element->series_buffer, start_sample - buffer_start_sample, length);

	case LAL_D_TYPE_CODE:
		return XLALCutREAL8TimeSeries(element->series_buffer, start_sample - buffer_start_sample, length);

	default:
		/* impossible, would've been caught above */
		return NULL;
	}
}


/*
 * ============================================================================
 *
 *                          GStreamer Source Element
 *
 * ============================================================================
 */


/*
 * Properties
 */


enum property {
	ARG_SRC_LOCATION = 1,
	ARG_SRC_BUFFER_DURATION,
	ARG_SRC_INSTRUMENT,
	ARG_SRC_CHANNEL_NAME,
	ARG_SRC_START_TIME_GPS,
	ARG_SRC_STOP_TIME_GPS
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	switch(id) {
	case ARG_SRC_LOCATION:
		free(element->location);
		element->location = g_value_dup_string(value);
		break;

	case ARG_SRC_BUFFER_DURATION:
		element->series_buffer_duration = g_value_get_int(value);
		break;

	case ARG_SRC_INSTRUMENT:
		free(element->instrument);
		element->instrument = g_value_dup_string(value);
		free(element->full_channel_name);
		element->full_channel_name = build_full_channel_name(element->instrument, element->channel_name);
		break;

	case ARG_SRC_CHANNEL_NAME:
		free(element->channel_name);
		element->channel_name = g_value_dup_string(value);
		free(element->full_channel_name);
		element->full_channel_name = build_full_channel_name(element->instrument, element->channel_name);
		break;

	case ARG_SRC_START_TIME_GPS:
		if(XLALStrToGPS(&element->start_time, g_value_get_string(value), NULL) < 0) {
			GST_ERROR("invalid start_time_gps \"%s\"", g_value_get_string(value));
		}
		break;

	case ARG_SRC_STOP_TIME_GPS:
		if(XLALStrToGPS(&element->stop_time, g_value_get_string(value), NULL) < 0) {
			GST_ERROR("invalid stop_time_gps \"%s\"", g_value_get_string(value));
		}
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	switch(id) {
	case ARG_SRC_LOCATION:
		g_value_set_string(value, element->location);
		break;

	case ARG_SRC_BUFFER_DURATION:
		g_value_set_int(value, element->series_buffer_duration);
		break;

	case ARG_SRC_INSTRUMENT:
		g_value_set_string(value, element->instrument);
		break;

	case ARG_SRC_CHANNEL_NAME:
		g_value_set_string(value, element->channel_name);
		break;

	case ARG_SRC_START_TIME_GPS:
		g_value_take_string(value, XLALGPSToStr(NULL, &element->start_time));
		break;

	case ARG_SRC_STOP_TIME_GPS:
		g_value_take_string(value, XLALGPSToStr(NULL, &element->stop_time));
		break;
	}
}


/*
 * start()
 */


static gboolean start(GstBaseSrc *object)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(object);
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);
	GstPad *srcpad = GST_BASE_SRC_PAD(basesrc);
	FrCache *cache;
	GstCaps *caps;
	double buffer_duration;

	/*
	 * Open frame stream.
	 */

	cache = XLALFrImportCache(element->location);
	if(!cache) {
		GST_ERROR("XLALFrImportCache() failed");
		return FALSE;
	}
	element->stream = XLALFrCacheOpen(cache);
	XLALFrDestroyCache(cache);
	if(!element->stream) {
		GST_ERROR("XLALFrCacheOpen() failed");
		return FALSE;
	}

	element->series_type = XLALFrGetTimeSeriesType(element->full_channel_name, element->stream);

	/*
	 * Turn on checking for missing data.
	 */

	element->stream->mode = LAL_FR_VERBOSE_MODE;

	/*
	 * Prime the series buffer.
	 */

	buffer_duration = XLALGPSDiff(&element->stop_time, &element->start_time);
	if(buffer_duration > element->series_buffer_duration)
		buffer_duration = element->series_buffer_duration;

	switch(element->series_type) {
	case LAL_I4_TYPE_CODE:
		element->series_buffer = XLALFrReadINT4TimeSeries(element->stream, element->full_channel_name, &element->start_time, buffer_duration, 0);
		break;

	case LAL_S_TYPE_CODE:
		element->series_buffer = XLALFrReadREAL4TimeSeries(element->stream, element->full_channel_name, &element->start_time, buffer_duration, 0);
		break;

	case LAL_D_TYPE_CODE:
		element->series_buffer = XLALFrReadREAL8TimeSeries(element->stream, element->full_channel_name, &element->start_time, buffer_duration, 0);
		break;

	case -1:
		GST_ERROR("XLALFrGetTimeSeriesType() failed");
		goto error;

	default:
		GST_ERROR("unsupported data type (LALTYPECODE=%d) in channel \"%s\"", element->series_type, element->full_channel_name);
		goto error;
	}

	if(!element->series_buffer) {
		GST_ERROR("XLALFrRead*TimeSeries() failed");
		goto error;
	}

	caps = series_to_caps(element->instrument, element->channel_name, element->series_buffer, element->series_type);
	if(!gst_pad_set_caps(srcpad, caps)) {
		GST_DEBUG("gst_pad_set_caps() failed");
		DestroyTimeSeries(element->series_buffer, element->series_type);
		element->series_buffer = NULL;
		element->series_type = -1;
		gst_caps_unref(caps);
		goto error;
	}
	gst_caps_unref(caps);

	/*
	 * Done
	 */

	return TRUE;

error:
	XLALFrClose(element->stream);
	element->stream = NULL;
	element->series_buffer = NULL;
	element->series_type = -1;
	return FALSE;
}


/*
 * get_caps()
 */


static GstCaps *get_caps(GstBaseSrc *object)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	if(element->series_buffer)
		return series_to_caps(element->instrument, element->channel_name, element->series_buffer, element->series_type);
	else
		return gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_SRC_PAD(object)));
}


/*
 * create()
 */


static GstFlowReturn create(GstPushSrc *object, GstBuffer **buffer)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(object);
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);
	GstPad *srcpad = GST_BASE_SRC_PAD(basesrc);
	GstFlowReturn result;

	switch(element->series_type) {
	case LAL_I4_TYPE_CODE: {
		INT4TimeSeries *chunk = read_series(element, element->next_sample, basesrc->blocksize / sizeof(*chunk->data->data));
		if(!chunk)
			/* EOS */
			return GST_FLOW_UNEXPECTED;
		result = gst_pad_alloc_buffer_and_set_caps(srcpad, element->next_sample, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(srcpad), buffer);
		if(result != GST_FLOW_OK) {
			GST_DEBUG("gst_pad_alloc_buffer() failed");
			XLALDestroyINT4TimeSeries(chunk);
			return result;
		}
		memcpy(GST_BUFFER_DATA(*buffer), chunk->data->data, GST_BUFFER_SIZE(*buffer));
		GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + chunk->data->length - 1;
		GST_BUFFER_TIMESTAMP(*buffer) = (GstClockTime) XLALGPSToINT8NS(&element->start_time) + (GstClockTime) (element->next_sample * GST_SECOND * chunk->deltaT + 0.5);
		GST_BUFFER_DURATION(*buffer) = (GstClockTime) (chunk->data->length * GST_SECOND * chunk->deltaT + 0.5);
		if(element->next_sample == 0)
			GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);
		element->next_sample += chunk->data->length;
		XLALDestroyINT4TimeSeries(chunk);
		break;
	}

	case LAL_S_TYPE_CODE: {
		REAL4TimeSeries *chunk = read_series(element, element->next_sample, basesrc->blocksize / sizeof(*chunk->data->data));
		if(!chunk)
			/* EOS */
			return GST_FLOW_UNEXPECTED;
		result = gst_pad_alloc_buffer_and_set_caps(srcpad, element->next_sample, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(srcpad), buffer);
		if(result != GST_FLOW_OK) {
			GST_DEBUG("gst_pad_alloc_buffer() failed");
			XLALDestroyREAL4TimeSeries(chunk);
			return result;
		}
		memcpy(GST_BUFFER_DATA(*buffer), chunk->data->data, GST_BUFFER_SIZE(*buffer));
		GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + chunk->data->length - 1;
		GST_BUFFER_TIMESTAMP(*buffer) = (GstClockTime) XLALGPSToINT8NS(&element->start_time) + (GstClockTime) (element->next_sample * GST_SECOND * chunk->deltaT + 0.5);
		GST_BUFFER_DURATION(*buffer) = (GstClockTime) (chunk->data->length * GST_SECOND * chunk->deltaT + 0.5);
		if(element->next_sample == 0)
			GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);
		element->next_sample += chunk->data->length;
		XLALDestroyREAL4TimeSeries(chunk);
		break;
	}

	case LAL_D_TYPE_CODE: {
		REAL8TimeSeries *chunk = read_series(element, element->next_sample, basesrc->blocksize / sizeof(*chunk->data->data));
		if(!chunk)
			/* EOS */
			return GST_FLOW_UNEXPECTED;
		result = gst_pad_alloc_buffer_and_set_caps(srcpad, element->next_sample, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(srcpad), buffer);
		if(result != GST_FLOW_OK) {
			GST_DEBUG("gst_pad_alloc_buffer() failed");
			XLALDestroyREAL8TimeSeries(chunk);
			return result;
		}
		memcpy(GST_BUFFER_DATA(*buffer), chunk->data->data, GST_BUFFER_SIZE(*buffer));
		GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + chunk->data->length - 1;
		GST_BUFFER_TIMESTAMP(*buffer) = (GstClockTime) XLALGPSToINT8NS(&element->start_time) + (GstClockTime) (element->next_sample * GST_SECOND * chunk->deltaT + 0.5);
		GST_BUFFER_DURATION(*buffer) = (GstClockTime) (chunk->data->length * GST_SECOND * chunk->deltaT + 0.5);
		if(element->next_sample == 0)
			GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);
		element->next_sample += chunk->data->length;
		XLALDestroyREAL8TimeSeries(chunk);
		break;
	}

	default:
		break;
	}

	return GST_FLOW_OK;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	if(element->stream) {
		XLALFrClose(element->stream);
		element->stream = NULL;
	}
	DestroyTimeSeries(element->series_buffer, element->series_type);
	element->series_buffer = NULL;
	element->series_type = -1;

	return TRUE;
}


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance dispose function.  See ???
 */


static void dispose(GObject *object)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	free(element->location);
	element->location = NULL;
	free(element->instrument);
	element->instrument = NULL;
	free(element->channel_name);
	element->channel_name = NULL;
	free(element->full_channel_name);
	element->full_channel_name = NULL;
	if(element->stream) {
		XLALFrClose(element->stream);
		element->stream = NULL;
	}
	DestroyTimeSeries(element->series_buffer, element->series_type);
	element->series_buffer = NULL;
	element->series_type = -1;

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static GstElementDetails plugin_details = {
		"GWF Frame File Source",
		"Source",
		"LAL cache-based .gwf frame file source element",
		"Kipp Cannon <kcannon@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64}; " \
				"audio/x-raw-int, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 32, " \
				"depth = (int) 32, " \
				"signed = (boolean) true"
			)
		)
	);

	gst_element_class_set_details(element_class, &plugin_details);
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
	GstPushSrcClass *gstpush_src_class = GST_PUSH_SRC_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;

	g_object_class_install_property(gobject_class, ARG_SRC_LOCATION, g_param_spec_string("location", "Location", "Path to LAL cache file (see LSCdataFind for more information).", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SRC_BUFFER_DURATION, g_param_spec_int("buffer-duration", "Buffer duration", "Size of input buffer in seconds.", 1, 2048, DEFAULT_BUFFER_DURATION, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SRC_INSTRUMENT, g_param_spec_string("instrument", "Instrument", "Instrument name (e.g., \"H1\").", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SRC_CHANNEL_NAME, g_param_spec_string("channel-name", "Channel name", "Channel name (e.g., \"LSC-STRAIN\").", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	/* FIXME:  "string" is not the best type for these ... */
	g_object_class_install_property(gobject_class, ARG_SRC_START_TIME_GPS, g_param_spec_string("start-time-gps", "Start time", "Start time in GPS seconds.", "0.000000000", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SRC_STOP_TIME_GPS, g_param_spec_string("stop-time-gps", "Stop time", "Stop time in GPS seconds.", "0.000000000", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/*
	 * pushsrc method overrides
	 */

	gstbasesrc_class->start = start;
	gstbasesrc_class->get_caps = get_caps;
	gstpush_src_class->create = create;
	gstbasesrc_class->stop = stop;
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	element->location = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->full_channel_name = NULL;
	XLALGPSSet(&element->start_time, 0, 0);
	XLALGPSSet(&element->stop_time, 0, 0);
	element->next_sample = 0;
	element->stream = NULL;
	element->series_type = -1;
	element->series_buffer_duration = DEFAULT_BUFFER_DURATION;
	element->series_buffer = NULL;

	gst_base_src_set_format(GST_BASE_SRC(object), GST_FORMAT_TIME);
}


/*
 * gstlal_framesrc_get_type().
 */


GType gstlal_framesrc_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALFrameSrcClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALFrameSrc),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_PUSH_SRC, "lal_framesrc", &info, 0);
	}

	return type;
}

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


#include <math.h>
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
#define DEFAULT_UNITS_STRING "strain"
#define DEFAULT_UNITS_UNIT lalStrainUnit


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


/*
 * Type-agnostic time series deallocator
 */


static void replace_input_buffer(GSTLALFrameSrc *element, void *new_input_buffer, LALTYPECODE new_series_type)
{
	if(element->input_buffer) {
		switch(element->series_type) {
		case LAL_I4_TYPE_CODE:
			XLALDestroyINT4TimeSeries(element->input_buffer);
			break;
		case LAL_S_TYPE_CODE:
			XLALDestroyREAL4TimeSeries(element->input_buffer);
			break;
		case LAL_D_TYPE_CODE:
			XLALDestroyREAL8TimeSeries(element->input_buffer);
			break;
		default:
			GST_ERROR("unsupported LAL type code (%d)", element->series_type);
			break;
		}
	}
	element->input_buffer = new_input_buffer;
	element->series_type = new_series_type;
}


/*
 * Convert a generic time series to the matching GstCaps.  Returns the
 * output of gst_caps_new_simple() --- ref()ed caps that need to be
 * unref()ed when done.
 */


static gint series_to_rate(const void *series, LALTYPECODE type)
{
	double deltaT;

	switch(type) {
	case LAL_I4_TYPE_CODE:
		deltaT = ((const INT4TimeSeries *) series)->deltaT;
		break;

	case LAL_S_TYPE_CODE:
		deltaT = ((const REAL4TimeSeries *) series)->deltaT;
		break;

	case LAL_D_TYPE_CODE:
		deltaT = ((const REAL8TimeSeries *) series)->deltaT;
		break;

	default:
		GST_ERROR("unsupported LAL type code (%d)", type);
		return -1;
	}

	return round(1.0 / deltaT);
}


static GstCaps *series_to_caps_and_taglist(const char *instrument, const char *channel_name, const void *series, LALTYPECODE type, GstTagList **taglist)
{
	char units[100];
	GstCaps *caps;

	switch(type) {
	case LAL_I4_TYPE_CODE: {
		const INT4TimeSeries *local_series = series;
		XLALUnitAsString(units, sizeof(units), &local_series->sampleUnits);
		caps = gst_caps_new_simple(
			"audio/x-raw-int",
			"rate", G_TYPE_INT, series_to_rate(series, type),
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 32,
			"depth", G_TYPE_INT, 32,
			"signed", G_TYPE_BOOLEAN, TRUE,
			NULL
		);
		break;
	}

	case LAL_S_TYPE_CODE: {
		const REAL4TimeSeries *local_series = series;
		XLALUnitAsString(units, 100, &local_series->sampleUnits);
		caps = gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", G_TYPE_INT, series_to_rate(series, type),
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 32,
			NULL
		);
		break;
	}

	case LAL_D_TYPE_CODE: {
		const REAL8TimeSeries *local_series = series;
		XLALUnitAsString(units, 100, &local_series->sampleUnits);
		caps = gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", G_TYPE_INT, series_to_rate(series, type),
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 64,
			NULL
		);
		break;
	}

	default:
		GST_ERROR("unsupported LAL type code (%d)", type);
		caps = NULL;
		break;
	}

	if(caps)
		GST_DEBUG("constructed caps:  %" GST_PTR_FORMAT, caps);
	else
		GST_ERROR("failure constructing caps");

	*taglist = gst_tag_list_new_full(
		GSTLAL_TAG_INSTRUMENT, instrument,
		GSTLAL_TAG_CHANNEL_NAME, channel_name,
		GSTLAL_TAG_UNITS, units,
		NULL
	);

	if(*taglist)
		GST_DEBUG("constructed taglist: %" GST_PTR_FORMAT, *taglist);
	else
		GST_ERROR("failure constructing taglist");

	return caps;
}


/*
 * Retrieve a chunk of data from the series buffer, loading more data if
 * needed.
 */


static void *read_series(GSTLALFrameSrc *element, guint64 start_sample, guint64 length)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(element);
	double deltaT;
	guint64 segment_length;
	LIGOTimeGPS input_buffer_epoch;
	guint64 input_buffer_offset;
	guint64 input_buffer_length;

	/*
	 * retrieve the sample rate, timestamp, offset, and size of the
	 * input buffer
	 */

	switch(element->series_type) {
	case LAL_I4_TYPE_CODE: {
		INT4TimeSeries *input_buffer = element->input_buffer;
		deltaT = input_buffer->deltaT;
		input_buffer_epoch = input_buffer->epoch;
		input_buffer_length = input_buffer->data->length;
		break;
	}

	case LAL_S_TYPE_CODE: {
		REAL4TimeSeries *input_buffer = element->input_buffer;
		deltaT = input_buffer->deltaT;
		input_buffer_epoch = input_buffer->epoch;
		input_buffer_length = input_buffer->data->length;
		break;
	}

	case LAL_D_TYPE_CODE: {
		REAL8TimeSeries *input_buffer = element->input_buffer;
		deltaT = input_buffer->deltaT;
		input_buffer_epoch = input_buffer->epoch;
		input_buffer_length = input_buffer->data->length;
		break;
	}

	default:
		GST_ERROR_OBJECT(element, "unsupported LAL type code (%d)", element->series_type);
		return NULL;
	}
	input_buffer_offset = gst_util_uint64_scale_int_round(XLALGPSToINT8NS(&input_buffer_epoch) - basesrc->segment.start, (gint) round(1.0 / deltaT), GST_SECOND);

	/*
	 * segment length (now that we know the sample rate)
	 */

	if(GST_CLOCK_TIME_IS_VALID(basesrc->segment.stop))
		segment_length = gst_util_uint64_scale_int_round(basesrc->segment.stop - basesrc->segment.start, (gint) round(1.0 / deltaT), GST_SECOND);
	else
		segment_length = G_MAXUINT64;

	/*
	 * check for EOF
	 */

	if(start_sample >= segment_length) {
		GST_ERROR_OBJECT(element, "requested interval lies outside input domain");
		return NULL;
	}

	/*
	 * does the requested interval start in the input buffer?  if not,
	 * read a new buffer
	 */

	if(start_sample < input_buffer_offset || start_sample - input_buffer_offset >= input_buffer_length) {
		void *new_input_buffer;

		/*
		 * compute the bounds of the new buffer, using the
		 * requested start time as the buffer's start time
		 */

		input_buffer_offset = start_sample;
		input_buffer_length = round(element->input_buffer_duration / deltaT);
		if(segment_length - input_buffer_offset < input_buffer_length)
			input_buffer_length = segment_length - input_buffer_offset;
		XLALINT8NSToGPS(&input_buffer_epoch, basesrc->segment.start);
		XLALGPSAdd(&input_buffer_epoch, input_buffer_offset * deltaT);

		/*
		 * load the buffer
		 *
		 * NOTE:  frame files cannot be relied on to provide the
		 * correct units, so we unconditionally override them with
		 * a user-supplied value.
		 */

		GST_LOG_OBJECT(element, "read %g seconds of channel %s at %d%.09u\n", input_buffer_length * deltaT, element->full_channel_name, input_buffer_epoch.gpsSeconds, input_buffer_epoch.gpsNanoSeconds);
		switch(element->series_type) {
		case LAL_I4_TYPE_CODE:
			new_input_buffer = XLALFrReadINT4TimeSeries(element->stream, element->full_channel_name, &input_buffer_epoch, input_buffer_length * deltaT, 0);
			if(!new_input_buffer) {
				GST_ERROR_OBJECT(element, "XLALFrReadINT4TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
				XLALClearErrno();
				return NULL;
			}
			((INT4TimeSeries *) (new_input_buffer))->sampleUnits = element->units;
			break;

		case LAL_S_TYPE_CODE:
			new_input_buffer = XLALFrReadREAL4TimeSeries(element->stream, element->full_channel_name, &input_buffer_epoch, input_buffer_length * deltaT, 0);
			if(!new_input_buffer) {
				GST_ERROR_OBJECT(element, "XLALFrReadREAL4TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
				XLALClearErrno();
				return NULL;
			}
			((REAL4TimeSeries *) (new_input_buffer))->sampleUnits = element->units;
			break;

		case LAL_D_TYPE_CODE:
			new_input_buffer = XLALFrReadREAL8TimeSeries(element->stream, element->full_channel_name, &input_buffer_epoch, input_buffer_length * deltaT, 0);
			if(!new_input_buffer) {
				GST_ERROR_OBJECT(element, "XLALFrReadREAL8TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
				XLALClearErrno();
				return NULL;
			}
			((REAL8TimeSeries *) (new_input_buffer))->sampleUnits = element->units;
			break;

		default:
			GST_ERROR_OBJECT(element, "unsupported LAL type code (%d)", element->series_type);
			return NULL;
		}

		/*
		 * new buffer successfully loaded.  replace the old input
		 * buffer with the new one
		 */

		replace_input_buffer(element, new_input_buffer, element->series_type);
	}

	/*
	 * clip the requested interval against the top of the input buffer
	 */

	if(start_sample + length - input_buffer_offset > input_buffer_length)
		length = input_buffer_offset + input_buffer_length - start_sample;

	/*
	 * extract the requested interval
	 */

	switch(element->series_type) {
	case LAL_I4_TYPE_CODE:
		return XLALCutINT4TimeSeries(element->input_buffer, start_sample - input_buffer_offset, length);

	case LAL_S_TYPE_CODE:
		return XLALCutREAL4TimeSeries(element->input_buffer, start_sample - input_buffer_offset, length);

	case LAL_D_TYPE_CODE:
		return XLALCutREAL8TimeSeries(element->input_buffer, start_sample - input_buffer_offset, length);

	default:
		GST_ERROR_OBJECT(element, "unsupported LAL type code (%d)", element->series_type);
		return NULL;
	}
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_SRC_LOCATION = 1,
	ARG_SRC_BUFFER_DURATION,
	ARG_SRC_INSTRUMENT,
	ARG_SRC_CHANNEL_NAME,
	ARG_SRC_UNITS
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SRC_LOCATION:
		free(element->location);
		element->location = g_value_dup_string(value);
		break;

	case ARG_SRC_BUFFER_DURATION:
		element->input_buffer_duration = g_value_get_int(value);
		break;

	case ARG_SRC_INSTRUMENT:
		free(element->instrument);
		element->instrument = g_value_dup_string(value);
		free(element->full_channel_name);
		element->full_channel_name = gstlal_build_full_channel_name(element->instrument, element->channel_name);
		break;

	case ARG_SRC_CHANNEL_NAME:
		free(element->channel_name);
		element->channel_name = g_value_dup_string(value);
		free(element->full_channel_name);
		element->full_channel_name = gstlal_build_full_channel_name(element->instrument, element->channel_name);
		break;

	case ARG_SRC_UNITS: {
		const char *units = g_value_get_string(value);
		if(!units || !strlen(units))
			element->units = lalDimensionlessUnit;
		else
			XLALParseUnitString(&element->units, units);
		break;
	}
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SRC_LOCATION:
		g_value_set_string(value, element->location);
		break;

	case ARG_SRC_BUFFER_DURATION:
		g_value_set_int(value, element->input_buffer_duration);
		break;

	case ARG_SRC_INSTRUMENT:
		g_value_set_string(value, element->instrument);
		break;

	case ARG_SRC_CHANNEL_NAME:
		g_value_set_string(value, element->channel_name);
		break;

	case ARG_SRC_UNITS: {
		char units[100];
		XLALUnitAsString(units, sizeof(units), &element->units);
		g_value_set_string(value, units);
		break;
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
 * get_caps()
 */


static GstCaps *get_caps(GstBaseSrc *object)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);
	GstCaps *caps;

	if(element->input_buffer) {
		GstTagList *taglist;
		caps = series_to_caps_and_taglist(element->instrument, element->channel_name, element->input_buffer, element->series_type, &taglist);
		gst_tag_list_free(taglist);
	} else
		caps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_SRC_PAD(object)));
	return caps;
}


/*
 * start()
 */


static gboolean start(GstBaseSrc *object)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);
	FrCache *cache;
	LIGOTimeGPS stream_start;
	GstCaps *caps;
	GstTagList *taglist;

	/*
	 * Open frame stream.
	 */

	cache = XLALFrImportCache(element->location);
	if(!cache) {
		GST_ERROR_OBJECT(element, "XLALFrImportCache() failed: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
		return FALSE;
	}
	element->stream = XLALFrCacheOpen(cache);
	XLALFrDestroyCache(cache);
	if(!element->stream) {
		GST_ERROR_OBJECT(element, "XLALFrCacheOpen() failed: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
		return FALSE;
	}

	/*
	 * Be verbose, and make gaps in the cache and seeks beyond the end
	 * of the cache errors.
	 */

	XLALFrSetMode(element->stream, (LAL_FR_DEFAULT_MODE | LAL_FR_VERBOSE_MODE) & ~(LAL_FR_IGNOREGAP_MODE | LAL_FR_IGNORETIME_MODE));

	/*
	 * Get the series type and start time.
	 */

	element->series_type = XLALFrGetTimeSeriesType(element->full_channel_name, element->stream);
	/* FIXME:  series_type should not be an unsigned type if negative
	 * values are used to indicate failure */
	if((int) element->series_type < 0) {
		GST_ERROR_OBJECT(element, "XLALFrGetTimeSeriesType() failed: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALFrClose(element->stream);
		element->stream = NULL;
		XLALClearErrno();
		return FALSE;
	}
	XLALGPSSet(&stream_start, element->stream->file->toc->GTimeS[element->stream->pos], element->stream->file->toc->GTimeN[element->stream->pos]);

	/*
	 * Create a zero-length I/O buffer, and populate its metadata
	 *
	 * NOTE:  frame files cannot be relied on to provide the correct
	 * units, so we unconditionally override them with a user-supplied
	 * value.
	 */

	switch(element->series_type) {
	case LAL_I4_TYPE_CODE:
		element->input_buffer = XLALCreateINT4TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!element->input_buffer) {
			GST_ERROR_OBJECT(element, "XLALCreateINT4TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetINT4TimeSeriesMetadata(element->input_buffer, element->stream)) {
			GST_ERROR_OBJECT(element, "XLALFrGetINT4TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALDestroyINT4TimeSeries(element->input_buffer);
			element->input_buffer = NULL;
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		((INT4TimeSeries *) (element->input_buffer))->sampleUnits = element->units;
		break;

	case LAL_S_TYPE_CODE:
		element->input_buffer = XLALCreateREAL4TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!element->input_buffer) {
			GST_ERROR_OBJECT(element, "XLALCreateREAL4TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetREAL4TimeSeriesMetadata(element->input_buffer, element->stream)) {
			GST_ERROR_OBJECT(element, "XLALFrGetREAL4TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALDestroyREAL4TimeSeries(element->input_buffer);
			element->input_buffer = NULL;
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		((REAL4TimeSeries *) (element->input_buffer))->sampleUnits = element->units;
		break;

	case LAL_D_TYPE_CODE:
		element->input_buffer = XLALCreateREAL8TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!element->input_buffer) {
			GST_ERROR_OBJECT(element, "XLALCreateREAL8TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetREAL8TimeSeriesMetadata(element->input_buffer, element->stream)) {
			GST_ERROR_OBJECT(element, "XLALFrGetREAL8TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALDestroyREAL8TimeSeries(element->input_buffer);
			element->input_buffer = NULL;
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		((REAL8TimeSeries *) (element->input_buffer))->sampleUnits = element->units;
		break;

	default:
		GST_ERROR_OBJECT(element, "unsupported data type (LALTYPECODE=%d) for channel \"%s\"", element->series_type, element->full_channel_name);
		XLALFrClose(element->stream);
		element->stream = NULL;
		return FALSE;
	}

	/*
	 * Try setting the caps on the source pad.
	 */

	caps = series_to_caps_and_taglist(element->instrument, element->channel_name, element->input_buffer, element->series_type, &taglist);
	if(!caps) {
		GST_ERROR_OBJECT(element, "unable to construct caps");
		XLALFrClose(element->stream);
		element->stream = NULL;
		replace_input_buffer(element, NULL, -1);
		return FALSE;
	}
	if(!gst_pad_set_caps(GST_BASE_SRC_PAD(object), caps)) {
		gst_caps_unref(caps);
		GST_ERROR_OBJECT(element, "unable to set caps on %s", GST_PAD_NAME(GST_BASE_SRC_PAD(object)));
		XLALFrClose(element->stream);
		element->stream = NULL;
		replace_input_buffer(element, NULL, -1);
		return FALSE;
	}
	gst_caps_unref(caps);

	/*
	 * Transmit the tag list.
	 */

	if(!gst_pad_push_event(GST_BASE_SRC_PAD(object), gst_event_new_tag(taglist)))
		GST_ERROR_OBJECT(element, "unable to push taglist %" GST_PTR_FORMAT " on %s", taglist, GST_PAD_NAME(GST_BASE_SRC_PAD(object)));

	/*
	 * Done
	 */

	return TRUE;
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
	replace_input_buffer(element, NULL, -1);

	return TRUE;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(basesrc);
	GstPad *srcpad = GST_BASE_SRC_PAD(basesrc);
	GstFlowReturn result;

	/*
	 * Just in case
	 */

	*buffer = NULL;

	/*
	 * Read data
	 */

	switch(element->series_type) {
	case LAL_I4_TYPE_CODE: {
		INT4TimeSeries *chunk;
		if(basesrc->blocksize % sizeof(*chunk->data->data)) {
			GST_ERROR_OBJECT(element, "block size not an integer multiple of the sample size");
			return GST_FLOW_ERROR;
		}
		chunk = read_series(element, basesrc->offset, basesrc->blocksize / sizeof(*chunk->data->data));
		if(!chunk) {
			/*
			 * EOS
			 */
			return GST_FLOW_UNEXPECTED;
		}
		result = gst_pad_alloc_buffer(srcpad, basesrc->offset, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(srcpad), buffer);
		if(result != GST_FLOW_OK) {
			XLALDestroyINT4TimeSeries(chunk);
			return result;
		}
		if(basesrc->offset != GST_BUFFER_OFFSET(*buffer)) {
			/* FIXME:  didn't get the buffer offset we asked
			 * for, do something about it */
		}
		memcpy(GST_BUFFER_DATA(*buffer), chunk->data->data, GST_BUFFER_SIZE(*buffer));
		GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + chunk->data->length;
		GST_BUFFER_TIMESTAMP(*buffer) = (GstClockTime) XLALGPSToINT8NS(&chunk->epoch);
		GST_BUFFER_DURATION(*buffer) = (GstClockTime) round(chunk->data->length * GST_SECOND * chunk->deltaT);
		if(basesrc->offset == 0)
			GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);
		basesrc->offset += chunk->data->length;
		XLALDestroyINT4TimeSeries(chunk);
		break;
	}

	case LAL_S_TYPE_CODE: {
		REAL4TimeSeries *chunk;
		if(basesrc->blocksize % sizeof(*chunk->data->data)) {
			GST_ERROR_OBJECT(element, "block size not an integer multiple of the sample size");
			return GST_FLOW_ERROR;
		}
		chunk = read_series(element, basesrc->offset, basesrc->blocksize / sizeof(*chunk->data->data));
		if(!chunk) {
			/*
			 * EOS
			 */
			return GST_FLOW_UNEXPECTED;
		}
		result = gst_pad_alloc_buffer(srcpad, basesrc->offset, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(srcpad), buffer);
		if(result != GST_FLOW_OK) {
			XLALDestroyREAL4TimeSeries(chunk);
			return result;
		}
		if(basesrc->offset != GST_BUFFER_OFFSET(*buffer)) {
			/* FIXME:  didn't get the buffer offset we asked
			 * for, do something about it */
		}
		memcpy(GST_BUFFER_DATA(*buffer), chunk->data->data, GST_BUFFER_SIZE(*buffer));
		GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + chunk->data->length;
		GST_BUFFER_TIMESTAMP(*buffer) = (GstClockTime) XLALGPSToINT8NS(&chunk->epoch);
		GST_BUFFER_DURATION(*buffer) = (GstClockTime) round(chunk->data->length * GST_SECOND * chunk->deltaT);
		if(basesrc->offset == 0)
			GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);
		basesrc->offset += chunk->data->length;
		XLALDestroyREAL4TimeSeries(chunk);
		break;
	}

	case LAL_D_TYPE_CODE: {
		REAL8TimeSeries *chunk;
		if(basesrc->blocksize % sizeof(*chunk->data->data)) {
			GST_ERROR_OBJECT(element, "block size not an integer multiple of the sample size");
			return GST_FLOW_ERROR;
		}
		chunk = read_series(element, basesrc->offset, basesrc->blocksize / sizeof(*chunk->data->data));
		if(!chunk) {
			/*
			 * EOS
			 */
			return GST_FLOW_UNEXPECTED;
		}
		result = gst_pad_alloc_buffer(srcpad, basesrc->offset, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(srcpad), buffer);
		if(result != GST_FLOW_OK) {
			XLALDestroyREAL8TimeSeries(chunk);
			return result;
		}
		if(basesrc->offset != GST_BUFFER_OFFSET(*buffer)) {
			/* FIXME:  didn't get the buffer offset we asked
			 * for, do something about it */
		}
		memcpy(GST_BUFFER_DATA(*buffer), chunk->data->data, GST_BUFFER_SIZE(*buffer));
		GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + chunk->data->length;
		GST_BUFFER_TIMESTAMP(*buffer) = (GstClockTime) XLALGPSToINT8NS(&chunk->epoch);
		GST_BUFFER_DURATION(*buffer) = (GstClockTime) round(chunk->data->length * GST_SECOND * chunk->deltaT);
		if(basesrc->offset == 0)
			GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);
		basesrc->offset += chunk->data->length;
		XLALDestroyREAL8TimeSeries(chunk);
		break;
	}

	default:
		break;
	}

	return GST_FLOW_OK;
}


/*
 * is_seekable()
 */


static gboolean is_seekable(GstBaseSrc *object)
{
	return TRUE;
}


/*
 * do_seek()
 */


static gboolean do_seek(GstBaseSrc *basesrc, GstSegment *segment)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(basesrc);
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
		GST_WARNING_OBJECT(element, "XLALFrSeek() to GPS time %d.%09d s failed: %s", epoch.gpsSeconds, epoch.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
	}

	/*
	 * Done
	 */

	basesrc->offset = 0;
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
 * Parent class.
 */


static GstBaseSrcClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
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
	replace_input_buffer(element, NULL, -1);

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
		"GWF Frame File Source",
		"Source",
		"LAL cache-based .gwf frame file source element",
		"Kipp Cannon <kcannon@ligo.caltech.edu>"
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

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->finalize = finalize;

	g_object_class_install_property(gobject_class, ARG_SRC_LOCATION, g_param_spec_string("location", "Location", "Path to LAL cache file (see LSCdataFind for more information).", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SRC_BUFFER_DURATION, g_param_spec_int("buffer-duration", "Buffer duration", "Size of input buffer in seconds.", 1, 2048, DEFAULT_BUFFER_DURATION, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SRC_INSTRUMENT, g_param_spec_string("instrument", "Instrument", "Instrument name (e.g., \"H1\").", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SRC_CHANNEL_NAME, g_param_spec_string("channel-name", "Channel name", "Channel name (e.g., \"LSC-STRAIN\").", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SRC_UNITS, g_param_spec_string("units", "Units", "Units string parsable by LAL's Units code (e.g., \"strain\" or \"counts\"). null or an empty string means dimensionless.", DEFAULT_UNITS_STRING, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/*
	 * GstBaseSrc method overrides
	 */

	gstbasesrc_class->get_caps = get_caps;
	gstbasesrc_class->start = start;
	gstbasesrc_class->stop = stop;
	gstbasesrc_class->create = create;
	gstbasesrc_class->is_seekable = is_seekable;
	gstbasesrc_class->do_seek = do_seek;
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(object);
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	basesrc->offset = 0;
	element->location = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->full_channel_name = NULL;
	element->stream = NULL;
	element->units = DEFAULT_UNITS_UNIT;
	element->input_buffer_duration = DEFAULT_BUFFER_DURATION;
	element->input_buffer = NULL;
	element->series_type = -1;

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
		type = g_type_register_static(GST_TYPE_BASE_SRC, "lal_framesrc", &info, 0);
	}

	return type;
}

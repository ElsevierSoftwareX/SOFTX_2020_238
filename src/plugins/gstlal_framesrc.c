/*
 * LAL cache-based .gwf frame file src element
 *
 * Copyright (C) 2008  Kipp C. Cannon
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
 * Convert a generic time series to the matching GstCaps.  Returns the
 * output of gst_caps_new_simple() --- ref()ed caps that need to be
 * unref()ed when done.
 */


static GstCaps *series_to_caps(gint rate, LALTYPECODE type)
{
	GstCaps *caps;

	switch(type) {
	case LAL_I4_TYPE_CODE:
		caps = gst_caps_new_simple(
			"audio/x-raw-int",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 32,
			"depth", G_TYPE_INT, 32,
			"signed", G_TYPE_BOOLEAN, TRUE,
			NULL
		);
		break;

	case LAL_S_TYPE_CODE:
		caps = gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 32,
			NULL
		);
		break;

	case LAL_D_TYPE_CODE:
		caps = gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 64,
			NULL
		);
		break;

	default:
		GST_ERROR("unsupported LAL type code (%d)", type);
		caps = NULL;
		break;
	}

	if(caps)
		GST_DEBUG("constructed caps:  %" GST_PTR_FORMAT, caps);
	else
		GST_ERROR("failure constructing caps");

	return caps;
}


/*
 * Retrieve a chunk of data.
 */


static GstFlowReturn read_series(GSTLALFrameSrc *element, guint64 offset, guint64 length, void** out_series)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(element);
	guint64 segment_length;
	LIGOTimeGPS start_time;
	void *series;

	/*
	 * segment length
	 */

	if(GST_CLOCK_TIME_IS_VALID(basesrc->segment.stop))
		segment_length = gst_util_uint64_scale_int_round(basesrc->segment.stop - basesrc->segment.start, element->rate, GST_SECOND);
	else
		segment_length = G_MAXUINT64;

	/*
	 * check for EOF, clip requested length to segment
	 */

	if(offset >= segment_length) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, ("requested interval lies outside input domain"), (NULL));
		return GST_FLOW_ERROR;
	}
	if(segment_length - offset < length)
		length = segment_length - offset;

	/*
	 * convert the start sample to a start time
	 */

	XLALINT8NSToGPS(&start_time, basesrc->segment.start + gst_util_uint64_scale_int_round(offset, GST_SECOND, element->rate));

	/*
	 * load the buffer
	 *
	 * NOTE:  frame files cannot be relied on to provide the correct
	 * units, so we unconditionally override them with a user-supplied
	 * value.
	 */

	GST_LOG_OBJECT(element, "reading %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" at %d.%09u s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds);
	switch(element->series_type) {
	case LAL_I4_TYPE_CODE:
		series = XLALFrReadINT4TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, ("XLALFrReadINT4TimeSeries() %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" at %d.%09u s failed", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		break;

	case LAL_S_TYPE_CODE:
		series = XLALFrReadREAL4TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, ("XLALFrReadREAL4TimeSeries() %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" at %d.%09u s failed", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		break;

	case LAL_D_TYPE_CODE:
		series = XLALFrReadREAL8TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, ("XLALFrReadREAL8TimeSeries() %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" at %d.%09u s failed", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		break;

	default:
		GST_ELEMENT_ERROR(element, RESOURCE, READ, ("unsupported LAL type code (%d)", element->series_type), (NULL));
		return GST_FLOW_ERROR;
	}

	/*
	 * done
	 */

	/* FIXME: Can XLALFrRead?????TimeSeries can return NULL on success? */

	*out_series = series;
	return GST_FLOW_OK;
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
		g_free(element->location);
		element->location = g_value_dup_string(value);
		break;

	case ARG_SRC_INSTRUMENT:
		g_free(element->instrument);
		element->instrument = g_value_dup_string(value);
		g_free(element->full_channel_name);
		element->full_channel_name = gstlal_build_full_channel_name(element->instrument, element->channel_name);
		break;

	case ARG_SRC_CHANNEL_NAME:
		g_free(element->channel_name);
		element->channel_name = g_value_dup_string(value);
		g_free(element->full_channel_name);
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
 * start()
 */


static gboolean start(GstBaseSrc *object)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);
	FrCache *cache;
	LIGOTimeGPS stream_start;
	gint rate, width;
	GstCaps *caps;

	/*
	 * Make a note to push tags before emitting next frame.
	 */

	element->need_tags = TRUE;

	/*
	 * Open frame stream.
	 */

	cache = XLALFrImportCache(element->location);
	if(!cache) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALFrImportCache() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		return FALSE;
	}
	element->stream = XLALFrCacheOpen(cache);
	XLALFrDestroyCache(cache);
	if(!element->stream) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALFrCacheOpen() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
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
	if((int) element->series_type < 0) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALFrGetTimeSeriesType() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
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
	case LAL_I4_TYPE_CODE: {
		INT4TimeSeries *series = XLALCreateINT4TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALCreateINT4TimeSeries() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetINT4TimeSeriesMetadata(series, element->stream)) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALFrGetINT4TimeSeriesMetadata() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALDestroyINT4TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		width = 32;
		caps = series_to_caps(rate, element->series_type);
		XLALDestroyINT4TimeSeries(series);
		break;
	}

	case LAL_S_TYPE_CODE: {
		REAL4TimeSeries *series = XLALCreateREAL4TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALCreateREAL4TimeSeries() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetREAL4TimeSeriesMetadata(series, element->stream)) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALFrGetREAL4TimeSeriesMetadata() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALDestroyREAL4TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		width = 32;
		caps = series_to_caps(rate, element->series_type);
		XLALDestroyREAL4TimeSeries(series);
		break;
	}

	case LAL_D_TYPE_CODE: {
		REAL8TimeSeries *series = XLALCreateREAL8TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALCreateREAL8TimeSeries() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetREAL8TimeSeriesMetadata(series, element->stream)) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("XLALFrGetREAL8TimeSeriesMetadata() failed"), ("%s", XLALErrorString(XLALGetBaseErrno())));
			XLALDestroyREAL8TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		width = 64;
		caps = series_to_caps(rate, element->series_type);
		XLALDestroyREAL8TimeSeries(series);
		break;
	}

	default:
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("unsupported data type (LALTYPECODE=%d) for channel \"%s\"", element->series_type, element->full_channel_name), (NULL));
		XLALFrClose(element->stream);
		element->stream = NULL;
		return FALSE;
	}

	/*
	 * Try setting the caps on the source pad.
	 */

	if(!caps) {
		GST_ERROR_OBJECT(element, "unable to construct caps");
		XLALFrClose(element->stream);
		element->stream = NULL;
		return FALSE;
	}
	if(!gst_pad_set_caps(GST_BASE_SRC_PAD(object), caps)) {
		gst_caps_unref(caps);
		GST_ERROR_OBJECT(element, "unable to set caps %" GST_PTR_FORMAT " on %s", caps, GST_PAD_NAME(GST_BASE_SRC_PAD(object)));
		XLALFrClose(element->stream);
		element->stream = NULL;
		return FALSE;
	}
	gst_caps_unref(caps);

	/*
	 * Done
	 */

	element->rate = rate;
	element->width = width;

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

	return TRUE;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(basesrc);
	GstFlowReturn result;

	/*
	 * Push tag list if we haven't already
	 */

	if (element->need_tags) {
		GstEvent *evt;
		GstTagList *taglist;
		char units[100];
		XLALUnitAsString(units, sizeof(units), &element->units);

		taglist = gst_tag_list_new_full(
			GSTLAL_TAG_INSTRUMENT, element->instrument,
			GSTLAL_TAG_CHANNEL_NAME, element->channel_name,
			GSTLAL_TAG_UNITS, units,
			NULL
		);

		if (!taglist) {
			GST_ELEMENT_ERROR(element, CORE, TAG, ("failure constructing taglist"), (NULL));
			return GST_FLOW_ERROR;
		}

		evt = gst_event_new_tag(taglist);

		if (!evt) {
			GST_ELEMENT_ERROR(element, CORE, TAG, ("failure constructing tag event"), (NULL));
			return GST_FLOW_ERROR;
		}

		if (!gst_pad_push_event(GST_BASE_SRC_PAD(basesrc), evt)) {
			GST_ELEMENT_ERROR(element, CORE, TAG, ("failure pusing tag event"), (NULL));
			return GST_FLOW_ERROR;
		}

		element->need_tags = FALSE;
	}

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
		if(gst_base_src_get_blocksize(basesrc) % sizeof(*chunk->data->data)) {
			GST_ERROR_OBJECT(element, "block size %lu is not an integer multiple of the sample size %lu", gst_base_src_get_blocksize(basesrc), sizeof(*chunk->data->data));
			return GST_FLOW_ERROR;
		}
		result = read_series(element, basesrc->offset, gst_base_src_get_blocksize(basesrc) / sizeof(*chunk->data->data), &chunk);
		if (result != GST_FLOW_OK)
			return result;
		result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
		if(result != GST_FLOW_OK) {
			XLALDestroyINT4TimeSeries(chunk);
			return result;
		}
		if(basesrc->offset != GST_BUFFER_OFFSET(*buffer) || chunk->data->length * sizeof(*chunk->data->data) != GST_BUFFER_SIZE(*buffer)) {
			/* FIXME:  didn't get the buffer offset we asked
			 * for, do something about it */
			GST_FIXME_OBJECT(element, "gst_pad_alloc_buffer() didn't give us the offset we asked for.  do something about it, but what?");
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
		if(gst_base_src_get_blocksize(basesrc) % sizeof(*chunk->data->data)) {
			GST_ERROR_OBJECT(element, "block size %lu is not an integer multiple of the sample size %lu", gst_base_src_get_blocksize(basesrc), sizeof(*chunk->data->data));
			return GST_FLOW_ERROR;
		}
		result = read_series(element, basesrc->offset, gst_base_src_get_blocksize(basesrc) / sizeof(*chunk->data->data), &chunk);
		if (result != GST_FLOW_OK)
			return result;
		result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
		if(result != GST_FLOW_OK) {
			XLALDestroyREAL4TimeSeries(chunk);
			return result;
		}
		if(basesrc->offset != GST_BUFFER_OFFSET(*buffer) || chunk->data->length * sizeof(*chunk->data->data) != GST_BUFFER_SIZE(*buffer)) {
			/* FIXME:  didn't get the buffer offset we asked
			 * for, do something about it */
			GST_FIXME_OBJECT(element, "gst_pad_alloc_buffer() didn't give us the offset we asked for.  do something about it, but what?");
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
		if(gst_base_src_get_blocksize(basesrc) % sizeof(*chunk->data->data)) {
			GST_ERROR_OBJECT(element, "block size %lu is not an integer multiple of the sample size %lu", gst_base_src_get_blocksize(basesrc), sizeof(*chunk->data->data));
			return GST_FLOW_ERROR;
		}
		result = read_series(element, basesrc->offset, gst_base_src_get_blocksize(basesrc) / sizeof(*chunk->data->data), &chunk);
		if (result != GST_FLOW_OK)
			return result;
		result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, chunk->data->length * sizeof(*chunk->data->data), GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
		if(result != GST_FLOW_OK) {
			XLALDestroyREAL8TimeSeries(chunk);
			return result;
		}
		if(basesrc->offset != GST_BUFFER_OFFSET(*buffer) || chunk->data->length * sizeof(*chunk->data->data) != GST_BUFFER_SIZE(*buffer)) {
			/* FIXME:  didn't get the buffer offset we asked
			 * for, do something about it */
			GST_FIXME_OBJECT(element, "gst_pad_alloc_buffer() didn't give us the offset we asked for.  do something about it, but what?");
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
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, ("seek failed:  start time is required"), (NULL));
		return FALSE;
	}
	XLALINT8NSToGPS(&epoch, segment->start);

	/*
	 * Try doing the seek
	 */

	if(XLALFrSeek(element->stream, &epoch)) {
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, ("XLALFrSeek() to %d.%09u s failed", epoch.gpsSeconds, epoch.gpsNanoSeconds), ("%s", XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		return FALSE;
	}

	/*
	 * Done
	 */

	basesrc->offset = 0;
	return TRUE;
}


/*
 * query
 */


static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(basesrc);

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
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	g_free(element->location);
	element->location = NULL;
	g_free(element->instrument);
	element->instrument = NULL;
	g_free(element->channel_name);
	element->channel_name = NULL;
	g_free(element->full_channel_name);
	element->full_channel_name = NULL;
	if(element->stream) {
		XLALFrClose(element->stream);
		element->stream = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"GWF Frame File Source",
		"Source",
		"LAL cache-based .gwf frame file source element",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

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

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_SRC_LOCATION,
		g_param_spec_string(
			"location",
			"Location",
			"Path to LAL cache file (see ligo_data_find for more information).",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_SRC_INSTRUMENT,
		g_param_spec_string(
			"instrument",
			"Instrument",
			"Instrument name (e.g., \"H1\").",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_SRC_CHANNEL_NAME,
		g_param_spec_string(
			"channel-name",
			"Channel name",
			"Channel name (e.g., \"LSC-STRAIN\").",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_SRC_UNITS,
		g_param_spec_string(
			"units",
			"Units",
			"Units string parsable by LAL's Units code (e.g., \"strain\" or \"counts\"). null or an empty string means dimensionless.",
			DEFAULT_UNITS_STRING,
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
	gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);
	gstbasesrc_class->check_get_range = GST_DEBUG_FUNCPTR(check_get_range);
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

	gst_pad_use_fixed_caps(GST_BASE_SRC_PAD(basesrc));

	basesrc->offset = 0;
	element->location = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->full_channel_name = NULL;
	element->rate = 0;
	element->width = 0;
	element->stream = NULL;
	element->units = DEFAULT_UNITS_UNIT;
	element->series_type = -1;

	gst_base_src_set_format(GST_BASE_SRC(object), GST_FORMAT_TIME);

	/*
	 * Make a note to push tags before emitting next frame.
	 */

	element->need_tags = TRUE;

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

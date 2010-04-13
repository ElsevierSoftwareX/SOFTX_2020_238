/*
 * LAL online h(t) src element
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
#include <gstlal_onlinehoftsrc.h>


/*
 * Parent class.
 */


static GstPushSrcClass *parent_class = NULL;


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


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


static GstCaps *series_to_caps_and_taglist(const char *instrument, const char *channel_name, LALUnit sampleUnits, gint rate, LALTYPECODE type, GstTagList **taglist)
{
	char units[100];
	GstCaps *caps;

	XLALUnitAsString(units, sizeof(units), &sampleUnits);

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

	if(taglist) {
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
	}

	return caps;
}


/*
 * Retrieve a chunk of data.
 */


#if 0
static void *read_series(GSTLALOnlineHoftSrc *element, guint64 offset, guint64 length)
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
		GST_ERROR_OBJECT(element, "requested interval lies outside input domain");
		return NULL;
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

	GST_LOG_OBJECT(element, "reading %lu samples (%g seconds) of channel \"%s\" at %d.%09u s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds);
	switch(element->series_type) {
	case LAL_I4_TYPE_CODE:
		series = XLALFrReadINT4TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ERROR_OBJECT(element, "XLALFrReadINT4TimeSeries() %lu samples (%g seconds) of channel \"%s\" at %d.%09u s failed: %s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return NULL;
		}
		break;

	case LAL_S_TYPE_CODE:
		series = XLALFrReadREAL4TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ERROR_OBJECT(element, "XLALFrReadREAL4TimeSeries() %lu samples (%g seconds) of channel \"%s\" at %d.%09u s failed: %s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return NULL;
		}
		break;

	case LAL_D_TYPE_CODE:
		series = XLALFrReadREAL8TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ERROR_OBJECT(element, "XLALFrReadREAL8TimeSeries() %lu samples (%g seconds) of channel \"%s\" at %d.%09u s failed: %s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return NULL;
		}
		break;

	default:
		GST_ERROR_OBJECT(element, "unsupported LAL type code (%d)", element->series_type);
		return NULL;
	}

	/*
	 * done
	 */

	return series;
}
#endif



/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_SRC_INSTRUMENT = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SRC_INSTRUMENT:
		g_free(element->instrument);
		element->instrument = g_value_dup_string(value);
		break;

	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SRC_INSTRUMENT:
		g_value_set_string(value, element->instrument);
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

#if 0
static gboolean start(GstBaseSrc *object)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);
	FrCache *cache;
	LIGOTimeGPS stream_start;
	gint rate, width;
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

	taglist = NULL;
	switch(element->series_type) {
	case LAL_I4_TYPE_CODE: {
		INT4TimeSeries *series = XLALCreateINT4TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ERROR_OBJECT(element, "XLALCreateINT4TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetINT4TimeSeriesMetadata(series, element->stream)) {
			GST_ERROR_OBJECT(element, "XLALFrGetINT4TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALDestroyINT4TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		width = 32;
		caps = series_to_caps_and_taglist(element->instrument, element->channel_name, element->units, rate, element->series_type, &taglist);
		XLALDestroyINT4TimeSeries(series);
		break;
	}

	case LAL_S_TYPE_CODE: {
		REAL4TimeSeries *series = XLALCreateREAL4TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ERROR_OBJECT(element, "XLALCreateREAL4TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetREAL4TimeSeriesMetadata(series, element->stream)) {
			GST_ERROR_OBJECT(element, "XLALFrGetREAL4TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALDestroyREAL4TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		width = 32;
		caps = series_to_caps_and_taglist(element->instrument, element->channel_name, element->units, rate, element->series_type, &taglist);
		XLALDestroyREAL4TimeSeries(series);
		break;
	}

	case LAL_D_TYPE_CODE: {
		REAL8TimeSeries *series = XLALCreateREAL8TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ERROR_OBJECT(element, "XLALCreateREAL8TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetREAL8TimeSeriesMetadata(series, element->stream)) {
			GST_ERROR_OBJECT(element, "XLALFrGetREAL8TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALDestroyREAL8TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		width = 64;
		caps = series_to_caps_and_taglist(element->instrument, element->channel_name, element->units, rate, element->series_type, &taglist);
		XLALDestroyREAL8TimeSeries(series);
		break;
	}

	default:
		GST_ERROR_OBJECT(element, "unsupported data type (LALTYPECODE=%d) for channel \"%s\"", element->series_type, element->full_channel_name);
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
	 * Transmit the tag list.
	 */

	if(taglist && !gst_pad_push_event(GST_BASE_SRC_PAD(object), gst_event_new_tag(taglist)))
		GST_ERROR_OBJECT(element, "unable to push taglist %" GST_PTR_FORMAT " on %s", taglist, GST_PAD_NAME(GST_BASE_SRC_PAD(object)));
	taglist = NULL;	/* gst_event_new_tag() took ownership */

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
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	if(element->stream) {
		XLALFrClose(element->stream);
		element->stream = NULL;
	}

	return TRUE;
}
#endif


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
    return GST_FLOW_ERROR; /// TODO
}


#if 0
static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(basesrc);
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
		if(gst_base_src_get_blocksize(basesrc) % sizeof(*chunk->data->data)) {
			GST_ERROR_OBJECT(element, "block size %u is not an integer multiple of the sample size %u", gst_base_src_get_blocksize(basesrc), sizeof(*chunk->data->data));
			return GST_FLOW_ERROR;
		}
		chunk = read_series(element, basesrc->offset, gst_base_src_get_blocksize(basesrc) / sizeof(*chunk->data->data));
		if(!chunk) {
			/*
			 * EOS
			 */
			return GST_FLOW_UNEXPECTED;
		}
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
			GST_ERROR_OBJECT(element, "block size %u is not an integer multiple of the sample size %u", gst_base_src_get_blocksize(basesrc), sizeof(*chunk->data->data));
			return GST_FLOW_ERROR;
		}
		chunk = read_series(element, basesrc->offset, gst_base_src_get_blocksize(basesrc) / sizeof(*chunk->data->data));
		if(!chunk) {
			/*
			 * EOS
			 */
			return GST_FLOW_UNEXPECTED;
		}
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
			GST_ERROR_OBJECT(element, "block size %u is not an integer multiple of the sample size %u", gst_base_src_get_blocksize(basesrc), sizeof(*chunk->data->data));
			return GST_FLOW_ERROR;
		}
		chunk = read_series(element, basesrc->offset, gst_base_src_get_blocksize(basesrc) / sizeof(*chunk->data->data));
		if(!chunk) {
			/*
			 * EOS
			 */
			return GST_FLOW_UNEXPECTED;
		}
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
#endif



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
    /// TODO
    return FALSE;
}


#if 0
static gboolean do_seek(GstBaseSrc *basesrc, GstSegment *segment)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(basesrc);
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
 * query
 */


#if 0
static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(basesrc);

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
#endif


/*
 * check_get_range()
 */


#if 0
static gboolean check_get_range(GstBaseSrc *basesrc)
{
	return TRUE;
}
#endif


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
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	g_free(element->instrument);
	element->instrument = NULL;

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
		"Online h(t) Source",
		"Source",
		"LAL online h(t) source",
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
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64} " \
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

	parent_class = g_type_class_ref(GST_TYPE_PUSH_SRC);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

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

	/*
	 * GstBaseSrc method overrides
	 */

	//gstbasesrc_class->start = GST_DEBUG_FUNCPTR(start);
	//gstbasesrc_class->stop = GST_DEBUG_FUNCPTR(stop);
	gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
	gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR(is_seekable);
	gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
	//gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);
	//gstbasesrc_class->check_get_range = GST_DEBUG_FUNCPTR(check_get_range);
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(object);
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	gst_pad_use_fixed_caps(GST_BASE_SRC_PAD(basesrc));

	basesrc->offset = 0;
	element->instrument = NULL;
	element->rate = 0;
	element->width = 0;
	element->stream = NULL;
	element->series_type = -1;

	gst_base_src_set_format(GST_BASE_SRC(object), GST_FORMAT_TIME);
}


/*
 * gstlal_onlinehoftsrc_get_type().
 */


GType gstlal_onlinehoftsrc_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALOnlineHoftSrcClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALOnlineHoftSrc),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_PUSH_SRC, "lal_onlinehoftsrc", &info, 0);
	}

	return type;
}

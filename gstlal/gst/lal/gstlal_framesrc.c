/*
 * LAL cache-based .gwf frame file src element
 *
 * Copyright (C) 2008-2011  Kipp Cannon
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
#include <gst/base/gstbasesrc.h>


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
#include <gstlal_debug.h>
#include <gstlal_segments.h>
#include <gstlal_tags.h>
#include <gstlal_framesrc.h>


/*
 * ========================================================================
 *
 *                                Boilerplate
 *
 * ========================================================================
 */


#define GST_CAT_DEFAULT lal_framesrc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_framesrc", 0, "lal_framesrc element");
}


GST_BOILERPLATE_FULL(GSTLALFrameSrc, gstlal_framesrc, GstBaseSrc, GST_TYPE_BASE_SRC, additional_initializations);


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
 * Get the unit size.
 */


static guint unit_size(GSTLALFrameSrc *element)
{
	return element->width / 8;
}


/*
 * Convert a generic time series to the matching GstCaps.  Returns the
 * output of gst_caps_new_simple() --- ref()ed caps that need to be
 * unref()ed when done.
 */


static GstCaps *series_to_caps(gint rate, LALTYPECODE type)
{
	GstCaps *caps;

	switch(type) {
	case LAL_I2_TYPE_CODE:
	case LAL_I4_TYPE_CODE:
		caps = gst_caps_new_simple(
			"audio/x-raw-int",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, type == LAL_I2_TYPE_CODE ? 16 : 32,
			"depth", G_TYPE_INT, type == LAL_I2_TYPE_CODE ? 16 : 32,
			"signed", G_TYPE_BOOLEAN, TRUE,
			NULL
		);
		break;

	case LAL_S_TYPE_CODE:
	case LAL_D_TYPE_CODE:
		caps = gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, type == LAL_S_TYPE_CODE ? 32 : 64,
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
 * Convert sample offsets to/from times
 */


static GstClockTime offset_to_time(GSTLALFrameSrc *element, guint64 offset)
{
	g_assert(GST_CLOCK_TIME_IS_VALID(GST_BASE_SRC(element)->segment.start));
	return GST_BASE_SRC(element)->segment.start + gst_util_uint64_scale_int_round(offset, GST_SECOND, element->rate);
}


static guint64 time_to_offset(GSTLALFrameSrc *element, GstClockTime t)
{
	g_assert(GST_CLOCK_TIME_IS_VALID(GST_BASE_SRC(element)->segment.start));
	if(t < (GstClockTime) GST_BASE_SRC(element)->segment.start)
		return 0;
	return gst_util_uint64_scale_int_round(t - GST_BASE_SRC(element)->segment.start, element->rate, GST_SECOND);
}


/*
 * Determine the size of the next output buffer, and whether or not it's a
 * gap
 */


static guint64 get_next_buffer_length(GSTLALFrameSrc *element, guint64 offset, gboolean *gap)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(element);
	guint64 segment_length;
	guint64 length;

	/*
	 * use the blocksize to set the target buffer length
	 */

	if(gst_base_src_get_blocksize(basesrc) % unit_size(element))
		GST_WARNING_OBJECT(element, "block size %lu is not an integer multiple of the sample size %lu, rounding up", gst_base_src_get_blocksize(basesrc), unit_size(element));
	length = (gst_base_src_get_blocksize(basesrc) + unit_size(element) - 1) / unit_size(element);	/* ceil */
	if(!length)
		length = 1;

	/*
	 * check for EOF, clip buffer length to seek segment
	 */

	if(GST_CLOCK_TIME_IS_VALID(basesrc->segment.stop))
		segment_length = time_to_offset(element, basesrc->segment.stop);
	else
		segment_length = G_MAXUINT64;

	if(offset >= segment_length) {
		GST_WARNING_OBJECT(element, "requested interval lies outside input domain");
		length = 0;
	}
	if(segment_length - offset < length)
		length = segment_length - offset;

	/*
	 * if a segment list is set, figure out if we are in a gap and clip
	 * the buffer length to the current segment
	 */

	if(element->segmentlist) {
		gint index = gstlal_segment_list_index(element->segmentlist, offset_to_time(element, offset));
		guint64 end_offset;

		if(!gstlal_segment_list_length(element->segmentlist)) {
			/* no segments in list */
			*gap = TRUE;
			end_offset = G_MAXUINT64;
		} else if(index < 0) {
			/* current time precedes segments in list */
			*gap = TRUE;
			end_offset = time_to_offset(element, gstlal_segment_list_get(element->segmentlist, 0)->start);
		} else if(time_to_offset(element, gstlal_segment_list_get(element->segmentlist, index)->stop) <= offset) {
			/* current time is past end of most recent segment
			 * in list */
			*gap = TRUE;
			if(index < gstlal_segment_list_length(element->segmentlist) - 1)
				/* there is another segment after that one */
				end_offset = time_to_offset(element, gstlal_segment_list_get(element->segmentlist, index + 1)->start);
			else
				/* that was the last segment */
				end_offset = G_MAXUINT64;
		} else {
			/* current time is in most recent segment */
			*gap = FALSE;
			end_offset = time_to_offset(element, gstlal_segment_list_get(element->segmentlist, index)->stop);
		}

		if(end_offset - offset < length)
			length = end_offset - offset;
	} else {
		*gap = FALSE;
	}

	/*
	 * done
	 */

	GST_DEBUG_OBJECT(element, "at offset %" G_GUINT64_FORMAT " buffer will be %s of %" G_GUINT64_FORMAT " samples", offset, *gap ? "gap" : "non-gap", length);
	return length;
}


/*
 * Retrieve a chunk of data.
 */


static GstFlowReturn read_series(GSTLALFrameSrc *element, guint64 offset, guint64 length, void *dst)
{
	LIGOTimeGPS start_time;

	/*
	 * convert the start sample to a start time
	 */

	XLALINT8NSToGPS(&start_time, offset_to_time(element, offset));

	/*
	 * load the buffer
	 *
	 * NOTE:  frame files cannot be relied on to provide the correct
	 * units, so we unconditionally override them with a user-supplied
	 * value.
	 *
	 * NOTE:  we do our own time stamp calculation, we do not rely on
	 * the timestamps returned y LAL for the buffer metadata.  we do
	 * this to ensure we have control over the rounding direction when
	 * the starting sample of a buffer does not correspond to an
	 * integer nanosecond.  however, we do include a safety check that
	 * requires the timestamp we compute and the timestamp LAL computes
	 * to never disagree by more than 1 ns.
	 */

	GST_LOG_OBJECT(element, "reading %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" starting at %d.%09u s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds);
	switch(element->series_type) {
	case LAL_I2_TYPE_CODE: {
		INT2TimeSeries *series = XLALFrReadINT2TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("XLALFrReadINT2TimeSeries() %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" at %d.%09u s failed: %s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno())));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		g_assert_cmpuint(llabs((gint64) offset_to_time(element, offset) - (gint64) XLALGPSToINT8NS(&series->epoch)), <=, 1);
		g_assert_cmpuint(round(1.0 / series->deltaT), ==, element->rate);
		g_assert_cmpuint(series->data->length, ==, length);
		g_assert_cmpuint(unit_size(element), ==, sizeof(*series->data->data));
		memcpy(dst, series->data->data, length * unit_size(element));
		XLALDestroyINT2TimeSeries(series);
	}
		break;

	case LAL_I4_TYPE_CODE: {
		INT4TimeSeries *series = XLALFrReadINT4TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("XLALFrReadINT4TimeSeries() %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" at %d.%09u s failed: %s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno())));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		g_assert_cmpuint(llabs((gint64) offset_to_time(element, offset) - (gint64) XLALGPSToINT8NS(&series->epoch)), <=, 1);
		g_assert_cmpuint(round(1.0 / series->deltaT), ==, element->rate);
		g_assert_cmpuint(series->data->length, ==, length);
		g_assert_cmpuint(unit_size(element), ==, sizeof(*series->data->data));
		memcpy(dst, series->data->data, length * unit_size(element));
		XLALDestroyINT4TimeSeries(series);
	}
		break;

	case LAL_S_TYPE_CODE: {
		REAL4TimeSeries *series = XLALFrReadREAL4TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("XLALFrReadREAL4TimeSeries() %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" at %d.%09u s failed: %s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno())));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		g_assert_cmpuint(llabs((gint64) offset_to_time(element, offset) - (gint64) XLALGPSToINT8NS(&series->epoch)), <=, 1);
		g_assert_cmpuint(round(1.0 / series->deltaT), ==, element->rate);
		g_assert_cmpuint(series->data->length, ==, length);
		g_assert_cmpuint(unit_size(element), ==, sizeof(*series->data->data));
		memcpy(dst, series->data->data, length * unit_size(element));
		XLALDestroyREAL4TimeSeries(series);
	}
		break;

	case LAL_D_TYPE_CODE: {
		REAL8TimeSeries *series = XLALFrReadREAL8TimeSeries(element->stream, element->full_channel_name, &start_time, (double) length / element->rate, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("XLALFrReadREAL8TimeSeries() %" G_GUINT64_FORMAT " samples (%g seconds) of channel \"%s\" at %d.%09u s failed: %s", length, (double) length / element->rate, element->full_channel_name, start_time.gpsSeconds, start_time.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno())));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		g_assert_cmpuint(llabs((gint64) offset_to_time(element, offset) - (gint64) XLALGPSToINT8NS(&series->epoch)), <=, 1);
		g_assert_cmpuint(round(1.0 / series->deltaT), ==, element->rate);
		g_assert_cmpuint(series->data->length, ==, length);
		g_assert_cmpuint(unit_size(element), ==, sizeof(*series->data->data));
		memcpy(dst, series->data->data, length * unit_size(element));
		XLALDestroyREAL8TimeSeries(series);
	}
		break;

	default:
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("unsupported LAL type code (%d)", element->series_type));
		return GST_FLOW_ERROR;
	}

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * Open the LAL frame cache and sieve for an instrument
 */


static FrStream *open_frstream(GSTLALFrameSrc *element, const char *filename, const char *cache_src_regex, const char *cache_dsc_regex)
{
	FrCacheSieve sieve;
	FrCache *fullcache;
	FrCache *cache;
	FrStream *stream;

	/* 
	 * Set up seive params
	 */

	sieve.earliestTime = 0;
	sieve.latestTime = G_MAXINT32;
	sieve.urlRegEx = NULL;
	sieve.dscRegEx = cache_dsc_regex;
	sieve.srcRegEx = cache_src_regex;

	/*
	 * Open frame stream
	 */

	fullcache = XLALFrImportCache(filename);
	if(!fullcache) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALFrImportCache() failed: %s", XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		return NULL;
	}

	/*
	 * Sieve the cache for the cache_src_regex of interest
	 */

	cache = XLALFrSieveCache(fullcache, &sieve);
	XLALFrDestroyCache(fullcache);
	if(!cache) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALFrSieveCache() failed: %s", XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		return NULL;
	}

	/*
	 * Open the stream
	 */

	stream = XLALFrCacheOpen(cache);
	XLALFrDestroyCache(cache);
	if(!stream) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALFrCacheOpen() failed: %s", XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		return NULL;
	}

	/*
	 * Success
	 */

	return stream;
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
	LIGOTimeGPS stream_start;
	gint rate, width;
	GstCaps *caps;
	
	/*
	 * Make a note to push tags before emitting next frame.
	 */

	element->need_tags = TRUE;

	/*
	 * Open the frame cache sieved by instrument
	 */

	element->stream = open_frstream(element, element->location, element->cache_src_regex, element->cache_dsc_regex);
	if(!element->stream)
		return FALSE;

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
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALFrGetTimeSeriesType() failed: %s", XLALErrorString(XLALGetBaseErrno())));
		XLALFrClose(element->stream);
		element->stream = NULL;
		XLALClearErrno();
		return FALSE;
	}
	XLALGPSSet(&stream_start, element->stream->file->toc->GTimeS[element->stream->pos], element->stream->file->toc->GTimeN[element->stream->pos]);

	/*
	 * Create a zero-length I/O buffer and populate its metadata
	 *
	 * NOTE:  frame files cannot be relied on to provide the correct
	 * units, so we unconditionally override them with a user-supplied
	 * value.
	 */

	switch(element->series_type) {
	case LAL_I2_TYPE_CODE: {
		INT2TimeSeries *series = XLALCreateINT2TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALCreateINT2TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno())));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetINT2TimeSeriesMetadata(series, element->stream)) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALFrGetINT2TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno())));
			XLALDestroyINT2TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		if(fabs(1.0 / series->deltaT - rate) / rate > 1e-16) {
			GST_ERROR_OBJECT(element, "non-integer sample rate in frame file:  1 / %g s != %d Hz", series->deltaT, element->rate);
			XLALDestroyINT2TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			return FALSE;
		}
		width = 16;
		caps = series_to_caps(rate, element->series_type);
		XLALDestroyINT2TimeSeries(series);
		break;
	}

	case LAL_I4_TYPE_CODE: {
		INT4TimeSeries *series = XLALCreateINT4TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALCreateINT4TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno())));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetINT4TimeSeriesMetadata(series, element->stream)) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALFrGetINT4TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno())));
			XLALDestroyINT4TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		if(fabs(1.0 / series->deltaT - rate) / rate > 1e-16) {
			GST_ERROR_OBJECT(element, "non-integer sample rate in frame file:  1 / %g s != %d Hz", series->deltaT, element->rate);
			XLALDestroyINT4TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			return FALSE;
		}
		width = 32;
		caps = series_to_caps(rate, element->series_type);
		XLALDestroyINT4TimeSeries(series);
		break;
	}

	case LAL_S_TYPE_CODE: {
		REAL4TimeSeries *series = XLALCreateREAL4TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALCreateREAL4TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno())));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetREAL4TimeSeriesMetadata(series, element->stream)) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALFrGetREAL4TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno())));
			XLALDestroyREAL4TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		if(fabs(1.0 / series->deltaT - rate) / rate > 1e-16) {
			GST_ERROR_OBJECT(element, "non-integer sample rate in frame file:  1 / %g s != %d Hz", series->deltaT, element->rate);
			XLALDestroyREAL4TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			return FALSE;
		}
		width = 32;
		caps = series_to_caps(rate, element->series_type);
		XLALDestroyREAL4TimeSeries(series);
		break;
	}

	case LAL_D_TYPE_CODE: {
		REAL8TimeSeries *series = XLALCreateREAL8TimeSeries(element->full_channel_name, &stream_start, 0.0, 0.0, &lalDimensionlessUnit, 0);
		if(!series) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALCreateREAL8TimeSeries() failed: %s", XLALErrorString(XLALGetBaseErrno())));
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		if(XLALFrGetREAL8TimeSeriesMetadata(series, element->stream)) {
			GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("XLALFrGetREAL8TimeSeriesMetadata() failed: %s", XLALErrorString(XLALGetBaseErrno())));
			XLALDestroyREAL8TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			XLALClearErrno();
			return FALSE;
		}
		rate = round(1.0 / series->deltaT);
		if(fabs(1.0 / series->deltaT - rate) / rate > 1e-16) {
			GST_ERROR_OBJECT(element, "non-integer sample rate in frame file:  1 / %g s != %d Hz", series->deltaT, element->rate);
			XLALDestroyREAL8TimeSeries(series);
			XLALFrClose(element->stream);
			element->stream = NULL;
			return FALSE;
		}
		width = 64;
		caps = series_to_caps(rate, element->series_type);
		XLALDestroyREAL8TimeSeries(series);
		break;
	}

	default:
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("unsupported data type (LALTYPECODE=%d) for channel \"%s\"", element->series_type, element->full_channel_name));
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
	 * Set up default segment using start time of frame cache, so that
	 * if the user does not provide any seek event playback will start
	 * from the first frame
	 */

	gst_segment_set_newsegment(&object->segment, FALSE, 1.0, GST_FORMAT_TIME, (GstClockTime) XLALGPSToINT8NS(&stream_start), GST_CLOCK_TIME_NONE, (GstClockTime) XLALGPSToINT8NS(&stream_start));

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
	guint64 buffer_length;
	gboolean next_is_gap = FALSE;
	GstFlowReturn result;

	/*
	 * Push tag list if we haven't already
	 */

	if(element->need_tags) {
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

		if(!taglist) {
			GST_ELEMENT_ERROR(element, CORE, TAG, (NULL), ("failure constructing taglist"));
			return GST_FLOW_ERROR;
		}

		evt = gst_event_new_tag(taglist);

		if(!evt) {
			GST_ELEMENT_ERROR(element, CORE, TAG, (NULL), ("failure constructing tag event"));
			return GST_FLOW_ERROR;
		}

		if(!gst_pad_push_event(GST_BASE_SRC_PAD(basesrc), evt)) {
			GST_ELEMENT_ERROR(element, CORE, TAG, (NULL), ("failure pushing tag event"));
			return GST_FLOW_ERROR;
		}

		element->need_tags = FALSE;
	}

	/*
	 * Just in case
	 */

	*buffer = NULL;

	/*
	 * Determine the size of the next output buffer, and whether or not
	 * it's a gap
	 */

	buffer_length = get_next_buffer_length(element, basesrc->offset, &next_is_gap);
	if(buffer_length == 0) {
		GST_LOG_OBJECT(element, "end of stream");
		/* end of stream */
		return GST_FLOW_UNEXPECTED;
	}

	/*
	 * Construct and populate output buffer
	 */

	/* FIXME:  gst_pad_alloc_buffer() will lock up if called too soon
	 * after the pipeline is put into the playing state.  it is my
	 * belief (Kipp) that this is due to the screwed up way gstlal
	 * pipelines seek the framesrc element.  sorting the mess out is
	 * going to take time because it relies on at least one patch to
	 * gstreamer proper being accepted (Steve P's patch to fix the seek
	 * event flood induced by tee elements) before it can be
	 * investigated farther.  once that patch is into the version of
	 * gstreamer on the clusters we can take this stupid pause out and
	 * look again at what's really going on here */

	{
	static gboolean first = TRUE;
	/* 5 seconds.  seems to be enough */
	if(first) g_usleep(10000000);
	first = FALSE;
	}

	result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, buffer_length * unit_size(element), GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
	if(result != GST_FLOW_OK) {
		GST_ERROR_OBJECT(element, "gst_pad_alloc_buffer() returned %d (%s)", result, gst_flow_get_name(result));
		return result;
	}
	if(basesrc->offset != GST_BUFFER_OFFSET(*buffer) || buffer_length * unit_size(element) != GST_BUFFER_SIZE(*buffer)) {
		/* FIXME:  didn't get the buffer offset we asked
		 * for, do something about it */
		GST_FIXME_OBJECT(element, "gst_pad_alloc_buffer() didn't give us the offset we asked for.  do something about it, but what?");
	}
	if(!next_is_gap) {
		GST_LOG_OBJECT(element, "populating %" G_GUINT64_FORMAT " sample non-gap buffer", buffer_length);
		result = read_series(element, basesrc->offset, buffer_length, GST_BUFFER_DATA(*buffer));
		if(result != GST_FLOW_OK) {
			GST_ERROR_OBJECT(element, "failure reading data");
			gst_buffer_unref(*buffer);
			*buffer = NULL;
			return result;
		}
	} else {
		GST_LOG_OBJECT(element, "populating %" G_GUINT64_FORMAT " sample gap buffer", buffer_length);
		memset(GST_BUFFER_DATA(*buffer), 0, GST_BUFFER_SIZE(*buffer));
	}

	/*
	 * Set the buffer metadata.  The buffer duration is computed by
	 * imposing "timestamp + duration = next timestamp", and we require
	 * the result to not disagree by more than 1 ns from the duration
	 * obtained by converting the count of samples to an exactly
	 * rounded duration.
	 */

	GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + buffer_length;
	GST_BUFFER_TIMESTAMP(*buffer) = offset_to_time(element, GST_BUFFER_OFFSET(*buffer));
	GST_BUFFER_DURATION(*buffer) = offset_to_time(element, GST_BUFFER_OFFSET_END(*buffer)) - GST_BUFFER_TIMESTAMP(*buffer);
	g_assert_cmpuint(llabs((gint64) GST_BUFFER_DURATION(*buffer) - (gint64) gst_util_uint64_scale_int_round(GST_BUFFER_SIZE(*buffer) / unit_size(element), GST_SECOND, element->rate)), <=, 1);
	if(basesrc->offset == 0)
		GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);
	if(next_is_gap)
		GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(*buffer, GST_BUFFER_FLAG_GAP);
	basesrc->offset += buffer_length;
	GST_LOG_OBJECT(element, "constructed %s buffer spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(*buffer, GST_BUFFER_FLAG_GAP) ? "gap" : "non-gap", GST_BUFFER_BOUNDARIES_ARGS(*buffer));

	/*
	 * Done
	 */

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
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, (NULL), ("start time is required"));
		return FALSE;
	}
	XLALINT8NSToGPS(&epoch, segment->start);

	/*
	 * Try doing the seek
	 */

	if(XLALFrSeek(element->stream, &epoch)) {
		GST_WARNING_OBJECT(element, "XLALFrSeek() to %d.%09u s failed: %s", epoch.gpsSeconds, epoch.gpsNanoSeconds, XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
		GST_FIXME_OBJECT(element, "ignoring XLALFrSeek() failure");
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
		guint64 offset;

		gst_query_parse_convert(query, &src_format, &src_value, &dest_format, &dest_value);

		/*
		 * convert all source formats to a sample offset
		 */

		switch(src_format) {
		case GST_FORMAT_DEFAULT:
		case GST_FORMAT_TIME:
			if(src_value < basesrc->segment.start) {
				GST_WARNING_OBJECT(element, "requested time precedes start of segment, clipping to start of segment");
				offset = 0;
			} else
				offset = time_to_offset(element, src_value);
			break;

		case GST_FORMAT_BYTES:
			offset = src_value / unit_size(element);
			break;

		case GST_FORMAT_BUFFERS:
			offset = gst_util_uint64_scale_int_round(src_value, gst_base_src_get_blocksize(basesrc), unit_size(element));
			break;

		case GST_FORMAT_PERCENT:
			if(src_value < 0) {
				GST_WARNING_OBJECT(element, "requested percentage < 0, clipping to 0");
				offset = 0;
			} else if(src_value > 100) {
				GST_WARNING_OBJECT(element, "requested percentage > 100, clipping to 100");
				offset = time_to_offset(element, basesrc->segment.stop);
			} else
				offset = gst_util_uint64_scale_int_round(src_value, time_to_offset(element, basesrc->segment.stop), 100);
			break;

		default:
			g_assert_not_reached();
			return FALSE;
		}

		/*
		 * convert sample offset to destination format
		 */

		switch(dest_format) {
		case GST_FORMAT_DEFAULT:
		case GST_FORMAT_TIME:
			dest_value = offset_to_time(element, offset);
			break;

		case GST_FORMAT_BYTES:
			dest_value = offset * unit_size(element);
			break;

		case GST_FORMAT_BUFFERS:
			dest_value = gst_util_uint64_scale_int_ceil(offset, unit_size(element), gst_base_src_get_blocksize(basesrc));
			break;

		case GST_FORMAT_PERCENT:
			dest_value = gst_util_uint64_scale_int_round(offset, 100, time_to_offset(element, basesrc->segment.stop));
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
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum property {
	ARG_SRC_LOCATION = 1,
	ARG_SRC_INSTRUMENT,
	ARG_CACHE_SRC_REGEX,
	ARG_CACHE_DSC_REGEX,
	ARG_SRC_CHANNEL_NAME,
	ARG_SRC_UNITS,
	ARG_SEGMENT_LIST
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

	case ARG_CACHE_SRC_REGEX:
		g_free(element->cache_src_regex);
		element->cache_src_regex = g_value_dup_string(value);
		break;

	case ARG_CACHE_DSC_REGEX:
		g_free(element->cache_dsc_regex);
		element->cache_dsc_regex = g_value_dup_string(value);
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

	case ARG_SEGMENT_LIST:
		gstlal_segment_list_free(element->segmentlist);
		element->segmentlist = gstlal_segment_list_from_g_value_array(g_value_get_boxed(value));
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
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

	case ARG_CACHE_DSC_REGEX:
		g_value_set_string(value, element->cache_dsc_regex);
		break;

	case ARG_CACHE_SRC_REGEX:
		g_value_set_string(value, element->cache_src_regex);
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

	case ARG_SEGMENT_LIST:
		if(element->segmentlist)
			g_value_take_boxed(value, g_value_array_from_gstlal_segment_list(element->segmentlist));
		/* FIXME:  else? */
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * Instance finalize function.
 */


static void finalize(GObject *object)
{
	GSTLALFrameSrc *element = GSTLAL_FRAMESRC(object);

	g_free(element->location);
	element->location = NULL;
	g_free(element->instrument);
	element->instrument = NULL;
	g_free(element->cache_src_regex);
	element->cache_src_regex = NULL;
	g_free(element->cache_dsc_regex);
	element->cache_dsc_regex = NULL;
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
 * Base init function.
 */


static void gstlal_framesrc_base_init(gpointer class)
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
				"signed = (boolean) true; " \
				"audio/x-raw-int, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 16, " \
				"depth = (int) 16, " \
				"signed = (boolean) true"
			)
		)
	);
}


/*
 * Class init function.
 */


static void gstlal_framesrc_class_init(GSTLALFrameSrcClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(klass);

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
		ARG_CACHE_SRC_REGEX,
		g_param_spec_string(
			"cache-src-regex",
			"Pattern",
			"Description regex for sieving cache (e.g. \"H.*\")",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_CACHE_DSC_REGEX,
		g_param_spec_string(
			"cache-dsc-regex",
			"Pattern",
			"Source/Observatory regex for sieving cache (e.g. \"H.*\")",
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
	g_object_class_install_property(
		gobject_class,
		ARG_SEGMENT_LIST,
		g_param_spec_value_array(
			"segment-list",
			"Segment List",
			"List of Segments.  This is an Nx2 array where N (the rows) is the number of segments.  The columns are the start and stop times of each segment in integer nanoseconds.",
			g_param_spec_value_array(
				"segment",
				"[start, stop)",
				"Start and stop time of segment in nanoseconds.",
				g_param_spec_uint64(
					"time",
					"Time",
					"Time in nanoseconds.",
					0, G_MAXUINT64, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
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
 * Instance init function.
 */


static void gstlal_framesrc_init(GSTLALFrameSrc *element, GSTLALFrameSrcClass *klass)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(element);

	gst_pad_use_fixed_caps(GST_BASE_SRC_PAD(basesrc));

	basesrc->offset = 0;
	element->location = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->cache_src_regex = NULL;
	element->cache_dsc_regex = NULL;
	element->full_channel_name = NULL;
	element->rate = 0;
	element->width = 0;
	element->stream = NULL;
	element->units = DEFAULT_UNITS_UNIT;
	element->series_type = -1;
	element->segmentlist = NULL;

	gst_base_src_set_format(basesrc, GST_FORMAT_TIME);

	/*
	 * Make a note to push tags before emitting next frame.
	 */

	element->need_tags = TRUE;

}

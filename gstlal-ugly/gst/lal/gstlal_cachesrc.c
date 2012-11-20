/*
 * GstLALCacheSrc
 *
 * Copyright (C) 2012  Kipp Cannon
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
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>


#include <lal/XLALError.h>
#include <lal/FrameCache.h>


#include <gstlal/gstlal_debug.h>
#include <gstlal_cachesrc.h>



/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_cachesrc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_cachesrc", 0, "lal_cachesrc element");
}


GST_BOILERPLATE_FULL(GstLALCacheSrc, gstlal_cachesrc, GstBaseSrc, GST_TYPE_BASE_SRC, additional_initializations);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_LOCATION NULL
#define DEFAULT_CACHE_SRC_REGEX NULL
#define DEFAULT_CACHE_DSC_REGEX NULL
#define DEFAULT_USE_MMAP FALSE


/*
 * ============================================================================
 *
 *                             Internal Functions
 *
 * ============================================================================
 */


static GstFlowReturn read_buffer(GstPad *pad, int fd, guint64 offset, size_t size, GstBuffer **buf)
{
	size_t read_offset;
	GstFlowReturn result = GST_FLOW_OK;

	result = gst_pad_alloc_buffer(pad, offset, size, GST_PAD_CAPS(pad), buf);
	if(result != GST_FLOW_OK)
		goto done;

	read_offset = 0;
	do {
		ssize_t bytes_read = read(fd, GST_BUFFER_DATA(*buf) + read_offset, GST_BUFFER_SIZE(*buf) - read_offset);
		if(bytes_read < 0) {
			gst_buffer_unref(*buf);
			*buf = NULL;
			result = GST_FLOW_ERROR;
			goto done;
		}
		read_offset += bytes_read;
	} while(read_offset < size);

done:
	return result;
}


static void munmap_buffer(GstBuffer *buf)
{
	if(buf) {
		g_assert(GST_IS_BUFFER(buf));
		munmap(GST_BUFFER_DATA(buf), GST_BUFFER_SIZE(buf));
	}
}


static GstFlowReturn mmap_buffer(GstPad *pad, int fd, guint64 offset, size_t size, GstBuffer **buf)
{
	GstFlowReturn result = GST_FLOW_OK;

	*buf = gst_buffer_new();
	if(!*buf) {
		result = GST_FLOW_ERROR;
		goto done;
	}
	GST_BUFFER_FLAG_SET(*buf, GST_BUFFER_FLAG_READONLY);
	GST_BUFFER_DATA(*buf) = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
	if(!GST_BUFFER_DATA(*buf)) {
		gst_buffer_unref(*buf);
		*buf = NULL;
		result = GST_FLOW_ERROR;
		goto done;
	}
	GST_BUFFER_SIZE(*buf) = size;
	GST_BUFFER_OFFSET(*buf) = offset;
	gst_buffer_set_caps(*buf, GST_PAD_CAPS(pad));

	/*
	 * hack to get both the data pointer and the size to the munmap()
	 * call.  the mallocdata pointer is set to the buffer object
	 * itself, and the freefunc looks inside to get the real pointer
	 * and the size.  this probably goes really wrong if a subbuffer is
	 * ever made from the buffer, but then the application can choose
	 * not to use mmap()ed buffers if it wants to be able to do
	 * subbuffering.  for the .gwf frame file use case, subbuffering is
	 * (mostly) nonsensical anyway.
	 */

	GST_BUFFER_MALLOCDATA(*buf) = (void *) *buf;
	GST_BUFFER_FREE_FUNC(*buf) = (GFreeFunc) munmap_buffer;

done:
	return result;
}


static GstClockTime cache_entry_start_time(GstLALCacheSrc *element, guint i)
{
	return element->cache->frameFiles[i].startTime * GST_SECOND;
}


static GstClockTime cache_entry_duration(GstLALCacheSrc *element, guint i)
{
	return element->cache->frameFiles[i].duration * GST_SECOND;
}


static GstClockTime cache_entry_end_time(GstLALCacheSrc *element, guint i)
{
	return (element->cache->frameFiles[i].startTime + element->cache->frameFiles[i].duration) * GST_SECOND;
}


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/*
 * ============================================================================
 *
 *                             GstBaseSrc Methods
 *
 * ============================================================================
 */


/*
 * start()
 */


static gboolean start(GstBaseSrc *basesrc)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(basesrc);
	FrCache *cache;
	FrCacheSieve sieve = {
		.earliestTime = 0,
		.latestTime = G_MAXINT32,
		.urlRegEx = NULL,
		.srcRegEx = element->cache_src_regex,
		.dscRegEx = element->cache_dsc_regex,
	};

	g_return_val_if_fail(element->location != NULL, FALSE);

	element->cache = XLALFrImportCache(element->location);
	if(!element->cache) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("error reading '%s': %s", element->location, XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		return FALSE;
	}
	GST_DEBUG_OBJECT(element, "loaded '%s': %d items in cache", element->location, element->cache->numFrameFiles);

	/* sieving also puts the files in time order */
	cache = XLALFrSieveCache(element->cache, &sieve);
	if(!cache) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("error sieving cache: %s", element->location, XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		XLALFrDestroyCache(element->cache);
		element->cache = NULL;
		return FALSE;
	}
	element->cache = cache;
	GST_DEBUG_OBJECT(element, "%d items remain in cache after sieve", element->cache->numFrameFiles);

	basesrc->offset = 0;
	element->index = 0;
	element->need_discont = TRUE;

	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *basesrc)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(basesrc);

	XLALFrDestroyCache(element->cache);
	element->cache = NULL;

	return TRUE;
}


/*
 * is_seekable()
 */


static gboolean is_seekable(GstBaseSrc *basesrc)
{
	return TRUE;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buf)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(basesrc);
	GError *error = NULL;
	gchar *host = NULL;
	gchar *path = NULL;
	int fd;
	struct stat statinfo;
	GstFlowReturn result = GST_FLOW_OK;

	g_assert(element->cache != NULL);

	if(element->index >= element->cache->numFrameFiles || cache_entry_start_time(element, element->index) >= (GstClockTime) basesrc->segment.stop)
		return GST_FLOW_UNEXPECTED;

	path = g_filename_from_uri(element->cache->frameFiles[element->index].url, &host, &error);
	g_free(host);
	if(error) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("error parsing uri '%s': %s", element->cache->frameFiles[element->index].url, error->message));
		g_error_free(error);
		result = GST_FLOW_ERROR;
		goto done;
	}

	fd = open(path, O_RDONLY);
	if(fd < 0) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("open('%s') failed: %s", path, sys_errlist[errno]));
		result = GST_FLOW_ERROR;
		goto done;
	}

	if(fstat(fd, &statinfo)) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("fstat('%s') failed: %s", path, sys_errlist[errno]));
		result = GST_FLOW_ERROR;
		close(fd);
		goto done;
	}

	if(element->use_mmap)
		result = mmap_buffer(GST_BASE_SRC_PAD(basesrc), fd, basesrc->offset, statinfo.st_size, buf);
	else
		result = read_buffer(GST_BASE_SRC_PAD(basesrc), fd, basesrc->offset, statinfo.st_size, buf);
	close(fd);
	if(result != GST_FLOW_OK) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("%s('%s') failed: %s", element->use_mmap ? "mmap" : "read", path, sys_errlist[errno]));
		goto done;
	}

	GST_BUFFER_TIMESTAMP(*buf) = cache_entry_start_time(element, element->index);
	GST_BUFFER_DURATION(*buf) = cache_entry_duration(element, element->index);
	GST_BUFFER_OFFSET_END(*buf) = GST_BUFFER_OFFSET(*buf) + GST_BUFFER_SIZE(*buf);
	basesrc->offset += GST_BUFFER_SIZE(*buf);
	if(element->need_discont || GST_BUFFER_TIMESTAMP(*buf) != cache_entry_end_time(element, element->index - 1)) {
		GST_BUFFER_FLAG_SET(*buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}

	GST_DEBUG_OBJECT(element, "pushing '%s' %" GST_BUFFER_BOUNDARIES_FORMAT, path, GST_BUFFER_BOUNDARIES_ARGS(*buf));

	element->index++;
done:
	g_free(path);
	return result;
}


/*
 * do_seek()
 */


static gboolean do_seek(GstBaseSrc *basesrc, GstSegment *segment)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(basesrc);
	guint i;
	gboolean success = TRUE;

	GST_DEBUG_OBJECT(element, "requested segment is [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT "), stream time %" GST_TIME_SECONDS_FORMAT ", accum %" GST_TIME_SECONDS_FORMAT ", last_stop %" GST_TIME_SECONDS_FORMAT ", duration %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(segment->start), GST_TIME_SECONDS_ARGS(segment->stop), GST_TIME_SECONDS_ARGS(segment->time), GST_TIME_SECONDS_ARGS(segment->accum), GST_TIME_SECONDS_ARGS(segment->last_stop), GST_TIME_SECONDS_ARGS(segment->duration));

	g_return_val_if_fail(element->cache != NULL, FALSE);

	/*
	 * require a start time
	 */

	if(!GST_CLOCK_TIME_IS_VALID(segment->start)) {
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, (NULL), ("start time is required"));
		success = FALSE;
		goto done;
	}

	/*
	 * do the seek.  the loop assumes the cache entries are in time
	 * order and searches for the first file whose end is past the
	 * requested time.
	 */

	for(i = 0; i < element->cache->numFrameFiles; i++) {
		GstClockTime min = cache_entry_start_time(element, i);
		GstClockTime max = cache_entry_end_time(element, i);
		if(i)
			g_assert_cmpuint(cache_entry_start_time(element, i), >=, cache_entry_start_time(element, i - 1));
		if((GstClockTime) segment->start < max) {
			GST_DEBUG_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT ": found uri '%s' spanning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(segment->start), element->cache->frameFiles[i].url, GST_TIME_SECONDS_ARGS(min), GST_TIME_SECONDS_ARGS(max));
			if(i != element->index) {
				basesrc->offset = 0;
				element->index = i;
				element->need_discont = TRUE;
			}
			if((GstClockTime) segment->start < min)
				GST_WARNING_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT " uri starts at %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(segment->start), GST_TIME_SECONDS_ARGS(min));
			goto done;
		}
	}
	GST_WARNING_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT " beyond end of cache", GST_TIME_SECONDS_ARGS(segment->start));
	if(i != element->index) {
		basesrc->offset = 0;
		element->index = i;
		element->need_discont = TRUE;
	}

	/*
	 * done
	 */

done:
	return success;
}


/*
 * query()
 */


static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(basesrc);
	gboolean success = TRUE;

	if(!element->cache || !element->cache->numFrameFiles)
		success = parent_class->query(basesrc, query);
	else {
		switch(GST_QUERY_TYPE(query)) {
		case GST_QUERY_POSITION:
			if(element->index < element->cache->numFrameFiles)
				/* report start of next file */
				gst_query_set_position(query, GST_FORMAT_TIME, cache_entry_start_time(element, element->index) - cache_entry_start_time(element, 0));
			else
				/* special case for EOS:  report end of last file */
				gst_query_set_position(query, GST_FORMAT_TIME, cache_entry_end_time(element, element->cache->numFrameFiles - 1) - cache_entry_start_time(element, 0));
			break;

		case GST_QUERY_DURATION:
			gst_query_set_duration(query, GST_FORMAT_TIME, cache_entry_end_time(element, element->cache->numFrameFiles - 1) - cache_entry_start_time(element, 0));
			break;

		case GST_QUERY_SEGMENT:
			gst_query_set_segment(query, 1.0, GST_FORMAT_TIME, cache_entry_start_time(element, 0), cache_entry_end_time(element, element->cache->numFrameFiles - 1));
			break;

		case GST_QUERY_URI:
			gst_query_set_uri(query, element->cache->frameFiles[element->index].url);
			break;

		default:
			success = parent_class->query(basesrc, query);
			break;
		}
	}

	if(success)
		GST_DEBUG_OBJECT(element, "result: %" GST_PTR_FORMAT, query);
	else
		GST_ERROR_OBJECT(element, "query failed");
	return success;
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


enum property {
	PROP_LOCATION = 1,
	PROP_CACHE_SRC_REGEX,
	PROP_CACHE_DSC_REGEX,
	PROP_USE_MMAP,
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(object);

	GST_OBJECT_LOCK(object);

	switch(id) {
	case PROP_LOCATION:
		g_free(element->location);
		element->location = g_value_dup_string(value);
		break;

	case PROP_CACHE_SRC_REGEX:
		g_free(element->cache_src_regex);
		element->cache_src_regex = g_value_dup_string(value);
		break;

	case PROP_CACHE_DSC_REGEX:
		g_free(element->cache_dsc_regex);
		element->cache_dsc_regex = g_value_dup_string(value);
		break;

	case PROP_USE_MMAP:
		element->use_mmap = g_value_get_boolean(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(object);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(object);

	GST_OBJECT_LOCK(object);

	switch(id) {
	case PROP_LOCATION:
		g_value_set_string(value, element->location);
		break;

	case PROP_CACHE_SRC_REGEX:
		g_value_set_string(value, element->cache_src_regex);
		break;

	case PROP_CACHE_DSC_REGEX:
		g_value_set_string(value, element->cache_dsc_regex);
		break;

	case PROP_USE_MMAP:
		g_value_set_boolean(value, element->use_mmap);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(object);
}


static void finalize(GObject *object)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(object);

	g_free(element->location);
	element->location = NULL;
	g_free(element->cache_src_regex);
	element->cache_src_regex = NULL;
	g_free(element->cache_dsc_regex);
	element->cache_dsc_regex = NULL;
	XLALFrDestroyCache(element->cache);
	element->cache = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void gstlal_cachesrc_base_init(gpointer klass)
{
	/* no-op */
}


static void gstlal_cachesrc_class_init(GstLALCacheSrcClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_set_details_simple(
		element_class,
		"LAL Frame Cache File Source",
		"Source",
		"Retrieve frame files from locations recorded in a LAL cache file.",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-igwd-frame, " \
				"framed = (boolean) true"
			)
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_LOCATION,
		g_param_spec_string(
			"location",
			"Location",
			"Path to LAL cache file.",
			DEFAULT_LOCATION,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_CACHE_SRC_REGEX,
		g_param_spec_string(
			"cache-src-regex",
			"Pattern",
			"Source/Observatory regex for sieving cache (e.g. \"H.*\").",
			DEFAULT_CACHE_SRC_REGEX,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_CACHE_DSC_REGEX,
		g_param_spec_string(
			"cache-dsc-regex",
			"Pattern",
			"Description regex for sieving cache (e.g. \".*RDS_C03.*\").",
			DEFAULT_CACHE_DSC_REGEX,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_USE_MMAP,
		g_param_spec_boolean(
			"use-mmap",
			"Use mmap() instead of read()",
			"Use mmap() instead of read().",
			DEFAULT_USE_MMAP,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	gstbasesrc_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesrc_class->stop = GST_DEBUG_FUNCPTR(stop);
	gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR(is_seekable);
	gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
	gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
	gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);
}


static void gstlal_cachesrc_init(GstLALCacheSrc *element, GstLALCacheSrcClass *klass)
{
	gst_base_src_set_format(GST_BASE_SRC(element), GST_FORMAT_TIME);

	element->location = NULL;
	element->cache_src_regex = NULL;
	element->cache_dsc_regex = NULL;
	element->cache = NULL;
}

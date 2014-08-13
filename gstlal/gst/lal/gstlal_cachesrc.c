/*
 * GstLALCacheSrc
 *
 * Copyright (C) 2012--2013  Kipp Cannon
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


/**
 * SECTION:gstlal_cachesrc
 * @short_description:  Retrieve frame files from locations recorded in a LAL cache file.
 *
 * Reviewed:  a922d6dd59d0b58442c0bf7bc4cc4d740b8c6a43 2014-08-12 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
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
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>


#include <lal/XLALError.h>
#include <lal/LALCache.h>


#include <gstlal/gstlal_debug.h>
#include <gstlal_cachesrc.h>


/*
 * ========================================================================
 *
 *                               GstURIHandler
 *
 * ========================================================================
 */


#define URI_SCHEME "lalcache"


static GstURIType uri_get_type(GType type)
{
	return GST_URI_SRC;
}


/* 1.0:  this becomes static const gchar *const * */
static gchar **uri_get_protocols(GType type)
{
	/* 1.0:  this becomes
	static const gchar *protocols[] = {URI_SCHEME, NULL};
	*/
	static gchar *protocols[] = {(gchar *) URI_SCHEME, NULL};

	return protocols;
}


/* 1.0:  this becomes static gchar * */
static const gchar *uri_get_uri(GstURIHandler *handler)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(handler);
	GString *uri = g_string_new(URI_SCHEME "://");
	gchar separator = '?';

	g_string_append_uri_escaped(uri, element->location, NULL, FALSE);

	if(element->cache_src_regex) {
		g_string_append_c(uri, separator);
		g_string_append(uri, "cache-src-regex=");
		g_string_append_uri_escaped(uri, element->cache_src_regex, NULL, FALSE);
		separator = '&';
	}

	if(element->cache_dsc_regex) {
		g_string_append_c(uri, separator);
		g_string_append(uri, "cache-dsc-regex=");
		g_string_append_uri_escaped(uri, element->cache_dsc_regex, NULL, FALSE);
	}

	/* 1.0:  returning this value won't be a memory leak */
	return g_string_free(uri, FALSE);
}


static gchar *g_uri_unescape_string_inplace(gchar **s)
{
	gchar *unescaped = g_uri_unescape_string(*s, NULL);
	g_free(*s);
	*s = unescaped;
	return *s;
}


/* 1.0:  this gets a GError ** argument */
static gboolean uri_set_uri(GstURIHandler *handler, const gchar *uri)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(handler);
	gchar *scheme = g_uri_parse_scheme(uri);
	int query_offset;
	gchar *location = NULL;
	gchar *cache_src_regex = NULL;
	gchar *cache_dsc_regex = NULL;
	gboolean success = TRUE;

	success = !strcmp(scheme, URI_SCHEME);
	if(success) {
		success &= sscanf(uri, URI_SCHEME "://%m[^?]%n", &location, &query_offset) >= 1;
		if(!success)
			GST_ERROR_OBJECT(element, "bad uri '%s'", uri);
	} else
		GST_ERROR_OBJECT(element, "wrong scheme '%s'", scheme);
	if(success && uri[query_offset] == '?') {
		gchar **fragments = g_strsplit(&uri[++query_offset], "&", 0);
		gchar **fragment;
		for(fragment = fragments; *fragment; fragment++) {
			gchar **namevalue = g_strsplit(*fragment, "=", 2);
			success &= namevalue[0] && namevalue[1];
			if(success) {
				g_uri_unescape_string_inplace(&namevalue[0]);
				g_uri_unescape_string_inplace(&namevalue[1]);
				if(!g_strcmp0("cache-src-regex", namevalue[0])) {
					g_free(cache_src_regex);
					cache_src_regex = g_strdup(namevalue[1]);
				} else if(!g_strcmp0("cache-dsc-regex", namevalue[0])) {
					g_free(cache_dsc_regex);
					cache_dsc_regex = g_strdup(namevalue[1]);
				} else {
					GST_ERROR_OBJECT(element, "query '%s' not recognized", namevalue[0]);
					success = FALSE;
				}
			} else
				GST_ERROR_OBJECT(element, "invalid query '%s'", *fragment);
			g_strfreev(namevalue);
		}
		g_strfreev(fragments);
	}
	if(success)
		g_object_set(G_OBJECT(element), "location", g_uri_unescape_string_inplace(&location), "cache-src-regex", cache_src_regex, "cache-dsc-regex", cache_dsc_regex, NULL);

	g_free(location);
	g_free(cache_src_regex);
	g_free(cache_dsc_regex);
	g_free(scheme);
	return success;
}


static void uri_handler_init(gpointer g_iface, gpointer iface_data)
{
	GstURIHandlerInterface *iface = (GstURIHandlerInterface *) g_iface;

	iface->get_uri = GST_DEBUG_FUNCPTR(uri_get_uri);
	iface->set_uri = GST_DEBUG_FUNCPTR(uri_set_uri);
	/* 1.0:  this is ->get_type */
	iface->get_type_full = GST_DEBUG_FUNCPTR(uri_get_type);
	/* 1.0:  this is ->get_protocols */
	iface->get_protocols_full = GST_DEBUG_FUNCPTR(uri_get_protocols);
}


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
	static const GInterfaceInfo uri_handler_info = {
		uri_handler_init,
		NULL,
		NULL
	};
	g_type_add_interface_static(type, GST_TYPE_URI_HANDLER, &uri_handler_info);

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


static GstFlowReturn read_buffer(GstBaseSrc *basesrc, const char *path, int fd, guint64 offset, size_t size, GstBuffer **buf)
{
	GstPad *pad = GST_BASE_SRC_PAD(basesrc);
	size_t read_offset;
	GstFlowReturn result = GST_FLOW_OK;

	result = gst_pad_alloc_buffer(pad, offset, size, GST_PAD_CAPS(pad), buf);
	if(result != GST_FLOW_OK)
		goto done;
	g_assert_cmpuint(GST_BUFFER_OFFSET(*buf), ==, offset);
	g_assert_cmpuint(GST_BUFFER_SIZE(*buf), ==, size);

	read_offset = 0;
	do {
		ssize_t bytes_read = read(fd, GST_BUFFER_DATA(*buf) + read_offset, GST_BUFFER_SIZE(*buf) - read_offset);
		if(bytes_read < 0) {
			GST_ELEMENT_ERROR(basesrc, RESOURCE, READ, (NULL), ("read('%s') failed: %s", path, strerror(errno)));
			gst_buffer_unref(*buf);
			*buf = NULL;
			result = GST_FLOW_ERROR;
			goto done;
		}
		read_offset += bytes_read;
	} while(read_offset < size);
	g_assert_cmpuint(read_offset, ==, size);

	GST_BUFFER_OFFSET_END(*buf) = offset + size;

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


static GstFlowReturn mmap_buffer(GstBaseSrc *basesrc, const char *path, int fd, guint64 offset, size_t size, GstBuffer **buf)
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
		GST_ELEMENT_ERROR(basesrc, RESOURCE, READ, (NULL), ("mmap('%s') failed: %s", path, strerror(errno)));
		gst_buffer_unref(*buf);
		*buf = NULL;
		result = GST_FLOW_ERROR;
		goto done;
	}
	GST_BUFFER_SIZE(*buf) = size;
	GST_BUFFER_OFFSET(*buf) = offset;
	GST_BUFFER_OFFSET_END(*buf) = offset + size;
	gst_buffer_set_caps(*buf, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)));

	/*
	 * hack to get both the data pointer and the size to the munmap()
	 * call.  the mallocdata pointer is set to the buffer object
	 * itself, and the freefunc looks inside to get the real pointer
	 * and the size.
	 */

	GST_BUFFER_MALLOCDATA(*buf) = (void *) *buf;
	GST_BUFFER_FREE_FUNC(*buf) = (GFreeFunc) munmap_buffer;

done:
	return result;
}


static GstClockTime cache_entry_start_time(GstLALCacheSrc *element, guint i)
{
	return element->cache->list[i].t0 * GST_SECOND;
}


static GstClockTime cache_entry_duration(GstLALCacheSrc *element, guint i)
{
	return element->cache->list[i].dt * GST_SECOND;
}


static GstClockTime cache_entry_end_time(GstLALCacheSrc *element, guint i)
{
	return (element->cache->list[i].t0 + element->cache->list[i].dt) * GST_SECOND;
}


static guint time_to_index(GstLALCacheSrc *element, GstClockTime t)
{
	guint i;

	g_assert(element->cache != NULL);

	/*
	 * the loop assumes the cache entries are in time order and
	 * searches for the first file whose end is past the requested
	 * time
	 */

	for(i = 0; i < element->cache->length; i++)
		if(cache_entry_end_time(element, i) > t)
			break;

	return i;
}


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

	g_return_val_if_fail(element->location != NULL, FALSE);
	g_return_val_if_fail(element->cache == NULL, FALSE);

	element->cache = XLALCacheImport(element->location);
	if(!element->cache) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("error reading '%s': %s", element->location, XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		return FALSE;
	}
	GST_DEBUG_OBJECT(element, "loaded '%s': %d item(s) in cache", element->location, element->cache->length);

	if(XLALCacheSieve(element->cache, 0, 0, element->cache_src_regex, element->cache_dsc_regex, NULL)) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("error sieving cache '%s': %s", element->location, XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		XLALDestroyCache(element->cache);
		element->cache = NULL;
		return FALSE;
	}
	GST_DEBUG_OBJECT(element, "%d item(s) remain in cache after sieve", element->cache->length);

	if(!element->cache->length)
		GST_WARNING_OBJECT(element, "cache is empty!");
	else if(XLALCacheSort(element->cache)) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("error sorting cache '%s': %s", element->location, XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		XLALDestroyCache(element->cache);
		element->cache = NULL;
		return FALSE;
	}

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

	if(element->cache) {
		XLALDestroyCache(element->cache);
		element->cache = NULL;
	} else
		g_assert_not_reached();

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

	*buf = NULL;	/* just in case */

	if(element->index >= element->cache->length || (GST_CLOCK_TIME_IS_VALID(basesrc->segment.stop) && cache_entry_start_time(element, element->index) >= (GstClockTime) basesrc->segment.stop)) {
		GST_DEBUG_OBJECT(element, "EOS");
		return GST_FLOW_UNEXPECTED;
	}

	GST_DEBUG_OBJECT(element, "loading '%s'", element->cache->list[element->index].url);
	path = g_filename_from_uri(element->cache->list[element->index].url, &host, &error);
	g_free(host);
	if(error) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("error parsing uri '%s': %s", element->cache->list[element->index].url, error->message));
		g_error_free(error);
		result = GST_FLOW_ERROR;
		goto done;
	}

	fd = open(path, O_RDONLY);
	if(fd < 0) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("open('%s') failed: %s", path, strerror(errno)));
		result = GST_FLOW_ERROR;
		goto done;
	}

	if(fstat(fd, &statinfo)) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("fstat('%s') failed: %s", path, strerror(errno)));
		result = GST_FLOW_ERROR;
		close(fd);
		goto done;
	}

	if(element->use_mmap)
		result = mmap_buffer(basesrc, path, fd, basesrc->offset, statinfo.st_size, buf);
	else
		result = read_buffer(basesrc, path, fd, basesrc->offset, statinfo.st_size, buf);
	close(fd);
	if(result != GST_FLOW_OK)
		goto done;

	/*
	 * finish setting buffer metadata.  need_discont is TRUE for the
	 * first buffer, so cache_entry_end_time() won't be invoked with an
	 * index of -1
	 */

	GST_BUFFER_TIMESTAMP(*buf) = cache_entry_start_time(element, element->index);
	GST_BUFFER_DURATION(*buf) = cache_entry_duration(element, element->index);
	basesrc->offset = GST_BUFFER_OFFSET_END(*buf);
	if(element->need_discont || GST_BUFFER_TIMESTAMP(*buf) != cache_entry_end_time(element, element->index - 1)) {
		GST_BUFFER_FLAG_SET(*buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}

	GST_DEBUG_OBJECT(element, "pushing '%s' spanning %" GST_BUFFER_BOUNDARIES_FORMAT, path, GST_BUFFER_BOUNDARIES_ARGS(*buf));

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

	if(!element->cache) {
		GST_ERROR_OBJECT(element, "no file cache loaded");
		success = FALSE;
		goto done;
	}

	/*
	 * require a start time
	 */

	if(!GST_CLOCK_TIME_IS_VALID(segment->start)) {
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, (NULL), ("start time is required"));
		success = FALSE;
		goto done;
	}

	/*
	 * do the seek
	 */

	i = time_to_index(element, segment->start);
	if(i >= element->cache->length)
		GST_WARNING_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT " beyond end of cache", GST_TIME_SECONDS_ARGS(segment->start));
	else if(GST_CLOCK_TIME_IS_VALID(segment->stop) && (GstClockTime) segment->stop <= cache_entry_start_time(element, i)) {
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, (NULL), ("no data available for segment"));
		success = FALSE;
		goto done;
	} else if((GstClockTime) segment->start < cache_entry_start_time(element, i))
		GST_WARNING_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT ": found uri '%s' spanning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(segment->start), element->cache->list[i].url, GST_TIME_SECONDS_ARGS(cache_entry_start_time(element, i)), GST_TIME_SECONDS_ARGS(cache_entry_end_time(element, i)));
	else
		GST_DEBUG_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT ": found uri '%s' spanning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(segment->start), element->cache->list[i].url, GST_TIME_SECONDS_ARGS(cache_entry_start_time(element, i)), GST_TIME_SECONDS_ARGS(cache_entry_end_time(element, i)));

	/*
	 * done
	 */

	if(i != element->index) {
		basesrc->offset = 0;
		element->index = i;
		element->need_discont = TRUE;
	}
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

	if(!element->cache || !element->cache->length)
		success = parent_class->query(basesrc, query);
	else {
		switch(GST_QUERY_TYPE(query)) {
		case GST_QUERY_FORMATS:
			gst_query_set_formats(query, 3, GST_FORMAT_TIME, GST_FORMAT_BUFFERS, GST_FORMAT_PERCENT);
			break;

		case GST_QUERY_CONVERT: {
			GstFormat src_format, dst_format;
			gint64 src_value, dst_value;
			GstClockTime time;

			gst_query_parse_convert(query, &src_format, &src_value, &dst_format, &dst_value);

			/*
			 * convert all source formats to a time
			 */

			switch(src_format) {
			case GST_FORMAT_TIME:
				time = src_value;
				break;

			case GST_FORMAT_BUFFERS:
				if(src_value < 0)
					time = cache_entry_start_time(element, 0);
				else if(src_value < element->cache->length)
					time = cache_entry_start_time(element, src_value);
				else
					time = cache_entry_end_time(element, element->cache->length - 1);
				break;

			case GST_FORMAT_PERCENT:
				if(src_value < 0)
					time = cache_entry_start_time(element, 0);
				else if(src_value <= GST_FORMAT_PERCENT_MAX)
					time = cache_entry_start_time(element, 0) + gst_util_uint64_scale_round(cache_entry_end_time(element, element->cache->length - 1) - cache_entry_start_time(element, 0), src_value, GST_FORMAT_PERCENT_MAX);
				else
					time = cache_entry_end_time(element, element->cache->length - 1);
				break;

			default:
				g_assert_not_reached();
				success = FALSE;
				break;
			}

			/*
			 * convert time to destination format
			 */

			switch(dst_format) {
			case GST_FORMAT_TIME:
				dst_value = time;
				break;

			case GST_FORMAT_BUFFERS:
				time = MIN(MAX(time, cache_entry_start_time(element, 0)), cache_entry_end_time(element, element->cache->length - 1));
				dst_value = time_to_index(element, time);
				break;

			case GST_FORMAT_PERCENT:
				time = MIN(MAX(time, cache_entry_start_time(element, 0)), cache_entry_end_time(element, element->cache->length - 1));
				dst_value = gst_util_uint64_scale_round(cache_entry_end_time(element, element->cache->length - 1) - cache_entry_start_time(element, 0), GST_FORMAT_PERCENT_MAX, time - cache_entry_start_time(element, 0));
				break;

			default:
				g_assert_not_reached();
				success = FALSE;
				break;
			}

			gst_query_set_convert(query, src_format, src_value, dst_format, dst_value);
			break;
		}

		case GST_QUERY_POSITION:
			if(element->index < element->cache->length)
				/* report start of next file */
				gst_query_set_position(query, GST_FORMAT_TIME, cache_entry_start_time(element, element->index) - cache_entry_start_time(element, 0));
			else
				/* special case for EOS:  report end of last file */
				gst_query_set_position(query, GST_FORMAT_TIME, cache_entry_end_time(element, element->cache->length - 1) - cache_entry_start_time(element, 0));
			break;

		case GST_QUERY_DURATION:
			gst_query_set_duration(query, GST_FORMAT_TIME, cache_entry_end_time(element, element->cache->length - 1) - cache_entry_start_time(element, 0));
			break;

		case GST_QUERY_SEEKING:
			gst_query_set_seeking(query, GST_FORMAT_TIME, TRUE, cache_entry_start_time(element, 0), cache_entry_end_time(element, element->cache->length - 1));
			break;

		case GST_QUERY_SEGMENT: {
			GstClockTime segstart = GST_CLOCK_TIME_IS_VALID(basesrc->segment.start) ? MAX((GstClockTime) basesrc->segment.start, cache_entry_start_time(element, 0)) : cache_entry_start_time(element, 0);
			GstClockTime segstop = GST_CLOCK_TIME_IS_VALID(basesrc->segment.stop) ? MIN((GstClockTime) basesrc->segment.stop, cache_entry_end_time(element, element->cache->length - 1)) : cache_entry_end_time(element, element->cache->length - 1);
			gst_query_set_segment(query, 1.0, GST_FORMAT_TIME, segstart, segstop);
			break;
		}

		case GST_QUERY_URI:
			gst_query_set_uri(query, element->cache->list[element->index].url);
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
	XLALDestroyCache(element->cache);
	element->cache = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void gstlal_cachesrc_base_init(gpointer klass)
{
}


static void gstlal_cachesrc_class_init(GstLALCacheSrcClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstbasesrc_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesrc_class->stop = GST_DEBUG_FUNCPTR(stop);
	gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR(is_seekable);
	gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
	gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
	gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);

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
}


static void gstlal_cachesrc_init(GstLALCacheSrc *element, GstLALCacheSrcClass *klass)
{
	gst_base_src_set_format(GST_BASE_SRC(element), GST_FORMAT_TIME);

	element->location = NULL;
	element->cache_src_regex = NULL;
	element->cache_dsc_regex = NULL;
	element->cache = NULL;
}

/*
 * GstLALCacheSrc
 *
 * Copyright (C) 2012--2016  Kipp Cannon
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
 * Loads files in order from a list of files stored in a LAL cache.  Each
 * file is placed into its own #GstBuffer either by reading it into memory
 * or mmap()ing the file.
 *
 * Regular expressions can be used to select a subset of files from the
 * cache.  One regular expression can be applied to the "source" column,
 * and one to the "description" column.  The regular expression syntax is
 * the syntax recognized by the XLALCacheSieve() function in LAL.  At the
 * time of writing, this function is implemented using the regcomp() POSIX
 * standard library routine (see regex(7)).
 *
 * The element advertises support for the "lalcache" URI protocol.  The URI
 * format is
 * <quote>lalcache:///path/to/cachefile?cache-src-regex=...?cache-dsc-regex=...</quote>
 * where the source and description regex components are both optional.
 * See the #dataurisrc element for more information.
 *
 * Reviewed:  a922d6dd59d0b58442c0bf7bc4cc4d740b8c6a43 2014-08-12 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 *
 * Action:
 * - Provide link to lal_cachesrc and other similar files on the status page
 *
 * Completed Actions:
 * - are the warnings and errors related to lack of data in do_seek() correct?
 * i.e., are warnings and errors needed for these conditions?  
 * done: lack of start time is no longer an error, element seeks to start of cache in this case
 * - Wrote a unit test
 *
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
#ifdef HAVE_GSTREAMER_1_6
#include <gst/allocators/gstfdmemory.h>
#else
/* FIXME:  remove when we can rely on having 1.6 */
#include "gstfdmemory.h"
#include "gstfdmemory.c"
#undef GST_CAT_DEFAULT
#endif


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


static const gchar *const *uri_get_protocols(GType type)
{
	static const gchar *protocols[] = {URI_SCHEME, NULL};

	return protocols;
}


static gchar *uri_get_uri(GstURIHandler *handler)
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

	return g_string_free(uri, FALSE);
}


static gchar *g_uri_unescape_string_inplace(gchar **s)
{
	gchar *unescaped = g_uri_unescape_string(*s, NULL);
	g_free(*s);
	*s = unescaped;
	return *s;
}


static gboolean uri_set_uri(GstURIHandler *handler, const gchar *uri, GError **err)
{
	/* FIXME:  report errors via err argument */
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
	iface->get_type = GST_DEBUG_FUNCPTR(uri_get_type);
	iface->get_protocols = GST_DEBUG_FUNCPTR(uri_get_protocols);
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


G_DEFINE_TYPE_WITH_CODE(GstLALCacheSrc, gstlal_cachesrc, GST_TYPE_BASE_SRC, additional_initializations(g_define_type_id));


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
	GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS(G_OBJECT_GET_CLASS(basesrc));
	GstMapInfo mapinfo;
	size_t read_offset;
	GstFlowReturn result = GST_FLOW_OK;

	basesrc_class->alloc(basesrc, offset, size, buf);
	if(!buf) {
		result = GST_FLOW_ERROR;
		goto done;
	}

	gst_buffer_map(*buf, &mapinfo, GST_MAP_WRITE);
	read_offset = 0;
	do {
		ssize_t bytes_read = read(fd, mapinfo.data + read_offset, mapinfo.size - read_offset);
		if(bytes_read < 0) {
			GST_ELEMENT_ERROR(basesrc, RESOURCE, READ, (NULL), ("read('%s') failed: %s", path, strerror(errno)));
			gst_buffer_unmap(*buf, &mapinfo);
			gst_buffer_unref(*buf);
			*buf = NULL;
			result = GST_FLOW_ERROR;
			goto done;
		}
		read_offset += bytes_read;
	} while(read_offset < size);
	g_assert_cmpuint(read_offset, ==, size);
	gst_buffer_unmap(*buf, &mapinfo);

	GST_BUFFER_OFFSET(*buf) = offset;
	GST_BUFFER_OFFSET_END(*buf) = offset + size;

done:
	close(fd);
	return result;
}


static GstFlowReturn mmap_buffer(GstBaseSrc *basesrc, const char *path, int fd, guint64 offset, size_t size, GstBuffer **buf)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(basesrc);
	GstMemory *memory;
	GstFlowReturn result = GST_FLOW_OK;

	*buf = gst_buffer_new();
	if(!*buf) {
		close(fd);
		result = GST_FLOW_ERROR;
		goto done;
	}
	/* takes ownership of fd, but only on success.  allocator closes
	 * file when GstMemory is deallocated, but if this operation fails
	 * we must do it ourselves. */
	memory = gst_fd_allocator_alloc(element->fdallocator, fd, size, GST_MEMORY_FLAG_READONLY | GST_FD_MEMORY_FLAG_KEEP_MAPPED | GST_FD_MEMORY_FLAG_MAP_PRIVATE);
	if(!memory) {
		GST_ELEMENT_ERROR(basesrc, RESOURCE, READ, (NULL), ("gst_fd_allocator_alloc('%s') failed", path));
		gst_buffer_unref(*buf);
		close(fd);
		result = GST_FLOW_ERROR;
		goto done;
	}

	gst_buffer_append_memory(*buf, memory);

	GST_BUFFER_OFFSET(*buf) = offset;
	GST_BUFFER_OFFSET_END(*buf) = offset + size;

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


static char *cache_entry_src(GstLALCacheSrc *element, guint i)
{
	return element->cache->list[i].src;
}


static char *cache_entry_dsc(GstLALCacheSrc *element, guint i)
{
	return element->cache->list[i].dsc;
}


/*
 * return TRUE if element j is a fail-over copy of i, FALSE if not.
 * returns FALSE if either i or j is beyond the end of the cache, if i ==
 * j, or if one or both of the cache entries has incomplete metadata.
 */


static gboolean cache_entry_is_failover(GstLALCacheSrc *element, guint i, guint j)
{
	if(i != j && i < element->cache->length && j < element->cache->length) {
		const char *src_i = cache_entry_src(element, i);
		const char *src_j = cache_entry_src(element, j);
		const char *dsc_i = cache_entry_dsc(element, i);
		const char *dsc_j = cache_entry_dsc(element, j);
		GstClockTime t0_i = cache_entry_start_time(element, i);
		GstClockTime t0_j = cache_entry_start_time(element, j);
		GstClockTime dt_i = cache_entry_duration(element, i);
		GstClockTime dt_j = cache_entry_duration(element, j);

		/*
		 * NOTE: LAL's cache reading code translates undefined
		 * start times and durations to 0 and undefined source and
		 * description fields to NULL.
		 */

		return src_i && src_j && dsc_i && dsc_j && t0_i && t0_j && dt_i && dt_j && !g_strcmp0(src_i, src_j) && !g_strcmp0(dsc_i, dsc_j) && t0_i == t0_j && dt_i == dt_j;
	}

	return FALSE;
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
	gboolean result = TRUE;

	g_return_val_if_fail(element->location != NULL, FALSE);
	g_return_val_if_fail(element->cache == NULL, FALSE);

	element->fdallocator = gst_fd_allocator_new();

	element->cache = XLALCacheImport(element->location);
	if(!element->cache) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("error reading '%s': %s", element->location, XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		result = FALSE;
		goto done;
	}
	GST_DEBUG_OBJECT(element, "loaded '%s': %d item(s) in cache", element->location, element->cache->length);

	if(XLALCacheSieve(element->cache, 0, 0, element->cache_src_regex, element->cache_dsc_regex, NULL)) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("error sieving cache '%s': %s", element->location, XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		XLALDestroyCache(element->cache);
		element->cache = NULL;
		result = FALSE;
		goto done;
	}
	GST_DEBUG_OBJECT(element, "%d item(s) remain in cache after sieve", element->cache->length);

	if(!element->cache->length)
		GST_WARNING_OBJECT(element, "cache is empty!");
	else if(XLALCacheSort(element->cache)) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("error sorting cache '%s': %s", element->location, XLALErrorString(XLALGetBaseErrno())));
		XLALClearErrno();
		XLALDestroyCache(element->cache);
		element->cache = NULL;
		result = FALSE;
		goto done;
	}

	element->last_index = 0;
	element->index = 0;
	element->need_discont = TRUE;

done:
	gst_base_src_start_complete(basesrc, result ? GST_FLOW_OK : GST_FLOW_ERROR);
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

	gst_object_unref(element->fdallocator);
	element->fdallocator = NULL;

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

	/*
	 * skip entries in cache that are fail-over copies of the last file
	 * successfully loaded.  this is the first part of the machinery
	 * that allows caches to provide redundant fail-over entries.
	 * NOTE:  cache_entry_is_failover() includes bounds check and also
	 * a check that the two indexes are actually different (i.e., that
	 * we have already loaded a file previously), so we don't need any
	 * extra checks here.
	 */

next:
	while(cache_entry_is_failover(element, element->last_index, element->index)) {
		GST_WARNING_OBJECT(element, "skipping cache entry '%s': fail-over copy of '%s'.", element->cache->list[element->index].url, element->cache->list[element->last_index].url);
		element->index++;
	}

	/*
	 * check for EOS
	 */

	if(element->index >= element->cache->length || (GST_CLOCK_TIME_IS_VALID(basesrc->segment.stop) && cache_entry_start_time(element, element->index) >= (GstClockTime) basesrc->segment.stop)) {
		GST_DEBUG_OBJECT(element, "EOS");
		return GST_FLOW_EOS;
	}

	/*
	 * load the file
	 */

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
		/*
		 * failed to open file.  if the next entry in the cache has
		 * the same observatory, description, and spans exactly the
		 * same time interval, try openning it instead.  this is
		 * the second part of the machinery that allows caches to
		 * provide redundant fail-over copies.
		 */

		if(cache_entry_is_failover(element, element->index, element->index + 1)) {
			GST_WARNING_OBJECT(element, "open('%s') failed: %s.  trying fail-over to next cache entry", path, strerror(errno));
			element->index++;
			g_free(path);
			goto next;
		}
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("open('%s') failed: %s.  no fail-over copies available.", path, strerror(errno)));
		result = GST_FLOW_ERROR;
		goto done;
	}

	if(fstat(fd, &statinfo)) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("fstat('%s') failed: %s", path, strerror(errno)));
		result = GST_FLOW_ERROR;
		close(fd);
		goto done;
	}

	/* these functions take ownership of fd;  we do not close */
	if(element->use_mmap)
		result = mmap_buffer(basesrc, path, fd, offset, statinfo.st_size, buf);
	else
		result = read_buffer(basesrc, path, fd, offset, statinfo.st_size, buf);
	if(result != GST_FLOW_OK)
		goto done;

	/*
	 * finish setting buffer metadata.  need_discont is TRUE for the
	 * first buffer, after which ->last_index will be meaningful, so no
	 * need to check for nonsensical ->last_index value.
	 */

	GST_BUFFER_TIMESTAMP(*buf) = cache_entry_start_time(element, element->index);
	GST_BUFFER_DURATION(*buf) = cache_entry_duration(element, element->index);
	if(element->need_discont || GST_BUFFER_TIMESTAMP(*buf) != cache_entry_end_time(element, element->last_index)) {
		GST_BUFFER_FLAG_SET(*buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}

	GST_DEBUG_OBJECT(element, "pushing '%s' spanning %" GST_BUFFER_BOUNDARIES_FORMAT, path, GST_BUFFER_BOUNDARIES_ARGS(*buf));

	element->last_index = element->index;
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

	GST_DEBUG_OBJECT(element, "requested segment is [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT "), stream time %" GST_TIME_SECONDS_FORMAT ", position %" GST_TIME_SECONDS_FORMAT ", duration %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(segment->start), GST_TIME_SECONDS_ARGS(segment->stop), GST_TIME_SECONDS_ARGS(segment->time), GST_TIME_SECONDS_ARGS(segment->position), GST_TIME_SECONDS_ARGS(segment->duration));

	if(!element->cache) {
		GST_ERROR_OBJECT(element, "no file cache loaded");
		success = FALSE;
		goto done;
	}

	/*
	 * do the seek
	 */

	i = GST_CLOCK_TIME_IS_VALID(segment->start) ? time_to_index(element, segment->start) : 0;
	if(i >= element->cache->length)
		GST_WARNING_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT " beyond end of cache", GST_TIME_SECONDS_ARGS(segment->start));
	else if(GST_CLOCK_TIME_IS_VALID(segment->stop) && (GstClockTime) segment->stop <= cache_entry_start_time(element, i)) {
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, (NULL), ("no data available for segment"));
		success = FALSE;
		goto done;
	} else if(GST_CLOCK_TIME_IS_VALID(segment->start) && (GstClockTime) segment->start < cache_entry_start_time(element, i))
		GST_WARNING_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT ": found uri '%s' spanning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(segment->start), element->cache->list[i].url, GST_TIME_SECONDS_ARGS(cache_entry_start_time(element, i)), GST_TIME_SECONDS_ARGS(cache_entry_end_time(element, i)));
	else
		GST_DEBUG_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT ": found uri '%s' spanning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(segment->start), element->cache->list[i].url, GST_TIME_SECONDS_ARGS(cache_entry_start_time(element, i)), GST_TIME_SECONDS_ARGS(cache_entry_end_time(element, i)));

	/*
	 * done
	 */

	if(i != element->index) {
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
		success = GST_BASE_SRC_CLASS(gstlal_cachesrc_parent_class)->query(basesrc, query);
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
			success = GST_BASE_SRC_CLASS(gstlal_cachesrc_parent_class)->query(basesrc, query);
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

	G_OBJECT_CLASS(gstlal_cachesrc_parent_class)->finalize(object);
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
			GST_CAPS_ANY
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


static void gstlal_cachesrc_init(GstLALCacheSrc *element)
{
	gst_base_src_set_format(GST_BASE_SRC(element), GST_FORMAT_TIME);

	element->fdallocator = NULL;
	element->location = NULL;
	element->cache_src_regex = NULL;
	element->cache_dsc_regex = NULL;
	element->cache = NULL;
}

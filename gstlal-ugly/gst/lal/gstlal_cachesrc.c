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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>


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


/*
 * ============================================================================
 *
 *                             Internal Functions
 *
 * ============================================================================
 */


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


static gboolean start(GstBaseSrc *src)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(src);
	FrCache *cache;
	FrCacheSieve params = {0,};

	g_return_val_if_fail(element->location != NULL, FALSE);

	element->cache = XLALFrImportCache(element->location);
	/* this puts the files in time order */
	cache = XLALFrSieveCache(element->cache, &params);
	if(!cache) {
		XLALFrDestroyCache(element->cache);
		element->cache = NULL;
		return FALSE;
	}
	element->cache = cache;

	element->index = 0;

	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *src)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(src);

	XLALFrDestroyCache(element->cache);
	element->cache = NULL;

	return TRUE;
}


/*
 * is_seekable()
 */


static gboolean is_seekable(GstBaseSrc *src)
{
	return TRUE;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *src, guint64 offset, guint size, GstBuffer **buf)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(src);
	GError *error = NULL;
	gchar *host = NULL;
	gchar *path;
	int fd;
	struct stat statinfo;
	size_t read_offset;
	GstFlowReturn result = GST_FLOW_OK;

	g_assert(element->cache != NULL);

	if(element->index >= element->cache->numFrameFiles)
		return GST_FLOW_UNEXPECTED;

	path = g_filename_from_uri(element->cache->frameFiles[element->index].url, &host, &error);
	if(error) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("error parsing URI '%s': %s", element->cache->frameFiles[element->index].url, error->message));
		g_error_free(error);
		result = GST_FLOW_ERROR;
		goto done;
	}

	fd = open(path, O_RDONLY);
	if(fd < 0) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("open('%s') failed: %s", path, sys_errlist[errno]));
		result = GST_FLOW_ERROR;
		g_free(path);
		goto done;
	}
	g_free(path);

	if(fstat(fd, &statinfo)) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("fstat('%s') failed: %s", path, sys_errlist[errno]));
		result = GST_FLOW_ERROR;
		close(fd);
		goto done;
	}

	result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(src), offset, statinfo.st_size, GST_PAD_CAPS(GST_BASE_SRC_PAD(src)), buf);
	if(result != GST_FLOW_OK)
		goto done;

	for(read_offset = 0; read_offset < GST_BUFFER_SIZE(*buf); ) {
		ssize_t bytes_read = read(fd, GST_BUFFER_DATA(*buf) + read_offset, GST_BUFFER_SIZE(*buf) - read_offset);
		if(bytes_read < 0) {
			GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("read('%s') failed: %s", path, sys_errlist[errno]));
			result = GST_FLOW_ERROR;
			close(fd);
			goto done;
		}
		read_offset += bytes_read;
	}
	close(fd);

	GST_BUFFER_TIMESTAMP(*buf) = element->cache->frameFiles[element->index].startTime * GST_SECOND;
	GST_BUFFER_DURATION(*buf) = element->cache->frameFiles[element->index].duration * GST_SECOND;
	GST_BUFFER_OFFSET(*buf) = offset;
	GST_BUFFER_OFFSET_END(*buf) = GST_BUFFER_OFFSET(*buf) + GST_BUFFER_SIZE(*buf);
	if(element->index == 0 || GST_BUFFER_TIMESTAMP(*buf) != (GstClockTime) (element->cache->frameFiles[element->index - 1].startTime + element->cache->frameFiles[element->index - 1].duration) * GST_SECOND)
		GST_BUFFER_FLAG_SET(*buf, GST_BUFFER_FLAG_DISCONT);

	GST_DEBUG_OBJECT(element, "pushing %s %" GST_BUFFER_BOUNDARIES_FORMAT, element->cache->frameFiles[element->index].url, GST_BUFFER_BOUNDARIES_ARGS(*buf));

	element->index++;
done:
	return result;
}


/*
 * do_seek()
 */


static gboolean do_seek(GstBaseSrc *basesrc, GstSegment *segment)
{
	GstLALCacheSrc *element = GSTLAL_CACHESRC(basesrc);
	GstClockTime min, max;
	guint i;
	gboolean success = TRUE;

	g_return_val_if_fail(element->cache != NULL, FALSE);

	/*
	 * require a start time
	 */

	if(!GST_CLOCK_TIME_IS_VALID(segment->start)) {
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, (NULL), ("start time is required"));
		/*success = FALSE;*/
		goto done;
	}

	/*
	 * check range
	 */

	if(element->cache->numFrameFiles < 1) {
		GST_ELEMENT_ERROR(element, RESOURCE, SEEK, (NULL), ("seek impossible: no files in cache"));
		/*success = FALSE;*/
		goto done;
	}
	min = element->cache->frameFiles[0].startTime * GST_SECOND;
	max = (element->cache->frameFiles[element->cache->numFrameFiles - 1].startTime + element->cache->frameFiles[element->cache->numFrameFiles - 1].duration) * GST_SECOND;
	if((GstClockTime) segment->start < min || (GstClockTime) segment->start >= max) {
		/*GST_ELEMENT_ERROR(element, RESOURCE, SEEK, (NULL), ("seek to %" GST_TIME_SECONDS_FORMAT " out of range: cache spans [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(segment->start), GST_TIME_SECONDS_ARGS(min), GST_TIME_SECONDS_ARGS(max)));*/
		/*success = FALSE;*/
		goto done;
	}

	/*
	 * do the seek.  the loop assumes the cache entries are in time
	 * order and searches for the first file whose end is past the
	 * requested time.
	 */

	for(i = 0; i < element->cache->numFrameFiles; i++) {
		if(i)
			g_assert_cmpuint(element->cache->frameFiles[i].startTime, >=, element->cache->frameFiles[i - 1].startTime);
		min = element->cache->frameFiles[i].startTime * GST_SECOND;
		max = (element->cache->frameFiles[i].startTime + element->cache->frameFiles[i].duration) * GST_SECOND;
		if((GstClockTime) segment->start < max) {
			GST_DEBUG_OBJECT(element, "seek to %" GST_TIME_SECONDS_FORMAT ":  found file '%s' spanning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(segment->start), element->cache->frameFiles[i].url, GST_TIME_SECONDS_ARGS(min), GST_TIME_SECONDS_ARGS(max));
			element->index = i;
			goto done;
		}
	}
	/*success = FALSE;*/

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
	else
		switch(GST_QUERY_TYPE(query)) {
		case GST_QUERY_POSITION:
			if(element->index < element->cache->numFrameFiles)
				/* report start of next file */
				gst_query_set_position(query, GST_FORMAT_TIME, (element->cache->frameFiles[element->index].startTime - element->cache->frameFiles[0].startTime) * GST_SECOND);
			else
				/* special case for EOS:  report end of last file */
				gst_query_set_position(query, GST_FORMAT_TIME, (element->cache->frameFiles[element->cache->numFrameFiles - 1].startTime + element->cache->frameFiles[element->cache->numFrameFiles - 1].duration - element->cache->frameFiles[0].startTime) * GST_SECOND);
			break;

		case GST_QUERY_DURATION: {
			GstClockTime min = element->cache->frameFiles[0].startTime * GST_SECOND;
			GstClockTime max = (element->cache->frameFiles[element->cache->numFrameFiles - 1].startTime + element->cache->frameFiles[element->cache->numFrameFiles - 1].duration) * GST_SECOND;
			gst_query_set_duration(query, GST_FORMAT_TIME, max - min);
			break;
		}

		case GST_QUERY_SEGMENT: {
			GstClockTime min = element->cache->frameFiles[0].startTime * GST_SECOND;
			GstClockTime max = (element->cache->frameFiles[element->cache->numFrameFiles - 1].startTime + element->cache->frameFiles[element->cache->numFrameFiles - 1].duration) * GST_SECOND;
			gst_query_set_segment(query, 1.0, GST_FORMAT_TIME, min, max);
			break;
		}

		case GST_QUERY_URI:
			gst_query_set_uri(query, element->cache->frameFiles[element->index].url);
			break;

		default:
			success = parent_class->query(basesrc, query);
			break;
		}

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

	g_object_class_install_property(
		gobject_class,
		PROP_LOCATION,
		g_param_spec_string(
			"location",
			"Location",
			"Path to LAL frame file cache.",
			DEFAULT_LOCATION,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

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

	element->cache = NULL;
}

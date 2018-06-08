/* 
 * Copyright (C) 2014 Qi Chu <qi.chu@ligo.org>
 * most of this file is copied from gstfilesink.c 0.10.32 and 0.10.36
 *
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *                    2000 Wim Taymans <wtay@chello.be>
 *                    2006 Wim Taymans <wim@fluendo.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/* This element will synchronize the snr sequencies from all detectors, find 
 * peaks from all detectors and for each peak, do null stream analysis.
 */

#include <glib/gstdio.h>
#include <gst/gst.h>
#include <errno.h>

#include <string.h>
#include <math.h>
#include "postcoh_filesink.h"
#include "postcohinspiral_table_utils.h"

#define EPSILON 5.0
enum 
{
	PROP_0,
	PROP_LOCATION,
	PROP_COMPRESS,
	PROP_SNAPSHOT_INTERVAL
};

G_DEFINE_TYPE_WITH_CODE (
		PostcohFilesink, 
		postcoh_filesink, 
    		GST_TYPE_BASE_SINK, 
		GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "postcoh filesink", 0, "postcoh filesink element"););

/* The following content is mostly copied from gstfilesink.c */

#if 0
/* Copy of glib's g_fopen due to win32 libc/cross-DLL brokenness: we can't
 * use the 'file pointer' opened in glib (and returned from this function)
 * in this library, as they may have unrelated C runtimes. */
static FILE *
gst_fopen (const gchar * filename, const gchar * mode)
{
#ifdef G_OS_WIN32
  wchar_t *wfilename = g_utf8_to_utf16 (filename, -1, NULL, NULL, NULL);
  wchar_t *wmode;
  FILE *retval;
  int save_errno;

  if (wfilename == NULL) {
    errno = EINVAL;
    return NULL;
  }

  wmode = g_utf8_to_utf16 (mode, -1, NULL, NULL, NULL);

  if (wmode == NULL) {
    g_free (wfilename);
    errno = EINVAL;
    return NULL;
  }

  retval = _wfopen (wfilename, wmode);
  save_errno = errno;

  g_free (wfilename);
  g_free (wmode);

  errno = save_errno;
  return retval;
#else
  return fopen (filename, mode);
#endif
}
#endif

static void postcoh_filesink_dispose (GObject * object);

static void postcoh_filesink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void postcoh_filesink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

// We don't need file operations any more as is taken care of by xml writer
//static gboolean postcoh_filesink_open_file (PostcohFilesink * sink);
//static void postcoh_filesink_close_file (PostcohFilesink * sink);

static gboolean postcoh_filesink_start (GstBaseSink * sink);
static gboolean postcoh_filesink_stop (GstBaseSink * sink);
static gboolean postcoh_filesink_event (GstBaseSink * sink, GstEvent * event);
static GstFlowReturn postcoh_filesink_render (GstBaseSink * sink,
    GstBuffer * buffer);

//static gboolean postcoh_filesink_do_seek (PostcohFilesink * filesink,
//    guint64 new_offset);
//static gboolean postcoh_filesink_get_current_offset (PostcohFilesink * filesink,
//    guint64 * p_pos);

//static gboolean postcoh_filesink_query (GstBaseSink * bsink, GstQuery * query);

static void postcoh_filesink_uri_handler_init (gpointer g_iface,
    gpointer iface_data);

//#define postcoh_filesink_parent_class parent_class

#define GST_CAT_DEFAULT postcoh_filesink_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

static void
_do_init (GType filesink_type)
{
  static const GInterfaceInfo urihandler_info = {
    postcoh_filesink_uri_handler_init,
    NULL,
    NULL
  };

  g_type_add_interface_static (filesink_type, GST_TYPE_URI_HANDLER,
      &urihandler_info);
  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "postcoh filesink", 0,
      "postcoh filesink element");
}


#if 0
static void
postcoh_filesink_base_init(gpointer g_class)
{

  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (g_class);
  gst_element_class_set_details_simple (gstelement_class,
      "Postcoh File Sink",
      "Sink/File", "Write postcoh tables to a xml file",
      "Qi Chu <qi.chu at ligo dot org>");

  gst_element_class_add_pad_template (
		  gstelement_class,
		  gst_pad_template_new(
			  "sink",
			  GST_PAD_SINK,
			  GST_PAD_ALWAYS,
			  gst_caps_from_string(
				  "application/x-lal-postcoh"
			  )
			  )
		  );

}
#endif
static void
postcoh_filesink_class_init (PostcohFilesinkClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS (klass);

  postcoh_filesink_parent_class = g_type_class_ref(GST_TYPE_BASE_SINK);
  
  gobject_class->dispose = postcoh_filesink_dispose;

  gobject_class->set_property = postcoh_filesink_set_property;
  gobject_class->get_property = postcoh_filesink_get_property;

  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  gst_element_class_set_details_simple (gstelement_class,
      "Postcoh File Sink",
      "Sink/File", "Write postcoh tables to a xml file",
      "Qi Chu <qi.chu at ligo dot org>");

  gst_element_class_add_pad_template (
		  gstelement_class,
		  gst_pad_template_new(
			  "sink",
			  GST_PAD_SINK,
			  GST_PAD_ALWAYS,
			  gst_caps_from_string(
				  "application/x-lal-postcoh"
			  )
			  )
		  );

  g_object_class_install_property (gobject_class, PROP_LOCATION,
      g_param_spec_string ("location", "File Location",
          "Location prefix of the file to write", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_COMPRESS,
      g_param_spec_int ("compression", "Whether to compress or not",
          "(0) Non-compression, (1) Compression", 0, 1, 1,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SNAPSHOT_INTERVAL,
      g_param_spec_int ("snapshot-interval", "How often to store postcoh table",
          "How often to store postcoh table: (0) At the end, (N) Every N seconds", 0, G_MAXINT, 0,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  
  gstbasesink_class->start = GST_DEBUG_FUNCPTR (postcoh_filesink_start);
  gstbasesink_class->stop = GST_DEBUG_FUNCPTR (postcoh_filesink_stop);
//  gstbasesink_class->query = GST_DEBUG_FUNCPTR (postcoh_filesink_query);
  gstbasesink_class->render = GST_DEBUG_FUNCPTR (postcoh_filesink_render);
  gstbasesink_class->event = GST_DEBUG_FUNCPTR (postcoh_filesink_event);

}

static void
postcoh_filesink_init (PostcohFilesink * sink)
{
  sink->filename = NULL;
  sink->file = NULL;
  sink->uri = NULL;

  sink->cur_filename = NULL;
  gst_base_sink_set_sync (GST_BASE_SINK (sink), FALSE);
}

static void
postcoh_filesink_dispose (GObject * object)
{
  PostcohFilesink *sink = POSTCOH_FILESINK (object);

  G_OBJECT_CLASS (postcoh_filesink_parent_class)->dispose (object);

  if (sink->uri) {
	  g_free (sink->uri);
	  sink->uri = NULL;
  }
  if (sink->filename) {
	  g_free (sink->filename);
	  sink->filename = NULL;
  }
}

/* copied from gstreamer-0.10.36/ gstreamer/ gst/ gsturi.c */

static gchar *
gst_file_utils_canonicalise_path (const gchar * path)
{
  gchar **parts, **p, *clean_path;

#ifdef G_OS_WIN32
  {
    GST_WARNING ("FIXME: canonicalise win32 path");
    return g_strdup (path);
  }
#endif

  parts = g_strsplit (path, "/", -1);

  p = parts;
  while (*p != NULL) {
    if (strcmp (*p, ".") == 0) {
      /* just move all following parts on top of this, incl. NUL terminator */
      g_free (*p);
      g_memmove (p, p + 1, (g_strv_length (p + 1) + 1) * sizeof (gchar *));
      /* re-check the new current part again in the next iteration */
      continue;
    } else if (strcmp (*p, "..") == 0 && p > parts) {
      /* just move all following parts on top of the previous part, incl.
       * NUL terminator */
      g_free (*(p - 1));
      g_free (*p);
      g_memmove (p - 1, p + 1, (g_strv_length (p + 1) + 1) * sizeof (gchar *));
      /* re-check the new current part again in the next iteration */
      --p;
      continue;
    }
    ++p;
  }
  if (*path == '/') {
    guint num_parts;

    num_parts = g_strv_length (parts) + 1;      /* incl. terminator */
    parts = g_renew (gchar *, parts, num_parts + 1);
    g_memmove (parts + 1, parts, num_parts * sizeof (gchar *));
    parts[0] = g_strdup ("/");
  }

  clean_path = g_build_filenamev (parts);
  g_strfreev (parts);
  return clean_path;
}

static gboolean
file_path_contains_relatives (const gchar * path)
{
  return (strstr (path, "/./") != NULL || strstr (path, "/../") != NULL ||
      strstr (path, G_DIR_SEPARATOR_S "." G_DIR_SEPARATOR_S) != NULL ||
      strstr (path, G_DIR_SEPARATOR_S ".." G_DIR_SEPARATOR_S) != NULL);
}

/**
 * gst_filename_to_uri:
 * @filename: absolute or relative file name path
 * @error: pointer to error, or NULL
 *
 * Similar to g_filename_to_uri(), but attempts to handle relative file paths
 * as well. Before converting @filename into an URI, it will be prefixed by
 * the current working directory if it is a relative path, and then the path
 * will be canonicalised so that it doesn't contain any './' or '../' segments.
 *
 * On Windows #filename should be in UTF-8 encoding.
 *
 * Since: 0.10.33
 */
static gchar *
gst_filename_to_uri_local (const gchar * filename, GError ** error)
{
  gchar *abs_location = NULL;
  gchar *uri, *abs_clean;

  g_return_val_if_fail (filename != NULL, NULL);
  g_return_val_if_fail (error == NULL || *error == NULL, NULL);

  if (g_path_is_absolute (filename)) {
    if (!file_path_contains_relatives (filename)) {
      uri = g_filename_to_uri (filename, NULL, error);
      goto beach;
    }

    abs_location = g_strdup (filename);
  } else {
    gchar *cwd;

    cwd = g_get_current_dir ();
    abs_location = g_build_filename (cwd, filename, NULL);
    g_free (cwd);

    if (!file_path_contains_relatives (abs_location)) {
      uri = g_filename_to_uri (abs_location, NULL, error);
      goto beach;
    }
  }

  /* path is now absolute, but contains '.' or '..' */
  abs_clean = gst_file_utils_canonicalise_path (abs_location);
  GST_LOG ("'%s' -> '%s' -> '%s'", filename, abs_location, abs_clean);
  uri = g_filename_to_uri (abs_clean, NULL, error);
  g_free (abs_clean);

beach:

  g_free (abs_location);
  GST_DEBUG ("'%s' -> '%s'", filename, uri);
  return uri;
}

static gboolean
postcoh_filesink_set_location (PostcohFilesink * sink, const gchar * location)
{
  if (sink->file)
    goto was_open;

  g_free (sink->filename);
  g_free (sink->uri);
  if (location != NULL) {
    /* we store the filename as we received it from the application. On Windows
     * this should be in UTF8 */
    sink->filename = g_strdup (location);
    sink->uri = gst_filename_to_uri_local (location, NULL);
//    sink->uri = g_filename_to_uri (location, NULL, NULL);
//    sink->uri = gst_uri_construct ("file", sink->filename);
//    printf ("filename : %s", sink->filename);
//    printf ("uri      : %s", sink->uri);
  } else {
    sink->filename = NULL;
    sink->uri = NULL;
  }

  return TRUE;

  /* ERRORS */
was_open:
  {
    g_warning ("Changing the `location' property on filesink when a file is "
        "open is not supported.");
    return FALSE;
  }
}

static void
postcoh_filesink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  PostcohFilesink *sink = POSTCOH_FILESINK (object);

  switch (prop_id) {
    case PROP_LOCATION:
      postcoh_filesink_set_location (sink, g_value_get_string (value));
      break;
    case PROP_COMPRESS:
      sink->compress = g_value_get_int(value);
      break;
    case PROP_SNAPSHOT_INTERVAL:
      sink->snapshot_interval = g_value_get_int(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
postcoh_filesink_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  PostcohFilesink *sink = POSTCOH_FILESINK (object);

  switch (prop_id) {
    case PROP_LOCATION:
      g_value_set_string (value, sink->filename);
      break;
    case PROP_COMPRESS:
      g_value_set_int (value, sink->compress);
      break;
    case PROP_SNAPSHOT_INTERVAL:
      g_value_set_int (value, sink->snapshot_interval);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}
#if 0
static gboolean
postcoh_filesink_open_file (PostcohFilesink * sink)
{
  gint mode;

  /* open the file */
  if (sink->filename == NULL || sink->filename[0] == '\0')
    goto no_filename;

  sink->file = gst_fopen (sink->filename, "wb");
  if (sink->file == NULL)
    goto open_failed;

  return TRUE;

  /* ERRORS */
no_filename:
  {
    GST_ELEMENT_ERROR (sink, RESOURCE, NOT_FOUND,
        (_("No file name specified for writing.")), (NULL));
    return FALSE;
  }
open_failed:
  {
    GST_ELEMENT_ERROR (sink, RESOURCE, OPEN_WRITE,
        (_("Could not open file \"%s\" for writing."), sink->filename),
        GST_ERROR_SYSTEM);
    return FALSE;
  }
}

static void
postcoh_filesink_close_file (PostcohFilesink * sink)
{
  if (sink->file) {
    if (fclose (sink->file) != 0)
      goto close_failed;

    GST_DEBUG_OBJECT (sink, "closed file");
    sink->file = NULL;

  }
  return;

  /* ERRORS */
close_failed:
  {
    GST_ELEMENT_ERROR (sink, RESOURCE, CLOSE,
        (_("Error closing file \"%s\"."), sink->filename), GST_ERROR_SYSTEM);
    return;
  }
}
#endif

static gboolean 
postcoh_filesink_start_xml(PostcohFilesink *sink)
{
    sink->writer = xmlNewTextWriterFilename(sink->cur_filename->str, sink->compress);
    xmlTextWriterPtr writer = sink->writer;
    if (writer == NULL) {
        printf("testXmlwriterFilename: Error creating the xml writer\n");
        return FALSE;
    }

    int rc;
    rc = xmlTextWriterSetIndent(writer, 1);
    rc = xmlTextWriterSetIndentString(writer, BAD_CAST "\t");

    /* Start the document with the xml default for the version,
     * encoding utf-8 and the default for the standalone
     * declaration. */
    rc = xmlTextWriterStartDocument(writer, NULL, MY_ENCODING, NULL);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartDocument\n");
        return FALSE;
    }

    rc = xmlTextWriterWriteDTD(writer, BAD_CAST "LIGO_LW", NULL, BAD_CAST "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt", NULL);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteDTD\n");
        return FALSE;
    }

    /* Start an element named "LIGO_LW". Since thist is the first
     * element, this will be the root element of the document. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return FALSE;
    }

    /* Start an element named "LIGO_LW" as child of EXAMPLE. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return FALSE;
    }

  sink->xtable = (XmlTable *)malloc(sizeof(XmlTable));
  XmlTable * xtable = sink->xtable;
  postcohinspiral_table_init(xtable);
    // Add the Table Node
    rc = xmlTextWriterStartElement(writer, BAD_CAST "Table");

    // Add Attribute "Name"
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST xtable->tableName->str);

    // Write Column Nodes
    int i;
    for (i = 0; i < (int)xtable->names->len; ++i)
    {
        rc = xmlTextWriterStartElement(writer, BAD_CAST "Column");

        GString *colName = &g_array_index(xtable->names, GString, i);
        GString *type = &g_array_index(xtable->type_names, GString, i);


        rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST type->str);
        rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST colName->str);
        
        rc = xmlTextWriterEndElement(writer);
    }

    // Write Stream Nodes
    rc = xmlTextWriterStartElement(writer, BAD_CAST "Stream"); 
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Delimiter", BAD_CAST xtable->delimiter->str);
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Type", BAD_CAST "Local");
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name", BAD_CAST xtable->tableName->str);
    
    rc = xmlTextWriterWriteString(writer, BAD_CAST "\n");
    if (rc < 0)
	    return FALSE;
    else
	    return TRUE;
}


static gboolean postcoh_filesink_end_xml(PostcohFilesink *sink)
{
  xmlTextWriterPtr writer = sink->writer;
  int rc = 0;

  /* for stream */
  rc = xmlTextWriterEndElement(writer);
  if (rc < 0) {
      printf
          ("Error at xmlTextWriterEndElement\n");
      return FALSE;
  }



  /* for table */
  xmlTextWriterEndElement(writer);
  if (rc < 0) {
      printf
          ("Error at xmlTextWriterEndElement\n");
      return FALSE;
  }


  /* for the whole document */
  xmlTextWriterEndDocument(writer);
  if (rc < 0) {
      printf
          ("Error at xmlTextWriterEndDocument\n");
      return FALSE;
  }
  return TRUE;
}

static gboolean
postcoh_filesink_cleanup_xml (PostcohFilesink * sink)
{

  if (sink->writer)
    xmlFreeTextWriter(sink->writer);
//  postcoh_filesink_close_file (sink);
  if (sink->xtable) {
	  free(sink->xtable);
	  sink->xtable = NULL;
  }
  if (sink->cur_filename) {
    g_string_free(sink->cur_filename, TRUE);
    sink->cur_filename = NULL;
  }
  return TRUE;
}

static gboolean postcoh_filesink_is_invalid_background(PostcohInspiralTable *table)
{
	gboolean is_invalid = FALSE;
	if (table->is_background == 1 && table->cohsnr < EPSILON)
		is_invalid = TRUE;
	return is_invalid;

}
static GstFlowReturn
postcoh_filesink_write_table_from_buf(PostcohFilesink *sink, GstBuffer *buf)
{
  GstMapInfo mapinfo;
  gst_buffer_map(buf, &mapinfo, GST_MAP_WRITE);

  PostcohInspiralTable *table = (PostcohInspiralTable *)&mapinfo.data;
  PostcohInspiralTable *table_end = (PostcohInspiralTable *) (&mapinfo.data + mapinfo.size);

  XmlTable *xtable = sink->xtable;
  int rc;
  gboolean is_invalid = FALSE;

  GST_LOG_OBJECT(sink, "start to write postcoh table");
  for(; table<table_end; table++) {
	//is_invalid = postcoh_filesink_is_invalid_background(table);
	if (!is_invalid) {
        GString *line = g_string_new("\t\t\t\t");
	postcohinspiral_table_set_line(line, table, xtable);
        rc = xmlTextWriterWriteFormatRaw(sink->writer, line->str);
	if (rc < 0)
		return GST_FLOW_ERROR;
        g_string_free(line, TRUE);
	}
  }
  gst_buffer_unmap(buf, &mapinfo);

  GST_LOG_OBJECT (sink,
		"Writen a buffer (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
		GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
		G_GUINT64_FORMAT,  (unsigned int) gst_buffer_get_size(buf),
		GST_TIME_ARGS (GST_BUFFER_PTS(buf)),
		GST_TIME_ARGS (GST_BUFFER_DURATION (buf)),
		GST_BUFFER_OFFSET (buf), GST_BUFFER_OFFSET_END (buf));

  return GST_FLOW_OK;
}

/* handle events (search) */
static gboolean
postcoh_filesink_event (GstBaseSink * basesink, GstEvent * event)
{
  GstEventType type;
  PostcohFilesink *sink;

  sink = POSTCOH_FILESINK (basesink);

  type = GST_EVENT_TYPE (event);

  switch (type) {
    case GST_EVENT_EOS:
//      if (fflush (sink->file))
//        goto flush_failed;

    GST_LOG_OBJECT(sink, "EVENT EOS. Finish writing document");

    gboolean rt = postcoh_filesink_end_xml(sink);
    if (rt == FALSE) {
      GST_ERROR_OBJECT(sink, "postcoh xml end writing failed");
      return FALSE;
    }
    /* close current file */
    if (sink->writer) {
      xmlFreeTextWriter(sink->writer);
      sink->writer = NULL;
    }

    guint duration = (sink->t_end - sink->t_start) / GST_SECOND;
    if (duration != (unsigned) sink->snapshot_interval) {
      GString *new_filename = g_string_new(sink->uri);
      gint gps_time = sink->t_start / GST_SECOND;
      g_string_append_printf(new_filename, "_%d_%d.xml.gz", gps_time, duration);
      /* rename does not recognize the first 7 chars "file://" so we need to remove that */
      gchar *tmp_new_filename_str = g_strdup(&(new_filename->str[7]));
      gchar *tmp_cur_filename_str = g_strdup(&(sink->cur_filename->str[7]));
      /* rename the current file to a proper name */
      if(g_rename(tmp_cur_filename_str, tmp_new_filename_str) == -1) {
        perror("Error when renaming");
      }
      g_string_free(new_filename, TRUE);
      new_filename = NULL;
      g_free(tmp_new_filename_str);
      g_free(tmp_cur_filename_str);
    }
      break;
    default:
      break;
  }

  return TRUE;

  // return GST_BASE_SINK_CLASS (parent_class)->event (sink, event);
#if 0
flush_failed:
  {
    GST_ELEMENT_ERROR (sink, RESOURCE, WRITE,
        (_("Error while writing to file \"%s\"."), sink->filename),
        GST_ERROR_SYSTEM);
    gst_event_unref (event);
    return FALSE;
  }
#endif
}



static GstFlowReturn
postcoh_filesink_render (GstBaseSink * basesink, GstBuffer * buf)
{
  PostcohFilesink *sink;
  sink = POSTCOH_FILESINK (basesink);
  sink->t_end = GST_BUFFER_PTS(buf) + GST_BUFFER_DURATION(buf);

  if (!GST_CLOCK_TIME_IS_VALID(sink->t_start)) {
    sink->t_start = GST_BUFFER_PTS(buf);
    // This is the filename prefix.
    g_assert(sink->uri);
    sink->cur_filename = g_string_new(sink->uri);
    gint gps_time = sink->t_start / GST_SECOND;
    if (sink->snapshot_interval)
      g_string_append_printf(sink->cur_filename, "_%d_%d.xml.gz", gps_time, sink->snapshot_interval);
    else 
      g_string_append_printf(sink->cur_filename, "_%d_end.xml.gz", gps_time);

    gboolean rt = postcoh_filesink_start_xml(sink);
    if (rt == FALSE) {
      GST_ERROR_OBJECT(sink, "postcoh xml header writing failed");
      return GST_FLOW_ERROR;
    }
    GstFlowReturn rs = postcoh_filesink_write_table_from_buf(sink, buf);
    return rs;
  }

  GstClockTime t_cur = GST_BUFFER_PTS(buf);
  if (sink->snapshot_interval > 0 && (t_cur - sink->t_start)/GST_SECOND > (unsigned) sink->snapshot_interval) {

    gboolean rt = postcoh_filesink_end_xml(sink);
    if (rt == FALSE) {
      GST_ERROR_OBJECT(sink, "postcoh xml end writing failed");
      return GST_FLOW_ERROR;
    }

    postcoh_filesink_cleanup_xml(sink);

    /* Create a new xml file for postcoh table */
    sink->t_start = t_cur;
    // This is the filename prefix.
    g_assert(sink->uri);
    sink->cur_filename = g_string_new(sink->uri);
    gint gps_time = sink->t_start / GST_SECOND;
    g_string_append_printf(sink->cur_filename, "_%d_%d.xml.gz", gps_time, sink->snapshot_interval);
    rt = postcoh_filesink_start_xml(sink);
    if (rt == FALSE) {
      GST_ERROR_OBJECT(sink, "postcoh xml header writing failed");
      return GST_FLOW_ERROR;
    }

  }
    GstFlowReturn rs = postcoh_filesink_write_table_from_buf(sink, buf);
    return rs;
}

static gboolean
postcoh_filesink_start (GstBaseSink * basesink)
{
  PostcohFilesink *sink = POSTCOH_FILESINK (basesink);
  sink->t_start = GST_CLOCK_TIME_NONE;
  return TRUE;
}

static gboolean
postcoh_filesink_stop (GstBaseSink * basesink)
{

  PostcohFilesink *sink = POSTCOH_FILESINK (basesink);
  postcoh_filesink_cleanup_xml(sink);
  return TRUE;
}
/*** GSTURIHANDLER INTERFACE *************************************************/

static GstURIType
postcoh_filesink_uri_get_type (void)
{
  return GST_URI_SINK;
}

static gchar **
postcoh_filesink_uri_get_protocols (void)
{
  static gchar *protocols[] = { (char *) "file", NULL };

  return protocols;
}

static const gchar *
postcoh_filesink_uri_get_uri (GstURIHandler * handler)
{
  PostcohFilesink *sink = POSTCOH_FILESINK (handler);

  return sink->uri;
}

static gboolean
postcoh_filesink_uri_set_uri (GstURIHandler * handler, const gchar * uri)
{
  gchar *protocol, *location;
  gboolean ret;
  PostcohFilesink *sink = POSTCOH_FILESINK (handler);

  protocol = gst_uri_get_protocol (uri);
  if (strcmp (protocol, "file") != 0) {
    g_free (protocol);
    return FALSE;
  }
  g_free (protocol);

  /* allow file://localhost/foo/bar by stripping localhost but fail
   * for every other hostname */
  if (g_str_has_prefix (uri, "file://localhost/")) {
    char *tmp;

    /* 16 == strlen ("file://localhost") */
    tmp = g_strconcat ("file://", uri + 16, NULL);
    /* we use gst_uri_get_location() although we already have the
     * "location" with uri + 16 because it provides unescaping */
    location = gst_uri_get_location (tmp);
    g_free (tmp);
  } else if (strcmp (uri, "file://") == 0) {
    /* Special case for "file://" as this is used by some applications
     *  to test with gst_element_make_from_uri if there's an element
     *  that supports the URI protocol. */
    postcoh_filesink_set_location (sink, NULL);
    return TRUE;
  } else {
    location = gst_uri_get_location (uri);
  }

  if (!location)
    return FALSE;
  if (!g_path_is_absolute (location)) {
    g_free (location);
    return FALSE;
  }

  ret = postcoh_filesink_set_location (sink, location);
  g_free (location);

  return ret;
}

static void
postcoh_filesink_uri_handler_init (gpointer g_iface, gpointer iface_data)
{
  GstURIHandlerInterface *iface = (GstURIHandlerInterface *) g_iface;

  iface->get_type = postcoh_filesink_uri_get_type;
  iface->get_protocols = postcoh_filesink_uri_get_protocols;
  iface->get_uri = postcoh_filesink_uri_get_uri;
  iface->set_uri = postcoh_filesink_uri_set_uri;
}


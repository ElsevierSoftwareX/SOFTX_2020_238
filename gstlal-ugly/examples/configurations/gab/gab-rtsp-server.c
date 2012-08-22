/*
 * GStreamer Audio Bridge, based on gst-rtsp-server example code test-readme.c
 * GStreamer
 * Copyright (C) 2008 Wim Taymans <wim.taymans at gmail.com>
 * Copyright (C) 2010 Leo Singer <leo.singer@ligo.org>
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

#include <gst/gst.h>

#include <gst/rtsp-server/rtsp-server.h>


/*
 * ============================================================================
 *
 *                              GabMediaMapping
 *
 * ============================================================================
 */


GType gab_media_mapping_get_type();

#define GAB_TYPE_MEDIA_MAPPING              (gab_media_mapping_get_type ())
#define GAB_IS_MEDIA_MAPPING(obj)           (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GAB_TYPE_MEDIA_MAPPING))
#define GAB_IS_MEDIA_MAPPING_CLASS(klass)   (G_TYPE_CHECK_CLASS_TYPE ((klass), GAB_TYPE_MEDIA_MAPPING))
#define GAB_MEDIA_MAPPING_GET_CLASS(obj)    (G_TYPE_INSTANCE_GET_CLASS ((obj), GAB_TYPE_MEDIA_MAPPING, GabMediaMappingClass))
#define GAB_MEDIA_MAPPING(obj)              (G_TYPE_CHECK_INSTANCE_CAST ((obj), GAB_TYPE_MEDIA_MAPPING, GabMediaMapping))
#define GAB_MEDIA_MAPPING_CLASS(klass)      (G_TYPE_CHECK_CLASS_CAST ((klass), GAB_TYPE_MEDIA_MAPPING, GabMediaMappingClass))
#define GAB_MEDIA_MAPPING_CAST(obj)         ((GabMediaMapping*)(obj))
#define GAB_MEDIA_MAPPING_CLASS_CAST(klass) ((GabMediaMappingClass*)(klass))

typedef struct {
  GstRTSPMediaMapping parent;
  GstRTSPMediaFactory *factory;
} GabMediaMapping;

typedef struct {
  GstRTSPMediaMappingClass parent;
} GabMediaMappingClass;

/* override GstRTSPMediaFactory->find_media, always return our factory regardless of path */
static GstRTSPMediaFactory *find_media (GstRTSPMediaMapping *mapping, const GstRTSPUrl *url)
{
  GabMediaMapping *gab_mapping = GAB_MEDIA_MAPPING(mapping);
  return gab_mapping->factory;
}

static void gab_media_mapping_class_init (gpointer class, gpointer class_data)
{
  GstRTSPMediaMappingClass *parent_class = GST_RTSP_MEDIA_MAPPING_CLASS (class);
  parent_class->find_media = GST_DEBUG_FUNCPTR (find_media);
}

static void gab_media_mapping_instance_init (GTypeInstance *object, gpointer class)
{
  GabMediaMapping *mapping = GAB_MEDIA_MAPPING(object);
  mapping->factory = NULL;
}

GType gab_media_mapping_get_type()
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GabMediaMappingClass),
			.class_init = gab_media_mapping_class_init,
			.instance_size = sizeof(GabMediaMapping),
			.instance_init = gab_media_mapping_instance_init,
		};
		type = g_type_register_static(GST_TYPE_RTSP_MEDIA_MAPPING, "GabMediaMapping", &info, 0);
	}

	return type;
}

GabMediaMapping *
gab_media_mapping_new (void)
{
  GabMediaMapping *result;

  result = g_object_new (GAB_TYPE_MEDIA_MAPPING, NULL);

  return result;
}


/*
 * ============================================================================
 *
 *                              GabMediaFactory
 *
 * ============================================================================
 */


GType gab_media_factory_get_type();

#define GAB_TYPE_MEDIA_FACTORY              (gab_media_factory_get_type ())
#define GAB_IS_MEDIA_FACTORY(obj)           (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GAB_TYPE_MEDIA_FACTORY))
#define GAB_IS_MEDIA_FACTORY_CLASS(klass)   (G_TYPE_CHECK_CLASS_TYPE ((klass), GAB_TYPE_MEDIA_FACTORY))
#define GAB_MEDIA_FACTORY_GET_CLASS(obj)    (G_TYPE_INSTANCE_GET_CLASS ((obj), GAB_TYPE_MEDIA_FACTORY, GabMediaFactoryClass))
#define GAB_MEDIA_FACTORY(obj)              (G_TYPE_CHECK_INSTANCE_CAST ((obj), GAB_TYPE_MEDIA_FACTORY, GabMediaFactory))
#define GAB_MEDIA_FACTORY_CLASS(klass)      (G_TYPE_CHECK_CLASS_CAST ((klass), GAB_TYPE_MEDIA_FACTORY, GabMediaFactoryClass))
#define GAB_MEDIA_FACTORY_CAST(obj)         ((GabMediaFactory*)(obj))
#define GAB_MEDIA_FACTORY_CLASS_CAST(klass) ((GabMediaFactoryClass*)(klass))


typedef struct {
  GstRTSPMediaFactory parent;

  /* hostname and port for NDS server */
  gchar *host;
  gint port;
} GabMediaFactory;

typedef struct {
  GstRTSPMediaFactoryClass parent;
} GabMediaFactoryClass;

static GstElement * (*base_get_element) (GstRTSPMediaFactory *factory, const GstRTSPUrl *url);

static GstElement *
get_element (GstRTSPMediaFactory *factory, const GstRTSPUrl *url)
{
  GabMediaFactory *gab_factory = GAB_MEDIA_FACTORY (factory);
  GstElement *element = base_get_element(factory, url);
  GstIterator *iterator = gst_bin_iterate_sources (GST_BIN (element));
  GstElement *source;
  GValue channel_name_value = {0,};
  GValue host_value = {0,};
  GValue port_value = {0,};

  GstIteratorResult result = gst_iterator_next (iterator, (gpointer) &source);
  gst_iterator_free(iterator);

  if (result != GST_ITERATOR_OK)
  {
    GST_ERROR ("could not get source element in pipeline");
    gst_object_unref(source);
    gst_object_unref(element);
    return NULL;
  }

  g_value_init (&host_value, G_TYPE_STRING);
  g_value_set_string (&host_value, gab_factory->host);
  g_object_set_property (G_OBJECT (source), "host", &host_value);

  g_value_init (&port_value, G_TYPE_INT);
  g_value_set_int (&port_value, gab_factory->port);
  g_object_set_property (G_OBJECT (source), "port", &port_value);

  g_value_init (&channel_name_value, G_TYPE_STRING);
  g_value_take_string (&channel_name_value, g_path_get_basename(url->abspath));
  g_object_set_property (G_OBJECT (source), "channel-name", &channel_name_value);

  gst_object_unref(source);

  return element;
}

static void gab_media_factory_class_init (gpointer class, gpointer class_data)
{
  GstRTSPMediaFactoryClass *parent_class = GST_RTSP_MEDIA_FACTORY_CLASS (class);
  base_get_element = parent_class->get_element;
  parent_class->get_element = GST_DEBUG_FUNCPTR (get_element);
}

GType gab_media_factory_get_type()
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GabMediaFactoryClass),
			.class_init = gab_media_factory_class_init,
			.instance_size = sizeof(GabMediaFactory),
		};
		type = g_type_register_static(GST_TYPE_RTSP_MEDIA_FACTORY, "GabMediaFactory", &info, 0);
	}

	return type;
}

GabMediaFactory *
gab_media_factory_new (void)
{
  GabMediaFactory *result;

  result = g_object_new (GAB_TYPE_MEDIA_FACTORY, NULL);

  return result;
}


/*
 * ============================================================================
 *
 *                               Main program
 *
 * ============================================================================
 */


int
main (int argc, char *argv[])
{
  GMainLoop *loop;
  GstRTSPServer *server;
  GabMediaMapping *mapping;
  GabMediaFactory *factory;
  GError *err = NULL;
  gchar *host = NULL;
  gint port = 31200;

  GOptionEntry options[] = {
    {"host", 'n', 0, G_OPTION_ARG_STRING, &host, "NDS server hostname", NULL},
    {"port", 'p', 0, G_OPTION_ARG_INT, &port, "NDS server port", NULL},
    {NULL}
  };

  GOptionContext *ctx;

  if (!g_thread_supported ())
    g_thread_init (NULL);

  ctx = g_option_context_new ("[GAB ARGUMENTS]");
  g_option_context_add_main_entries (ctx, options, NULL);
  g_option_context_add_group (ctx, gst_init_get_option_group ());
  if (!g_option_context_parse (ctx, &argc, &argv, &err)) {
    g_print ("Error initializing: %s\n", GST_STR_NULL (err->message));
    exit (1);
  }
  g_option_context_free (ctx);

  gst_init (&argc, &argv);

  loop = g_main_loop_new (NULL, FALSE);

  /* create a server instance */
  server = gst_rtsp_server_new ();

  /* create new mapping for this server, every server has a default mapper object
   * that be used to map uri mount points to media factories, but we are going
   * to replace it with our own custom mapping */
  mapping = gab_media_mapping_new ();
  gst_rtsp_server_set_media_mapping (server, GST_RTSP_MEDIA_MAPPING (mapping));

  /* make a media factory for a test stream. The default media factory can use
   * gst-launch syntax to create pipelines.
   * any launch line works as long as it contains elements named pay%d. Each
   * element with pay%d names will be a stream */
  factory = gab_media_factory_new ();
  factory->host = host;
  factory->port = port;
  gst_rtsp_media_factory_set_launch (GST_RTSP_MEDIA_FACTORY (factory),
    "( ndssrc channel-type=online " /* NDS client */
    "! adder " /* throws away timestamps */
    "! audioconvert " /* convert to floating point */
    "! audiochebband lower-frequency=40 upper-frequency=1000 " /* band-pass filter */
    "! audioresample " /* downsample (or upsample) */
    "! lal_nofakedisconts " /* fix fake discontinuity flags (it's a long story) */
    "! audioconvert " /* convert back to integer */
    "! rtpL16pay name=pay0 pt=97 )" /* payload raw audio */
   );

  gst_rtsp_media_factory_set_shared (GST_RTSP_MEDIA_FACTORY (factory), TRUE);

  /* attach the test factory to the maping */
  mapping->factory = GST_RTSP_MEDIA_FACTORY (factory);

  /* don't need the ref to the mapper anymore */
  g_object_unref (mapping);

  /* attach the server to the default maincontext */
  gst_rtsp_server_attach (server, NULL);

  /* start serving */
  g_main_loop_run (loop);

  return 0;
}

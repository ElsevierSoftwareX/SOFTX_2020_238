
#ifndef __POSTCOH_FILESINK_H__
#define __POSTCOH_FILESINK_H__

#include <stdio.h>

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

#include <LIGOLw_xmllib/LIGOLwHeader.h>

G_BEGIN_DECLS

#define POSTCOH_TYPE_FILESINK \
  (postcoh_filesink_get_type())
#define POSTCOH_FILESINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),POSTCOH_TYPE_FILESINK,PostcohFilesink))
#define POSTCOH_FILESINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),POSTCOH_TYPE_FILESINK,PostcohFilesinkClass))
#define GST_IS_POSTCOH_FILESINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),POSTCOH_TYPE_FILESINK))
#define GST_IS_POSTCOH_FILESINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),POSTCOH_TYPE_FILESINK))

typedef struct _PostcohFilesink PostcohFilesink;
typedef struct _PostcohFilesinkClass PostcohFilesinkClass;

/**
 * PostcohFilesink:
 *
 * Opaque #PostcohFilesink structure.
 */
struct _PostcohFilesink {
  GstBaseSink parent;

  /*< private >*/
  gchar *filename;
  gchar *uri;
  FILE *file;

  gint compress;
  xmlTextWriterPtr writer;
  XmlTable *xtable;
  
};

struct _PostcohFilesinkClass {
  GstBaseSinkClass parent_class;
};

G_GNUC_INTERNAL GType postcoh_filesink_get_type (void);

G_END_DECLS

#endif

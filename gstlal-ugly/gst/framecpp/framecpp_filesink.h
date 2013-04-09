#ifndef __FRAMECPP_FILESINK_H__
#define __FRAMECPP_FILESINK_H__

#include <glib.h>
#include <gst/gst.h>

G_BEGIN_DECLS

#define FRAMECPP_FILESINK_TYPE \
        (framecpp_filesink_get_type())
#define FRAMECPP_FILESINK(obj) \
        (G_TYPE_CHECK_INSTANCE_CAST((obj), FRAMECPP_FILESINK_TYPE, FRAMECPPFilesink))
#define FRAMECPP_FILESINK_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_CAST((klass), FRAMECPP_FILESINK_TYPE, FRAMECPPFilesinkClass))               
#define GST_IS_FRAMECPP_FILESINK_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_TYPE((klass), FRAMECPP_FILESINK_TYPE))

typedef struct {
        GstBinClass parent_class;
} FRAMECPPFilesinkClass;

typedef struct {
        GstBin element;
        gchar *frame_type;
} FRAMECPPFilesink;

GType framecpp_filesink_get_type(void);

G_END_DECLS

#endif /* __FRAMECPP_FILESINK_H__ */

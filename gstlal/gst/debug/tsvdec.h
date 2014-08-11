#ifndef __GSTLAL_TSVDEC_H__
#define __GSTLAL_TSVDEC_H__


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


#define GSTLAL_TSVDEC_TYPE \
	(gstlal_tsvdec_get_type())
#define GSTLAL_TSVDEC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TSVDEC_TYPE, GSTLALTSVDec))
#define GSTLAL_TSVDEC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TSVDEC_TYPE, GSTLALTSVDecClass))
#define GST_IS_GSTLAL_TSVDEC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TSVDEC_TYPE))
#define GST_IS_GSTLAL_TSVDEC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TSVDEC_TYPE))


typedef struct _GSTLALTSVDec GSTLALTSVDec;
typedef struct _GSTLALTSVDecClass GSTLALTSVDecClass;

struct _GSTLALTSVDec {
	GstElement element;
	GstPad *sinkpad;
	GstPad *srcpad;
	char *data;
	guint size;
	guint rate;
	guint channels;
	guint64 offset;
	GstClockTime t0;
	char *FS;
	char *RS;
	/* FILE *fp; */
};

struct _GSTLALTSVDecClass {
	GstElementClass parent_class;
};

GType gstlal_tsvdec_get_type(void);

G_END_DECLS

#endif	/* __GSTLAL_TSVDEC_H__ */

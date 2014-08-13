#ifndef __GST_LAL_SEGMENTSRC_H__
#define __GST_LAL_SEGMENTSRC_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>
#include <gstlal/gstlal_segments.h>

G_BEGIN_DECLS

#define GSTLAL_SEGMENTSRC_TYPE \
    (gstlal_segmentsrc_get_type())
#define GSTLAL_SEGMENTSRC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_SEGMENTSRC_TYPE, GSTLALSegmentSrc))
#define GSTLAL_SEGMENTSRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_SEGMENTSRC_TYPE, GSTLALSegmentSrcClass))
#define GST_IS_GSTLAL_SEGMENTSRC(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_SEGMENTSRC_TYPE))
#define GST_IS_GSTLAL_SEGMENTSRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_SEGMENTSRC_TYPE))

typedef struct _GSTLALSegmentSrc GSTLALSegmentSrc;
typedef struct _GSTLALSegmentSrcClass GSTLALSegmentSrcClass;

/**
 * GSTLALSegmentSrc:
 */

struct _GSTLALSegmentSrc {
    GstBaseSrc			element;

    GMutex			*segment_matrix_lock;
    struct gstlal_segment_list	*seglist;
    gboolean			invert_output;
    gint			rate;
};

/**
 * GSTLALSegmentSrcClass:
 * @parent_class:  the parent class
 */

struct _GSTLALSegmentSrcClass {
    GstBaseSrcClass     parent_class;
};

GType gstlal_segmentsrc_get_type(void);

G_END_DECLS

#endif /* __GST_LAL_SEGMENTSRC_H__ */

#ifndef __GSTLAL_SNGLINSPIRAL_SPEARMAN_H__
#define __GSTLAL_SNGLINSPIRAL_SPEARMAN_H__

#include <glib.h>
#include <glib-object.h>
#include <gst/gst.h>
#include <gstlal/gstlal_peakfinder.h>
#include <complex.h>
#include <string.h>
#include <math.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLInspiralRead.h>
#include <lal/LALStdlib.h>

G_BEGIN_DECLS

GstBuffer *gstlal_snglinspiral_new_buffer_from_peak_spearman(struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, void *pval);

G_END_DECLS
#endif	/* __GSTLAL_SNGLINSPIRAL_SPEARMAN_H__ */


#ifndef __GSTLAL_SNGLBURST_H__
#define __GSTLAL_SNGLBURST_H__

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


int gstlal_snglburst_array_from_file(char *bank_filename, SnglBurst **bankarray);
int gstlal_set_channel_in_snglburst_array(SnglBurst *bankarray, int length, char *channel);
int gstlal_set_instrument_in_snglburst_array(SnglBurst *bankarray, int length, char *instrument);

GstBuffer *gstlal_snglburst_new_buffer_from_peak(struct gstlal_double_complex_peak_samples_and_values *input, SnglBurst *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, guint64 *count);
GstBuffer *gstlal_snglburst_new_double_buffer_from_peak(struct gstlal_double_peak_samples_and_values *input, SnglBurst *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, guint64 *count);

G_END_DECLS
#endif	/* __GSTLAL_SNGLBURST_H__ */


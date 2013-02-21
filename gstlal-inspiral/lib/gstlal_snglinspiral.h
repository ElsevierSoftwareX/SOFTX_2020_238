#ifndef __GSTLAL_SNGLINSPIRAL_H__
#define __GSTLAL_SNGLINSPIRAL_H__

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

double gstlal_eta(double m1, double m2);
double gstlal_mchirp(double m1, double m2);
double gstlal_effective_distance(double snr, double sigmasq);

int gstlal_snglinspiral_array_from_file(char *bank_filename, SnglInspiralTable **bankarray);
int gstlal_set_channel_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, char *channel);
int gstlal_set_instrument_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, char *instrument);
int gstlal_set_sigmasq_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, double *sigmasq);

GstBuffer *gstlal_snglinspiral_new_buffer_from_peak(struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2);

G_END_DECLS
#endif	/* __GSTLAL_SNGLINSPIRAL_H__ */


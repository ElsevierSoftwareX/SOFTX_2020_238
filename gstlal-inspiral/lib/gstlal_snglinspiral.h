/*
 * Copyright (C) 2012,2013  Chad Hanna
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#ifndef __GSTLAL_SNGLINSPIRAL_H__
#define __GSTLAL_SNGLINSPIRAL_H__


#include <glib.h>
#include <gst/gst.h>
#include <lal/LIGOMetadataTables.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_matrix_complex_float.h>
#include <gstlal/gstlal_peakfinder.h>


G_BEGIN_DECLS

double gstlal_eta(double m1, double m2);
double gstlal_mchirp(double m1, double m2);
double gstlal_effective_distance(double snr, double sigmasq);

int gstlal_snglinspiral_array_from_file(const char *bank_filename, SnglInspiralTable **bankarray);
int gstlal_set_channel_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, char *channel);
int gstlal_set_instrument_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, char *instrument);
int gstlal_set_sigmasq_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, double *sigmasq);
int gstlal_set_min_offset_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, GstClockTimeDiff *difftime);

void gstlal_snglinspiral_array_free(SnglInspiralTable *bankarray);

/*
 * FIXME: only support single precision SNR snippets at the moment
 */
GstBuffer *gstlal_snglinspiral_new_buffer_from_peak(struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *H1_snr_matrix_view, gsl_matrix_complex_float_view *K1_snr_matrix_view, gsl_matrix_complex_float_view *L1_snr_matrix_view, gsl_matrix_complex_float_view *V1_snr_matrix_view, GstClockTimeDiff);
int gstlal_snglinspiral_append_peak_to_buffer(GstBuffer *srcbuf, struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *H1_snr_matrix_view, gsl_matrix_complex_float_view *K1_snr_matrix_view, gsl_matrix_complex_float_view *L1_snr_matrix_view, gsl_matrix_complex_float_view *V1_snr_matrix_view);


G_END_DECLS
#endif	/* __GSTLAL_SNGLINSPIRAL_H__ */


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


#ifndef __GSTLAL_SNGLTRIGGER_H__
#define __GSTLAL_SNGLTRIGGER_H__


#include <glib.h>
#include <gst/gst.h>
#include <lal/LIGOMetadataTables.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_matrix_complex_float.h>
#include <gstlal/gstlal_peakfinder.h>


#define LIGOMETA_IFO_MAX 8
#define LIGOMETA_SEARCH_MAX 25
#define LIGOMETA_CHANNEL_MAX 65

G_BEGIN_DECLS

typedef struct
tagSnglTriggerTable
{
	struct tagSnglTriggerTable *next;
	CHAR ifo[LIGOMETA_IFO_MAX];
	CHAR channel[LIGOMETA_CHANNEL_MAX];
	guint channel_index;
	LIGOTimeGPS end;
	REAL4 phase;
	REAL4 snr;
	REAL4 chisq;
	REAL8 sigmasq;
}
SnglTriggerTable;


/*
 * FIXME: only support single precision SNR snippets at the moment
 */
GstBuffer *gstlal_sngltrigger_new_buffer_from_peak(struct gstlal_peak_state *input, char *channel_name, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *snr_matrix_view, GstClockTimeDiff, gboolean max_snr);

void add_buffer_from_channel(struct gstlal_peak_state *input, char *channel_name, GstPad *pad, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *snr_matrix_view, int channel, GstBuffer *srcbuf);

G_END_DECLS
#endif	/* __GSTLAL_SNGLTRIGGER_H__ */


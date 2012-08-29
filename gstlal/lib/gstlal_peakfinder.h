#ifndef __GSTLAL_PEAKFINDER_H__
#define __GSTLAL_PEAKFINDER_H__


#include <glib.h>
#include <glib-object.h>
#include <complex.h>

G_BEGIN_DECLS

typedef enum tag_gstlal_peak_type_specifier {
	GSTLAL_PEAK_FLOAT,
	GSTLAL_PEAK_DOUBLE,
	GSTLAL_PEAK_COMPLEX,
	GSTLAL_PEAK_DOUBLE_COMPLEX,
	GSTLAL_PEAK_TYPE_COUNT
} gstlal_peak_type_specifier;

struct gstlal_peak_state {
	guint channels;
	guint num_events;
	guint *samples;
	/* should sync with gstlal_peak_type */
	union	{
		float * as_float;
		double * as_double;
		float complex * as_float_complex;
		double complex * as_double_complex;
		} values;
	gstlal_peak_type_specifier type;
	guint unit;
	guint pad;
	double thresh;
};


/*
 * Type agnostic declarations
 */


/* make new structures */
struct gstlal_peak_state *gstlal_peak_state_new(guint channels, gstlal_peak_type_specifier type);

/* free structures */
int gstlal_peak_state_free(struct gstlal_peak_state *state);

/* clear contents of structures */
int gstlal_peak_state_clear(struct gstlal_peak_state *state);

/* Convenience functions */
GstBuffer *gstlal_new_buffer_from_peak(struct gstlal_peak_state *input, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate);

int gstlal_peak_over_window(struct gstlal_peak_state *state, const void *data, guint64 length);

int gstlal_series_around_peak(struct gstlal_peak_state *state, void *data, void *outputmat, guint n);


/*
 * Type specific declarations
 */


/* float */

#define TYPE_STRING float
#define TYPE float
#include "gstlal_peakfinder.ht"
#undef TYPE
#undef TYPE_STRING

/* double */

#define TYPE_STRING double
#define TYPE double
#include "gstlal_peakfinder.ht"
#undef TYPE
#undef TYPE_STRING

/* complex */

#define TYPE_STRING float_complex
#define TYPE float complex
#include "gstlal_peakfinder.ht"
#undef TYPE
#undef TYPE_STRING

/* double complex */

#define TYPE_STRING double_complex
#define TYPE double complex
#include "gstlal_peakfinder.ht"
#undef TYPE
#undef TYPE_STRING

G_END_DECLS
#endif	/* __GSTLAL_PEAKFINDER_H__ */

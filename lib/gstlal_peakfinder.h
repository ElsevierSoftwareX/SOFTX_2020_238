#ifndef __GSTLAL_PEAKFINDER_H__
#define __GSTLAL_PEAKFINDER_H__


#include <glib.h>
#include <glib-object.h>
#include <complex.h>

G_BEGIN_DECLS

struct gstlal_double_peak_samples_and_values {
	guint channels;
	guint num_events;
	guint *samples;
	double *values;
};

struct gstlal_double_complex_peak_samples_and_values {
	guint channels;
	guint num_events;
	guint *samples;
	double complex *values;
};

/* make new structures */
struct gstlal_double_peak_samples_and_values *gstlal_double_peak_samples_and_values_new(guint channels);
struct gstlal_double_complex_peak_samples_and_values *gstlal_double_complex_peak_samples_and_values_new(guint channels);

/* clear contents of structures */
int gstlal_double_peak_samples_and_values_clear(struct gstlal_double_peak_samples_and_values *);
int gstlal_double_complex_peak_samples_and_values_clear(struct gstlal_double_complex_peak_samples_and_values *);

/* find a peak over a fixed window */
int gstlal_double_peak_over_window(struct gstlal_double_peak_samples_and_values *output, const double *data, guint64 length);
int gstlal_double_fill_output_with_peak(struct gstlal_double_peak_samples_and_values *input, double *data, guint64 lenght);
GstBuffer *gstlal_double_new_buffer_from_peak(struct gstlal_double_peak_samples_and_values *input, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate);

/* clear an output buffer and populate with just the peaks */
int gstlal_double_complex_peak_over_window(struct gstlal_double_complex_peak_samples_and_values *output, const double *data, guint64 length);
int gstlal_double_complex_fill_output_with_peak(struct gstlal_double_complex_peak_samples_and_values *input, double complex *data, guint64 length);

G_END_DECLS
#endif	/* __GSTLAL_PEAKFINDER_H__ */

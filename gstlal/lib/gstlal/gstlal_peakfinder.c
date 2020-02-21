#include <glib.h>
#include <glib-object.h>
#include <gst/gst.h>
#include <gstlal_peakfinder.h>
#include <complex.h>
#include <string.h>
#include <math.h>
#include <lal/TriggerInterpolation.h>

/*
 * Type agnostic functions
 */


/* Init a structure to hold peak state */
struct gstlal_peak_state *gstlal_peak_state_new(guint channels, gstlal_peak_type_specifier type)
{
	struct gstlal_peak_state *new = g_new0(struct gstlal_peak_state, 1);
	if (!new) return NULL;
	new->channels = channels;
	new->samples = g_malloc0(sizeof(guint) * channels);
	new->interpsamples = g_malloc0(sizeof(double) * channels);
	new->num_events = 0;
	new->pad = GSTLAL_PEAK_INTERP_LENGTH;
	new->thresh = 0;
	new->type = type;
	new->is_gap = FALSE;
	new->no_peaks_past_threshold = TRUE;

	switch (new->type)
	{
		case GSTLAL_PEAK_FLOAT:
		new->unit = sizeof(float);
		new->values.as_float = (float *) g_malloc0(new->unit * channels);
		new->interpvalues.as_float = (float *) g_malloc0(new->unit * channels);
		break;
		
		case GSTLAL_PEAK_DOUBLE:
		new->unit = sizeof(double);
		new->values.as_double = (double *) g_malloc0(new->unit * channels);
		new->interpvalues.as_double = (double *) g_malloc0(new->unit * channels);
		break;
		
		case GSTLAL_PEAK_COMPLEX:
		new->unit = sizeof(float complex);
		new->values.as_float_complex = (float complex *) g_malloc0(new->unit * channels);
		new->interpvalues.as_float_complex = (float complex *) g_malloc0(new->unit * channels);
		break;

		case GSTLAL_PEAK_DOUBLE_COMPLEX:
		new->unit = sizeof(double complex);
		new->values.as_double_complex = (double complex *) g_malloc0(new->unit * channels);
		new->interpvalues.as_double_complex = (double complex *) g_malloc0(new->unit * channels);
		break;

		default:
		g_assert(new->type < GSTLAL_PEAK_TYPE_COUNT);
		return NULL;
	}

	/* Interpolator */
	/* FIXME perhaps expose this property ? */
	new->interp = XLALCreateLanczosTriggerInterpolant(GSTLAL_PEAK_INTERP_LENGTH);
	
	return new;
}

/* Free a structure to hold peak state */
int gstlal_peak_state_free(struct gstlal_peak_state *val)
{
	g_free(val->samples);
	g_free(val->interpsamples);
	g_free(val->values.as_float);
	g_free(val->interpvalues.as_float);
	XLALDestroyLanczosTriggerInterpolant(val->interp);
	return 0;
}

/* Clear a structure to hold peak state */
int gstlal_peak_state_clear(struct gstlal_peak_state *val)
{
	memset(val->samples, 0.0, val->channels * sizeof(guint));
	memset(val->interpsamples, 0.0, val->channels * sizeof(double));
	memset(val->values.as_float, 0.0, val->channels * val->unit);
	memset(val->interpvalues.as_float, 0.0, val->channels * val->unit);
	val->num_events = 0;
	val->is_gap = FALSE;
	// dont reset the value of no peaks past threshold, user is responsibile for that
	// FIXME This will be removed eventually, see itacac for more info
	//val->no_peaks_past_threshold = TRUE;
	return 0;
}

/* A convenience function to make a new buffer and populate it with peaks */
GstBuffer *gstlal_new_buffer_from_peak(struct gstlal_peak_state *state, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate)
{
	/* FIXME check errors */
	
	/* Size is length in samples times number of channels times number of
	 * bytes per sample
	 */

	gint size = state->unit * length * state->channels;
	GstBuffer *srcbuf = gst_buffer_new_allocate(NULL, size, NULL);
	GstMapInfo mapinfo;

	/* FIXME someday with better gap support don't actually allocate data
	 * in this case.  For now we just mark it as a gap but let the rest of
	 * this function do its thing so that we get a buffer allocated with
	 * zeros 
	 */

	if (state->is_gap)
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	/* set the offset */
        GST_BUFFER_OFFSET(srcbuf) = offset;
        GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

        /* set the time stamps */
        GST_BUFFER_PTS(srcbuf) = time;
        GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);

	gst_buffer_map(srcbuf, &mapinfo, GST_MAP_WRITE);
	switch (state->type)
	{
		case GSTLAL_PEAK_FLOAT:
		gstlal_float_fill_output_with_peak(state, (float *) mapinfo.data, length);
		break;
		
		case GSTLAL_PEAK_DOUBLE:
		gstlal_double_fill_output_with_peak(state, (double *) mapinfo.data, length);
		break;
		
		case GSTLAL_PEAK_COMPLEX:
		gstlal_float_complex_fill_output_with_peak(state, (float complex *) mapinfo.data, length);
		break;

		case GSTLAL_PEAK_DOUBLE_COMPLEX:
		gstlal_double_complex_fill_output_with_peak(state, (double complex *) mapinfo.data, length);
		break;

		default:
		g_assert(state->type < GSTLAL_PEAK_TYPE_COUNT);
	}
	gst_buffer_unmap(srcbuf, &mapinfo);

	return srcbuf;
}

/* A convenience function to find the peak over a window based on the type specified by state */
int gstlal_peak_over_window(struct gstlal_peak_state *state, const void *data, guint64 length)
{	
	switch (state->type)
	{
		case GSTLAL_PEAK_FLOAT:
		return gstlal_float_peak_over_window(state, (float *) data, length);
		break;
		
		case GSTLAL_PEAK_DOUBLE:
		return gstlal_double_peak_over_window(state, (double *) data, length);
		break;
		
		case GSTLAL_PEAK_COMPLEX:
		return gstlal_float_complex_peak_over_window(state, (float complex *) data, length);
		break;

		case GSTLAL_PEAK_DOUBLE_COMPLEX:
		return gstlal_double_complex_peak_over_window(state, (double complex *) data, length);
		break;

		default:
		g_assert(state->type < GSTLAL_PEAK_TYPE_COUNT);
	}

	return 1;
}

/* A convenience function to find the series around a peak based on the type specified by state */
int gstlal_series_around_peak(struct gstlal_peak_state *state, void *data, void *outputmat, guint n)
{	
	switch (state->type)
	{
		case GSTLAL_PEAK_FLOAT:
		return gstlal_float_series_around_peak(state, (float *) data, (float *) outputmat, n);
		break;
		
		case GSTLAL_PEAK_DOUBLE:
		return gstlal_double_series_around_peak(state, (double *) data, (double *) outputmat, n);
		break;
		
		case GSTLAL_PEAK_COMPLEX:
		return gstlal_float_complex_series_around_peak(state, (float complex *) data, (float complex *) outputmat, n);
		break;

		case GSTLAL_PEAK_DOUBLE_COMPLEX:
		return gstlal_double_complex_series_around_peak(state, (double complex *) data, (double complex *) outputmat, n);
		break;

		default:
		g_assert(state->type < GSTLAL_PEAK_TYPE_COUNT);
	}

	return 1;
}

int gstlal_peak_max_over_channels(struct gstlal_peak_state *state)
{
	int i, out = -1;
	switch (state->type)
	{
		case GSTLAL_PEAK_FLOAT:
		{
			float max_val = 0;
			float current_val = 0;
			/* Type casting unsigned int (guint) to int */
			for(i = 0; i < (int)state->channels; i++)
			{
				current_val = fabsf(state->values.as_float[i]);
				if(current_val > max_val)
				{
					max_val = current_val;
					out = i;
				}
			}
			break;
		}

		case GSTLAL_PEAK_DOUBLE:
		{
			double max_val = 0;
			double current_val = 0;
			/* Type casting unsigned int (guint) to int */
			for(i = 0; i < (int)state->channels; i++)
			{
				current_val = fabs(state->values.as_double[i]);
				if(current_val > max_val)
				{
					max_val = current_val;
					out = i;
				}
			}
			break;
		}

		case GSTLAL_PEAK_COMPLEX:
		{
			float max_val = 0;
			float current_val = 0;
			/* Type casting unsigned int (guint) to int */
			for(i = 0; i < (int)state->channels; i++)
			{
				current_val = cabsf(state->values.as_float_complex[i]);
				if(current_val > max_val)
				{
					max_val = current_val;
					out = i;
				}
			}
			break;
		}

		case GSTLAL_PEAK_DOUBLE_COMPLEX:
		{
			double max_val = 0;
			double current_val = 0;
			/* Type casting unsigned int (guint) to int */
			for(i = 0; i < (int)state->channels; i++)
			{
				current_val = cabs(state->values.as_double_complex[i]);
				if(current_val > max_val)
				{
					max_val = current_val;
					out = i;
				}
			}
			break;
		}

		default:
		g_assert(state->type < GSTLAL_PEAK_TYPE_COUNT);

	}

	return out;
}

/*
 * Type specific functions
 */

/* float */
#define ABSFUNC(x) x * x
#define TYPE_STRING float
#define XLAL_TYPE_STRING REAL4
#define TYPE float
#include "gstlal_peakfinder.ct"
#undef TYPE
#undef TYPE_STRING
#undef XLAL_TYPE_STRING
#undef ABSFUNC

/* double */
#define ABSFUNC(x) x * x
#define TYPE_STRING double
#define XLAL_TYPE_STRING REAL8
#define TYPE double
#include "gstlal_peakfinder.ct"
#undef TYPE
#undef TYPE_STRING
#undef XLAL_TYPE_STRING
#undef ABSFUNC

/* float complex */
#define ABSFUNC(x) ((float *) &x)[0] * ((float *) &x)[0] + ((float *) &x)[1] * ((float *) &x)[1]
#define TYPE_STRING float_complex
#define XLAL_TYPE_STRING COMPLEX8
#define TYPE float complex
#include "gstlal_peakfinder.ct"
#undef TYPE
#undef TYPE_STRING
#undef XLAL_TYPE_STRING
#undef ABSFUNC

/* double complex */
#define ABSFUNC(x) ((double *) &x)[0] * ((double *) &x)[0] + ((double *) &x)[1] * ((double *) &x)[1]
#define TYPE_STRING double_complex
#define XLAL_TYPE_STRING COMPLEX16
#define TYPE double complex
#include "gstlal_peakfinder.ct"
#undef TYPE
#undef TYPE_STRING
#undef XLAL_TYPE_STRING
#undef ABSFUNC


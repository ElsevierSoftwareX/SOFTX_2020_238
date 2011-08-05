/*
 * Copyright (C) 2009 Kipp Cannon <kipp.cannon@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from the C library
 */


#include <complex.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal.h>
#include <gstlal_plugins.h>
#include <gstlal_firbank.h>


/*
 * stuff from FFTW and GSL
 */


#include <fftw3.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * return the number of FIR vectors
 */


static unsigned fir_channels(const GSTLALFIRBank *element)
{
	return element->fir_matrix->size1;
}


/*
 * return the number of samples in the FIR vectors
 */


static unsigned fir_length(const GSTLALFIRBank *element)
{
	return element->fir_matrix->size2;
}


/*
 * return the number of time-domain samples in each FFT.  inverting this,
 *
 * block_stride = block_length - fir_length + 1
 *
 * the +1 is because when there is one complete filter-length of data one
 * can compute 1 sample of output.  we round the result up to an even
 * integer to simplify the question of how many samples of frequency-domain
 * data there will be.
 */


static unsigned fft_block_length(const GSTLALFIRBank *element)
{
	unsigned block_length = element->block_stride + fir_length(element) - 1;
	return block_length & 1 ? block_length + 1 : block_length;
}


/*
 * the number of samples from the start of one FFT block to the next, also
 * the number of samples produced from each FFT block.  note that we don't
 * use the element property because the corresponding block length might
 * get rounded up 1 sample.
 */


static unsigned fft_block_stride(const GSTLALFIRBank *element)
{
	return fft_block_length(element) - fir_length(element) + 1;
}


/*
 * number of samples that must be available to compute at least n output
 * samples
 */


static guint64 minimum_input_length(const GSTLALFIRBank *element, guint64 n)
{
	if(element->time_domain)
		return n + fir_length(element) - 1;
	else {
		guint64 stride = fft_block_stride(element);
		guint64 blocks = (n + stride - 1) / stride;
		return (blocks - 1) * stride + fft_block_length(element);
	}
}


/*
 * construct a buffer of zeros and push into adapter
 */


static int push_zeros(GSTLALFIRBank *element, unsigned samples)
{
	GstBuffer *zerobuf;

	if(samples) {
		zerobuf = gst_buffer_new_and_alloc(samples * sizeof(double));
		if(!zerobuf) {
			GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
			return -1;
		}
		memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
		gst_adapter_push(element->adapter, zerobuf);
		element->zeros_in_adapter += samples;
	}

	return 0;
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALFIRBank *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	GST_BUFFER_SIZE(buf) = outsamples * fir_channels(element) * sizeof(double);
	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
	if(element->need_discont) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
}


/*
 * the number of samples available in the adapter
 */


static guint64 get_available_samples(GSTLALFIRBank *element)
{
	return gst_adapter_available(element->adapter) / sizeof(double);
}


static guint64 get_available_nonzero_samples(GSTLALFIRBank *element)
{
	guint64 available_samples = get_available_samples(element);
	return available_samples > element->zeros_in_adapter ? available_samples - element->zeros_in_adapter : 0;
}


/*
 * create/free FFT workspace.  call with fir_matrix_lock held
 */


static int create_fft_workspace(GSTLALFIRBank *element)
{
	unsigned i;
	unsigned length_fd = fft_block_length(element) / 2 + 1;

	/*
	 * frequency-domain input
	 */

	g_mutex_lock(gstlal_fftw_lock);

	GST_DEBUG_OBJECT(element, "starting FFTW planning");
	element->input_fd = (complex double *) fftw_malloc(length_fd * sizeof(*element->input_fd));
	element->in_plan = fftw_plan_dft_r2c_1d(fft_block_length(element), (double *) element->input_fd, element->input_fd, FFTW_MEASURE);

	/*
	 * frequency-domain workspace
	 */

	element->workspace_fd = (complex double *) fftw_malloc(length_fd * sizeof(*element->workspace_fd));
	element->out_plan = fftw_plan_dft_c2r_1d(fft_block_length(element), element->workspace_fd, (double *) element->workspace_fd, FFTW_MEASURE);
	GST_DEBUG_OBJECT(element, "FFTW planning complete");

	/*
	 * loop over filters.  copy each time-domain filter to input_fd,
	 * zero-pad, transform to frequency domain, and save in
	 * fir_matrix_fd.  the frequency-domain filters are pre-scaled by
	 * 1/n and conjugated to save those operations inside the filtering
	 * loop.
	 */

	element->fir_matrix_fd = (complex double *) fftw_malloc(fir_channels(element) * length_fd * sizeof(*element->fir_matrix_fd));

	g_mutex_unlock(gstlal_fftw_lock);

	for(i = 0; i < fir_channels(element); i++) {
		unsigned j;
		memset(element->input_fd, 0, length_fd * sizeof(*element->input_fd));
		for(j = 0; j < fir_length(element); j++)
			((double *) element->input_fd)[j] = gsl_matrix_get(element->fir_matrix, i, j) / fft_block_length(element);
		fftw_execute(element->in_plan);
		for(j = 0; j < length_fd; j++)
			element->fir_matrix_fd[i * length_fd + j] = conj(element->input_fd[j]);
	}

	/*
	 * done
	 */

	return 0;
}


static void free_fft_workspace(GSTLALFIRBank *element)
{
	g_mutex_lock(gstlal_fftw_lock);

	fftw_free(element->fir_matrix_fd);
	element->fir_matrix_fd = NULL;
	fftw_free(element->input_fd);
	element->input_fd = NULL;
	fftw_destroy_plan(element->in_plan);
	element->in_plan = NULL;
	fftw_free(element->workspace_fd);
	element->workspace_fd = NULL;
	fftw_destroy_plan(element->out_plan);
	element->out_plan = NULL;

	g_mutex_unlock(gstlal_fftw_lock);
}


/*
 * transform input samples to output samples using a time-domain algorithm
 */


static unsigned td_output_length(const GSTLALFIRBank *element, unsigned available_length)
{
	/*
	 * how many samples can we construct from the contents of the
	 * adapter?  the +1 is because when there is 1 FIR-length of data
	 * in the adapter then we can produce 1 output sample, not 0.
	 */

	if(available_length < fir_length(element))
		return 0;
	return available_length - fir_length(element) + 1;
}


static unsigned tdfilter(GSTLALFIRBank *element, GstBuffer *outbuf, unsigned available_length)
{
	unsigned i;
	unsigned output_length;
	gsl_vector_view input;
	gsl_matrix_view output;

	/*
	 * how many samples can we construct from the contents of the
	 * adapter?
	 */

	output_length = td_output_length(element, available_length);
	if(!output_length)
		goto done;

	g_assert(GST_BUFFER_SIZE(outbuf) >= output_length * fir_channels(element) * sizeof(double));

	/*
	 * wrap the adapter's contents in a GSL vector view.  note that the
	 * wrapper vector's length is set to the fir_length length, not the
	 * length that has been peeked at, so that inner products work
	 * properly.
	 */

	input = gsl_vector_view_array((double *) gst_adapter_peek(element->adapter, available_length * sizeof(double)), fir_length(element));

	/*
	 * wrap output buffer in a GSL matrix view.
	 */

	output = gsl_matrix_view_array((double *) GST_BUFFER_DATA(outbuf), output_length, fir_channels(element));

	/*
	 * assemble the output sample time series as the columns of a
	 * matrix.
	 */

	for(i = 0; i < output_length; i++) {
		/*
		 * the current row (sample) in the output matrix
		 */

		gsl_vector_view output_sample = gsl_matrix_row(&(output.matrix), i);

		/*
		 * compute one vector of output samples --- the projection
		 * of the input onto each of the FIR filters
		 */

		gsl_blas_dgemv(CblasNoTrans, 1.0, element->fir_matrix, &(input.vector), 0.0, &(output_sample.vector));

		/*
		 * advance the input pointer
		 */

		input.vector.data++;
	}

	/*
	 * done
	 */

done:
	return output_length;
}


/*
 * transform input samples to output samples using a frequency-domain
 * algorithm
 */


static unsigned fdfilter(GSTLALFIRBank *element, GstBuffer *outbuf, unsigned available_length)
{
	unsigned stride;
	unsigned blocks;
	unsigned input_length;
	unsigned output_length;
	unsigned filter_length_fd;
	double *input;
	double *output_end;
	gsl_matrix_view output;
	gsl_vector_view workspace;

	/*
	 * how many FFT blocks can we construct from the contents of the
	 * adapter?
	 */

	if(available_length < fft_block_length(element))
		return 0;
	stride = fft_block_stride(element);
	blocks = (available_length - fft_block_length(element)) / stride + 1;
	output_length = blocks * stride;
	input_length = output_length + fir_length(element) - 1;
	filter_length_fd = fft_block_length(element) / 2 + 1;

	output_length = MIN(output_length, GST_BUFFER_SIZE(outbuf) / (fir_channels(element) * sizeof(double)));

	/*
	 * retrieve input samples
	 */

	input = (double *) gst_adapter_peek(element->adapter, input_length * sizeof(double));

	/*
	 * wrap workspace (as real numbers) in a GSL vector view.  note
	 * that vector has length fft_block_stride to affect the requisite
	 * overlap
	 */

	workspace = gsl_vector_view_array((double *) element->workspace_fd, stride);

	/*
	 * wrap first block of output buffer in a GSL matrix view
	 */

	output = gsl_matrix_view_array(((double *) GST_BUFFER_DATA(outbuf)), stride, fir_channels(element));
	output_end = output.matrix.data + output_length * fir_channels(element);

	/*
	 * loop over FFT blocks
	 */

	while(output.matrix.data < output_end) {
		complex double *filter;
		unsigned j;

		/*
		 * clip the output to the output buffer's length
		 */

		if(output_end - output.matrix.data < stride * fir_channels(element))
			workspace.vector.size = output.matrix.size1 = (output_end - output.matrix.data) / fir_channels(element);

		/*
		 * copy a block-length of data to input workspace and
		 * transform to frequency-domain
		 */

		memcpy(element->input_fd, input, fft_block_length(element) * sizeof(*input));
		fftw_execute(element->in_plan);

		/*
		 * loop over filters
		 */

		filter = element->fir_matrix_fd;
		for(j = 0; j < fir_channels(element); j++) {
			/*
			 * multiply input by filter, transform to
			 * time-domain, copy to output.  note that the
			 * workspace view and the output matrix view are
			 * only stride samples long, thus the end of the
			 * real workspace is not copied.  the
			 * frequency-domain filters are constructed so that
			 * the wrap-around transient lives in that part of
			 * the work space in the time domain.
			 */

			complex double *last_input_fd, *input_fd = element->input_fd;
			complex double *workspace_fd = element->workspace_fd;
			for(last_input_fd = input_fd + filter_length_fd; input_fd < last_input_fd; )
				*(workspace_fd++) = *(input_fd++) * *(filter++);
			fftw_execute(element->out_plan);
			gsl_matrix_set_col(&output.matrix, j, &workspace.vector);
		}

		/*
		 * advance to next block
		 */

		input += stride;
		output.matrix.data += stride * fir_channels(element);
	}

	/*
	 * done
	 */

	return output_length;
}


/*
 * filtering algorithm front-end
 */


static GstFlowReturn filter(GSTLALFIRBank *element, GstBuffer *outbuf)
{
	unsigned available_length;
	unsigned output_length;

	/*
	 * how much data is available?
	 */

	available_length = get_available_samples(element);

	/*
	 * run filtering code
	 */

	if(element->time_domain) {
		/*
		 * use time-domain filter implementation
		 */

		output_length = tdfilter(element, outbuf, available_length);
	} else {
		/*
		 * use frequency-domain filter implementation;  start by
		 * building frequency-domain filters if we don't have them
		 * yet
		 */

		if(!element->fir_matrix_fd)
			create_fft_workspace(element);

		output_length = fdfilter(element, outbuf, available_length);
	}

	/*
	 * output produced?
	 */

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/*
	 * flush the data from the adapter
	 */

	gst_adapter_flush(element->adapter, output_length * sizeof(double));
	if(element->zeros_in_adapter > available_length - output_length)
		/*
		 * some trailing zeros have been flushed from the adapter
		 */

		element->zeros_in_adapter = available_length - output_length;

	/*
	 * set buffer metadata
	 */

	set_metadata(element, outbuf, output_length, FALSE);

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * run filter code on adapter's contents putting result into a newly
 * allocated buffer, and push buffer downstream.
 */


static GstFlowReturn filter_and_push(GSTLALFIRBank *element, guint64 length)
{
	GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(element));
	GstBuffer *buf;
	GstFlowReturn result;

	if(!length)
		return GST_FLOW_OK;
	result = gst_pad_alloc_buffer(srcpad, element->next_out_offset, length * fir_channels(element) * sizeof(double), GST_PAD_CAPS(srcpad), &buf);
	if(result != GST_FLOW_OK)
		return result;
	result = filter(element, buf);
	g_assert(result == GST_FLOW_OK);
	return gst_pad_push(srcpad, buf);
}


/*
 * flush the remaining contents of the adapter.  e.g., at EOS, etc.
 */


static GstFlowReturn flush_history(GSTLALFIRBank *element)
{
	unsigned available_length;
	unsigned padding;
	unsigned output_length;
	unsigned final_gap_length;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * in time-domain mode, there's never enough stuff left in the
	 * adpater, after an invocation of the transform() method, to
	 * produce any more output data.
	 */

	if(element->time_domain)
		goto done;

	/*
	 * in frequency-domain mode, there's never enough stuff left in the
	 * adapter, after an invocation of the transform() method, to
	 * process one full FFT block-length, but there can be enough in
	 * the adapter to produce at least some samples of output.  figure
	 * out how many samples are in the adpater, how many output samples
	 * we can generate, how many zeros need to be pushed into the
	 * adpater to pad it to a full FFT block-length
	 */

	available_length = get_available_samples(element);

	output_length = td_output_length(element, available_length);
	if(output_length <= 0)
		/* not enough, infact, to generate any output */
		goto done;
	final_gap_length = element->zeros_in_adapter >= fir_length(element) ? element->zeros_in_adapter - fir_length(element) + 1 : 0;

	/* sanity checks */
	g_assert(available_length <= fft_block_length(element));
	g_assert(output_length <= fft_block_stride(element));
	g_assert(final_gap_length <= output_length);

	padding = fft_block_length(element) - available_length;

	/*
	 * push enough zeros to pad to an FFT block boundary
	 */

	if(push_zeros(element, padding) < 0) {
		GST_DEBUG_OBJECT(element, "failure padding to FFT block boundary");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * process contents
	 */

	result = filter_and_push(element, output_length - final_gap_length);
	if(result != GST_FLOW_OK)
		goto done;
	if(final_gap_length) {
		GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(element));
		GstBuffer *buf;

		result = gst_pad_alloc_buffer(srcpad, element->next_out_offset, final_gap_length * fir_channels(element) * sizeof(double), GST_PAD_CAPS(srcpad), &buf);
		if(result != GST_FLOW_OK)
			goto done;

		memset(GST_BUFFER_DATA(buf), 0, GST_BUFFER_SIZE(buf));
		set_metadata(element, buf, final_gap_length, TRUE);
		result = gst_pad_push(srcpad, buf);
	}

	/*
	 * done.  discard the adapter's contents regardless of success
	 */

done:
	gst_adapter_clear(element->adapter);
	element->zeros_in_adapter = 0;
	return result;
}


/*
 * constuct a new segment event and push downstream.  must be called with
 * the fir_matrix_lock held
 */


static GstFlowReturn do_new_segment(GSTLALFIRBank *element)
{
	gboolean update;
	gdouble rate;
	GstFormat format;
	gint64 start, stop, position;
	gint64 samples_lost;
	GstFlowReturn result = GST_FLOW_OK;

	if(!element->fir_matrix)
		goto done;
	if(!element->rate)
		goto done;
	if(!element->last_new_segment)
		goto done;

	gst_event_parse_new_segment(element->last_new_segment, &update, &rate, &format, &start, &stop, &position);

	samples_lost = fir_length(element) - 1;

	switch(format) {
	case GST_FORMAT_TIME:
		GST_DEBUG_OBJECT(element, "transforming [%" G_GINT64_FORMAT ", %" G_GINT64_FORMAT "), position = %" G_GINT64_FORMAT", rate = %d, latency = %" G_GINT64_FORMAT "\n", start, stop, position, element->rate, element->latency);
		start = gst_util_uint64_scale_int_round(start, element->rate, GST_SECOND);
		start += samples_lost - element->latency;
		start = gst_util_uint64_scale_int_round(start, GST_SECOND, element->rate);
		if(stop != -1) {
			stop = gst_util_uint64_scale_int_round(stop, element->rate, GST_SECOND);
			stop += -samples_lost - element->latency;
			stop = gst_util_uint64_scale_int_round(stop, GST_SECOND, element->rate);
		}
		position = start;
		GST_DEBUG_OBJECT(element, "to [%" G_GINT64_FORMAT ", %" G_GINT64_FORMAT "), position = %" G_GINT64_FORMAT", rate = %d, latency = %" G_GINT64_FORMAT "\n", start, stop, position, element->rate, element->latency);
		break;

	default:
		g_assert_not_reached();
		break;
	}

	result = gst_pad_push_event(GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(element)), gst_event_new_new_segment(update, rate, format, start, stop, position));

done:
	return result;
}


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


enum gstlal_firbank_signal {
	SIGNAL_RATE_CHANGED,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


static void rate_changed(GstElement *element, gint rate, void *data)
{
	/* FIXME: send updated segment downstream?  because latency now
	 * means something different */
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
	)
);


GST_BOILERPLATE(
	GSTLALFIRBank,
	gstlal_firbank,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_TIME_DOMAIN = 1,
	ARG_BLOCK_STRIDE,
	ARG_FIR_MATRIX,
	ARG_LATENCY
};


#define DEFAULT_TIME_DOMAIN FALSE
#define DEFAULT_BLOCK_STRIDE 1
#define DEFAULT_LATENCY 0


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint channels;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = sizeof(double) * channels;

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad's format is the same as the source pad's except
		 * it must have only 1 channel
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++)
			gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, 1, NULL);
		break;

	case GST_PAD_SINK:
		/*
		 * source pad's format is the same as the sink pad's except
		 * it can have any number of channels or, if the size of
		 * the FIR matrix is known, the number of channels must
		 * equal the number of FIR filters.
		 */

		g_mutex_lock(element->fir_matrix_lock);
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			if(element->fir_matrix)
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, fir_channels(element), NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		g_mutex_unlock(element->fir_matrix_lock);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return caps;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	guint unit_size;
	guint other_unit_size;
	gboolean success = TRUE;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_ERROR_OBJECT(element, "size not a multiple of %u", unit_size);
		return FALSE;
	}
	if(!get_unit_size(trans, othercaps, &other_unit_size))
		return FALSE;

	/*
	 * wait for FIR matrix
	 */

	g_mutex_lock(element->fir_matrix_lock);
	while(!element->fir_matrix) {
		GST_DEBUG_OBJECT(element, "fir matrix not available, waiting ...");
		g_cond_wait(element->fir_matrix_available, element->fir_matrix_lock);
		if(GST_STATE(GST_ELEMENT(trans)) == GST_STATE_NULL) {
			GST_DEBUG_OBJECT(element, "element now in null state, abandoning wait for fir matrix");
			success = FALSE;
			goto done;
		}
	}

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * number of input bytes required to produce an output
		 * buffer of (at least) the requested size
		 */

		*othersize = minimum_input_length(element, size / unit_size);
		if(*othersize <= get_available_samples(element))
			*othersize = 0;
		else
			*othersize = (*othersize - get_available_samples(element)) * other_unit_size;
		break;

	case GST_PAD_SINK:
		/*
		 * number of output bytes to be generated by the receipt of
		 * an input buffer of the given size.
		 */

		*othersize = size / unit_size + get_available_samples(element);
		if(*othersize < minimum_input_length(element, 1))
			*othersize = 0;
		else if(element->time_domain)
			/* + 1 because when we have one FIR-length of
			 * samples we can compute 1 output sample, not 0 */
			*othersize = (*othersize - (guint) fir_length(element) + 1) * other_unit_size;
		else {
			guint64 blocks = (*othersize - fft_block_length(element)) / fft_block_stride(element) + 1;
			*othersize = (blocks * fft_block_stride(element)) * other_unit_size;
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		success = FALSE;
		break;
	}

done:
	g_mutex_unlock(element->fir_matrix_lock);
	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	GstStructure *s;
	gint rate;
	gint channels;
	gboolean success = TRUE;

	s = gst_caps_get_structure(outcaps, 0);
	if(!gst_structure_get_int(s, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, outcaps);
		success = FALSE;
	} else if(!gst_structure_get_int(s, "rate", &rate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
		success = FALSE;
	} else if(element->fir_matrix && (channels != (gint) fir_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, fir_channels(element), outcaps);
		success = FALSE;
	}

	if(success) {
		gint old_rate = element->rate;
		element->rate = rate;
		if(element->rate != old_rate)
			g_signal_emit(G_OBJECT(trans), signals[SIGNAL_RATE_CHANGED], 0, element->rate, NULL);
	}

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	element->adapter = gst_adapter_new();
	element->zeros_in_adapter = 0;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->need_new_segment = TRUE;
	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseTransform *trans)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	g_object_unref(element->adapter);
	element->adapter = NULL;
	return TRUE;
}


/*
 * event()
 */


static gboolean event(GstBaseTransform *trans, GstEvent *event)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT: {
		if(element->last_new_segment)
			gst_event_unref(element->last_new_segment);
		gst_event_ref(event);
		element->last_new_segment = event;
		element->need_new_segment = TRUE;
		return FALSE;
	}

	case GST_EVENT_EOS:
		/*
		 * end-of-stream:  finish processing adapter's contents
		 */

		if(flush_history(element) != GST_FLOW_OK)
			GST_WARNING_OBJECT(element, "unable to process internal history, some data at end of stream has been discarded");
		return TRUE;

	default:
		return TRUE;
	}
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	guint64 in_length;
	guint64 adapter_length;
	guint64 nonzero_samples;
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	GstFlowReturn result;

	/*
	 * wait for FIR matrix
	 */

	g_mutex_lock(element->fir_matrix_lock);
	while(!element->fir_matrix) {
		GST_DEBUG_OBJECT(element, "fir matrix not available, waiting ...");
		g_cond_wait(element->fir_matrix_available, element->fir_matrix_lock);
		if(GST_STATE(GST_ELEMENT(trans)) == GST_STATE_NULL) {
			GST_DEBUG_OBJECT(element, "element now in null state, abandoning wait for fir matrix");
			result = GST_FLOW_WRONG_STATE;
			goto done;
		}
	}

	/*
	 * check for new segment
	 */
	/* FIXME:  we should do this whenever the sample rate, FIR matrix
	 * size or latency changes, but those cases should produce "update"
	 * new segment events */

	if(element->need_new_segment) {
		do_new_segment(element);
		element->need_new_segment = FALSE;
	}

	/*
	 * check for discontinuity
	 */

	if(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0)) {
		/*
		 * clear adapter
		 */

		gst_adapter_clear(element->adapter);
		element->zeros_in_adapter = 0;

		/*
		 * (re)sync timestamp and offset book-keeping
		 */

		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = element->offset0 + fir_length(element) - 1 - element->latency;

		/*
		 * be sure to flag the next output buffer as a discontinuity
		 */

		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * gap logic
	 */

	g_assert(GST_BUFFER_OFFSET_IS_VALID(inbuf));
	g_assert(GST_BUFFER_OFFSET_END_IS_VALID(inbuf));

	in_length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	adapter_length = get_available_samples(element);
	nonzero_samples = get_available_nonzero_samples(element);
	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s.
		 */

		gst_buffer_ref(inbuf);	/* don't let calling code free buffer */
		gst_adapter_push(element->adapter, inbuf);
		element->zeros_in_adapter = 0;
		result = filter(element, outbuf);
	} else if(!nonzero_samples) {
		/*
		 * input is 0s and adapter has no non-zero samples in it so
		 * output is all 0s.  the output nominally has the same
		 * number of samples as the input unless we are still in
		 * the start-up transient at the beginning of the stream
		 */

		if(adapter_length < fir_length(element) - 1) {
			/*
			 * input has not yet advanced beyond the start-up
			 * transient.  initialize adapter's contents by
			 * consuming input samples.  if there are no input
			 * samples left then we produce no output
			 */

			guint64 zeros = MIN(in_length, fir_length(element) - 1 - adapter_length);
			push_zeros(element, zeros);
			in_length -= zeros;
			if(!in_length) {
				result = GST_BASE_TRANSFORM_FLOW_DROPPED;
				goto done;
			}
		}
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, in_length, TRUE);
		result = GST_FLOW_OK;
	} else if(adapter_length + in_length <= minimum_input_length(element, nonzero_samples)) {
		/*
		 * input is 0s, adapter has at least 1 non-zero sample in
		 * it, but not enough input samples are available to
		 * consume the non-zero samples from the adapter and still
		 * have data left over:  output will be single buffer
		 * computed normally.  the buffer might be dropped if we
		 * don't have enough samples to compute anything
		 */

		push_zeros(element, in_length);
		result = filter(element, outbuf);
	} else {
		/*
		 * input is 0s, adapter has at least 1 non-zero sample in
		 * it, and enough input samples are available to consume
		 * the non-zero samples from the adapter and still have
		 * data left over:  output will be two buffers, the first
		 * computed normally to consume the non-zero samples, the
		 * second a gap buffer
		 */

		/*
		 * push enough zeros into the adapter to allow the
		 * filtering code to consume the non-zero samples
		 */

		guint64 zeros = minimum_input_length(element, nonzero_samples) - adapter_length;
		push_zeros(element, zeros);
		in_length -= zeros;

		/*
		 * generate the first output buffer and push downstream
		 */

		if(element->time_domain) {
			result = filter_and_push(element, nonzero_samples);
			if(result != GST_FLOW_OK)
				goto done;
		} else {
			/* FIXME:  the boundary between the two buffers should be
			 * set to where the zeros actually start, not where the FFT
			 * block boundary occurs */

			result = filter_and_push(element, nonzero_samples);
			if(result != GST_FLOW_OK)
				goto done;
		}

		/*
		 * generate the second, gap, buffer
		 */

		GST_BUFFER_SIZE(outbuf) = in_length * fir_channels(element) * sizeof(double);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, in_length, TRUE);
	}

	/*
	 * done
	 */

done:
	g_mutex_unlock(element->fir_matrix_lock);
	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_TIME_DOMAIN:
		g_mutex_lock(element->fir_matrix_lock);
		element->time_domain = g_value_get_boolean(value);
		if(element->time_domain) {
			/*
			 * invalidate frequency-domain filters
			 */

			free_fft_workspace(element);
		}
		g_mutex_unlock(element->fir_matrix_lock);
		break;

	case ARG_BLOCK_STRIDE: {
		gint block_stride;
		g_mutex_lock(element->fir_matrix_lock);
		block_stride = g_value_get_int(value);
		if(block_stride != element->block_stride)
			/*
			 * invalidate frequency-domain filters
			 */

			free_fft_workspace(element);
		element->block_stride = block_stride;
		g_mutex_unlock(element->fir_matrix_lock);
		break;
	}

	case ARG_FIR_MATRIX: {
		unsigned channels;
		g_mutex_lock(element->fir_matrix_lock);
		if(element->fir_matrix) {
			channels = fir_channels(element);
			gsl_matrix_free(element->fir_matrix);
		} else
			channels = 0;
		element->fir_matrix = gstlal_gsl_matrix_from_g_value_array(g_value_get_boxed(value));

		/*
		 * if the number of channels has changed, force a caps
		 * renegotiation
		 */

		if(fir_channels(element) != channels) {
			/* FIXME:  is this right? */
			gst_pad_set_caps(GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(object)), NULL);
			/*gst_base_transform_reconfigure(GST_BASE_TRANSFORM(object));*/
		}

		/*
		 * invalidate frequency-domain filters
		 */

		free_fft_workspace(element);

		/*
		 * signal availability of new time-domain filters
		 */

		g_cond_broadcast(element->fir_matrix_available);
		g_mutex_unlock(element->fir_matrix_lock);
		break;
	}

	case ARG_LATENCY: {
		gint64 latency = g_value_get_int64(value);
		if(latency != element->latency) {
			/* FIXME:  send updated segment downstream? */
		}
		element->latency = latency;
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_TIME_DOMAIN:
		g_value_set_boolean(value, element->time_domain);
		break;

	case ARG_BLOCK_STRIDE:
		g_value_set_int(value, element->block_stride);
		break;

	case ARG_FIR_MATRIX:
		g_mutex_lock(element->fir_matrix_lock);
		if(element->fir_matrix)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix(element->fir_matrix));
		/* FIXME:  else?  maybe return an empty array, or unset the gvalue? */
		g_mutex_unlock(element->fir_matrix_lock);
		break;

	case ARG_LATENCY:
		g_value_set_int64(value, element->latency);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * dispose()
 */


static void dispose(GObject *object)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(object);

	/*
	 * wake up any threads that are waiting for the fir matrix to
	 * become available;  since we are being finalized the element
	 * state should be NULL causing those threads to bail out
	 */

	g_mutex_lock(element->fir_matrix_lock);
	g_cond_broadcast(element->fir_matrix_available);
	g_mutex_unlock(element->fir_matrix_lock);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(object);

	/*
	 * free resources
	 */

	g_mutex_free(element->fir_matrix_lock);
	element->fir_matrix_lock = NULL;
	g_cond_free(element->fir_matrix_available);
	element->fir_matrix_available = NULL;
	if(element->fir_matrix) {
		gsl_matrix_free(element->fir_matrix);
		element->fir_matrix = NULL;
	}
	free_fft_workspace(element);

	/*
	 * chain to parent class' finalize() method
	 */

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_firbank_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "FIR Filter Bank", "Filter/Audio", "Projects a single audio channel onto a bank of FIR filters to produce a multi-channel output", "Kipp Cannon <kipp.cannon@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->stop = GST_DEBUG_FUNCPTR(stop);
	transform_class->event = GST_DEBUG_FUNCPTR(event);
}


/*
 * class_init()
 */


static void gstlal_load_fftw_wisdom(void)
{
	char *filename;
	int savederrno;

	g_mutex_lock(gstlal_fftw_lock);
	savederrno = errno;
	filename = getenv(GSTLAL_FFTW_WISDOM_ENV);
	if(filename) {
		FILE *f = fopen(filename, "r");
		if(!f)
			GST_ERROR("cannot open FFTW wisdom file \"%s\": %s", filename, strerror(errno));
		else {
			if(!fftw_import_wisdom_from_file(f))
				GST_ERROR("failed to import FFTW wisdom from \"%s\": wisdom not loaded", filename);
			fclose(f);
		}
	} else if(!fftw_import_system_wisdom())
		GST_WARNING("failed to import system default FFTW wisdom: %s", strerror(errno));
	errno = savederrno;
	g_mutex_unlock(gstlal_fftw_lock);
}


static void gstlal_firbank_class_init(GSTLALFIRBankClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(dispose);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	klass->rate_changed = GST_DEBUG_FUNCPTR(rate_changed);

	g_object_class_install_property(
		gobject_class,
		ARG_TIME_DOMAIN,
		g_param_spec_boolean(
			"time-domain",
			"Use time-domain convolution",
			"Set to true to use time-domain (a.k.a. direct) convolution, set to false to use FFT-based convolution.  For long filters FFT-based convolution is usually significantly faster than time-domain convolution but incurs a higher processing latency and requires more RAM.",
			DEFAULT_TIME_DOMAIN,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_BLOCK_STRIDE,
		g_param_spec_int(
			"block-stride",
			"Convolution block stride",
			"When using FFT convolutions, this many samples will be produced from each block.  Smaller values decrease latency but increase computational cost.  If very small values are desired, consider using time-domain convolution mode instead.",
			1, G_MAXINT, DEFAULT_BLOCK_STRIDE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_MATRIX,
		g_param_spec_value_array(
			"fir-matrix",
			"FIR Matrix",
			"Array of impulse response vectors.  Number of vectors (rows) in matrix sets number of output channels.  All filters must have the same length.",
			g_param_spec_value_array(
				"response",
				"Impulse Response",
				"Array of amplitudes.",
				g_param_spec_double(
					"amplitude",
					"Amplitude",
					"Impulse response sample",
					-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_LATENCY,
		g_param_spec_int64(
			"latency",
			"Latency",
			"Filter latency in samples.",
			G_MININT64, G_MAXINT64, DEFAULT_LATENCY,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	signals[SIGNAL_RATE_CHANGED] = g_signal_new(
		"rate-changed",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALFIRBankClass,
			rate_changed
		),
		NULL,
		NULL,
		g_cclosure_marshal_VOID__INT,
		G_TYPE_NONE,
		1,
		G_TYPE_INT
	);

	/*
	 * Load FFTW wisdom
	 */

	gstlal_load_fftw_wisdom();
}


/*
 * init()
 */


static void gstlal_firbank_init(GSTLALFIRBank *filter, GSTLALFIRBankClass *klass)
{
	filter->block_stride = 0;	/* must != DEFAULT_BLOCK_STRIDE */
	filter->latency = 0;
	filter->adapter = NULL;
	filter->time_domain = FALSE;
	filter->fir_matrix_lock = g_mutex_new();
	filter->fir_matrix_available = g_cond_new();
	filter->fir_matrix = NULL;
	filter->fir_matrix_fd = NULL;
	filter->input_fd = NULL;
	filter->workspace_fd = NULL;
	filter->in_plan = NULL;
	filter->out_plan = NULL;
	filter->last_new_segment = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

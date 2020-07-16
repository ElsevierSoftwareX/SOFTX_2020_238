/*
 * Copyright (C) 2017 Leo Tsukada <tsukada@resceu.s.u-tokyo.ac.jp>
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


#include <string.h>
#include <math.h>
#include <unistd.h>


/*
 * stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * stuff from gstlal
 */


#include <gstlal/gstaudioadapter.h>
#include <gstlal/gstlal.h>
#include <gstlal_tdwhiten.h>


/*
 * ============================================================================
 *
 *                                Kernel Info
 *
 * ============================================================================
 */

struct kernelinfo_t {
	/*
	 * input sample index and timestamp at which this kernel was added
	 */

	guint64 offset;
	GstClockTime timestamp;

	/*
	 * timestamp when the kernel is no longer valid
	 */

	GstClockTime endtime;

	/*
	 * the kernel
	 */

	gsize length;
	gdouble *kernel;
};


static struct kernelinfo_t *kernelinfo_new(gsize length)
{
	struct kernelinfo_t *kernelinfo = g_new(struct kernelinfo_t, 1);
	gdouble *kernel = g_malloc_n(sizeof(*kernel), length);

	if(!kernelinfo || (!kernel && length != 0)) {
		g_free(kernelinfo);
		g_free(kernel);
		return NULL;
	}

	kernelinfo->offset = GST_BUFFER_OFFSET_NONE;
	kernelinfo->timestamp = GST_CLOCK_TIME_NONE;
	kernelinfo->endtime = GST_CLOCK_TIME_NONE;
	kernelinfo->length = length;
	kernelinfo->kernel = kernel;

	return kernelinfo;
}


static void kernelinfo_free(struct kernelinfo_t *kernelinfo)
{
	if(kernelinfo)
		g_free(kernelinfo->kernel);
	g_free(kernelinfo);
}


static void _kernelinfo_save_longest(gpointer data, gpointer userdata)
{
	struct kernelinfo_t *kernelinfo = data;
	gsize *length = userdata;

	if(kernelinfo->length > *length)
		*length = kernelinfo->length;
}


static gsize kernelinfo_longest(GQueue *kernels)
{
	gsize longest = 0;

	g_queue_foreach(kernels, _kernelinfo_save_longest, &longest);

	return longest;
}


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


static unsigned get_input_length(GSTLALTDwhiten *element, unsigned output_length)
{
	return output_length + kernelinfo_longest(element->kernels) - 1;
}


/*
 * how many output samples can be generated from the given number of input
 * samples
 */


static unsigned get_output_length(const GSTLALTDwhiten *element, unsigned input_length)
{
	/*
	 * how many samples can we construct from the contents of the
	 * adapter?  the +1 is because when there is 1 FIR-length of data
	 * in the adapter then we can produce 1 output sample, not 0.
	 */

	if(input_length < kernelinfo_longest(element->kernels))
		return 0;
	return input_length - kernelinfo_longest(element->kernels) + 1;
}


/*
 * the number of samples available in the adapter
 */


static guint64 get_available_samples(GSTLALTDwhiten *element)
{
	guint size;

	g_object_get(element->adapter, "size", &size, NULL);

	return size;
}


/*
 * construct a buffer of zeros and push into adapter
 */


static int push_zeros(GSTLALTDwhiten *element, unsigned samples)
{
	GstBuffer *zerobuf = gst_buffer_new_allocate(NULL, samples * sizeof(double), NULL);
	GstMapInfo mapinfo;
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
		return -1;
	}
	gst_buffer_map(zerobuf, &mapinfo, GST_MAP_WRITE);
	memset(mapinfo.data, 0, mapinfo.size);
	gst_buffer_unmap(zerobuf, &mapinfo);
	gst_audioadapter_push(element->adapter, zerobuf);
	return 0;
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALTDwhiten *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_PTS(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, GST_AUDIO_INFO_RATE(&element->audio_info));
	element->next_pts = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, GST_AUDIO_INFO_RATE(&element->audio_info));
	GST_BUFFER_DURATION(buf) = element->next_pts - GST_BUFFER_PTS(buf);
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
 * obtain the two gains applied to output of two kernels for tapering them.
 */


static double get_gains(GSTLALTDwhiten *element, guint64 offset, struct kernelinfo_t *kernelinfo, double *gain1)
{
	double gain0;

	if(kernelinfo == g_queue_peek_head(element->kernels)) {
		gain0 = 0.0;
		*gain1 = 1.0;
	} else if(offset <= kernelinfo->offset) {
		gain0 = 1.0;
		*gain1 = 0.0;
	} else if(offset <= kernelinfo->offset + element->taper_length) {
		double phase = M_PI / 2 / element->taper_length * (offset - kernelinfo->offset);

		/* FIXME:  find a place to do this check that is less costly */
		g_assert(kernelinfo->offset != GST_BUFFER_OFFSET_NONE);

		gain0 = pow(cos(phase), 2);
		*gain1 = pow(sin(phase), 2);
	} else {
		gain0 = 0.0;
		*gain1 = 1.0;
	}

	return gain0;
}



/*
 * compute an innner product from two arrays
 */


static double inner_product(gsl_vector *vec1, gsl_vector *vec2)
{
	double output;

	gsl_blas_ddot(vec1, vec2, &output);
	return output;
}


/*
 * compute the output from kernel matrix in time domain on double precision.
 */


static unsigned tddfilter(GSTLALTDwhiten *element, GstMapInfo *mapinfo, unsigned output_length)
{
	struct kernelinfo_t *kernelinfo;
	gsize longest_kernel_length = kernelinfo_longest(element->kernels);
	unsigned input_length;
	unsigned i;
	unsigned j;
	double gain0;
	double gain1;
	unsigned poped_kernels;
	double *input;
	double *output = (double *) mapinfo->data;
	gsl_vector_view input_view;
	gsl_vector_view kernel_view;
	

	/*
	 * clip number of output samples to buffer size
	 */

	output_length = MIN(output_length, mapinfo->size / sizeof(*output));

	/*
	 * how many samples do we need from the adapter?
	 */

	input_length = output_length + longest_kernel_length - 1;

	/*
	 * retrieve input samples from the adapter
	 */

	input = g_malloc(input_length * sizeof(*input));
	gst_audioadapter_copy_samples(element->adapter, input, input_length, NULL, NULL);

	/*
	 * zero the output
	 */

	memset(output, 0, output_length * sizeof(*output));

	/*
	 * assemble the output sample time series as the output array.
	 */

	poped_kernels = 0;
	for(j = 0; j < g_queue_get_length(element->kernels); j++) {
		/*
		 * peek the nth kernelinfo from the head of the queue.
		 */

		kernelinfo = g_queue_peek_nth(element->kernels, j);

		/*
		 * taper the two output samples time series computed by two
		 * kernels.
		 */

		input_view = gsl_vector_view_array(input, kernelinfo->length);
		kernel_view = gsl_vector_view_array(kernelinfo->kernel, kernelinfo->length);
		for(i = 0; i < output_length; i++, input_view.vector.data++) {
			gain0 = get_gains(element, element->next_out_offset + i, kernelinfo, &gain1);

			output[i] = output[i] * gain0 + inner_product(&kernel_view.vector, &input_view.vector) * gain1;
		}

		/*
		 * update the count for the unnessary kernels.
		 */

		if(element->next_out_offset + i > kernelinfo->offset + element->taper_length)
			poped_kernels = j;
	}

	/*
	 * done
	 */

	while(poped_kernels--)
		kernelinfo_free(g_queue_pop_head(element->kernels));
	g_free(input);

	return output_length;
}


/*
 * filtering algorithm front-end
 */


static unsigned filter(GSTLALTDwhiten *element, GstBuffer *buf)
{
	unsigned output_length;

	GstMapInfo mapinfo;

	output_length = get_output_length(element, get_available_samples(element));

	gst_buffer_map(buf, &mapinfo, GST_MAP_WRITE);

	if(output_length > 0)
		output_length = tddfilter(element, &mapinfo, output_length);

	gst_buffer_unmap(buf, &mapinfo);

	/*
	 * flush the data from the adapter
	 */

	gst_audioadapter_flush_samples(element->adapter, output_length);

	/*
	 * set output metadata
	 */

	set_metadata(element, buf, output_length, FALSE);

	/*
	 * done
	 */

	return output_length;
}


/*
 * run filter code on adapter's contents putting result into a newly
 * allocated buffer, and push buffer downstream.  the fir_matrix_lock must
 * be held when calling this function
 */


static GstFlowReturn filter_and_push(GSTLALTDwhiten *element, guint64 output_length)
{
	GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(element));
	GstBuffer *buf;
	unsigned filter_output_length;

	/* FIXME:  if we released the fir matrix lock the matrix might
	 * change while we do this.  but we probably shouldn't hold the
	 * lock while leaving the element */

	if(!output_length)
		return GST_FLOW_OK;

	buf = gst_buffer_new_allocate(NULL, output_length * GST_AUDIO_INFO_BPF(&element->audio_info), NULL);
	if(!buf)
		return GST_FLOW_ERROR;

	filter_output_length = filter(element, buf);
	g_assert_cmpuint(filter_output_length, ==, output_length);
	return gst_pad_push(srcpad, buf);
}


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


enum gstlal_tdwhiten_signal {
	SIGNAL_RATE_CHANGED,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


static void rate_changed(GstElement *element, gint rate, void *data)
{
	/* FIXME: the timestamp book-keeping needs to be reset on the next
	 * input buffer */
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


G_DEFINE_TYPE(
	GSTLALTDwhiten,
	gstlal_tdwhiten,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *                    Gst BaseTransform Method Overriddes
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
	GstAudioInfo info;
	gboolean success = gst_audio_info_from_caps(&info, caps);

	if(success)
		*size = GST_AUDIO_INFO_BPF(&info);

	return success;
}


/*
 * tranform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *incaps, gsize insize, GstCaps *outcaps, gsize *outsize)
{
	GSTLALTDwhiten *element = GSTLAL_TDWHITEN(trans);
	gsize in_unit_size;
	gsize out_unit_size;
	gboolean success = TRUE;

	if(!get_unit_size(trans, incaps, &in_unit_size))
		return FALSE;
	if(insize % in_unit_size) {
		GST_ERROR_OBJECT(element, "size not a multiple of %" G_GSIZE_FORMAT, in_unit_size);
		return FALSE;
	}
	if(!get_unit_size(trans, outcaps, &out_unit_size))
		return FALSE;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * number of input bytes required to produce an output
		 * buffer of (at least) the requested size
		 */

		*outsize = get_input_length(element, insize / in_unit_size);
		if(*outsize <= get_available_samples(element))
			*outsize = 0;
		else
			*outsize = (*outsize - get_available_samples(element)) * out_unit_size;
		break;

	case GST_PAD_SINK:
		/*
		 * number of output bytes to be generated by the receipt of
		 * an input buffer of the given size.
		 */

		*outsize = get_output_length(element, insize / in_unit_size + get_available_samples(element)) * out_unit_size;
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		success = FALSE;
		break;
	}

	/*
	 * done
	 */

	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALTDwhiten *element = GSTLAL_TDWHITEN(trans);
	gint old_rate = GST_AUDIO_INFO_RATE(&element->audio_info);
	gboolean success = gst_audio_info_from_caps(&element->audio_info, incaps);
	if(success && GST_AUDIO_INFO_RATE(&element->audio_info) != old_rate)
		g_signal_emit(G_OBJECT(trans), signals[SIGNAL_RATE_CHANGED], 0, GST_AUDIO_INFO_RATE(&element->audio_info), NULL);

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALTDwhiten *element = GSTLAL_TDWHITEN(trans);
	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, "unit-size", 8, NULL);
	element->t0 = GST_CLOCK_TIME_NONE;
	element->next_pts = 0;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	g_mutex_init(&element->kernel_lock);
	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseTransform *trans)
{
	GSTLALTDwhiten *element = GSTLAL_TDWHITEN(trans);
	g_object_unref(element->adapter);
	element->adapter = NULL;
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALTDwhiten *element = GSTLAL_TDWHITEN(trans);
	struct kernelinfo_t *kernelinfo;
	gboolean history_is_gap, input_is_gap;
	GstMapInfo mapinfo;
	guint output_length, nonzero_output_length;
	GstFlowReturn result = GST_FLOW_OK;

	g_assert(GST_BUFFER_PTS_IS_VALID(inbuf));
	g_assert(GST_BUFFER_DURATION_IS_VALID(inbuf));
	g_assert(GST_BUFFER_OFFSET_IS_VALID(inbuf));
	g_assert(GST_BUFFER_OFFSET_END_IS_VALID(inbuf));

	/*
	 * Check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		GST_INFO_OBJECT(element, "encountered discont.  reason:  %s", GST_BUFFER_IS_DISCONT(inbuf) ? "discont flag set in input" : GST_BUFFER_OFFSET(inbuf) != element->next_in_offset ? "input offset mismatch" : "internal clock not yet set");

		/*
		 * flush adapter
		 */

		gst_audioadapter_clear(element->adapter);

		/*
		 * remove all but the newest kernel
		 */

		while(g_queue_get_length(element->kernels) > 1)
			kernelinfo_free(g_queue_pop_head(element->kernels));

		/*
		 * (re)sync timestamp and offset book-keeping
		 */

		element->t0 = GST_BUFFER_PTS(inbuf);
		element->next_pts = GST_BUFFER_PTS(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = element->offset0 + kernelinfo_longest(element->kernels) - 1 - element->latency;

		/*
		 * be sure to flag the next output buffer as a discontinuity
		 */

		element->need_discont = TRUE;
	} else if(!gst_audioadapter_is_empty(element->adapter))
		g_assert_cmpuint(GST_BUFFER_PTS(inbuf), ==, gst_audioadapter_expected_timestamp(element->adapter));
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * Check if we are past the end time of the latest kernel in use.
	 * If so, put a waiting kernel into use if available.
	 */

	/*
	 * FIXME: the purpose of the wait loop below is to make the element
	 * wait for a new filter kernel if the current one is "expired."
	 * This should instead be done using the controller:
	 * https://gstreamer.freedesktop.org/documentation/application-development/advanced/dparams.html
	 * https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstTimedValueControlSource.html
	 */

	kernelinfo = g_queue_peek_tail(element->kernels);
	if(kernelinfo->endtime != GST_CLOCK_TIME_NONE && element->next_pts >= kernelinfo->endtime) {
		GstClockTime endtime = kernelinfo->endtime;
		/* Wait until an appropriate filter arrives before continuing */
		while(element->next_pts >= endtime) {
			/* First check if we changed our mind about the expiration date for the current kernel */
			kernelinfo = g_queue_peek_tail(element->kernels);
			if(kernelinfo->endtime > element->next_pts)
				goto setoffsets;
			kernelinfo = g_queue_peek_head(element->waiting_kernels);
			if(kernelinfo) {
				if(kernelinfo->endtime != GST_CLOCK_TIME_NONE) {
					/* Throw away ones we won't use */
					if(kernelinfo->endtime <= element->next_pts)
						kernelinfo_free(g_queue_pop_head(element->waiting_kernels));
					else
						endtime = kernelinfo->endtime;
				}
			}
			/* Don't waste CPUs */
			sleep(1);
		}
		/* Dump any kernels we haven't used and won't use */
		g_mutex_lock(&element->kernel_lock);
		kernelinfo = g_queue_peek_tail(element->kernels);
		if(kernelinfo->offset == GST_BUFFER_OFFSET_NONE)
			kernelinfo_free(g_queue_pop_head(element->kernels));
		g_queue_push_tail(element->kernels, g_queue_pop_head(element->waiting_kernels));
		g_mutex_unlock(&element->kernel_lock);
	}
setoffsets:

	/*
	 * set offsets and timestamps for kernels that are new.
	 * set_property() will ensure that there is only one new kernel at
	 * a time, so we can just check the tail of the queue.  don't need
	 * to check the rest.
	 */

	kernelinfo = g_queue_peek_tail(element->kernels);
	if(kernelinfo->offset == GST_BUFFER_OFFSET_NONE) {
		kernelinfo->offset = GST_BUFFER_OFFSET(inbuf);
		kernelinfo->timestamp = GST_BUFFER_PTS(inbuf);
	}

	/*
	 * build output buffer(s)
	 */

	input_is_gap = GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP);
	history_is_gap = gst_audioadapter_is_gap(element->adapter);

	GST_INFO_OBJECT(element, "%u+%u history+input samples in hand", (guint) get_available_samples(element), (guint) (GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf)));
	gst_buffer_ref(inbuf);	/* don't let calling code free buffer */
	gst_audioadapter_push(element->adapter, inbuf);

	output_length = get_output_length(element, get_available_samples(element));

	/* how much input data is required to consume all the oldest
	 * non-zero samples in the adapter, and if we had that much input
	 * data how much output data would we produce?  if the answer is
	 * "less than we're going to produce" then we will make two output
	 * buffers */
	/*nonzero_output_length = get_output_length(element, get_input_length(element, gst_audioadapter_head_nongap_length(element->adapter)));*/
	nonzero_output_length = get_output_length(element, get_input_length(element, get_available_samples(element) - gst_audioadapter_tail_gap_length(element->adapter)));

	GST_INFO_OBJECT(element, "state: history is %s, input is %s, zeros in adapter = %u", history_is_gap ? "gap" : "not gap", input_is_gap ? "gap" : "not gap", gst_audioadapter_tail_gap_length(element->adapter));
	if(!input_is_gap) {
		/*
		 * because the history that remains in the adapter cannot
		 * be large enough to compute even 1 sample, the output is
		 * a single non-gap buffer whether or not the history is
		 * known to be all 0s
		 */

		guint samples = filter(element, outbuf);
		g_assert_cmpuint(output_length, ==, samples);
		GST_LOG_OBJECT(element, "output is %u samples", output_length);
	} else if(history_is_gap) {
		/*
		 * all data in hand is known to be 0s, the output is a
		 * single gap buffer
		 */

		gst_audioadapter_flush_samples(element->adapter, output_length);
		gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
		memset(mapinfo.data, 0, mapinfo.size);
		gst_buffer_unmap(outbuf, &mapinfo);
		set_metadata(element, outbuf, output_length, TRUE);
		GST_LOG_OBJECT(element, "output is %u sample gap", output_length);
	} else if(nonzero_output_length >= output_length) {
		/*
		 * at least some of the start of the history contains
		 * non-zero data, so we must produce at least 1 non-gap
		 * buffer, and there aren't enough 0s at the end of the
		 * history to allow a full gap buffer to be produced, so
		 * the output is a single non-gap buffer.  (the test is "do
		 * we have enough input to consume all non-zero samples
		 * from the history?")
		 */

		guint samples = filter(element, outbuf);
		g_assert_cmpuint(output_length, ==, samples);
		GST_LOG_OBJECT(element, "output is %u samples", output_length);
	} else {
		/*
		 * the tailing zeros in the history combined with the input
		 * data are together large enough to yield 0s in the
		 * output. the output will be two buffers, a non-gap buffer
		 * to finish off the non-zero data in the history followed
		 * by a gap buffer.
		 */

		guint gap_length = output_length - nonzero_output_length;

		GST_LOG_OBJECT(element, "output is %u samples followed by %u sample gap", output_length - gap_length, gap_length);

		/*
		 * generate the first output buffer and push downstream
		 */

		result = filter_and_push(element, nonzero_output_length);
		if(result != GST_FLOW_OK) {
			gst_buffer_unmap(outbuf, &mapinfo);
			goto done;
		}

		/*
		 * generate the second, gap, buffer
		 */

		gst_audioadapter_flush_samples(element->adapter, gap_length);
		gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
		memset(mapinfo.data, 0, mapinfo.size);
		gst_buffer_unmap(outbuf, &mapinfo);
		set_metadata(element, outbuf, gap_length, TRUE);
		gst_buffer_set_size(outbuf, gap_length * GST_AUDIO_INFO_BPF(&element->audio_info));
	}

	/*
	 * done
	 */

done:
	return result;
}


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_TAPER_LENGTH 0
#define DEFAULT_LATENCY 0


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum property {
	PROP_TAPER_LENGTH = 1,
	PROP_KERNEL,
	PROP_LATENCY,
	PROP_KERNEL_ENDTIME
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTDwhiten *element = GSTLAL_TDWHITEN(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id){
	case PROP_TAPER_LENGTH:
		element->taper_length = g_value_get_uint(value);
		break;

	case PROP_KERNEL: {
		GValueArray *va = g_value_get_boxed(value);
		struct kernelinfo_t *kernelinfo, *old_kernelinfo;
		/* dump any kernels that haven't yet been used */
		for(kernelinfo = g_queue_peek_tail(element->kernels); kernelinfo && kernelinfo->endtime == GST_CLOCK_TIME_NONE && kernelinfo->offset == GST_BUFFER_OFFSET_NONE; kernelinfo = g_queue_peek_tail(element->kernels))
			kernelinfo_free(g_queue_pop_tail(element->kernels));
		/* construct new kernelinfo object */
		kernelinfo = kernelinfo_new(va->n_values);
		gstlal_doubles_from_g_value_array(va, kernelinfo->kernel, NULL);
		/* push the new kernel into the queue */
		old_kernelinfo = g_queue_peek_tail(element->kernels);
		if(!old_kernelinfo) {
			if(element->kernel_endtime < G_MAXUINT64)
				kernelinfo->endtime = element->kernel_endtime;
			g_queue_push_tail(element->kernels, kernelinfo);
		} else if(old_kernelinfo->endtime == GST_CLOCK_TIME_NONE)
			g_queue_push_tail(element->kernels, kernelinfo);
		else
			/* this kernel should wait until the end time of the previous one */
			g_queue_push_tail(element->waiting_kernels, kernelinfo);
		break;
	}

	case PROP_LATENCY: {
		gint64 latency = g_value_get_int64(value);
		element->latency = latency;
		break;
	}

	case PROP_KERNEL_ENDTIME: {
		/*
		 * FIXME: This should instead be handled with the controller. Then this property would be unnecessary.
		 * https://gstreamer.freedesktop.org/documentation/application-development/advanced/dparams.html
		 * https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstTimedValueControlSource.html
		 */
		GstClockTime kernel_endtime = (GstClockTime) g_value_get_uint64(value);
		if(!g_queue_is_empty(element->kernels) && kernel_endtime < G_MAXUINT64) {
			g_mutex_lock(&element->kernel_lock);
			struct kernelinfo_t *kernelinfo;
			/*
			 * If this endtime is less than the current incoming data timestamps
			 * or less than the end time of the last kernel, we will throw the
			 * kernel away.
			 */
			GstClockTime min_endtime = element->next_pts;
			guint i;
			for(i = 0; i < g_queue_get_length(element->waiting_kernels); i++) {
				kernelinfo = g_queue_peek_nth(element->waiting_kernels, i);
				if(kernelinfo->endtime != GST_CLOCK_TIME_NONE)
					min_endtime = min_endtime > kernelinfo->endtime ? min_endtime : kernelinfo->endtime;
			}
			for(i = 0; i < g_queue_get_length(element->kernels); i++) {
				kernelinfo = g_queue_peek_nth(element->kernels, i);
				if(kernelinfo->endtime != GST_CLOCK_TIME_NONE)
					min_endtime = min_endtime > kernelinfo->endtime ? min_endtime : kernelinfo->endtime;
			}
			/* First check if there are kernels waiting to be used */
			if(!g_queue_is_empty(element->waiting_kernels)) {
				if(kernel_endtime > min_endtime) {
					kernelinfo = g_queue_peek_tail(element->waiting_kernels);
					kernelinfo->endtime = kernel_endtime;
					element->kernel_endtime = kernel_endtime;
					/* In this case, also throw away any waiting kernels without end times */
					for(i = 0; i < g_queue_get_length(element->waiting_kernels); i++) {
						kernelinfo = g_queue_peek_nth(element->waiting_kernels, i);
						if(kernelinfo->endtime == GST_CLOCK_TIME_NONE)
							kernelinfo_free(g_queue_pop_nth(element->waiting_kernels, i));
					}
				} else
					kernelinfo_free(g_queue_pop_tail(element->waiting_kernels));
			} else if(!g_queue_is_empty(element->kernels)) {
				/* If there are no waiting kernels, apply this to the most recent kernel being used */
				if(kernel_endtime > min_endtime || g_queue_get_length(element->kernels) == 1) {
					kernelinfo = g_queue_peek_tail(element->kernels);
					kernelinfo->endtime = kernel_endtime;
					element->kernel_endtime = kernel_endtime;
				} else
					kernelinfo_free(g_queue_pop_tail(element->kernels));
			}
			g_mutex_unlock(&element->kernel_lock);
		} else if (kernel_endtime < G_MAXUINT64)
			/* The first kernel has not arrived yet, so store it for when it does */
			element->kernel_endtime = kernel_endtime;
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
	GSTLALTDwhiten *element = GSTLAL_TDWHITEN(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id){
	case PROP_TAPER_LENGTH:
		g_value_set_uint(value, element->taper_length);
		break;

	case PROP_KERNEL: {
		if (g_queue_is_empty(element->kernels))
			g_value_take_boxed(value, g_value_array_new(0));
		else {
			struct kernelinfo_t *head_kernelinfo = g_queue_peek_tail(element->kernels);
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(head_kernelinfo->kernel, head_kernelinfo->length));
		}
		break;
	}

	case PROP_LATENCY:
		g_value_set_int64(value, element->latency);
		break;

	case PROP_KERNEL_ENDTIME: {
		/* returns the most recent end time in the queue, or G_MAXUINT64 if no times are present */
		g_value_set_uint64(value, element->kernel_endtime);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALTDwhiten *element = GSTLAL_TDWHITEN(object);

	g_queue_free_full(element->kernels, (GDestroyNotify) kernelinfo_free);
	element->kernels = NULL;
	g_queue_free_full(element->waiting_kernels, (GDestroyNotify) kernelinfo_free);
	element->waiting_kernels = NULL;
	g_mutex_clear(&element->kernel_lock);
	if(element->adapter) {
		g_object_unref(element->adapter);
		element->adapter = NULL;
	}

	/*
	 * chain to parent class' finalize() method
	 */

	G_OBJECT_CLASS(gstlal_tdwhiten_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		GST_AUDIO_CAPS_MAKE(GST_AUDIO_NE(F64)) ", " \
		"channels = (int) 1, " \
		"layout = (string) interleaved"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		GST_AUDIO_CAPS_MAKE(GST_AUDIO_NE(F64)) ", " \
		"channels = (int) 1, " \
		"layout = (string) interleaved"
	)
);


static void gstlal_tdwhiten_class_init(GSTLALTDwhitenClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Audio FIR Filter",
		"Filter/Audio",
		"Generic audio FIR filter with custom filter kernel and smooth kernel updates",
		"Leo Tsukada <tsukada@resceu.s.u-tokyo.ac.jp>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->stop = GST_DEBUG_FUNCPTR(stop);

	klass->rate_changed = GST_DEBUG_FUNCPTR(rate_changed);

	g_object_class_install_property(
		gobject_class,
		PROP_TAPER_LENGTH,
		g_param_spec_uint(
			"taper-length",
			"Taper length",
			"Number of samples for kernel transition.",
			0, G_MAXUINT, DEFAULT_TAPER_LENGTH,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_KERNEL,
		g_param_spec_value_array(
			"kernel",
			"Kernel",
			"The newest kernel.",
			g_param_spec_double(
				"sample",
				"Sample",
				"Samples of a kernel",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_LATENCY,
		g_param_spec_int64(
			"latency",
			"Latency",
			"Filter latency in samples.",
			G_MININT64, G_MAXINT64, DEFAULT_LATENCY,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT | GST_PARAM_CONTROLLABLE
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_KERNEL_ENDTIME,
		g_param_spec_uint64(
			"kernel-endtime",
			"Kernel end time",
			"Timestamp at which most recently acquired filter kernel becomes invalid.\n\t\t\t"
			"At this time, element will wait for another filter before processing\n\t\t\t"
			"more data.  This feature can be enabled at any time during data flow,\n\t\t\t"
			"but should not be disabled once enabled.  Default is to leave disabled.",
			0, G_MAXUINT64, G_MAXUINT64,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT | GST_PARAM_CONTROLLABLE
		)
	);

	signals[SIGNAL_RATE_CHANGED] = g_signal_new(
		"rate-changed",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALTDwhitenClass,
			rate_changed
		),
		NULL,
		NULL,
		g_cclosure_marshal_VOID__INT,
		G_TYPE_NONE,
		1,
		G_TYPE_INT
	);
}


/*
 * init()
 */


static void gstlal_tdwhiten_init(GSTLALTDwhiten *filter)
{
	filter->audio_info.bpf = 0;	/* impossible value */
	filter->adapter = NULL;
	filter->latency = 0;
	filter->kernels = g_queue_new();
	filter->waiting_kernels = g_queue_new();
	filter->kernel_endtime = G_MAXUINT64;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

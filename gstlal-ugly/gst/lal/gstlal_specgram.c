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
#include <complex.h>
#include <math.h>

/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <glibconfig.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>


/*
 * stuff from gstlal
 */

#include <gstlal/gstlal.h>
#include <gstlal_specgram.h>

/*
 * Extra stuff
 */

#include <fftw3.h>

/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */

/*
 * Compute the stride when in rate is higher than out rate
 */

static guint32 stride(const GSTLALSpecgram *element)
{
	return element->rate / element->outrate;
}

/*
 * Compute the minimum input legth that is required to form an output buffer of size n
 * We only form buffers in units of stride to make book keeping easier
 */

static guint64 minimum_input_length(const GSTLALSpecgram *element, guint64 n)
{
	return n + stride(element) + (guint64) element->n  - 1;
}

/*
 * Number of output channels produced
 */

static guint32 num_channels(GSTLALSpecgram *element)
{
	return element->n / 2;
}

/*
 * the number of samples available in the adapter
 */

static guint64 get_available_samples(GSTLALSpecgram *element)
{
	return gst_adapter_available(element->adapter) / sizeof(double); /*FIXME support other caps, don't rely on double */
}

/*
 * input to output size
 */

static guint64 get_output_size_from_input_size(GSTLALSpecgram *element, guint64 input_size, guint64 input_unit_size, guint64 output_unit_size)
{
	guint64 length = input_size / input_unit_size + get_available_samples(element);
	guint64 thestride = stride(element);

	if(length < minimum_input_length(element, 1))
		return 0;
	else
		/* buffers can only be made on stride boundaries */
		return  ((length - minimum_input_length(element, 1))) / thestride * output_unit_size;
}

/*
 * output length from input length
 */

static guint64 get_output_length_from_input_length(GSTLALSpecgram *element, guint64 input_length)
{
	guint64 length = input_length + get_available_samples(element);
	guint64 thestride = stride(element);

	if(length < minimum_input_length(element, 1))
		return 0;
	else
		/* FIXME include stride */
		return  (length - minimum_input_length(element, 1)) / thestride;
}

/*
 * output to input size
 */

static guint64 get_input_size_from_output_size(GSTLALSpecgram *element, guint64 output_size, guint64 output_unit_size, guint64 input_unit_size)
{
	guint64 length = output_size / output_unit_size;
	if (minimum_input_length(element, length) - get_available_samples(element) > 0)
		return (minimum_input_length(element, length) - get_available_samples(element) ) * input_unit_size;
	else return 0;

}

/*
 * construct a buffer of zeros and push into adapter
 */

static int push_zeros(GSTLALSpecgram *element, unsigned samples)
{
	/* FIXME, don't we need to give this buffer proper time stamps etc? I guess maybe we won't rely on the adapter to have proper time stamps */
	GstBuffer *zerobuf = gst_buffer_new_and_alloc(samples * sizeof(double)); /* FIXME support other caps, don't rely on double */
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
		return -1;
	}
	memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
	gst_adapter_push(element->adapter, zerobuf);
	return 0;
}

/*
 * set the metadata on an output buffer
 */

static void set_metadata(GSTLALSpecgram *element, GstBuffer *buf, guint64 outsamples, guint64 channels, gboolean gap)
{
	GST_BUFFER_SIZE(buf) = outsamples * channels * sizeof(double);	/* FIXME support other caps, don't rely on double */
	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->outrate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->outrate) - GST_BUFFER_TIMESTAMP(buf);
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
 * flush adapter
 */

static void flush(GSTLALSpecgram *element, guint64 available_length)
{
	if(available_length > element->n - 1)
		gst_adapter_flush(element->adapter, (available_length - (element->n - 1)) * sizeof(double));
}

/*
 * spec gram production
 */

static GstFlowReturn filter(GSTLALSpecgram *element, GstBuffer *outbuf, guint64 in_length)
{
	double *in, *out;
	guint64 available_length, output_length;
	guint64 i;
	guint64 j;
	guint64 thestride = stride(element);
	double norm = sqrt(num_channels(element));

	/* only ever pull out buffers in multiples of the stride */
	available_length = (get_available_samples(element) / thestride) * thestride;

	output_length = get_output_length_from_input_length(element, 0); /*FIXME the input buffer was already placed in the adapter, is that okay, probably but check */

	/* compute output samples FIXME support other caps, not just double */

	in = (double *) gst_adapter_peek(element->adapter, available_length * sizeof(double));
	out = (double *) GST_BUFFER_DATA(outbuf);

	for(i = 0; i < output_length; i++) {
		memcpy(element->infft, &(in[i*thestride]), element->n * sizeof(double));
		fftw_execute(element->fftplan);
		for (j = 0; j < num_channels(element); j++) {
			/* FIXME check that the index is right, this might start with DC, should it? */
			out[i * num_channels(element) + j] = (double) cabs(element->outfft[j]) / norm;
		}
	}

	/* output produced? */

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/* flush data from the adapter if the logic works we should have used all of the data we peeked */

	flush(element, available_length);

	/* set buffer metadata */
	/* Number of channels output is always element->n / 2 */

	set_metadata(element, outbuf, output_length, element->n / 2, FALSE);

	return GST_FLOW_OK;
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
		"channels = (int) [2, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
	)
);

GST_BOILERPLATE(
	GSTLALSpecgram,
	gstlal_specgram,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);

enum property {
	ARG_N = 1
};

#define DEFAULT_N 2

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

static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(trans);
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad's format is the same as the source pad's except
		 * it must have only 1 channel and can have a different rate
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *s = gst_caps_get_structure(caps, n);
			gst_structure_set(s, "channels", G_TYPE_INT, 1, NULL);
			gst_structure_set(s, "rate", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL); /*FIXME do the actual range calculation here */
		}
		break;

	case GST_PAD_SINK:

		/*
		 * source pad's format is the same as the sink pad's except
		 * it can have any number of channels > 2 or, if the n has been set
		 * the number of channels must be n / 2
		 * The rate can also be different
		 */

		/* g_mutex_lock(element->fir_matrix_lock); */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *s = gst_caps_get_structure(caps, n);
			gst_structure_set(s, "rate", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL); /*FIXME do the actual range calculation here */
			if(element->n)
				gst_structure_set(s, "channels", G_TYPE_INT, num_channels(element), NULL);
			else
				gst_structure_set(s, "channels", GST_TYPE_INT_RANGE, 2, G_MAXINT, NULL);
		}
		/* g_mutex_unlock(element->fir_matrix_lock); */
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return caps;
}

/*
 * Taken from audioresample
 */

static void fixate_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *othercaps)
{
	GstStructure *s;
	gint rate;

	s = gst_caps_get_structure (caps, 0);
	if (G_UNLIKELY (!gst_structure_get_int (s, "rate", &rate))) return;

	s = gst_caps_get_structure (othercaps, 0);
	gst_structure_fixate_field_nearest_int (s, "rate", rate);
	return;
}

/*
 * transform_size()
 */

static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(trans);
	guint unit_size;
	guint other_unit_size;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_DEBUG_OBJECT(element, "size not a multiple of %u", unit_size);
		return FALSE;
	}
	if(!get_unit_size(trans, othercaps, &other_unit_size))
		return FALSE;

	switch(direction) {
	case GST_PAD_SRC:

		/*
		 * number of input bytes required to produce an output
		 * buffer of (at least) the requested size
		 */
		*othersize = get_input_size_from_output_size(element, size, unit_size, other_unit_size);
		break;

	case GST_PAD_SINK:
		/*
		 * number of output bytes to be generated by the receipt of
		 * an input buffer of the given size.
		 */

		*othersize = get_output_size_from_input_size(element, size, unit_size, other_unit_size);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
}

/*
 * set_caps()
 */

static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(trans);
	GstStructure *s;
	gint inrate;
	gint outrate;
	gboolean success = TRUE;

	s = gst_caps_get_structure(incaps, 0);
	if(!gst_structure_get_int(s, "rate", &inrate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, incaps);
		success = FALSE;
	}

	s = gst_caps_get_structure(outcaps, 0);
	if(!gst_structure_get_int(s, "rate", &outrate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
		success = FALSE;
	}

	/* FIXME, this logic should be moved to the transform caps */

	if (outrate > inrate) {
		GST_DEBUG_OBJECT(element, "outrate must be less than or equal to inrate %" GST_PTR_FORMAT " %" GST_PTR_FORMAT, outcaps, incaps);
		success = FALSE;
	}

	if (outrate < (inrate / (gint) element->n)) {
		GST_DEBUG_OBJECT(element, "outrate must be greater than or equal to in rate / n %" GST_PTR_FORMAT " %" GST_PTR_FORMAT, outcaps, incaps);
		success = FALSE;
	}

	if(success) {
		element->rate = inrate;
		element->outrate = outrate;
	}

	return success;
}

/*
 * start()
 */

static gboolean start(GstBaseTransform *trans)
{
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(trans);
	element->adapter = gst_adapter_new();
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
        if (!(element->outfft)) {
		gstlal_fftw_lock();
		element->outfft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (element->n+1));

		element->infft = (double *) fftw_malloc(sizeof(double) * (2 * element->n));
		element->fftplan = fftw_plan_dft_r2c_1d((int) element->n, (double *) element->infft, (fftw_complex *) element->outfft, FFTW_MEASURE);
		gstlal_fftw_unlock();
	}

	return TRUE;
}

/*
 * stop()
 */

static gboolean stop(GstBaseTransform *trans)
{
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(trans);
	g_object_unref(element->adapter);
	element->adapter = NULL;
	return TRUE;
}

/*
 * transform()
 */

static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(trans);
	guint64 in_length;
	GstFlowReturn result;

	/* check for discontinuity */

	if(GST_BUFFER_IS_DISCONT(inbuf)) {

		/* flush adapter */

		gst_adapter_clear(element->adapter);

		/* (re)sync timestamp and offset book-keeping */

		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = element->offset0;

		/* be sure to flag the next output buffer as a discontinuity */

		element->need_discont = TRUE;
	}


	/* gap logic */

	/* FIXME no computation is saved during a gap, this might be wrong */

	in_length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	if (GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		push_zeros(element, in_length);
	} else {
		gst_buffer_ref(inbuf);
		gst_adapter_push(element->adapter, inbuf);
	}


	result = filter(element, outbuf, in_length);

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
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_N: {
		guint32 old_n = element->n;
		element->n = g_value_get_uint(value);
		if(element->n != old_n)
			g_object_notify(object, "n");
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
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_N:
		g_value_set_uint(value, element->n);
		break;

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
	GSTLALSpecgram *element = GSTLAL_SPECGRAM(object);


	/* free resources */

	if(element->adapter) {
		g_object_unref(element->adapter);
		element->adapter = NULL;
	}

	if (element->fftplan) {
		gstlal_fftw_lock();
		fftw_destroy_plan(element->fftplan);
		gstlal_fftw_unlock();
	}

	if (element->outfft) {
		fftw_free(element->outfft);
	}

	if (element->workspacefft) {
		fftw_free(element->workspacefft);
	}
	/* chain to parent class' finalize() method */

	G_OBJECT_CLASS(parent_class)->finalize(object);
}

/*
 * base_init()
 */

static void gstlal_specgram_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "Spectrogram", "Filter/Audio", 
	"produce n/2 channels of frequency power\n"
	"\tNote that you can change the rate at which\n"
	"\tthe output is generated by using a capsfilter.\n"
	"\tThe resulting FFTs will have less overlap and\n"
	"\tthe code will run faster. The output sample rate\n" 
	"\tmust be lower than the input sample rate and \n"
	"\thigher than the input sample rate / n", 
	"Chad Hanna <chad.hanna@ligo.org>");
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->fixate_caps = GST_DEBUG_FUNCPTR(fixate_caps);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->stop = GST_DEBUG_FUNCPTR(stop);
}

/*
 * class_init()
 */

static void gstlal_specgram_class_init(GSTLALSpecgramClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_N,
		g_param_spec_uint(
			"n",
			"n",
			"Number of samples to include in FFT",
			0, G_MAXUINT, DEFAULT_N,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}

/*
 * init()
 */

static void gstlal_specgram_init(GSTLALSpecgram *filter, GSTLALSpecgramClass *kclass)
{
	filter->rate = 0;
	filter->adapter = NULL;
	filter->n = DEFAULT_N;
	filter->infft = NULL;
	filter->outfft = NULL;
	filter->workspacefft = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

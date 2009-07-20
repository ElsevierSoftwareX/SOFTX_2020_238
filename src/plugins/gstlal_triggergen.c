#include <complex.h>
#include <math.h>
#include <glib.h>
#include <gst/gst.h>
#include <gsl/gsl_matrix.h>
#include <gstlal.h>
#include <gstlal_triggergen.h>
#include <gst/base/gstbasesink.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LALStdlib.h>


/*
 * ============================================================================
 *
 *                             Trigger Generator
 *
 * ============================================================================
 */


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_SNR_THRESH 5.5


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


static double eta(double m1, double m2)
{
	return m1 * m2 / pow(m1 + m2, 2);
}


static double mchirp(double m1, double m2)
{
	return pow(m1 * m2, 0.6) / pow(m1 + m2, 0.2);
}


static double effective_distance(double snr, double sigmasq)
{
	return sqrt(sigmasq) / snr;
}


static int setup_bankfile_input(GSTLALTriggerGen *element, char *bank_filename)
{
	SnglInspiralTable *bank = NULL;
	int i;

	element->bank_filename = bank_filename;
	element->num_templates = LALSnglInspiralTableFromLIGOLw(&bank, element->bank_filename, -1, -1);
	element->bank = calloc(element->num_templates, sizeof(*element->bank));
	if(!element->bank) {
		/* FIXME:  free template bank */
		return 0;
	}

	for(i = 0; bank; i++) {
		SnglInspiralTable *prev = bank;
		element->bank[i] = *bank;
		element->bank[i].next = NULL;
		bank = bank->next;
		free(prev);
	}

	return 0;
}


static void free_bankfile(GSTLALTriggerGen *element)
{
	if(element->bank_filename) {
		free(element->bank_filename);
		element->bank_filename = NULL;
	}
	if(element->bank) {
		free(element->bank);
		element->bank = NULL;
	}
	element->num_templates = 0;
}


static SnglInspiralTable *new_event(SnglInspiralTable *dest, LIGOTimeGPS end_time, double complex z, double complex chisq, int channel, GSTLALTriggerGen *element)
{
	double xi;

	*dest = element->bank[channel];

	dest->snr = cabs(z);
	dest->coa_phase = carg(z);
	dest->chisq = creal(chisq) + cimag(chisq);
	dest->chisq_dof = 1;
	dest->end_time = end_time;
	dest->end_time_gmst = XLALGreenwichMeanSiderealTime(&end_time);
	dest->eff_distance = effective_distance(dest->snr, dest->sigmasq);

	xi = dest->chisq / (1 + 0.1 * dest->snr * dest->snr);

	return dest;
}


/*
 * ============================================================================
 *
 *                                    Caps
 *
 * ============================================================================
 */


static void gen_set_bytes_per_sample(GstPad *pad, GstCaps *caps)
{
	GstStructure *structure;
	gint width, channels;

	structure = gst_caps_get_structure(caps, 0);
	gst_structure_get_int(structure, "width", &width);
	gst_structure_get_int(structure, "channels", &channels);

	gstlal_collect_pads_set_bytes_per_sample(pad, (width / 8) * channels);
}


static gboolean gen_setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(GST_PAD_PARENT(pad));
	GstStructure *structure;
	gboolean result = TRUE;

	GST_OBJECT_LOCK(element);
	gen_set_bytes_per_sample(pad, caps);
	structure = gst_caps_get_structure(caps, 0);
	gst_structure_get_int(structure, "rate", &element->rate);
	GST_OBJECT_UNLOCK(element);
	return result;
}


/*
 * ============================================================================
 *
 *                              Event Generation
 *
 * ============================================================================
 */


static GstFlowReturn gen_collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(user_data);
	guint64 earliest_input_offset, earliest_input_offset_end;
	GstBuffer *snrbuf;
	GstBuffer *chisqbuf;
	GstBuffer *srcbuf;
	LIGOTimeGPS epoch;
	const double complex *snrdata;
	const double complex *chisqdata;
	int length;
	int sample, channel;
	SnglInspiralTable event;
	GstFlowReturn result;

	/*
	 * check for new segment
	 */

	if(element->segment_pending) {
		GstEvent *event;
		GstSegment *segment = gstlal_collect_pads_get_segment(element->collect);
		if(!segment) {
			/* FIXME:  failure getting bounding segment, do
			 * something about it */
		}
		element->segment = *segment;
		gst_segment_free(segment);
		element->offset = GST_BUFFER_OFFSET_NONE;

		event = gst_event_new_new_segment_full(FALSE, element->segment.rate, 1.0, GST_FORMAT_TIME, element->segment.start, element->segment.stop, element->segment.start);
		if(!event) {
			/* FIXME:  failure getting event, do something
			 * about it */
		}
		gst_pad_push_event(element->srcpad, event);

		element->segment_pending = FALSE;
	}

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	if(!gstlal_collect_pads_get_earliest_offsets(element->collect, &earliest_input_offset, &earliest_input_offset_end, element->rate, element->segment.start)) {
		GST_ERROR_OBJECT(element, "cannot deduce input timestamp offset information");
		return GST_FLOW_ERROR;
	}

	/*
	 * check for EOS
	 */

	if(earliest_input_offset == GST_BUFFER_OFFSET_NONE)
		goto eos;

	/*
	 * don't let time go backwards.  in principle we could be smart and
	 * handle this, but the audiorate element can be used to correct
	 * screwed up time series so there is no point in re-inventing its
	 * capabilities here.
	 */

	if((element->offset != GST_BUFFER_OFFSET_NONE) && (earliest_input_offset < element->offset)) {
		GST_ERROR_OBJECT(element, "detected time reversal in at least one input stream:  expected nothing earlier than offset %lu, found sample at offset %lu", element->offset, earliest_input_offset);
		return GST_FLOW_ERROR;
	}

	/*
	 * get buffers upto the desired end offset.
	 */

	snrbuf = gstlal_collect_pads_take_buffer(pads, element->snrcollectdata, earliest_input_offset_end, element->rate, element->segment.start);
	chisqbuf = gstlal_collect_pads_take_buffer(pads, element->chisqcollectdata, earliest_input_offset_end, element->rate, element->segment.start);

	/*
	 * NULL means EOS.
	 */

	if(!snrbuf || !chisqbuf) {
		/* FIXME:  handle EOS */
	}

	/*
	 * FIXME:  rethink the collect pads system so that this doesn't
	 * happen  (I think the second part already cannot happen because
	 * we get the collect pads system to tell us the upper bound of
	 * available offsets before asking for the buffers, but need to
	 * check this)
	 */

	if(GST_BUFFER_OFFSET(snrbuf) != GST_BUFFER_OFFSET(chisqbuf) || GST_BUFFER_OFFSET_END(snrbuf) != GST_BUFFER_OFFSET_END(chisqbuf)) {
		gst_buffer_unref(snrbuf);
		gst_buffer_unref(chisqbuf);
		GST_ERROR_OBJECT(element, "misaligned buffer boundaries:  requested offsets upto %lu, got snr offsets %lu--%lu and \\chi^{2} offsets %lu--%lu", earliest_input_offset_end, GST_BUFFER_OFFSET(snrbuf), GST_BUFFER_OFFSET_END(snrbuf), GST_BUFFER_OFFSET(chisqbuf), GST_BUFFER_OFFSET_END(chisqbuf));
		return GST_FLOW_ERROR;
	}

	/*
	 * GAP --> no-op
	 */

	if(GST_BUFFER_FLAG_IS_SET(snrbuf, GST_BUFFER_FLAG_GAP) || GST_BUFFER_FLAG_IS_SET(chisqbuf, GST_BUFFER_FLAG_GAP)) {
		srcbuf = gst_buffer_new();
		gst_buffer_copy_metadata(srcbuf, snrbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
		if((element->offset == GST_BUFFER_OFFSET_NONE) || (element->offset != GST_BUFFER_OFFSET(srcbuf)))
			GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);
		gst_buffer_unref(snrbuf);
		gst_buffer_unref(chisqbuf);
		element->offset = GST_BUFFER_OFFSET_END(srcbuf);
		return gst_pad_push(element->srcpad, srcbuf);
	}

	/*
	 * Find events
	 */

	XLALINT8NSToGPS(&epoch, GST_BUFFER_TIMESTAMP(snrbuf));
	length = GST_BUFFER_OFFSET_END(snrbuf) - GST_BUFFER_OFFSET(snrbuf);
	snrdata = (const double complex *) GST_BUFFER_DATA(snrbuf);
	chisqdata = (const double complex *) GST_BUFFER_DATA(chisqbuf);

	event.snr = 0;

	for(sample = 0; sample < length; sample++) {
		for(channel = 0; channel < element->num_templates; channel++) {
			int index = sample * element->num_templates + channel;
			if(cabs(snrdata[index]) > event.snr) {
				LIGOTimeGPS t = epoch;
				XLALGPSAdd(&t, (double) sample / element->rate);
				new_event(&event, t, snrdata[index], chisqdata[index], channel, element);
			}
		}
	}

	/*
	 * Push event downstream
	 */

	if(event.snr > element->snr_thresh) {
		result = gst_pad_alloc_buffer(element->srcpad, GST_BUFFER_OFFSET(snrbuf), sizeof(event), GST_PAD_CAPS(element->srcpad), &srcbuf);
		gst_buffer_copy_metadata(srcbuf, snrbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);
		memcpy(GST_BUFFER_DATA(srcbuf), &event, sizeof(event));
	} else {
		srcbuf = gst_buffer_new();
		gst_buffer_copy_metadata(srcbuf, snrbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
	}
	if((element->offset == GST_BUFFER_OFFSET_NONE) || (element->offset != GST_BUFFER_OFFSET(srcbuf)))
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);

	gst_buffer_unref(snrbuf);
	gst_buffer_unref(chisqbuf);

	element->offset = GST_BUFFER_OFFSET_END(srcbuf);
	return gst_pad_push(element->srcpad, srcbuf);

eos:
	GST_DEBUG_OBJECT(element, "no data available (EOS)");
	gst_pad_push_event(element->srcpad, gst_event_new_eos());
	return GST_FLOW_UNEXPECTED;
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum gen_property {
	ARG_SNR_THRESH = 1, 
	ARG_BANK_FILENAME
};


static void gen_set_property(GObject *object, enum gen_property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);

	switch(id) {
	case ARG_SNR_THRESH:
		element->snr_thresh = g_value_get_double(value);
		break;

	case ARG_BANK_FILENAME:
		free_bankfile(element);
		setup_bankfile_input(element, g_value_dup_string(value));
		break;
	}
}


static void gen_get_property(GObject * object, enum gen_property id, GValue * value, GParamSpec * pspec)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);

	switch(id) {
	case ARG_SNR_THRESH:
		g_value_set_double(value,element->snr_thresh);
		break;

	case ARG_BANK_FILENAME:
		g_value_set_string(value,element->bank_filename);
		break;
	}
}


/*
 * ============================================================================
 *
 *                              Element Support
 *
 * ============================================================================
 */


static GstElementClass *gen_parent_class = NULL;


static void gen_finalize(GObject *object)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
	free_bankfile(element);
	G_OBJECT_CLASS(gen_parent_class)->finalize(object);
}


static GstStateChangeReturn gen_change_state(GstElement *element, GstStateChange transition)
{
	GSTLALTriggerGen *triggergen = GSTLAL_TRIGGERGEN(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		triggergen->segment_pending = TRUE;
		gst_segment_init(&triggergen->segment, GST_FORMAT_UNDEFINED);
		triggergen->offset = GST_BUFFER_OFFSET_NONE;
		gst_collect_pads_start(triggergen->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(triggergen->collect);
		break;

	default:
		break;
	}

	return gen_parent_class->change_state(element, transition);
}


static void gen_base_init(gpointer g_class)
{
	static GstElementDetails plugin_details = {
		"Trigger Generator",
		"Filter",
		"SNRs in Triggers out",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"
	};

	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);
	gst_element_class_set_details (element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"snr",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) [1, MAX], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
			)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"chisquare",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) [1, MAX], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
			)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-snglinspiral"
			)
		)
	);
}


static void gen_class_init(gpointer klass, gpointer class_data)
{
        GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

        gen_parent_class = g_type_class_ref(GST_TYPE_ELEMENT);
	gobject_class->set_property = gen_set_property;
	gobject_class->get_property = gen_get_property;
        gobject_class->finalize = gen_finalize;
	gstelement_class->change_state = gen_change_state;

        g_object_class_install_property(gobject_class, ARG_BANK_FILENAME, g_param_spec_string("bank-filename", "Bank file name", "Path to XML file used to generate the template bank", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SNR_THRESH, g_param_spec_double("snr-thresh", "SNR Threshold", "SNR Threshold that determines a trigger", 0, G_MAXDOUBLE, DEFAULT_SNR_THRESH, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}


static void gen_instance_init(GTypeInstance *object, gpointer klass)
{
        GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
        GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));
	element->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(element->collect, gen_collected, element);

        /* configure snr pad */
        pad = gst_element_get_static_pad(GST_ELEMENT(element), "snr");
        gst_pad_set_setcaps_function(pad, gen_setcaps);
	element->snrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->snrcollectdata));
	element->snrpad = pad;

	/* configure chisquare pad */
        pad = gst_element_get_static_pad(GST_ELEMENT(element), "chisquare");
        gst_pad_set_setcaps_function(pad, gen_setcaps);
	element->chisqcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->chisqcollectdata));
	element->chisqpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	element->srcpad = pad;

        /* internal data */
	element->rate = 0;
	element->bank_filename = NULL;
        element->bank = NULL;
	element->num_templates = 0;
	element->snr_thresh = DEFAULT_SNR_THRESH;
}


GType gstlal_triggergen_get_type(void)
{
        static GType type = 0;

        if(!type) {
                static const GTypeInfo info = {
                        .class_size = sizeof(GSTLALTriggerGenClass),
                        .class_init = gen_class_init,
                        .base_init = gen_base_init,
                        .instance_size = sizeof(GSTLALTriggerGen),
                        .instance_init = gen_instance_init,
                };
                type = g_type_register_static(GST_TYPE_ELEMENT, "lal_triggergen", &info, 0);
        }

        return type;
}


/*
 * ============================================================================
 *
 *                             Trigger XML Writer
 *
 * ============================================================================
 */


static gboolean xmlwriter_start(GstBaseSink *sink)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	memset(&status, 0, sizeof(status));

	element->xml = XLALOpenLIGOLwXMLFile(element->location);
	LALBeginLIGOLwXMLTable(&status, element->xml, sngl_inspiral_table);

	return TRUE;
}


static gboolean xmlwriter_stop(GstBaseSink *sink)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	memset(&status, 0, sizeof(status));

	if(element->xml) {
		LALEndLIGOLwXMLTable(&status, element->xml);
		XLALCloseLIGOLwXMLFile(element->xml);
		element->xml = NULL;
	}

	return TRUE;
}


static GstFlowReturn xmlwriter_render(GstBaseSink *sink, GstBuffer *buffer)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	MetadataTable table = {
		.snglInspiralTable = (SnglInspiralTable *) GST_BUFFER_DATA(buffer)
	};
	memset(&status, 0, sizeof(status));

	if(!GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_GAP))
		LALWriteLIGOLwXMLTable(&status, element->xml, table, sngl_inspiral_table);

	return GST_FLOW_OK;
}


enum xmlwriter_property {
	ARG_LOCATION = 1
};


static void xmlwriter_set_property(GObject * object, enum xmlwriter_property id, const GValue * value, GParamSpec * pspec)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	switch(id) {
	case ARG_LOCATION:
		free(element->location);
		element->location = g_value_dup_string(value);
		break;
	}
}


static void xmlwriter_get_property(GObject * object, enum xmlwriter_property id, GValue * value, GParamSpec * pspec)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	switch(id) {
	case ARG_LOCATION:
		g_value_set_string(value,element->location);
		break;
	}
}


static GstBaseSink *xmlwriter_parent_class = NULL;


static void xmlwriter_base_init(gpointer g_class)
{
	static GstElementDetails plugin_details = {
		"Trigger XML Writer",
		"Sink/File",
		"Writes LAL's SnglInspiralTable C structures to an XML file",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"
	};

	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);
	gst_element_class_set_details (element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-snglinspiral"
			)
		)
	);
}


static void xmlwriter_class_init(gpointer klass, gpointer class_data)
{
        GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

        xmlwriter_parent_class = g_type_class_ref(GST_TYPE_BASE_SINK);
	gobject_class->set_property = xmlwriter_set_property;
	gobject_class->get_property = xmlwriter_get_property;
	gstbasesink_class->start = xmlwriter_start;
	gstbasesink_class->stop = xmlwriter_stop;
	gstbasesink_class->render = xmlwriter_render;

        g_object_class_install_property(gobject_class, ARG_LOCATION, g_param_spec_string("location", "filename", "Path to output file", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}


static void xmlwriter_instance_init(GTypeInstance *object, gpointer klass)
{
        GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	/*
	 * Internal data
	 */

	element->location = NULL;
	element->xml = NULL;
}


GType gstlal_triggerxmlwriter_get_type(void)
{
        static GType type = 0;

        if(!type) {
                static const GTypeInfo info = {
                        .class_size = sizeof(GSTLALTriggerXMLWriterClass),
                        .class_init = xmlwriter_class_init,
                        .base_init = xmlwriter_base_init,
                        .instance_size = sizeof(GSTLALTriggerXMLWriter),
                        .instance_init = xmlwriter_instance_init,
                };
                type = g_type_register_static(GST_TYPE_BASE_SINK, "lal_triggerxmlwriter", &info, 0);
        }

        return type;
}

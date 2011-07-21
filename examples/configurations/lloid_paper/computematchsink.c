/*
 * Copyright (C) 2011  Leo Singer
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

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

#include <stdio.h>
#include <complex.h>
#include <math.h>

#define GST_TYPE_COMPUTE_MATCH_SINK (gst_compute_match_sink_get_type())
#define GST_COMPUTE_MATCH_SINK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_COMPUTE_MATCH_SINK, GstComputeMatchSink))
#define GST_IS_COMPUTE_MATCH_SINK(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_COMPUTE_MATCH_SINK))
#define GST_COMPUTE_MATCH_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_COMPUTE_MATCH_SINK, GstComputeMatchSinkClass))
#define GST_IS_COMPUTE_MATCH_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_COMPUTE_MATCH_SINK))
#define GST_COMPUTE_MATCH_SINK_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_COMPUTE_MATCH_SINK, GstComputeMatchSinkClass))

typedef struct {
	GstBaseSinkClass parent_class;
} GstComputeMatchSinkClass;

typedef struct {
	GstBaseSink element;

	/* properties */
	char *input_filename;
	char *output_filename;

	/* private */
	FILE *fid;
	int channels;
	double *impulse_norm;
	double *template_norm;
	double complex *dotprod;
} GstComputeMatchSink;

GST_BOILERPLATE(GstComputeMatchSink, gst_compute_match_sink, GstBaseSink, GST_TYPE_BASE_SINK)

/*
 * ========================================================================
 *
 *							 Utility Functions
 *
 * ========================================================================
 */


static double cnorm(double complex z)
{
	double re = creal(z);
	double im = cimag(z);
	return re * re + im * im;
}


/*
 * ============================================================================
 *
 *								 Properties
 *
 * ============================================================================
 */


enum property {
	PROP_INPUT_FILENAME = 1,
	PROP_OUTPUT_FILENAME
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GstComputeMatchSink *element = GST_COMPUTE_MATCH_SINK(object);

	GST_OBJECT_LOCK(element);

	switch (id)
	{
		case PROP_INPUT_FILENAME:
			g_free(element->input_filename);
			element->input_filename = g_value_dup_string(value);
			break;
		case PROP_OUTPUT_FILENAME:
			g_free(element->output_filename);
			element->output_filename = g_value_dup_string(value);
			break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GstComputeMatchSink *element = GST_COMPUTE_MATCH_SINK(object);

	GST_OBJECT_LOCK(element);

	switch (id) {
		case PROP_INPUT_FILENAME:
			g_value_set_string(value, element->input_filename);
			break;
		case PROP_OUTPUT_FILENAME:
			g_value_set_string(value, element->output_filename);
			break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *						GstBaseSink Method Overrides
 *
 * ============================================================================
 */


static gboolean set_caps(GstBaseSink *sink, GstCaps *caps)
{
	GstComputeMatchSink *element = GST_COMPUTE_MATCH_SINK(sink);

	return gst_structure_get_int(gst_caps_get_structure(caps, 0), "channels", &element->channels);
}


static gboolean start(GstBaseSink *sink)
{
	GstComputeMatchSink *element = GST_COMPUTE_MATCH_SINK(sink);

	element->fid = fopen(element->input_filename, "r");
	if (!element->fid)
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, ("could not open file %s", element->input_filename), GST_ERROR_SYSTEM);

	return (element->fid != NULL);
}

static GstFlowReturn render(GstBaseSink *sink, GstBuffer *buf)
{
	GstComputeMatchSink *element = GST_COMPUTE_MATCH_SINK(sink);
	double complex *impulse_data;
	double complex *template_data;
	double complex *template_data_ptr;
	gint64 len;

	if (!element->impulse_norm)
		element->impulse_norm = g_new0(double, element->channels);
	if (!element->template_norm)
		element->template_norm = g_new0(double, element->channels);
	if (!element->dotprod)
		element->dotprod = g_new0(double complex, element->channels);

	impulse_data = (double complex *) GST_BUFFER_DATA(buf);
	len = GST_BUFFER_SIZE(buf) / sizeof(*impulse_data);
	g_assert(2 * len % element->channels == 0);

	template_data = g_malloc(GST_BUFFER_SIZE(buf));
	template_data_ptr = template_data;
	len = fread(template_data, sizeof(*template_data), len, element->fid);
	g_assert(2 * len % element->channels == 0);

	for (; len > 0;)
	{
		double *impulse_norm = element->impulse_norm;
		double *template_norm = element->template_norm;
		double complex *dotprod = element->dotprod;
		int channel = 0;

		for (; channel < element->channels; channel++)
		{
			*(dotprod++) += (*impulse_data) * conj(*template_data);
			*(impulse_norm++) += cnorm(*(impulse_data++));
			*(template_norm++) += cnorm(*(template_data++));
			len--;
		}
	}

	g_free(template_data_ptr);
	return GST_FLOW_OK;
}


static gboolean event(GstBaseSink *sink, GstEvent *event)
{
	GstComputeMatchSink *element = GST_COMPUTE_MATCH_SINK(sink);
	gboolean success = TRUE;

	if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
	{
		double *impulse_norm = element->impulse_norm;
		double *template_norm = element->template_norm;
		double complex *dotprod = element->dotprod;
		FILE *fid;
		int channel;

		if (element->fid)
		{
			if (fclose(element->fid))
			{
				success = FALSE;
				GST_ELEMENT_ERROR(element, RESOURCE, CLOSE, ("could not close file %s", element->input_filename), GST_ERROR_SYSTEM);
			}
			element->fid = NULL;
		}


		if (impulse_norm && template_norm && dotprod)
		{
			if (success)
			{
				fid = fopen(element->output_filename, "w");
				if (fid)
				{
					for (channel = 0; channel < element->channels; channel++)
						fprintf(fid, "%0.23f\n", sqrt(cnorm(*(dotprod++)) / ((*(impulse_norm++)) * (*(template_norm++)))));
					if (fclose(fid))
					{
						success = FALSE;
						GST_ELEMENT_ERROR(element, RESOURCE, CLOSE, ("could not close file %s", element->output_filename), GST_ERROR_SYSTEM);
					}
				} else {
					success = FALSE;
					GST_ELEMENT_ERROR(element, RESOURCE, OPEN_WRITE, ("could not open file %s", element->output_filename), GST_ERROR_SYSTEM);
				}
			}

			g_free(element->impulse_norm);
			element->impulse_norm = NULL;
			g_free(element->template_norm);
			element->template_norm = NULL;
			g_free(element->dotprod);
			element->dotprod = NULL;
		}
	}

	return success;
}


/*
 * ============================================================================
 *
 *								Type Support
 *
 * ============================================================================
 */


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GstComputeMatchSink *element = GST_COMPUTE_MATCH_SINK(object);

	g_free(element->input_filename);
	element->input_filename = NULL;
	g_free(element->output_filename);
	element->output_filename = NULL;
	g_free(element->impulse_norm);
	element->impulse_norm = NULL;
	g_free(element->template_norm);
	element->template_norm = NULL;
	g_free(element->dotprod);
	element->dotprod = NULL;
	if (element->fid)
	{
		fclose(element->fid);
		element->fid = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void gst_compute_match_sink_base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Compute Match Sink",
		"Sink",
		"Special-purpose sink element",
		"Leo Singer <leo.singer@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"channels = (int) [1, MAX], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void gst_compute_match_sink_class_init(GstComputeMatchSinkClass *class)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(class);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	gstbasesink_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(render);
	gstbasesink_class->event = GST_DEBUG_FUNCPTR(event);

	g_object_class_install_property(
		gobject_class,
		PROP_INPUT_FILENAME,
		g_param_spec_string(
			"input-filename",
			"Input File Name",
			"Name of input file",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_OUTPUT_FILENAME,
		g_param_spec_string(
			"output-filename",
			"Output File Name",
			"Name of output file",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void gst_compute_match_sink_init(GstComputeMatchSink *object, GstComputeMatchSinkClass *class)
{
	GstBaseSink *basesink = GST_BASE_SINK(object);

	object->input_filename = NULL;
	object->output_filename = NULL;
	object->impulse_norm = NULL;
	object->template_norm = NULL;
	object->dotprod = NULL;
	object->fid = NULL;

	gst_pad_use_fixed_caps(GST_BASE_SINK_PAD(basesink));
}


static gboolean plugin_init(GstPlugin *plugin)
{
	return gst_element_register(plugin, "computematchsink", GST_RANK_NONE, GST_TYPE_COMPUTE_MATCH_SINK);
}


#define PACKAGE ""
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, "computematch", "special purpose plugin", plugin_init, "", "GPL", "", "http://www.lsc-group.phys.uwm.edu/daswg")

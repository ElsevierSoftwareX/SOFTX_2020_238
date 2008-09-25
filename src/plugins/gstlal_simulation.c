/*
 * An interface to LALSimulation.  
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


/*
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <stdint.h>
#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


/*
 * stuff from LAL
 */


#include <lal/LALComplex.h>
#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/LALSimulation.h>
#include <lal/LALSimInspiral.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/GenerateBurst.h>
#include <lal/FindChirp.h>
#include <lal/Date.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/Units.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_simulation.h>


/*
 * ============================================================================
 *
 *                                 XML Input
 *
 * ============================================================================
 */


/*
 * Clean up
 */


static void destroy_injection_document(struct injection_document *doc)
{
	if(doc) {
		XLALDestroyProcessTable(doc->process_table_head);
		XLALDestroyProcessParamsTable(doc->process_params_table_head);
		XLALDestroySearchSummaryTable(doc->search_summary_table_head);
		XLALDestroySimBurstTable(doc->sim_burst_table_head);
		while(doc->sim_inspiral_table_head) {
			SimInspiralTable *next = doc->sim_inspiral_table_head->next;
			XLALFree(doc->sim_inspiral_table_head);
			doc->sim_inspiral_table_head = next;
		}
	}
}


/*
 * Load document
 */


static struct injection_document *load_injection_document(const char *filename, LIGOTimeGPS start, LIGOTimeGPS end, double longest_injection)
{
	static const char func[] = "load_injection_document";
	struct injection_document *new;

	/*
	 * allocate the document
	 */

	new = malloc(sizeof(*new));
	if(!new) {
		XLALPrintError("%s(): malloc() failed\n", func);
		XLAL_ERROR_NULL(func, XLAL_ENOMEM);
	}

	/*
	 * adjust start and end times
	 */

	XLALGPSAdd(&start, -longest_injection);
	XLALGPSAdd(&end, longest_injection);

	/*
	 * load required tables
	 */

	new->process_table_head = XLALProcessTableFromLIGOLw(filename);
	if(!new->process_table_head)
		XLALPrintError("%s(): failure reading process table\n", func);
	else
		XLALPrintInfo("%s(): found process table\n", func);
	new->process_params_table_head = XLALProcessParamsTableFromLIGOLw(filename);
	if(!new->process_params_table_head)
		XLALPrintError("%s(): failure reading process_params table\n", func);
	else
		XLALPrintInfo("%s(): found process_params table\n", func);
	new->search_summary_table_head = XLALSearchSummaryTableFromLIGOLw(filename);
	if(!new->search_summary_table_head)
		XLALPrintError("%s(): non-fatal failure reading search_summary table\n", func);
	else
		XLALPrintInfo("%s(): found search_summary table\n", func);

	/*
	 * load optional sim_burst table
	 */

	new->has_sim_burst_table = XLALLIGOLwHasTable(filename, "sim_burst");
	if(new->has_sim_burst_table) {
		new->sim_burst_table_head = XLALSimBurstTableFromLIGOLw(filename, &start, &end);
		if(!new->sim_burst_table_head)
			XLALPrintError("%s(): failure reading sim_burst table\n", func);
		else
			XLALPrintInfo("%s(): found sim_burst table\n", func);
	} else
		new->sim_burst_table_head = NULL;

	/*
	 * load optional sim_inspiral table
	 */

	new->has_sim_inspiral_table = XLALLIGOLwHasTable(filename, "sim_inspiral");
	if(new->has_sim_inspiral_table) {
		new->sim_inspiral_table_head = NULL;
		if(SimInspiralTableFromLIGOLw(&new->sim_inspiral_table_head, filename, start.gpsSeconds - 1, end.gpsSeconds + 1) < 0) {
			XLALPrintError("%s(): failure reading sim_inspiral table\n", func);
			new->sim_inspiral_table_head = NULL;
		} else
			XLALPrintInfo("%s(): found sim_inspiral table\n", func);
	} else
		new->sim_inspiral_table_head = NULL;

	/*
	 * did we get it all?
	 */

	if(
		!new->process_table_head ||
		!new->process_params_table_head ||
		/* FIXME:  lalapps_inspinj doesn't include this table */
		/*!new->search_summary_table_head || */
		(new->has_sim_burst_table && !new->sim_burst_table_head) ||
		(new->has_sim_inspiral_table && !new->sim_inspiral_table_head)
	) {
		XLALPrintError("%s(): document is incomplete and/or malformed\n", func);
		destroy_injection_document(new);
		XLAL_ERROR_NULL(func, XLAL_EFUNC);
	}

	/*
	 * success
	 */

	return new;
}


/*
 * This is the function that gets called.
 */


static int add_xml_injections(REAL8TimeSeries *h, const struct injection_document *injection_document, COMPLEX8FrequencySeries *response)
{
	static const char func[] = "add_xml_injections";
	LIGOTimeGPS start;
	LIGOTimeGPS end;

	/*
	 * Compute bounds of injection interval
	 */

	start = end = h->epoch;
	XLALGPSAdd(&end, h->data->length * h->deltaT);

	/*
	 * sim_burst
	 */

	if(injection_document->sim_burst_table_head) {
		XLALPrintInfo("%s(): computing sim_burst injections ...\n", func);
		if(XLALBurstInjectSignals(h, injection_document->sim_burst_table_head, NULL))
			XLAL_ERROR(func, XLAL_EFUNC);
		XLALPrintInfo("%s(): done\n", func);
	}

	/*
	 * sim_inspiral
	 */

	if(injection_document->sim_inspiral_table_head) {
		LALStatus stat;
		double underflow_protection = 1.0;
		COMPLEX8FrequencySeries *inspiral_response = NULL;
		REAL4TimeSeries *mdc;
		unsigned i;

		if(!response) {
			LALUnit strain_per_count = gstlal_lalStrainPerADCCount();
			underflow_protection = 1e-20;
			inspiral_response = XLALCreateCOMPLEX8FrequencySeries(NULL, &h->epoch, 0.0, 1.0 / (h->data->length * h->deltaT), &strain_per_count, h->data->length / 2 + 1);
			if(!inspiral_response)
				XLAL_ERROR(func, XLAL_EFUNC);
			for(i = 0; i < inspiral_response->data->length; i++)
				inspiral_response->data->data[i] = XLALCOMPLEX8Rect(underflow_protection, 0.0);
		}

		mdc = XLALCreateREAL4TimeSeries(h->name, &h->epoch, h->f0, h->deltaT, &h->sampleUnits, h->data->length);
		if(!mdc) {
			XLALDestroyCOMPLEX8FrequencySeries(inspiral_response);
			XLAL_ERROR(func, XLAL_EFUNC);
		}
		memset(mdc->data->data, 0, mdc->data->length * sizeof(*mdc->data->data));

		XLALPrintInfo("%s(): computing sim_inspiral injections ...\n", func);
		/* FIXME: figure out how to do error handling like this */
		/*LAL_CALL(LALFindChirpInjectSignals(&stat, mdc, injection_document->sim_inspiral_table_head, response), &stat);*/
		XLALClearErrno();
		memset(&stat, 0, sizeof(stat));
		LALFindChirpInjectSignals(&stat, mdc, injection_document->sim_inspiral_table_head, response ? response : inspiral_response);
		XLALPrintInfo("%s(): done\n", func);

		for(i = 0; i < h->data->length; i++)
			h->data->data[i] += mdc->data->data[i] * underflow_protection;

		XLALDestroyCOMPLEX8FrequencySeries(inspiral_response);
		XLALDestroyREAL4TimeSeries(mdc);
	}

	/*
	 * done
	 */

	return 0;
}


static REAL8TimeSeries *compute_strain(double right_ascension, double declination, double psi, LALDetector * detector, LIGOTimeGPS * tc, double phic, double deltaT, double m1, double m2, double fmin, double r, double i, int order)
{
	REAL8TimeSeries *hplus = NULL;
	REAL8TimeSeries *hcross = NULL;
	if(XLALSimInspiralPN(&hplus, &hcross, tc, phic, deltaT, m1, m2, fmin, r, i, order)) {
		XLALDestroyREAL8TimeSeries(hplus);
		XLALDestroyREAL8TimeSeries(hcross);
		return XLALSimDetectorStrainREAL8TimeSeries(hplus, hcross, right_ascension, declination, psi, detector);
	} else {
		XLALDestroyREAL8TimeSeries(hplus);
		XLALDestroyREAL8TimeSeries(hcross);
		return NULL;
	}

}


/*
 * ============================================================================
 *
 *                             GStreamer Element
 *
 * ============================================================================
 */


/*
 * Properties
 */


enum property {
	ARG_XML_LOCATION = 1
};


static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec)
{

	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	switch (id) {
	case ARG_XML_LOCATION:
		free(element->xml_location);
		element->xml_location = g_value_dup_string(value);
		destroy_injection_document(element->injection_document);
		element->injection_document = NULL;
		break;
	}
}

static void get_property(GObject * object, enum property id, GValue * value, GParamSpec * pspec)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	switch (id) {
	case ARG_XML_LOCATION:
		g_value_set_string(value, element->xml_location);
		break;
	}
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *buf)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;
	REAL8TimeSeries *h;

	/*
	 * Load injections if needed
	 */

	if(!element->injection_document) {
		LIGOTimeGPS start = {-999999999, 0};
		LIGOTimeGPS end = {+999999999, 0};
		/* earliest and latest possible LIGOTimeGPS */
		/* FIXME:  hard-coded = BAD BAD BAD */
		/*XLALINT8NSToGPS(&start, (INT8) 1 << 63);
		XLALINT8NSToGPS(&end, ((INT8) 1 << 63) - 1);*/
		element->injection_document = load_injection_document(element->xml_location, start, end, 0.0);
		if(!element->injection_document) {
			GST_ERROR("error loading \"%s\"", element->xml_location);
			gst_buffer_unref(buf);
			result = GST_FLOW_ERROR;
			goto done;
		}
	}

	/*
	 * Wrap buffer in a LAL REAL8TimeSeries.
	 */

	h = gstlal_REAL8TimeSeries_from_buffer(buf);
	if(!h) {
		GST_ERROR("failure wrapping buffer in REAL8TimeSeries");
		gst_buffer_unref(buf);
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Add injections
	 */

	if(add_xml_injections(h, element->injection_document, NULL) < 0) {
		GST_ERROR("failure performing injections");
		h->data->data = NULL;
		XLALDestroyREAL8TimeSeries(h);
		gst_buffer_unref(buf);
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Free the wrapping.  Setting the data pointer to NULL prevents
	 * XLALDestroyREAL8TimeSeries() from free()ing the buffer's data.
	 */

	h->data->data = NULL;
	XLALDestroyREAL8TimeSeries(h);

	/*
	 * Push data out srcpad
	 */

	result = gst_pad_push(element->srcpad, buf);

	/*
	 * Done
	 */

done:
	gst_object_unref(element);
	return result;
}


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance dispose function.  See ???
 */


static void dispose(GObject * object)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	free(element->xml_location);
	element->xml_location = NULL;
	destroy_injection_document(element->injection_document);

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static GstElementDetails plugin_details = {
		"Simulation",
		"Filter",
		"An injection routine",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <channa@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;

	g_object_class_install_property(gobject_class, ARG_XML_LOCATION, g_param_spec_string("xml-location", "XML Location", "Name of LIGO Light Weight XML file containing list(s) of software injections", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance * object, gpointer class)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(object), "src");

	/* internal data */
	element->xml_location = NULL;
	element->injection_document = NULL;
}


/*
 * gstlal_simulation_get_type().
 */


GType gstlal_simulation_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALSimulationClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALSimulation),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_simulation", &info, 0);
	}

	return type;
}

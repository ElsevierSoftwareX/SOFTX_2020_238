/*
 * An interface to LALSimulation.  
 *
 * Copyright (C) 2008--2015  Kipp Cannon, Chad Hanna, Drew Keppel
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
 * ========================================================================
 *
 *				  Preamble
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
#include <gst/base/gstbasetransform.h>


/*
 * stuff from LAL
 */


#include <lal/LALConfig.h>
#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/LALSimulation.h>
#include <lal/LALSimInspiral.h>
#include <lal/SnglBurstUtils.h>
#include <lal/LIGOLwXMLBurstRead.h>
#include <lal/LIGOLwXMLInspiralRead.h>
#include <lal/GenerateBurst.h>
#include <lal/FindChirp.h>
#include <lal/Date.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/Units.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal_simulation.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_simulation_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_simulation", 0, "lal_simulation element");
}


GST_BOILERPLATE_FULL(
	GSTLALSimulation,
	gstlal_simulation,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM,
	additional_initializations
);


/*
 * ============================================================================
 *
 *				 XML Input
 *
 * ============================================================================
 */


/*
 * Clean up
 */


static void destroy_injection_document(struct injection_document *doc)
{
	if(doc) {
		XLALDestroySimBurstTable(doc->sim_burst_table_head);
		doc->sim_burst_table_head = NULL;
		XLALDestroyTimeSlideTable(doc->time_slide_table_head);
		doc->time_slide_table_head = NULL;
		while(doc->sim_inspiral_table_head) {
			SimInspiralTable *next = doc->sim_inspiral_table_head->next;
			XLALFree(doc->sim_inspiral_table_head);
			doc->sim_inspiral_table_head = next;
		}
	}
	g_free(doc);
}


/*
 * Load document
 */


static struct injection_document *load_injection_document(const char *filename, LIGOTimeGPS start, LIGOTimeGPS end, double longest_injection)
{
	int success = 1;
	struct injection_document *new;
	int nrows; 

	g_assert(filename != NULL);

	/*
	 * allocate the document
	 */

	new = g_new0(struct injection_document, 1);
	if(!new) {
		XLALPrintError("%s(): malloc() failed\n", __func__);
		XLAL_ERROR_NULL(XLAL_ENOMEM);
	}

	/*
	 * adjust start and end times
	 */

	XLALGPSAdd(&start, -longest_injection);
	XLALGPSAdd(&end, longest_injection);

	/*
	 * load optional sim_burst and time_slide table
	 */

	new->has_sim_burst_table = XLALLIGOLwHasTable(filename, "sim_burst");
	if(new->has_sim_burst_table < 0) {
		XLALPrintError("%s(): error searching for sim_burst table in \"%s\": %s\n", __func__, filename, XLALErrorString(xlalErrno));
		XLALClearErrno();
		new->has_sim_burst_table = 0;
		new->sim_burst_table_head = NULL;
		success = 0;
	} else if(new->has_sim_burst_table) {
		XLALClearErrno();
		new->sim_burst_table_head = XLALSimBurstTableFromLIGOLw(filename, &start, &end);
		if(XLALGetBaseErrno()) {
			XLALPrintError("%s(): failure reading sim_burst table from \"%s\"\n", __func__, filename);
			success = 0;
		} else
			XLALPrintInfo("%s(): found sim_burst table\n", __func__);
		XLALSortSimBurst(&new->sim_burst_table_head, XLALCompareSimBurstByGeocentTimeGPS);
	} else
		new->sim_burst_table_head = NULL;

	new->has_time_slide_table = XLALLIGOLwHasTable(filename, "time_slide");
	if(new->has_time_slide_table < 0) {
		XLALPrintError("%s(): error searching for time_slide table in \"%s\": %s\n", __func__, filename, XLALErrorString(xlalErrno));
		XLALClearErrno();
		new->has_time_slide_table = 0;
		new->time_slide_table_head = NULL;
		success = 0;
	} else if(new->has_time_slide_table) {
		XLALClearErrno();
		new->time_slide_table_head = XLALTimeSlideTableFromLIGOLw(filename);
		if(XLALGetBaseErrno()) {
			XLALPrintError("%s(): failure reading time_slide table from \"%s\"\n", __func__, filename);
			success = 0;
		} else
			XLALPrintInfo("%s(): found time_slide table\n", __func__);
	} else
		new->time_slide_table_head = NULL;

	/*
	 * skipping burst injections if no time_slide table present
	 */

	if(new->has_sim_burst_table && !new->has_time_slide_table) {
		XLALPrintWarning("%s(): Skipping burst injections since \"%s\" must contain a time_slide table for them.\n", __func__, filename);
		XLALDestroySimBurstTable(new->sim_burst_table_head);
		new->sim_burst_table_head = NULL;
	}

	/*
	 * load optional sim_inspiral table
	 */

	new->has_sim_inspiral_table = XLALLIGOLwHasTable(filename, "sim_inspiral");
	if(new->has_sim_inspiral_table < 0) {
		XLALPrintError("%s(): error searching for sim_inspiral table in \"%s\": %s\n", __func__, filename, XLALErrorString(xlalErrno));
		XLALClearErrno();
		new->has_sim_inspiral_table = 0;
		new->sim_inspiral_table_head = NULL;
		success = 0;
	} else if(new->has_sim_inspiral_table) {
		new->sim_inspiral_table_head = NULL;
		/* FIXME: we need a smarter way to load the injections. We want to pick up injections that have an end_time
		 * after end if the beginning of the waveform starts before end.
		 */
		nrows = SimInspiralTableFromLIGOLw(&new->sim_inspiral_table_head, filename, start.gpsSeconds - 1, end.gpsSeconds + 1);
		if(nrows < 0) {
			XLALPrintError("%s(): failure reading sim_inspiral table from \"%s\"\n", __func__, filename);
			new->sim_inspiral_table_head = NULL;
			success = 0;
		} else {
			/* FIXME no rows found raises an error we don't care about, but why ? */
			XLALPrintInfo("%s(): found sim_inspiral table\n", __func__);
			XLALClearErrno();
		}
		XLALSortSimInspiral(&new->sim_inspiral_table_head, XLALCompareSimInspiralByGeocentEndTime);
	} else
		new->sim_inspiral_table_head = NULL;

	/*
	 * did we get it all?
	 */

	if(!success) {
		XLALPrintError("%s(): document is incomplete and/or malformed reading \"%s\"\n", __func__, filename);
		destroy_injection_document(new);
		XLAL_ERROR_NULL(XLAL_EFUNC);
	}

	/*
	 * success
	 */

	return new;
}


/*
 * Create detector strain from sim_inspiral
 */


static int sim_inspiral_strain(REAL8TimeSeries **strain, SimInspiralTable *sim_inspiral, double deltaT, LALDetector detector)
{
	REAL8TimeSeries *hplus = NULL;
	REAL8TimeSeries *hcross = NULL;

	/*
	 * create waveform using lalinspiral's wrapping of lalsimulation.
	 * XLALInspiralTDWaveformFromSimInspiral() translates the
	 * sim_inspiral row into a function call into lalsimulation,
	 * collecting and returning a conditioned waveform suitable for
	 * injection into an h(t) stream (after projection onto an antenna
	 * respose).
	 */

	if(XLALInspiralTDWaveformFromSimInspiral(&hplus, &hcross, sim_inspiral, deltaT) == XLAL_FAILURE)
		XLAL_ERROR(XLAL_EFUNC);

	/* add the time of the injection at the geocentre to the
	 * start times of the h+ and hx time series.  after this,
	 * their epochs mark the start of those time series at the
	 * geocentre. */

	XLALGPSAddGPS(&hcross->epoch, &sim_inspiral->geocent_end_time);
	XLALGPSAddGPS(&hplus->epoch, &sim_inspiral->geocent_end_time);

	/*
	 * project waveform onto detector
	 */

	*strain = XLALSimDetectorStrainREAL8TimeSeries(hplus, hcross, sim_inspiral->longitude, sim_inspiral->latitude, sim_inspiral->polarization, &detector);
	XLALDestroyREAL8TimeSeries(hplus);
	XLALDestroyREAL8TimeSeries(hcross);
	if(!(*strain))
		XLAL_ERROR(XLAL_EFUNC);
	return XLAL_SUCCESS;
}


static int update_simulation_series(REAL8TimeSeries *h, GSTLALSimulation *element, const COMPLEX16FrequencySeries *response)
{
	double injTime, DeltaT, startMinusEnd, endMinusStart;
	REAL8 tmpREAL8;
	LIGOTimeGPS hStartTime, hEndTime, injStartTime, injEndTime;
	SimInspiralTable *thisSimInspiral;
	SimInspiralTable *prevSimInspiral = NULL;
	/* skip inspiral injections whose estimated geocentre start times are more than this many
	 * seconds outside of the target time series */
	const double injection_window = 100.0;

	/* to be deduced from the h's channel name */
	const LALDetector *detector;

	/*
	 * get detector information
	 */

	detector = XLALDetectorPrefixToLALDetector(h->name);
	if(!detector)
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * calculate h segment boundaries, h = [hStartTime, hEndTime)
	 */

	hStartTime = h->epoch;
	hEndTime = hStartTime;
	XLALGPSAdd(&hEndTime, h->data->length * h->deltaT);

	/*
	 * initialize simulation_series if NULL
	 */


	if(element->simulation_series == NULL) {
		size_t series_length = 0;
		LIGOTimeGPS startEpoch = hStartTime;
		XLALGPSAdd(&startEpoch, -injection_window);
		tmpREAL8 = XLALGPSDiff(&hStartTime, &startEpoch) / h->deltaT;
		tmpREAL8 -= floor(tmpREAL8);
		XLALGPSAdd(&startEpoch, -tmpREAL8);
		series_length += XLALGPSDiff(&hEndTime, &startEpoch) / h->deltaT;
		series_length += ceil(injection_window / h->deltaT);
		element->simulation_series = XLALCreateREAL8TimeSeries(h->name, &startEpoch, h->f0, h->deltaT, &h->sampleUnits, series_length);
		memset(element->simulation_series->data->data, 0, element->simulation_series->data->length * sizeof(*element->simulation_series->data->data));
	}

	/*
	 * resize simulation_series to cover this time
	 */

	DeltaT = XLALGPSDiff(&element->simulation_series->epoch, &hEndTime);
	DeltaT += element->simulation_series->deltaT * element->simulation_series->data->length;
	if(DeltaT < injection_window)
		element->simulation_series = XLALResizeREAL8TimeSeries(element->simulation_series, 0, element->simulation_series->data->length + (int) ceil((injection_window - DeltaT) / element->simulation_series->deltaT));
	if(!element->simulation_series)
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * loop over injections in file
	 */

	for(thisSimInspiral = element->injection_document->sim_inspiral_table_head; thisSimInspiral; ) {
		REAL8TimeSeries *inspiral_series = NULL;

		/*
		 * calculate start and end times for this series containing
		 * this injection.  NOTE:  this just uses the 0 PN formula
		 * and pads it to be safe
		 */

		/* the padding will no longer be sufficient at this mass */
		g_assert((thisSimInspiral->mass1 + thisSimInspiral->mass2) < 1e4);

		injTime = 1.0 + XLALSimInspiralTaylorF2ReducedSpinChirpTime(thisSimInspiral->f_lower, thisSimInspiral->mass1 * LAL_MSUN_SI, thisSimInspiral->mass2 * LAL_MSUN_SI, 0, 0);
		injStartTime = injEndTime = thisSimInspiral->geocent_end_time;
		XLALGPSAdd(&injStartTime, -1.9*injTime - 1.0);
		XLALGPSAdd(&injEndTime, 0.1*injTime + 1.0);

		/*
		 * check whether injection segment intersects h
		 */

		startMinusEnd = XLALGPSDiff(&injStartTime, &hEndTime);
		endMinusStart = XLALGPSDiff(&injEndTime, &hStartTime);

		if(endMinusStart < -injection_window) /* injection ends before h */ {
			if(prevSimInspiral)
				prevSimInspiral->next = thisSimInspiral->next;
			if(thisSimInspiral == element->injection_document->sim_inspiral_table_head)
				element->injection_document->sim_inspiral_table_head = thisSimInspiral->next;
			SimInspiralTable *tmpSimInspiral = thisSimInspiral;
			thisSimInspiral = thisSimInspiral->next;
			XLALFree(tmpSimInspiral);
			continue;
		}

		if(startMinusEnd > injection_window) /* injection starts after h */ {
			prevSimInspiral = thisSimInspiral;
			thisSimInspiral = thisSimInspiral->next;
			continue;
		}

		/*
		 * compute injection waveform
		 */

		if(sim_inspiral_strain(&inspiral_series, thisSimInspiral, h->deltaT, *detector))
			XLAL_ERROR(XLAL_EFUNC);

		/*
		 * resize simulation_series to cover this time
		 */

		DeltaT = XLALGPSDiff(&inspiral_series->epoch, &element->simulation_series->epoch);
		DeltaT += inspiral_series->data->length * inspiral_series->deltaT + injection_window;
		DeltaT -= element->simulation_series->data->length * element->simulation_series->deltaT;
		if(DeltaT > 0.)
			element->simulation_series = XLALResizeREAL8TimeSeries(element->simulation_series, 0, element->simulation_series->data->length + ceil(DeltaT / element->simulation_series->deltaT));
		if(!element->simulation_series) {
			XLALDestroyREAL8TimeSeries(inspiral_series);
			XLAL_ERROR(XLAL_EFUNC);
		}

		/*
		 * add detector strain to simulation_series
		 */

		if(XLALSimAddInjectionREAL8TimeSeries(element->simulation_series, inspiral_series, response)) {
			XLALDestroyREAL8TimeSeries(inspiral_series);
			XLAL_ERROR(XLAL_EFUNC);
		}
		XLALDestroyREAL8TimeSeries(inspiral_series);

		/*
		 * remove injection from list and continue
		 */

		if(prevSimInspiral)
			prevSimInspiral->next = thisSimInspiral->next;
		{
		SimInspiralTable *tmpSimInspiral = thisSimInspiral;
		thisSimInspiral = thisSimInspiral->next;
		if (tmpSimInspiral == element->injection_document->sim_inspiral_table_head)
			element->injection_document->sim_inspiral_table_head = thisSimInspiral;
		XLALFree(tmpSimInspiral);
		}
	}

	/*
	 * add burst injections to simulation_series
	 */

	if(element->injection_document->sim_burst_table_head) {
		/*
		 * We follow the procedure in here so we don't end up adding the same burst injection to the data multiple times.
		 * The burst_series buffer essentially windows the injections to those that are currently relevant.
		 */

		/*
		 * create a buffer to store burst injections
		 */

		REAL8TimeSeries *burst_series = XLALCreateREAL8TimeSeries(h->name, &h->epoch, h->f0, h->deltaT, &h->sampleUnits, h->data->length);
		if(!burst_series)
			XLAL_ERROR(XLAL_EFUNC);
		memset(burst_series->data->data, 0, burst_series->data->length * sizeof(*burst_series->data->data));

		/*
		 * inject waveforms into that buffer
		 */

		if(XLALBurstInjectSignals(burst_series, element->injection_document->sim_burst_table_head, element->injection_document->time_slide_table_head, response))
			XLAL_ERROR(XLAL_EFUNC);

		/*
		 * add waveforms buffer into simulation_series
		 */

		if(!XLALAddREAL8TimeSeries(element->simulation_series, burst_series)) {
			XLALDestroyREAL8TimeSeries(burst_series);
			XLAL_ERROR(XLAL_EFUNC);
		}
		XLALDestroyREAL8TimeSeries(burst_series);		
	}

	return 0;
}


/*
 * This is the function that gets called.
 */


static int add_simulation_series(REAL8TimeSeries *h, const GSTLALSimulation *element, const COMPLEX16FrequencySeries *response)
{
	size_t startIdx;

	/*
	 * add simulation_series to h
	 */

	if(!XLALAddREAL8TimeSeries(h, element->simulation_series))
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * shrink simulation_series removing anything before the end of h
	 */

	startIdx = (size_t) (XLALGPSDiff(&h->epoch, &element->simulation_series->epoch) / h->deltaT);
	startIdx += h->data->length;
	if(!XLALShrinkREAL8TimeSeries(element->simulation_series, startIdx, element->simulation_series->data->length - startIdx))
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * done
	 */

	return 0;
}


/*
 * ============================================================================
 *
 *                         GstBaseTransform Overrides
 *
 * ============================================================================
 */


/* FIXME:  or maybe as a source element, and let the adder do the mixing work */


/*
 * sink event()
 */


static gboolean event(GstBaseTransform *trans, GstEvent *event)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(trans);

	if(GST_EVENT_TYPE(event) == GST_EVENT_TAG) {
		GstTagList *taglist;
		gchar *instrument = NULL, *channel_name = NULL, *units = NULL;

		/*
		 * attempt to extract all 3 tags from the event's taglist
		 */

		gst_event_parse_tag(event, &taglist);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_UNITS, &units);

		if(instrument || channel_name || units) {
			/*
			 * if any of the 3 tags were provided discard
			 * all current values
			 */

			g_free(element->instrument);
			element->instrument = NULL;
			g_free(element->channel_name);
			element->channel_name = NULL;
			g_free(element->units);
			element->units = NULL;

			if(instrument && channel_name && units) {
				/*
				 * if all 3 tags were provided, save as the
				 * new values
				 */

				element->instrument = instrument;
				element->channel_name = channel_name;
				element->units = units;
			}

			/*
			 * do notifies
			 */

			g_object_notify(G_OBJECT(element), "instrument");
			g_object_notify(G_OBJECT(element), "channel-name");
			g_object_notify(G_OBJECT(element), "units");
		}
	}

	/*
	 * done.  forward all events
	 */

	return TRUE;
}


/*
 * chain()
 */


static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(trans);
	GstFlowReturn result = GST_FLOW_OK;
	REAL8TimeSeries *h;

	/*
	 * If no injection list, reduce to pass-through
	 */

	if(!element->xml_location)
		goto done;

	/*
	 * Load injections if needed
	 */

	if(!element->injection_document) {
		LIGOTimeGPS start = {0, 0};
		LIGOTimeGPS end = {+2000000000, 0};
		/* earliest and latest possible LIGOTimeGPS */
		/* FIXME:  hard-coded = BAD BAD BAD */
		/*XLALINT8NSToGPS(&start, (INT8) 1 << 63);
		XLALINT8NSToGPS(&end, ((INT8) 1 << 63) - 1);*/
		element->injection_document = load_injection_document(element->xml_location, start, end, 0.0);
		if(!element->injection_document) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("error loading \"%s\"", element->xml_location));
			result = GST_FLOW_ERROR;
			goto done;
		}
	}

	/*
	 * Make sure stream tags are sufficient
	 */

	if(!element->instrument || !element->channel_name || !element->units) {
		GST_ELEMENT_ERROR(element, STREAM, FORMAT, (NULL), ("stream metadata not available:  must receive tags \"%s\", \"%s\", \"%s\"", GSTLAL_TAG_INSTRUMENT, GSTLAL_TAG_CHANNEL_NAME, GSTLAL_TAG_UNITS));
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Wrap buffer in a LAL REAL8TimeSeries.
	 */

	h = gstlal_REAL8TimeSeries_from_buffer(buf, element->instrument, element->channel_name, element->units);
	if(!h) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("failure wrapping buffer in REAL8TimeSeries"));
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Update simulation_series
	 * FIXME: needs to update simulation_series when new injection file provided
	 */

	/* FIXME: do we need a series mutex lock here? */

	if(update_simulation_series(h, element, NULL) < 0) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("failure updating simulation_series"));
		result = GST_FLOW_ERROR;
		goto release_h;
	}

	/*
	 * Add injections
	 */

	if(add_simulation_series(h, element, NULL) < 0) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("failure performing injections"));
		result = GST_FLOW_ERROR;
		goto release_h;
	}

	/* FIXME: do we need a series mutex lock free here? */

	/*
	 * Free the wrapping.  Setting the data pointer to NULL prevents
	 * XLALDestroyREAL8TimeSeries() from free()ing the buffer's data.
	 */

release_h:
	h->data->data = NULL;
	XLALDestroyREAL8TimeSeries(h);

	/*
	 * Done
	 */

done:
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
 * Properties
 */


enum property {
	ARG_XML_LOCATION = 1,
	ARG_INSTRUMENT,
	ARG_CHANNEL_NAME,
	ARG_UNITS
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{

	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_XML_LOCATION:
		g_free(element->xml_location);
		element->xml_location = g_value_dup_string(value);
		destroy_injection_document(element->injection_document);
		element->injection_document = NULL;
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_XML_LOCATION:
		g_value_set_string(value, element->xml_location);
		break;

	case ARG_INSTRUMENT:
		g_value_set_string(value, element->instrument);
		break;

	case ARG_CHANNEL_NAME:
		g_value_set_string(value, element->channel_name);
		break;

	case ARG_UNITS:
		g_value_set_string(value, element->units);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject * object)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	g_free(element->xml_location);
	element->xml_location = NULL;
	destroy_injection_document(element->injection_document);
	element->injection_document = NULL;
	g_free(element->instrument);
	element->instrument = NULL;
	g_free(element->channel_name);
	element->channel_name = NULL;
	g_free(element->units);
	element->units = NULL;
	XLALDestroyREAL8TimeSeries(element->simulation_series);
	element->simulation_series = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_simulation_base_init(gpointer class)
{
}


/*
 * class_init()
 */


static void gstlal_simulation_class_init(GSTLALSimulationClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->event = GST_DEBUG_FUNCPTR(event);
	transform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);

	gst_element_class_set_details_simple(
		element_class,
		"Simulation",
		"Filter",
		"An injection routine calling lalsimulation waveform generators",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>, Drew Keppel <drew.keppel@ligo.org>"
	);

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

	g_object_class_install_property(
		gobject_class,
		ARG_XML_LOCATION,
		g_param_spec_string(
			"xml-location",
			"XML Location",
			"Name of LIGO Light Weight XML file containing list(s) of software injections",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_INSTRUMENT,
		g_param_spec_string(
			"instrument",
			"Instrument",
			"Name of instrument for which the injections are being simulated",
			NULL,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_CHANNEL_NAME,
		g_param_spec_string(
			"channel-name",
			"Channel name",
			"Name of the channel for which the injections are being simulated",
			NULL,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_UNITS,
		g_param_spec_string(
			"units",
			"Units",
			"Units in which the injections are being computed.  Units are a string in the format used by the LAL units package.",
			NULL,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * instance init
 */


static void gstlal_simulation_init(GSTLALSimulation *element, GSTLALSimulationClass *klass)
{
	element->xml_location = NULL;
	element->injection_document = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->units = NULL;
	element->simulation_series = NULL;

	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

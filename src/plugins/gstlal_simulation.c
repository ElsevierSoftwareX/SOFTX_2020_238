/*
 * An interface to LALSimulation.  
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna, Drew Keppel
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
#include <low_latency_inspiral_functions.h>

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
	int success = 1;
	struct injection_document *new;

	if(!filename) {
		XLALPrintError("%s(): filename not set\n");
		XLAL_ERROR_NULL(func, XLAL_EFAULT);
	}

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

	XLALClearErrno();
	new->process_table_head = XLALProcessTableFromLIGOLw(filename);
	if(XLALGetBaseErrno()) {
		XLALPrintError("%s(): failure reading process table from \"%s\"\n", func, filename);
		success = 0;
	} else
		XLALPrintInfo("%s(): found process table\n", func);

	XLALClearErrno();
	new->process_params_table_head = XLALProcessParamsTableFromLIGOLw(filename);
	if(XLALGetBaseErrno()) {
		XLALPrintError("%s(): failure reading process_params table from \"%s\"\n", func, filename);
		success = 0;
	} else
		XLALPrintInfo("%s(): found process_params table\n", func);

	XLALClearErrno();
	new->search_summary_table_head = XLALSearchSummaryTableFromLIGOLw(filename);
	if(XLALGetBaseErrno())
		XLALPrintError("%s(): non-fatal failure reading search_summary table from \"%s\"\n", func, filename);
	else
		XLALPrintInfo("%s(): found search_summary table\n", func);

	/*
	 * load optional sim_burst table
	 */

	new->has_sim_burst_table = XLALLIGOLwHasTable(filename, "sim_burst");
	if(new->has_sim_burst_table) {
		XLALClearErrno();
		new->sim_burst_table_head = XLALSimBurstTableFromLIGOLw(filename, &start, &end);
		if(XLALGetBaseErrno()) {
			XLALPrintError("%s(): failure reading sim_burst table from \"%s\"\n", func, filename);
			success = 0;
		} else
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
			XLALPrintError("%s(): failure reading sim_inspiral table from \"%s\"\n", func, filename);
			new->sim_inspiral_table_head = NULL;
			success = 0;
		} else
			XLALPrintInfo("%s(): found sim_inspiral table\n", func);
	} else
		new->sim_inspiral_table_head = NULL;

	/*
	 * did we get it all?
	 */

	if(!success) {
		XLALPrintError("%s(): document is incomplete and/or malformed reading \"%s\"\n", func, filename);
		destroy_injection_document(new);
		XLAL_ERROR_NULL(func, XLAL_EFUNC);
	}

	/*
	 * success
	 */

	return new;
}


static int update_injection_cache(REAL8TimeSeries *h, GSTLALSimulation *element, const COMPLEX16FrequencySeries *response)
{
	LALPNOrder order = LAL_PNORDER_THREE_POINT_FIVE;
	static const char func[] = "update_injection_cache";
	unsigned injection_made;
	double injTime;
	REAL8 tmpREAL8;
	LIGOTimeGPS hStartTime, hEndTime, injStartTime, injEndTime;
	SimInspiralTable *thisSimInspiral;
	SimBurst *thisSimBurst;
	struct GSTLALInjectionCache *thisInjectionCacheElement;
	struct GSTLALInjectionCache *previousInjectionCacheElement;
	/* skip burst injections whose geocentre times are more than this many
	 * seconds outside of the target time series */
	const double injection_window = 100.0;

	/*
	 * calculate h segment boundaries, h = [hStartTime, hEndTime)
	 */

	hStartTime = h->epoch;
	hEndTime = hStartTime;
	XLALGPSAdd(&hEndTime, h->data->length * h->deltaT);

	/*
	 * loop over injections in file
	 */

	thisSimInspiral = element->injection_document->sim_inspiral_table_head;
	while(thisSimInspiral) {

		/*
		 * calculate start and end times for this series containing this injection
		 */

		injTime = gstlal_spa_chirp_time((REAL8) thisSimInspiral->mass1 + thisSimInspiral->mass2, (REAL8) thisSimInspiral->eta, (REAL8) thisSimInspiral->f_lower, order);
		injStartTime = injEndTime = thisSimInspiral->geocent_end_time;
		XLALGPSAdd(&injStartTime, -(1.9*injTime - 1.0));
		XLALGPSAdd(&injEndTime, 0.1*injTime + 1.0);

		/*
		 * round these times to the nearest ones in h
		 */

		tmpREAL8 = XLALGPSDiff(&injStartTime, &hStartTime) / h->deltaT;
		tmpREAL8 -= floor(tmpREAL8);
		XLALGPSAdd(&injStartTime, -tmpREAL8);

		tmpREAL8 = XLALGPSDiff(&injEndTime, &hStartTime) / h->deltaT;
		tmpREAL8 -= floor(tmpREAL8);
		XLALGPSAdd(&injEndTime, h->deltaT - tmpREAL8);

		/*
		 * check whether injection segment intersects h
		 */

		if( (XLALGPSCmp(&injStartTime, &hStartTime) >= 0 && XLALGPSCmp(&injStartTime, &hEndTime) < 0) /* injection start within h */ ||
		    (XLALGPSCmp(&injEndTime, &hStartTime) > 0 && XLALGPSCmp(&injEndTime, &hEndTime) < 0) /* injection end within h */ ||
		    (XLALGPSCmp(&injStartTime, &hStartTime) <= 0 && XLALGPSCmp(&injEndTime, &hEndTime) >= 0) /* h within injection segment */ ) {
			thisInjectionCacheElement = element->injection_cache;
			injection_made = 0;

			/*
			 * loop over injections in cache and see if this injection has been created
			 */

			thisInjectionCacheElement = element->injection_cache;
			while(thisInjectionCacheElement) {
				if(thisSimInspiral == thisInjectionCacheElement->sim_inspiral_pointer)
					injection_made++;
				thisInjectionCacheElement = thisInjectionCacheElement->next;
			}

			/*
			 * create injection if non-existent
			 */

			if(!injection_made) {
				double underflow_protection = 1.0;
				LALUnit strain_per_count = gstlal_lalStrainPerADCCount();
				COMPLEX8FrequencySeries *inspiral_response;
				LALStatus stat;
				REAL4TimeSeries *series = NULL;
				unsigned i;

				/*
				 * create space for the new injection cache element
				 */

				struct GSTLALInjectionCache *newInjectionCacheElement = calloc(1,sizeof *thisInjectionCacheElement);

				/*
				 * copy sim_inspiral entry into newInjectionCacheElement and set next and sim_inspiral->next to NULL
				 */

				newInjectionCacheElement->sim_inspiral = calloc(1,sizeof *thisSimInspiral);
				memcpy(newInjectionCacheElement->sim_inspiral, thisSimInspiral, sizeof(*thisSimInspiral));
				newInjectionCacheElement->sim_inspiral_pointer = thisSimInspiral;
				newInjectionCacheElement->next = NULL;
				newInjectionCacheElement->sim_inspiral->next = NULL;
				newInjectionCacheElement->sim_burst = NULL;
				newInjectionCacheElement->sim_burst_pointer = NULL;

				/*
				 * create a response function, copy the double precision
				 * version if one was provided otherwise make a dummy flat
				 * response
				 */

				if(response) {
					/*
					 * FIXME:  because the time series constructed
					 * below has extra padding added to it, the
					 * resolution of this response is probably wrong.
					 */
					inspiral_response = XLALCreateCOMPLEX8FrequencySeries(response->name, &response->epoch, response->f0, response->deltaF, &response->sampleUnits, response->data->length);
				}
				else {
					inspiral_response = XLALCreateCOMPLEX8FrequencySeries(NULL, &h->epoch, 0.0, 1.0 / XLALGPSDiff(&injEndTime, &injStartTime), &strain_per_count, (int) (0.5 + XLALGPSDiff(&injEndTime, &injStartTime) / h->deltaT));
				}

				if(!inspiral_response) {
					free(newInjectionCacheElement);
					XLAL_ERROR(func, XLAL_EFUNC);
				}

				if(response) {
					for(i = 0; i < inspiral_response->data->length; i++)
						inspiral_response->data->data[i] = XLALCOMPLEX8Rect(LAL_REAL(response->data->data[i]), LAL_IMAG(response->data->data[i]));
				} else {
					underflow_protection = 1e-20;
					for(i = 0; i < inspiral_response->data->length; i++)
						inspiral_response->data->data[i] = XLALCOMPLEX8Rect(underflow_protection, 0.0);
				}

				/*
				 * create the time series in which to store the injection waveform
				 */

				series = XLALCreateREAL4TimeSeries(h->name, &injStartTime, h->f0, h->deltaT, &lalADCCountUnit, (int) (0.5 + XLALGPSDiff(&injEndTime, &injStartTime) / h->deltaT));
				if(!series) {
					free(newInjectionCacheElement);
					XLALDestroyCOMPLEX8FrequencySeries(inspiral_response);
					XLAL_ERROR(func, XLAL_EFUNC);
				}
				memset(series->data->data, 0, series->data->length * sizeof(*series->data->data));

				/*
				 * compute the injection waveform
				 */

				XLALPrintInfo("%s(): computing sim_inspiral injection ...\n", func);
				/* FIXME: figure out how to do error handling like this */
				/*LAL_CALL(LALFindChirpInjectSignals(&stat, series, sim_inspiral_head, response), &stat);*/
				XLALClearErrno();
				memset(&stat, 0, sizeof(stat));
				LALFindChirpInjectSignals(&stat, series, newInjectionCacheElement->sim_inspiral, inspiral_response);
				XLALPrintInfo("%s(): done\n", func);

				XLALDestroyCOMPLEX8FrequencySeries(inspiral_response);

				/*
				 * add the injection waveform and start and end times to the injection cache element
				 */

				newInjectionCacheElement->series = series;
				newInjectionCacheElement->startTime = injStartTime;
				newInjectionCacheElement->endTime = injEndTime;
				newInjectionCacheElement->underflow_protection = underflow_protection;

				/*
				 * add this injection cache element to the end of the injection cache linked list
				 */

				thisInjectionCacheElement = element->injection_cache;
				if(!thisInjectionCacheElement) {
					element->injection_cache = newInjectionCacheElement;
				}
				else {
					while(thisInjectionCacheElement->next)
						thisInjectionCacheElement = thisInjectionCacheElement->next;
					thisInjectionCacheElement->next = newInjectionCacheElement;
				}

				/* end if calculate injection */
			}
			/* end if injection segment overlaps h */
		}

		/*
		 * continue loop over injections
		 */

		thisSimInspiral = thisSimInspiral->next;
	}

	thisSimBurst = element->injection_document->sim_burst_table_head;
	while(thisSimBurst) {
		/*
		 * scroll through cache searching for this injection
		 */

		thisInjectionCacheElement = element->injection_cache;
		while(thisInjectionCacheElement) {
			if(thisSimBurst == thisInjectionCacheElement->sim_burst_pointer)
				break;
			thisInjectionCacheElement = thisInjectionCacheElement->next;
		}

		/*
		 * scroll to final burst injection in cache
		 */

		while(thisInjectionCacheElement) {
			if(thisInjectionCacheElement->sim_burst_pointer) {
				thisSimBurst = thisInjectionCacheElement->sim_burst_pointer;
			}
			thisInjectionCacheElement = thisInjectionCacheElement->next;
			if(!thisInjectionCacheElement)
				thisSimBurst = thisSimBurst->next;
		}
		if(!thisSimBurst)
			continue;

		/*
		 * check whether injection segment intersects h
		 */

		if(XLALGPSDiff(&h->epoch, &thisSimBurst->time_geocent_gps) > injection_window || XLALGPSDiff(&thisSimBurst->time_geocent_gps, &h->epoch) > (h->data->length * h->deltaT + injection_window)) {
			thisSimBurst = thisSimBurst->next;
			continue;
		}

		/*
		 * injection needs to be created if reach here
		 */

		{
			double underflow_protection = 1.0;
			/* to be deduced from the time series' channel name */
			const LALDetector *detector;
			/* FIXME:  fix the const entanglement so as to get rid of this */
			LALDetector detector_copy;
			/* + and x time series for injection waveform */
			REAL8TimeSeries *hplus, *hcross;
			/* injection time series as added to detector's */
			REAL8TimeSeries *strain;
			/* injection time series to be stored in cache */
			REAL4TimeSeries *series = NULL;
			unsigned i;

			/*
			 * create space for the new injection cache element
			 */

			struct GSTLALInjectionCache *newInjectionCacheElement = calloc(1,sizeof *thisInjectionCacheElement);

			/*
			 * copy sim_burst entry into newInjectionCacheElement and set next and sim_burst->next to NULL
			 */

			newInjectionCacheElement->sim_burst = calloc(1,sizeof *thisSimBurst);
			memcpy(newInjectionCacheElement->sim_burst, thisSimBurst, sizeof(*thisSimBurst));
			newInjectionCacheElement->sim_burst_pointer = thisSimBurst;
			newInjectionCacheElement->next = NULL;
			newInjectionCacheElement->sim_burst->next = NULL;
			newInjectionCacheElement->sim_inspiral_pointer = NULL;
			newInjectionCacheElement->sim_inspiral_pointer = NULL;

			/* turn the first two characters of the channel name into a
			 * detector */

			detector = XLALInstrumentNameToLALDetector(h->name);
			if(!detector)
				XLAL_ERROR(func, XLAL_EFUNC);
			XLALPrintInfo("%s(): channel name is '%s', instrument appears to be '%s'\n", func, h->name, detector->frDetector.prefix);
			detector_copy = *detector;

			/* construct the h+ and hx time series for the injection
			 * waveform.  in the time series produced by this function,
			 * t = 0 is the "time" of the injection. */

			if(XLALGenerateSimBurst(&hplus, &hcross, newInjectionCacheElement->sim_burst, h->deltaT))
				XLAL_ERROR(func, XLAL_EFUNC);

			/* add the time of the injection at the geocentre to the
			 * start times of the h+ and hx time series.  after this,
	 		 * their epochs mark the start of those time series at the
			 * geocentre. */

			XLALGPSAddGPS(&hcross->epoch, &newInjectionCacheElement->sim_burst->time_geocent_gps);
			XLALGPSAddGPS(&hplus->epoch, &newInjectionCacheElement->sim_burst->time_geocent_gps);

			/* project the wave onto the detector to produce the strain
			 * in the detector. */

			strain = XLALSimDetectorStrainREAL8TimeSeries(hplus, hcross, newInjectionCacheElement->sim_burst->ra, newInjectionCacheElement->sim_burst->dec, newInjectionCacheElement->sim_burst->psi, &detector_copy);
			XLALDestroyREAL8TimeSeries(hplus);
			XLALDestroyREAL8TimeSeries(hcross);
			if(!strain)
				XLAL_ERROR(func, XLAL_EFUNC);

			/*
			 * convert the REAL8TimeSeries to a REAL4TimeSeries
			 */

			series = XLALCreateREAL4TimeSeries(strain->name, &strain->epoch, strain->f0, strain->deltaT, &strain->sampleUnits, strain->data->length);
			if(!series)
				XLAL_ERROR(func, XLAL_EFUNC);
			underflow_protection = 1e-20;
			for(i=0; i<strain->data->length; i++)
				series->data->data[i] = (REAL4) (strain->data->data[i]/underflow_protection);
			XLALDestroyREAL8TimeSeries(strain);

			/*
			 * add the injection wavefore and start and end times to the injection cache element
			 */

			newInjectionCacheElement->series = series;
			newInjectionCacheElement->startTime = series->epoch;
			newInjectionCacheElement->endTime = series->epoch;
			XLALGPSAdd(&newInjectionCacheElement->endTime, series->data->length * series->deltaT);
			newInjectionCacheElement->underflow_protection = underflow_protection;

			/*
			 * add this injection cache element to the end of the injection cache linked list
			 */

			thisInjectionCacheElement = element->injection_cache;
			if(!thisInjectionCacheElement) {
				element->injection_cache = newInjectionCacheElement;
			}
			else {
				while(thisInjectionCacheElement->next)
					thisInjectionCacheElement = thisInjectionCacheElement->next;
				thisInjectionCacheElement->next = newInjectionCacheElement;
			}

			/* end if calculate injection */
		}
		
		/*
		 * continue loop over injections
		 */

		thisSimBurst = thisSimBurst->next;
	}

	/*
	 * remove injections from cache if no longer needed
	 */

	previousInjectionCacheElement = NULL;
	thisInjectionCacheElement = element->injection_cache;
	while(thisInjectionCacheElement) {
		unsigned int remove = 0;

		if(thisInjectionCacheElement->sim_burst_pointer) {
			if(XLALGPSDiff(&h->epoch, &thisInjectionCacheElement->sim_burst->time_geocent_gps) > injection_window || XLALGPSDiff(&thisInjectionCacheElement->sim_burst->time_geocent_gps, &h->epoch) > (h->data->length * h->deltaT + injection_window)) {
				remove = 1;
			}
		}
		else {
			if( XLALGPSCmp(&(thisInjectionCacheElement->endTime), &hStartTime) <= 0 ) {
				remove = 1;
			}
		}

		if(remove) {
			struct GSTLALInjectionCache *tmpInjectionCacheElement = thisInjectionCacheElement;
			if(previousInjectionCacheElement)
				previousInjectionCacheElement->next = thisInjectionCacheElement->next;
			else
				element->injection_cache = thisInjectionCacheElement->next;
			thisInjectionCacheElement = thisInjectionCacheElement->next;

			/*
			 * destroy the REAL4 time series and free the injection cache element
			 */

			XLALDestroyREAL4TimeSeries(tmpInjectionCacheElement->series);
			if(tmpInjectionCacheElement->sim_inspiral)
				free(tmpInjectionCacheElement->sim_inspiral);
			if(tmpInjectionCacheElement->sim_burst)
				free(tmpInjectionCacheElement->sim_burst);
			free(tmpInjectionCacheElement);
		}
		else {
			previousInjectionCacheElement = thisInjectionCacheElement;
			thisInjectionCacheElement = thisInjectionCacheElement->next;
		}
	}

	return 0;
}


/*
 * This is the function that gets called.
 */


static int add_xml_injections(REAL8TimeSeries *h, const GSTLALSimulation *element, const COMPLEX16FrequencySeries *response)
{
	struct GSTLALInjectionCache *thisInjectionCacheElement;

	/*
	 * loop over calculated injections in injection cache
	 */

	thisInjectionCacheElement = element->injection_cache;
	while(thisInjectionCacheElement) {
		REAL8 Delta_epoch = XLALGPSDiff(&(thisInjectionCacheElement->startTime), &(h->epoch));
		unsigned i, j;

		/*
		 * add injection time series to target time series
		 */

		/* set start indexes */
		if(Delta_epoch >= 0) {
			i = floor(Delta_epoch / h->deltaT + 0.5);
			j = 0;
		} else {
			i = 0;
			j = floor(-Delta_epoch / h->deltaT + 0.5);
		}

		/* add injection to h */
		for(; i < h->data->length && j < thisInjectionCacheElement->series->data->length; i++, j++) {
			h->data->data[i] += thisInjectionCacheElement->series->data->data[j] * thisInjectionCacheElement->underflow_protection;
		}

		thisInjectionCacheElement = thisInjectionCacheElement->next;
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
		REAL8TimeSeries *strain = XLALSimDetectorStrainREAL8TimeSeries(hplus, hcross, right_ascension, declination, psi, detector);
		XLALDestroyREAL8TimeSeries(hplus);
		XLALDestroyREAL8TimeSeries(hcross);
		return strain;
	} else {
		XLALDestroyREAL8TimeSeries(hplus);
		XLALDestroyREAL8TimeSeries(hcross);
		return NULL;
	}

}


/*
 * ============================================================================
 *
 *			     GStreamer Element
 *
 * ============================================================================
 */


/* FIXME:  re-write this as a subclass of the base transform class */


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
 * sink event()
 */


static gboolean sink_event(GstPad *pad, GstEvent *event)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(GST_PAD_PARENT(pad));
	gboolean success;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_TAG: {
		GstTagList *taglist;
		gchar *instrument, *channel_name, *units;
		gst_event_parse_tag(event, &taglist);
		success = gst_tag_list_get_string(taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
		success &= gst_tag_list_get_string(taglist, GSTLAL_TAG_CHANNEL, &channel_name);
		success &= gst_tag_list_get_string(taglist, GSTLAL_TAG_UNITS, &units);
		gst_tag_list_free(taglist);
		if(!success)
			GST_ERROR_OBJECT(element, "unable to parse instrument and/or channel and/or units from tag");
		else {
			g_free(element->instrument);
			element->instrument = instrument;
			g_free(element->channel_name);
			element->channel_name = channel_name;
			g_free(element->units);
			element->units = units;
			success = gst_pad_push_event(element->srcpad, event);
		}
		break;
	}

	default:
		success = gst_pad_event_default(pad, event);
		break;
	}

	return success;
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
			GST_ERROR_OBJECT(element, "error loading \"%s\"", element->xml_location);
			gst_buffer_unref(buf);
			result = GST_FLOW_ERROR;
			goto done;
		}
	}

	/*
	 * Wrap buffer in a LAL REAL8TimeSeries.
	 */

	h = gstlal_REAL8TimeSeries_from_buffer(buf, element->instrument, element->channel_name, element->units);
	if(!h) {
		GST_ERROR_OBJECT(element, "failure wrapping buffer in REAL8TimeSeries");
		gst_buffer_unref(buf);
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Update Injection Cache
	 * FIXME: needs to update injection cache when new injection file provided
	 */

	if(update_injection_cache(h, element, NULL) < 0) {
		GST_ERROR_OBJECT(element, "failure updating injection cache");
		h->data->data = NULL;
		XLALDestroyREAL8TimeSeries(h);
		gst_buffer_unref(buf);
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Add injections
	 */

	if(add_xml_injections(h, element, NULL) < 0) {
		GST_ERROR_OBJECT(element, "failure performing injections");
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
 * Instance finalize function.  See ???
 */


static void finalize(GObject * object)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	free(element->xml_location);
	element->xml_location = NULL;
	destroy_injection_document(element->injection_document);
	g_free(element->instrument);
	element->instrument = NULL;
	g_free(element->channel_name);
	element->channel_name = NULL;
	g_free(element->units);
	element->units = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
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
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"
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
	gobject_class->finalize = finalize;

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
	gst_pad_set_event_function(pad, sink_event);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(object), "src");

	/* internal data */
	element->xml_location = NULL;
	element->injection_document = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->units = NULL;
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

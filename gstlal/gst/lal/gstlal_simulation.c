/*
 * An interface to LALSimulation.  
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna, Drew Keppel
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


/*
 * stuff from LAL
 */


#include <lal/LALComplex.h>
#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/LALSimulation.h>
#include <lal/LALSimInspiral.h>
#include <lal/LIGOMetadataBurstUtils.h>
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


#include <gstlal.h>
#include <gstlal_tags.h>
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
	int success = 1;
	struct injection_document *new;
	int nrows; 

	g_assert(filename != NULL);

	/*
	 * allocate the document
	 */

	new = calloc(1, sizeof(*new));
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
	 * load optional sim_burst table
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


static int update_injection_cache(REAL8TimeSeries *h, GSTLALSimulation *element, const COMPLEX16FrequencySeries *response)
{
	LALPNOrder order = LAL_PNORDER_THREE_POINT_FIVE;
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
				} else {
					inspiral_response = XLALCreateCOMPLEX8FrequencySeries(NULL, &h->epoch, 0.0, 1.0 / XLALGPSDiff(&injEndTime, &injStartTime), &strain_per_count, (int) (0.5 + XLALGPSDiff(&injEndTime, &injStartTime) / h->deltaT));
				}

				if(!inspiral_response) {
					free(newInjectionCacheElement);
					XLAL_ERROR(XLAL_EFUNC);
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
					XLAL_ERROR(XLAL_EFUNC);
				}
				memset(series->data->data, 0, series->data->length * sizeof(*series->data->data));

				/*
				 * compute the injection waveform
				 */

				XLALPrintInfo("%s(): computing sim_inspiral injection ...\n", __func__);
				/* FIXME: figure out how to do error handling like this */
				/*LAL_CALL(LALFindChirpInjectSignals(&stat, series, sim_inspiral_head, response), &stat);*/
				XLALClearErrno();
				memset(&stat, 0, sizeof(stat));
				LALFindChirpInjectSignals(&stat, series, newInjectionCacheElement->sim_inspiral, inspiral_response);
				XLALPrintInfo("%s(): done\n", __func__);

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
				} else {
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
				XLAL_ERROR(XLAL_EFUNC);
			XLALPrintInfo("%s(): channel name is '%s', instrument appears to be '%s'\n", __func__, h->name, detector->frDetector.prefix);
			detector_copy = *detector;

			/* construct the h+ and hx time series for the injection
			 * waveform.  in the time series produced by this function,
			 * t = 0 is the "time" of the injection. */

			if(XLALGenerateSimBurst(&hplus, &hcross, newInjectionCacheElement->sim_burst, h->deltaT))
				XLAL_ERROR(XLAL_EFUNC);

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
				XLAL_ERROR(XLAL_EFUNC);

			/*
			 * convert the REAL8TimeSeries to a REAL4TimeSeries
			 */

			series = XLALCreateREAL4TimeSeries(strain->name, &strain->epoch, strain->f0, strain->deltaT, &strain->sampleUnits, strain->data->length);
			if(!series)
				XLAL_ERROR(XLAL_EFUNC);
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
			} else {
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
		} else {
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
		} else {
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
			i = round(Delta_epoch / h->deltaT);
			j = 0;
		} else {
			i = 0;
			j = round(-Delta_epoch / h->deltaT);
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


#if 0
/* unused function */
static REAL8TimeSeries *compute_strain(double right_ascension, double declination, double psi, LALDetector * detector, LIGOTimeGPS * tc, double phic, double deltaT, double m1, double m2, double fmin, double r, double i, int order)
{
	REAL8TimeSeries *hplus = NULL;
	REAL8TimeSeries *hcross = NULL;
	REAL8TimeSeries *strain = NULL;
	if(XLALSimInspiralPN(&hplus, &hcross, tc, phic, deltaT, m1, m2, fmin, r, i, order))
		strain = XLALSimDetectorStrainREAL8TimeSeries(hplus, hcross, right_ascension, declination, psi, detector);
	XLALDestroyREAL8TimeSeries(hplus);
	XLALDestroyREAL8TimeSeries(hcross);
	return strain;
}
#endif


/*
 * ============================================================================
 *
 *			     GStreamer Element
 *
 * ============================================================================
 */


/* FIXME:  re-write this as a subclass of the base transform class */
/* FIXME:  or maybe as a source element, and let the adder do the mixing work */


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
}

static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

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
}


/*
 * sink event()
 */


static gboolean sink_event(GstPad *pad, GstEvent *event)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(GST_PAD_PARENT(pad));

	if (GST_EVENT_TYPE(event) == GST_EVENT_TAG) {
		GstTagList *taglist;
		gchar *instrument = NULL, *channel_name = NULL, *units = NULL;

		/* Attempt to extract all 3 tags from the event's taglist. */
		gst_event_parse_tag(event, &taglist);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_UNITS, &units);

		if(instrument || channel_name || units) {
			/* If any of the 3 tags were provided, we discard the old, stored values. */
			g_free(element->instrument);
			element->instrument = NULL;
			g_free(element->channel_name);
			element->channel_name = NULL;
			g_free(element->units);
			element->units = NULL;

			if(instrument && channel_name && units) {
				/* If all 3 tags were provided, we save the new values. */
				element->instrument = instrument;
				element->channel_name = channel_name;
				element->units = units;
			}

			/* do notifies for all three */
			g_object_notify(G_OBJECT(element), "instrument");
			g_object_notify(G_OBJECT(element), "channel-name");
			g_object_notify(G_OBJECT(element), "units");
		}
	}

	/* Allow the default event handler to take over. Downstream elements may want to look at tags too. */
	return gst_pad_event_default(pad, event);
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
	 * If no injection list, reduce to pass-through
	 */

	if(!element->xml_location) {
		result = gst_pad_push(element->srcpad, buf);
		goto done;
	}

	/*
	 * Load injections if needed
	 */

	if(!element->injection_document) {
		LIGOTimeGPS start = {-2000000000, 0};
		LIGOTimeGPS end = {+2000000000, 0};
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
	 * If stream tags not sufficient, reduce to pass-through
	 */

	if(!element->instrument || !element->channel_name || !element->units) {
		GST_ERROR_OBJECT(element, "stream metadata not available, cannot construct injections:  must receive tags \"%s\", \"%s\", \"%s\"", GSTLAL_TAG_INSTRUMENT, GSTLAL_TAG_CHANNEL_NAME, GSTLAL_TAG_UNITS);
		result = gst_pad_push(element->srcpad, buf);
		goto done;
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
	g_free(element->xml_location);
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
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Simulation",
		"Filter",
		"An injection routine",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
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

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

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
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(object), "src");

	/* internal data */
	element->xml_location = NULL;
	element->injection_document = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->units = NULL;
	element->injection_cache = NULL;
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

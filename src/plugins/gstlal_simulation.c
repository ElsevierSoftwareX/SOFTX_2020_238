/*
 * A template bank.
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
 *  * stuff from gstreamer
 *   */


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


/*
 * stuff from LAL
 */
#include <lal/Date.h>
#include <lal/LALDatatypes.h>
#include <lal/FrameStream.h>
#include <lal/LALFrameIO.h>
#include <lal/LALStdlib.h>
#include <lal/Units.h>
#include <lal/TimeSeries.h>
#include <lal/LALSimulation.h>
#include <lal/LALSimInspiral.h>


/*
 * stuff from GSL
 */


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_templatebank.h>

#define DEFAULT_RIGHT_ASCENSION 0.0 	/* Radians    */
#define DEFAULT_DECLINATION 0.0     	/* Radians    */
#define DEFAULT_PSI 0.0		    	/* Radians    */
#define DEFAULT_PHIC 0.0	    	/* Radians    */
#define DEFAULT_MASS_ONE 1.40       	/* M_sun      */
#define DEFAULT_MASS_TWO 1.40       	/* M_sun      */
#define DEFAULT_F_LOW 30.0	    	/* Hz         */
#define DEFAULT_DISTANCE 1.0        	/* Mpc ???    */
#define DEFAULT_INCLINATION LAL_PI/4.0  /* Radians    */
#define DEFAULT_AMPLITUDE_ORDER 2       /* 2*PN order */
#define DEFAULT_PHASE_ORDER 7           /* 2*PN order */





static REAL8TimeSeries *compute_strain(double right_ascension, 
                                       double declination,
				       double psi,
				       LALDetector *detector,
				       LIGOTimeGPS *tc,
				       double phic,
				       double deltaT,
				       double m1,
				       double m2,
				       double fmin,
				       double r,
				       double i,
				       int amplitudeO,
				       int phaseO
				       )
{
	REAL8TimeSeries *hplus = NULL;
	REAL8TimeSeries *hcross = NULL;
	REAL8 x0 = 0;
	if (XLALSimInspiralPNGenerator(&hplus,&hcross,tc,x0,deltaT,m1,m2,fmin,r,i,amplitude0,phase0))
	{
	        XLALDestroyREAL8TimeSeries(hplus);
		XLALDestroyREAL8TimeSeries(hcross);
		return XLALSimDetectorStrainReal8TimeSeries(hplus,hcross,right_ascension,declination,psi,detector);
	}
	else 
	{
		XLALDestroyREAL8TimeSeries(hplus);
		XLALDestroyREAL8TimeSeries(hcross);
		return NULL;
	}
 
}

static int inject(REAL8TimeSeries *buffer, REAL8TimeSeries *h)
{

	/* Are multiple calls to XLALSimAddInjection safe?  Does the source 
	 * time series get corrupted or improved ? It is modified in place !
	 * Thus for now this function assumes that the entire injection can
	 * be added at once.  
	 */
	int status = 0;
	status = XLALSimAddInjectionREAL8TimeSeries(buffer,h,NULL);
	/* Once the injection is finished we don't need h, in fact we will
	 * check h in the future to see if a new injection should be computed
	 * */
        status += XLALDestroyREAL8TimeSeries(h);
	return status;
}


static void simulation_destroy(GSTLALTemplateBank *element)
{

    	/* FIXME: destroy allocated memory for the simulation */

}


static int simulation_create(GSTLALTemplateBank *element, int sample_rate)
{
        int verbose = 1;

        /* be sure we don't leak memory */

        simulation_destroy(element);

        /* FIXME: generate the simulation */


        /* done */

        return 0;
}


static void srcpads_destroy(GSTLALSimulation *element)
{
        GList *padlist;

        for(padlist = element->srcpads; padlist; padlist = g_list_next(padlist))
                gst_object_unref(GST_PAD(padlist->data));

        g_list_free(element->srcpads);
        element->srcpads = NULL;
}



enum property {
        ARG_RIGHT_ASCENSION = 1,
        ARG_DECLINATION,
        ARG_PSI,
        ARG_PHIC,
        ARG_MASS_ONE,
        ARG_MASS_TWO,
        ARG_F_LOW,
        ARG_DISTANCE,
        ARG_INCLINATION,
	ARG_AMPLITUDE_ORDER,
	ARG_PHASE_ORDER,
	ARG_DETECTOR,
	ARG_COALESCENCE_TIME
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{

        GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

        switch(id) {
        case ARG_RIGHT_ASCENSION:
                element->right_ascension = g_value_get_double(value);
                break;

        case ARG_DECLINATION:
                element->declination = g_value_get_double(value);
                break;

        case ARG_PSI:
                element->psi = g_value_get_double(value);
                break;

        case ARG_PHIC:
                element->phic = g_value_get_double(value);
                break;

        case ARG_MASS_ONE:
                element->m1 = g_value_get_double(value);
                break;

        case ARG_MASS_TWO:
                element->mass_two = g_value_get_double(value);
                break;

        case ARG_F_LOW:
                element->fmin = g_value_get_double(value);
                break;

        case ARG_DISTANCE:
                element->r = g_value_get_double(value);
                break;

        case ARG_INCLINATION:
                element->inclination = g_value_get_double(value);
                break;

        case ARG_AMPLITUDE_ORDER:
		element->inclination = g_value_get_int(value);
		break;

	case ARG_PHASE_ORDER:
		element->inclination = g_value_get_int(value);
		break;

        case ARG_COALESCENCE_TIME:
	        if(XLALStrToGPS(&element->tc, g_value_get_string(value), NULL) < 0) 
		{
		GST_DEBUG("invalid start_time_gps \"%s\"", g_value_get_string(value));
		}
		break;

	/*FIXME: The detector needs to be set up here too! */	
        }
}

static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
        GSTLALSimulation *element = GSTLAL_SIMULATION(object);

        switch(id) {
        case ARG_RIGHT_ASCENSION:
                g_value_set_double(value, element->right_ascension);
                break;

        case ARG_DECLINATION:
                g_value_set_double(value, element->declination);
                break;

        case ARG_PSI:
                g_value_set_double(value, element->psi);
                break;

        case ARG_PHIC:
                g_value_set_double(value, element->phic);
                break;

        case ARG_MASS_ONE:
                g_value_set_double(value, element->m1);
                break;

        case ARG_MASS_TWO:
                g_value_set_double(value, element->m2);
                break;

        case ARG_F_LOW:
                g_value_set_double(value, element->fmin);
                break;

        case ARG_DISTANCE:
                g_value_set_double(value, element->r);
                break;

        case ARG_INCLINATION:
                g_value_set_double(value, element->inclination);
                break; 

        case ARG_AMPLITUDE_ORDER:
		g_value_set_int(value, element->inclination);
		break;

        case ARG_PHASE_ORDER:
	        g_value_set_int(value, element->inclination);
		break;
        }
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
        GSTLALTemplateBank *element = GSTLAL_SIMULATION(gst_pad_get_parent(pad));
        gboolean result = TRUE;
        GList *padlist;

        for(padlist = element->srcpads; padlist; padlist = g_list_next(padlist)) {
                result = gst_pad_set_caps(GST_PAD(padlist->data), caps);
                if(result != TRUE)
                        goto done;
        }

done:
        gst_object_unref(element);
        return result;
}



/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
  {
  /* FIXME: Add the actual injection functionality here */
  }



/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;



/*
 * Instance dispose function.  See ???
 */


static void dispose(GObject *object)
{
        GSTLALSimulation *element = GSTLAL_SIMULATION(object);

        g_object_unref(element->adapter);
        element->adapter = NULL;
        srcpads_destroy(element);

        simulation_destroy(element);

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
                "Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <chann@ligo.caltech.edu>"
        };
        GstElementClass *element_class = GST_ELEMENT_CLASS(class);
        GstPadTemplate *sinkpad_template = gst_pad_template_new(
                "sink",
                GST_PAD_SINK,
                GST_PAD_ALWAYS,
                gst_caps_new_simple(
                        "audio/x-raw-float",
                        "rate", GST_TYPE_INT_RANGE, 1, SIMULATION_SAMPLE_RATE,
                        "channels", G_TYPE_INT, 1,
                        "endianness", G_TYPE_INT, G_BYTE_ORDER,
                        "width", G_TYPE_INT, 64,
                        NULL
                )
        );
        GstPadTemplate *srcpad_template = gst_pad_template_new(
                "injection",
                GST_PAD_SRC,
                GST_PAD_SOMETIMES,
                gst_caps_new_simple(
                        "audio/x-raw-float",
                        "rate", GST_TYPE_INT_RANGE, 1, SIMULATION_SAMPLE_RATE,
                        "channels", G_TYPE_INT, 1,
                        "endianness", G_TYPE_INT, G_BYTE_ORDER,
                        "width", G_TYPE_INT, 64,
                        NULL
                )
        );

        gst_element_class_set_details(element_class, &plugin_details);
        gst_element_class_add_pad_template(element_class, sinkpad_template);
        gst_element_class_add_pad_template(element_class, srcpad_template);
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

        /* FIXME THESE NEED TO BE SET TO THE APPROPRIATE PARAMS */
        g_object_class_install_property(gobject_class, ARG_RIGHT_ASCENSION, g_param_spec_double("right-ascension", "RA", "Right ascension of the injection in radians", 0, G_MAXDOUBLE, DEFAULT_RIGHT_ASCENSION, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, ARG_DECLINATION, g_param_spec_double("declination", "DEC", "Declination of the injection in radians", 0, G_MAXDOUBLE, DEFAULT_DECLINATION, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, ARG_PSI, g_param_spec_double("psi", "Psi", "Polarization angle of the injection in radians", 0, G_MAXDOUBLE, DEFAULT_PSI, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, ARG_PHIC, g_param_spec_double("phic", "coalescence phase", "Coalescence phase of the binary system in radians", 0, G_MAXDOUBLE, DEFAULT_PHIC, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, ARG_MASS_ONE, g_param_spec_double("m1", "mass one", "First component mass of the injection in Msun", 0, G_MAXDOUBLE, DEFAULT_MASS_ONE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, ARG_MASS_TWO, g_param_spec_double("m2", "mass two", "Second component mass of the injection in Msun", 0, G_MAXDOUBLE, DEFAULT_MASS_TWO, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, ARG_F_LOW, g_param_spec_double("f_low", "low frequency", "The starting frequency of the injection in Hz", 0, G_MAXDOUBLE, DEFAULT_F_LOW, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, ARG_DISTANCE, g_param_spec_double("r", "distance", "Distance of the injection in Mpc??? (check this)", 0, G_MAXDOUBLE, DEFAULT_MASS_ONE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, ARG_INCLINATION, g_param_spec_double("i", "inclination", "Inclination angle of the injection in radians", 0, G_MAXDOUBLE, DEFAULT_INCLINATION, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

        g_object_class_install_property(gobject_class, ARG_AMPLITUDE_ORDER, g_param_spec_init("amplitudeO", "amplitude order", "Twice the PN order for the amplitude of the injection", 0, G_MAXINT, DEFAULT_AMPLITUDE_ORDER, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

        g_object_class_install_property(gobject_class, ARG_PHASE_ORDER, g_param_spec_int("phaseO", "phase order", "Twice the PN order for the phase of the injection", 0, G_MAXINT, DEFAULT_PHASE_ORDER, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));


}


/*
 *  * Instance init function.  See
 *   *
 *    * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 *     */



static void instance_init(GTypeInstance *object, gpointer class)
{
        GSTLALTemplateBank *element = GSTLAL_SIMULATION(object);
        GstPad *pad;

        gst_element_create_all_pads(GST_ELEMENT(element));

        /* configure sink pad */
        pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
        gst_pad_set_setcaps_function(pad, setcaps);
        gst_pad_set_chain_function(pad, chain);
        gst_object_unref(pad);

        /* src pads */
        element->srcpads = NULL;
          {
          GstPadTemplate *template = gst_element_class_get_pad_template(class, "injection");
          pad = gst_pad_new_from_template(template, "injection");
          gst_object_ref(pad);    /* for our linked list */
          gst_element_add_pad(GST_ELEMENT(element), pad);
          element->srcpads = g_list_append(element->srcpads, pad);
          }

        /* internal data */
        element->adapter = gst_adapter_new();
        element->t_start = DEFAULT_T_START;
        element->t_end = DEFAULT_T_END;
        element->snr_length = DEFAULT_SNR_LENGTH;

        element->next_sample = 0;

        element->U = NULL;
        element->S = NULL;
        element->V = NULL;
        element->chifacs = NULL;
}




/*
 * gstlal_templatebank_get_type().
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


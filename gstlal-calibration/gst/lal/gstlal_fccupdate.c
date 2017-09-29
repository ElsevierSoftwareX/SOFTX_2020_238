/*Copyright (C) 2016 Kipp Cannon <kipp.cannon@ligo.org>, Theresa Chmiel,
 * Madeline Wade
 *
 *This program is free software; you can redistribute it and/or modify
 *it under the terms of the GNU General Public License as published by
 *the Free Software Foundation; either version 2 of the License, or
 *(at your option) any later version.
 *      
 *This program is distributed in the hope that it will be useful,
 *but WITHOUT ANY WARRANTY; without even the implied warranty of
 *MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *GNU General Public License for more details.
 *           
 *You should have received a copy of the GNU General Public License along
 *with this program; if not, write to the Free Software Foundation, Inc.,
 *51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*
 *============================================================================
 *
 *                                  Preamble
 *     
 *============================================================================
 */

/*
 *stuff from gobject/gstreamer
 */
//#include <gstlal_fccupdate.h>
#include <stdlib.h>
#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>

//extra stuff I added


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

#include <gstlal/gstaudioadapter.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal.h>

#include <complex.h>
#include <string.h>
#include <fftw3.h>
#include <math.h>

#include <lal/LALAtomicDatatypes.h>
#include <stddef.h>

/*
 *our own stuff
 */


#include <gstlal_fccupdate.h>


/*
 *============================================================================
 *   
 *                           GStreamer Boiler Plate
 *     
 *============================================================================
 */

#define GST_CAT_DEFAULT gstlal_fcc_update_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
        GSTLALFccUpdate,
        gstlal_fcc_update,
        GST_TYPE_BASE_TRANSFORM,
        GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_fcc_update", 0, "lal_fcc_update element")
);

/*
 *============================================================================
 *   
 *                                 Parameters
 *     
 *============================================================================
 */

#define DEFAULT_FIR_MATRIX 100.0
#define Pi 3.14159265358979323846

/*
 *============================================================================
 *
 *                                 Utilities
 *    
 *============================================================================
 */
//finds the length of a 1D gsl_matrix
int num_rows(gsl_matrix * A){
        int r = A->size2;
        return r;
}

//Computes the hanning window
float *hanning(int N, short itype)
{
    int half, i, idx, n;
    float *w;

    w = (float*) calloc(N, sizeof(float));
    memset(w, 0, N*sizeof(float));

    if(itype==1)    //periodic function
        n = N-1;
    else
        n = N;

    if(n%2==0)
    {
        half = n/2;
        for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
            w[i] = 0.5 * (1 - cos(2*Pi*(i+1) / (n+1)));

        idx = half-1;
        for(i=half; i<n; i++) {
            w[i] = w[idx];
            idx--;
        }
    }
    else
    {
        half = (n+1)/2;
        for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
            w[i] = 0.5 * (1 - cos(2*Pi*(i+1) / (n+1)));

        idx = half-2;
        for(i=half; i<n; i++) {
            w[i] = w[idx];
            idx--;
        }
    }

    if(itype==1)    //periodic function
    {
        for(i=N-1; i>=1; i--)
            w[i] = w[i-1];
        w[0] = 0.0;
    }
    return(w);
}

//Computes a tukey window
float *tukey(int N)
{
	float * HannWindow=malloc(sizeof(float)*N/2);
        HannWindow=hanning(N/2,0);
	float * TukeyWindow=malloc(sizeof(float)*N);
	int i=0;
	int j=N/4;
	for(i=0;i<N/4;i++)
		TukeyWindow[i]=HannWindow[i];
	for(i=N/4;i<3*N/4;i++)
		TukeyWindow[i]=1.0;
	for(i=3*N/4;i<N;i++) {
		TukeyWindow[i]=HannWindow[j];
		j++;
	}

	return(TukeyWindow);

}



//constructs the new filter with the computed average
long double* MakeFilter(GSTLALFccUpdate *element) {
	long double FcAverage=element->currentaverage;
	
	printf("FcAverage=%Lf\n,",FcAverage);

	long double FiltDur=(long double) element->filterduration;
	long double Filtdf=1.0L/(long double)FiltDur;
	long double Filtdt=1.0L/(long double) element->datarate;
	int filtlength= (int) (element->datarate/(2.0*Filtdf))+1;

	int i=0;
	long double Filtf[filtlength];
	while(i<filtlength) {
		Filtf[i]=(i*Filtdf);
		i++;
	}
	i=0;
	long double complex CavPoleFiltForTD[filtlength];
	double instrument_cavity_pole_frequency = element->fcmodel;

	while(i<filtlength) {
		CavPoleFiltForTD[i]=(1.0L+(I*(Filtf[i]/FcAverage)))/(1.0L+(I*(Filtf[i]/instrument_cavity_pole_frequency)));
		i++;
	}

	//adds delay to the filter
	i=0;
	double delaysamples=filtlength;
	long double complex Delay[filtlength];
	while(i<filtlength) {
		Delay[i]=cexpl(-2.0L*Pi*I*Filtf[i]*delaysamples*Filtdt);
		i++;
	}
	i=0;
	long double complex DelayedFilt[filtlength];
	while(i<filtlength) {
		DelayedFilt[i]=CavPoleFiltForTD[i]*Delay[i];
		i++;
	}

	//adds the negative frequencies to the filter to prepare for ifft
	i=0;
	long double complex NegFrequencies[filtlength];
	while(i<filtlength) {
		NegFrequencies[i]=conjl(DelayedFilt[(filtlength-1-i)]);
		i++;
	}

	long double complex CavPoleFiltTotal[filtlength*2-2];
	i=0;
	while(i<filtlength) {
		CavPoleFiltTotal[i]=DelayedFilt[i];
		i++;
	}
	while(i<(filtlength*2)-2) {
		CavPoleFiltTotal[i]=NegFrequencies[i-filtlength+1];
		i++;
	}


	// ifft (to make faster, don't create plan each time)
	int N=(filtlength*2)-2;
        fftw_complex *in, *out;
        fftw_plan p;
        in=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
        out=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
        p=fftw_plan_dft_1d(N,in,out,FFTW_BACKWARD,FFTW_ESTIMATE);
	
	i=0;
	while(i<N) {
		in[i]=CavPoleFiltTotal[i];
		i++;
	}

	fftw_execute(p);
		
	i=0;
	long double * CavPoleFiltTD=malloc(sizeof(long double)*N);

	while(i<N) {
                CavPoleFiltTD[i]=creall(out[i]/N);
                i++;
        }
	
	
	//adds Tukey window
	float * TukeyWindow=malloc(sizeof(float)*N);
	TukeyWindow=tukey(N);
	i=0;
        while(i<N) {
                CavPoleFiltTD[i]=CavPoleFiltTD[i]*TukeyWindow[i];
                i++;
        }

        element->fir_matrix=gsl_matrix_alloc(1,filtlength);
        for(i=0;i<filtlength;i++) {
        	gsl_matrix_set(element->fir_matrix,0,i,CavPoleFiltTD[i]);
        }


	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out); 


/*
	//prints out CavPoleFiltTD
	for(i=0;i<N;i++) {
		printf("%Lf,",CavPoleFiltTD[i]);
	}
*/

	return CavPoleFiltTD;

}

//computes the running average
void FindAverage(GSTLALFccUpdate *element, double newpoint, int i) {

	double updatedaverage;
	updatedaverage=(((i)/(i+1.0))*(element->currentaverage))+((1.0/(i+1.0))*newpoint);
        element->currentaverage=updatedaverage;
}


//sets up signaling

GstMessage *gstlal_fcc_update_message_fir_new(GSTLALFccUpdate *element,int filtlength)
{
        GArray *va=g_array_sized_new(FALSE,TRUE,sizeof(double),filtlength);

	int i=0;
	double data;
	for(i=0;i<filtlength;i++) {
		data=gsl_matrix_get(element->fir_matrix,0,i);
		g_array_append_val(va,data);
	}	
        GstStructure *s = gst_structure_new(
                "new_fir_matrix",
                "magnitude", G_TYPE_ARRAY, va,
                NULL
        );
        GstMessage *m = gst_message_new_element(GST_OBJECT(element), s);
        g_array_free(va,TRUE);

	printf("signal sent\n");

        //GST_MESSAGE_TIMESTAMP(m) = XLALGPSToINT8NS(&psd->epoch);

        return m;
}


static void rebuild_workspace_and_reset(GObject *object)
{
	return;
}
/*
*============================================================================
*   
*                                Signals
*     
*============================================================================
*/



/*
 *============================================================================
 *   
 *                     GstBaseTransform Method Overrides
 *     
 *============================================================================
 */


static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf) {

	GSTLALFccUpdate *element = GSTLAL_FCC_UPDATE(trans);
        GstMapInfo mapinfo;
        GstFlowReturn result = GST_FLOW_OK;


        GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
        gst_buffer_map(buf, &mapinfo, GST_MAP_READ);
        g_assert(mapinfo.size % sizeof(gdouble) == 0);

	gdouble *data, *end;
	data = (gdouble*) mapinfo.data;
	end = (gdouble*) (mapinfo.data+mapinfo.size);
	int averaging_length=floor(element->averaging_time*element->fccrate);
	int i=element->index;
	
	while(data<end) {

		if(i<averaging_length) {
			//continuously updated the average fcc_filter value
                        FindAverage(element,*data,i);
			i++;
			*data++;

		}
		else {
			printf("reached update length\n");
			//makes the new fir_matrix using the current average fcc_filter value
		        int filtlength=((int)(element->datarate*element->filterduration/2.0))*2;
		        long double *FccFilter=MakeFilter(element);

			//updated the values of the fir_matrix
			element->fir_matrix=gsl_matrix_alloc(1,filtlength);
			for(i=0;i<filtlength;i++) {
				gsl_matrix_set(element->fir_matrix,0,i,FccFilter[i]);
			}
			
			//sends a signal that the fir-matrix has been updated
			int messagesent;
                        g_object_notify(G_OBJECT(element), "fir-matrix");
                        messagesent=gst_element_post_message(
				GST_ELEMENT(element), 
				gstlal_fcc_update_message_fir_new(element,filtlength)
				);

			i=0;
		}

	}
	element->index=i;

	gst_buffer_unmap(buf,&mapinfo);
	return result;
}


/*
 *============================================================================
 *  _
 *                          GObject Method Overrides
 *     
 *============================================================================
 */


enum property {
        FIR_MATRIX = 1,
	DATA_RATE,
	FCC_RATE,
	FILTER_DURATION,
	FC_MODEL,
	AVERAGING_TIME
};


//Set FIR_MATRIX property

static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
        GSTLALFccUpdate *element = GSTLAL_FCC_UPDATE(object);

        GST_OBJECT_LOCK(element);
	
	switch (prop_id) {
        case FIR_MATRIX:
                if(element->fir_matrix)
                        gsl_matrix_free(element->fir_matrix);
                element->fir_matrix = gstlal_gsl_matrix_from_g_value_array(g_value_get_boxed(value));
                break;
	case DATA_RATE:
                element->datarate=g_value_get_int(value);
                break;
        case FCC_RATE:
                element->fccrate=g_value_get_int(value);
                break;
	case FILTER_DURATION:
                element->filterduration=g_value_get_double(value);
                break;
	case FC_MODEL:
		element->fcmodel = g_value_get_double(value);
		break;
	case AVERAGING_TIME:
		element->averaging_time = g_value_get_double(value);
		break;
        default:
                G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
        }

        GST_OBJECT_UNLOCK(element);
}

//get FIR_MATRIX property

static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
        GSTLALFccUpdate *element = GSTLAL_FCC_UPDATE(object);

        GST_OBJECT_LOCK(element);

        switch (prop_id) {
        case FIR_MATRIX:
                if(element->fir_matrix)
                        g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix(element->fir_matrix));
		break;
        case DATA_RATE:
                g_value_set_int(value,element->datarate);
		break;
        case FCC_RATE:
                g_value_set_int(value,element->fccrate);
                break;
        case FILTER_DURATION:
                g_value_set_double(value,element->filterduration);
                break;
	case FC_MODEL:
		g_value_set_double(value, element->fcmodel);
		break;
	case AVERAGING_TIME:
		g_value_set_double(value, element->averaging_time);
		break;
        default:
                G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
                break;
        }

        GST_OBJECT_UNLOCK(element);
}

//finalize

static void finalize(GObject *object)
{
        GSTLALFccUpdate *element = GSTLAL_FCC_UPDATE(object);
        if(element->fir_matrix) {
                gsl_matrix_free(element->fir_matrix);
                element->fir_matrix = NULL;
	}
        G_OBJECT_CLASS(gstlal_fcc_update_parent_class)->finalize(object);
}




//Class_init()
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
        GST_BASE_TRANSFORM_SINK_NAME,
        GST_PAD_SINK,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(
                "audio/x-raw, " \
                "rate = " GST_AUDIO_RATE_RANGE ", " \
                "channels = (int) 1, " \
                "format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
                "layout = (string) interleaved, " \
                "channel-mask = (bitmask) 0"
        )
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
        GST_BASE_TRANSFORM_SRC_NAME,
        GST_PAD_SRC,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(
                GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}") ", " \
                "layout = (string) interleaved, " \
                "channel-mask = (bitmask) 0"
        )
);



static void gstlal_fcc_update_class_init(GSTLALFccUpdateClass *klass)
{

        GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);    
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
        GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

        gst_element_class_set_details_simple(
                element_class,
                "Update the Fcc Filter",
                "Filter/Audio",
                "Makes a new fcc filter based on a new cavity pole frequency value",
                "Theresa Chmiel <theresa.chmiel@ligo.org>"
        );

        gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
        gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
        gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);


	g_object_class_install_property(
                gobject_class,
                FIR_MATRIX,
                g_param_spec_value_array(
                        "fir-matrix",
                        "FIR Matrix",
                        "Array of the cavity pole filter information in the time domain",
                        g_param_spec_value_array(
                                "response",
                                "Impulse Response",
                                "Array of amplitudes.",
                                g_param_spec_double(
                                        "amplitude",
                                        "Amplitude",
                                        "Impulse response sample",
                                        -G_MAXDOUBLE, G_MAXDOUBLE, DEFAULT_FIR_MATRIX,
                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
                                ),
                                G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
                        ),
                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
        	)
	);


        g_object_class_install_property(
                gobject_class,
                DATA_RATE,
                g_param_spec_int(
                        "data-rate",
                        "Data rate",
                        "The rate of the incoming data to be filtered (not the incoming fcc data).",
                        0, 
                        G_MAXINT,
                        16384, //Default rate
                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
                )
        );

        g_object_class_install_property(
                gobject_class,
                FCC_RATE,
                g_param_spec_int(
                        "fcc-rate",
                        "Fcc sample rate",
                        "The rate of the incoming fcc data (not the incoming data to be filtered).",
                        0,  
                        G_MAXINT,
                        16, //Default rate
                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
                )
        );

        g_object_class_install_property(
                gobject_class,
                FILTER_DURATION,
                g_param_spec_double(
                        "filter-duration",
                        "Filter duration",
                        "The the length of the desired filter to be generated in seconds.",
                        0.0,
                        G_MAXDOUBLE,
                        0.01, //Default filter duration
                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
                )
        );

	g_object_class_install_property(
		gobject_class,
		FC_MODEL,
		g_param_spec_double(
			"fcc-model",
			"F_cc model value",
			"The cavity pole frequency value from the static calibration model.",
			0.0,
			G_MAXDOUBLE,
			360.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		AVERAGING_TIME,
		g_param_spec_double(
			"averaging-time",
			"Averaging time",
			"The amount of time to averaging computed f_c values before constructing a new FIR filter.",
			1.0,
			G_MAXDOUBLE,
			1024.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

        gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
        gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);
}


//init

static void gstlal_fcc_update_init(GSTLALFccUpdate *element)
{
	g_signal_connect(G_OBJECT(element), "notify::fir-matrix",G_CALLBACK(rebuild_workspace_and_reset), NULL);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_prefer_passthrough (GST_BASE_TRANSFORM(element), TRUE);
}
      



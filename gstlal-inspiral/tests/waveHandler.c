//A handler script. Makes calls to LALSimulation C packages to generate spinning waveforms
//Called by cbc_template_iir.py
#include <Python.h>
//#include <numpy.h>
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <lal/LALConstants.h>
#include <lal/LALDatatypes.h>
#include <lal/Date.h>
#include <lal/FrequencySeries.h>
#include <lal/LALSimInspiral.h>
#include <lal/LALSimIMR.h>
#include <lal/XLALError.h>
#include <lal/LALAdaptiveRungeKutta4.h>




typedef struct tagGSParams {
    Approximant approximant;  /**< waveform family or "approximant" */
    LALSimulationDomain domain; /**< flag for time or frequency domain waveform */
    int phaseO;               /**< twice PN order of the phase */
    int ampO;                 /**< twice PN order of the amplitude */
    REAL8 phiRef;             /**< phase at fRef */
    REAL8 fRef;               /**< reference frequency */
    REAL8 deltaT;             /**< sampling interval */
    REAL8 deltaF;             /**< frequency resolution */
    REAL8 m1;                 /**< mass of companion 1 */
    REAL8 m2;                 /**< mass of companion 2 */
    REAL8 f_min;              /**< start frequency */
    REAL8 f_max;              /**< end frequency */
    REAL8 distance;           /**< distance of source */
    REAL8 inclination;        /**< inclination of L relative to line of sight */
    REAL8 s1x;                /**< (x,y,z) components of spin of m1 body */
    REAL8 s1y;                /**< z-axis along line of sight, L in x-z plane */
    REAL8 s1z;                /**< dimensionless spin, Kerr bound: |s1| <= 1 */
    REAL8 s2x;                /**< (x,y,z) component ofs spin of m2 body */
    REAL8 s2y;                /**< z-axis along line of sight, L in x-z plane */
    REAL8 s2z;                /**< dimensionless spin, Kerr bound: |s2| <= 1 */
    REAL8 lambda1;	      /**< (tidal deformability of mass 1) / (total mass)^5 (dimensionless) */
    REAL8 lambda2;	      /**< (tidal deformability of mass 2) / (total mass)^5 (dimensionless) */
    LALSimInspiralWaveformFlags *waveFlags; /**< Set of flags to control special behavior of some waveform families */
    LALSimInspiralTestGRParam *nonGRparams; /**< Linked list of non-GR parameters. Pass in NULL for standard GR waveforms */
    int axisChoice;           /**< flag to choose reference frame for spin coordinates */
    int inspiralOnly;         /**< flag to choose if generating only the the inspiral 1 or also merger and ring-down*/
    char outname[256];        /**< file to which output should be written */
    int ampPhase;
    int verbose;
} GSParams;



static PyObject *PySpinWaveGenerate(PyObject *self, PyObject *args)
{
//    npy_intp *dims = NULL;
    double m1, m2, dist, incl, f_min, f_max, fRef, s1X, s1Y, s1Z, s2X, s2Y, s2Z, nPN;
    PyArrayObject *dataAmp, *dataPhase;
    REAL8TimeSeries *hplus = NULL;
    REAL8TimeSeries *hcross = NULL;
    npy_intp dims[1] = {0};

    if(!PyArg_ParseTuple(args, "dddddddddddddd", &m1, &m2, &dist, &incl, &f_min, &f_max, &fRef, &nPN, &s1X, &s1Y, &s1Z, &s2X, &s2Y, &s2Z)) return NULL;



    //Source: ibs/src/lalsuite/lalsimulation/test/GenerateSimulation.c
    //Just using the parameter type to keep track of the variable types
    GSParams *params;
    params = (GSParams *) XLALMalloc(sizeof(GSParams));
    memset(params, 0, sizeof(GSParams));

    /* Set default values to the arguments */
    
    params->waveFlags = XLALSimInspiralCreateWaveformFlags();
    params->nonGRparams = NULL;
    params->approximant = SpinTaylorT4;
    params->domain = LAL_SIM_DOMAIN_TIME; //SpinTaylorF2 only has single component transverse spin
    params->phaseO = (int)nPN;
    params->ampO = 0;
    params->phiRef = 0;
    params->deltaT = 1./4096.; //0.000244140625 - is this below double precision? Don't think so
    params->deltaF = 0.125;
    params->m1 = m1 * LAL_MSUN_SI;
    params->m2 = m2 * LAL_MSUN_SI;
    params->f_min = f_min;
    params->fRef = fRef;
    params->f_max = f_max; // Generate as much as possible
    params->distance = dist * 1e6 * LAL_PC_SI;
    params->inclination = incl;
    params->s1x = s1X;
    params->s1y = s1Y;
    params->s1z = s1Z;
    params->s2x = s2X;
    params->s2y = s2Y;
    params->s2z = s2Z;
    params->lambda1 = 0.;
    params->lambda2 = 0.;
    strncpy(params->outname, "simulation.dat", 256); // output to this file //Not used
    params->ampPhase = 0; // output phase and amplitude //this program ONLY does amp/phase
    params->verbose = 0; // No verbosity

    XLALSimInspiralChooseTDWaveform(&hplus, &hcross, params->phiRef, 
	    params->deltaT, params->m1, params->m2, params->s1x, 
	    params->s1y, params->s1z, params->s2x, params->s2y, 
	    params->s2z, params->f_min, params->fRef, 
	    params->distance, params->inclination, params->lambda1, 
	    params->lambda2, params->waveFlags,
	    params->nonGRparams, params->ampO, params->phaseO,
	    params->approximant);


 //   printf("Make the data arrays \n");

/***** INLINE FUNCTIONS TO UNPACK PHASE ******/
    REAL8 *dataPtr1 = hplus->data->data;
    REAL8 *dataPtr2 = hcross->data->data;
    REAL8 thresh=3.14159; // Threshold to determine phase wrap-around //This was 5. Changed 23/07/2014
    REAL8 *phase;
    REAL8 *amp;
    REAL8 *phaseUW;

    phase = (REAL8 *) malloc(hplus->data->length*sizeof(REAL8));
    amp = (REAL8 *) malloc(hplus->data->length*sizeof(REAL8));
    phaseUW = (REAL8 *) malloc(hplus->data->length*sizeof(REAL8));

    size_t i;

    //Set amp and phase arrays
    for (i=0; i < hplus->data->length; i++)
    {
        amp[i] = sqrt( dataPtr1[i]*dataPtr1[i] + dataPtr2[i]*dataPtr2[i]);
        phase[i] = atan2(hcross->data->data[i], hplus->data->data[i]);
    }
    /**Unwind phase**/
    int cnt = 0; // # of times wrapped around branch cut
    phaseUW[0] = phase[0];
    for(i=1; i<hplus->data->length; i++) {
        if(phase[i-1] - phase[i] > thresh) // phase wrapped forward
            cnt += 1;
        else if(phase[i] - phase[i-1] > thresh) // phase wrapped backwards
            cnt -= 1;
        phaseUW[i] = phase[i] + cnt * LAL_TWOPI;
    }
    /**Scale phase to be zero at t=0**/
    for(i=0; i<hplus->data->length;i++){
	phaseUW[i] = phaseUW[i]-phaseUW[0];

    }

    //This is dodgy...
//    void *ptrAmp;
//    void *ptrPhaseUW;
//    ptrAmp = &amp;
//    ptrPhaseUW = &phaseUW;

/*********************************************/
    
    //Now repackage them as a Numpy tuple and return
    dims[0] = (npy_intp)hcross->data->length;

//    dataAmp = (PyArrayObject *) PyArray_SimpleNewFromData(1,dims,NPY_DOUBLE,ptrAmp);
//    dataPhase = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims,NPY_DOUBLE,ptrPhaseUW);
    
    dataAmp = (PyArrayObject *) PyArray_SimpleNewFromData(1,dims,NPY_DOUBLE,amp);
    dataPhase = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims,NPY_DOUBLE,phaseUW);

    PyObject *tupleresult = PyTuple_New(2);
    PyTuple_SetItem(tupleresult,0,PyArray_Return(dataAmp));
    PyTuple_SetItem(tupleresult,1,PyArray_Return(dataPhase));

    free(hcross);
    free(hplus);
    free(params);
    free(phase);
    return tupleresult;

}

static PyMethodDef methods[]={
    { "genwave", PySpinWaveGenerate, METH_VARARGS, "Generate a waveform using the SpinTaylorT4 method. Arguments are m1, m2, distance, inclination, flower, fupper, freference, s1x, s1y, s1z, s2x, s2y, s2z"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initwaveHandler(void){
    (void)Py_InitModule("waveHandler",methods);//, "A handler file for generating LAL simulation spinning waveforms");
    import_array()
}




int
main(int argc, char *argv[])
{
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initwaveHandler();

    return 0;
}


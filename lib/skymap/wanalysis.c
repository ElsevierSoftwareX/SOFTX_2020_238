// C headers

#include <math.h>
#include <string.h>

// LAL headers

#include <lal/LALConstants.h>
#include <lal/Skymap.h>
#include <lal/Random.h>

// Omega headers

#include "wanalysis.h"

void analysis_default_construct(analysis* a)
{
    a->n_detectors = 0;
    a->rate = 0;
    a->n_directions = 0;
    a->directions = 0;
    a->min_t = log(0);
    a->max_t = -log(0);
    a->delta_t = 1;
    a->log_skymap = 0;
    a->p_realloc = realloc;
    a->p_free = free;
    size_t i;
    for (i = 0; i != XLALSKYMAP_N; ++i)
    {
        a->xSw_real[i] = 0;
	a->xSw_imag[i] = 0;
        a->min_ts[i] = 0;
	a->max_ts[i] = 0;
        a->calibration_error[i] = 0.1;
    }
}

void analysis_default_directions(analysis* a)
{
    size_t u = 360 * 10 / 4;
    size_t v = 180 * 10 / 4;

    a->n_directions = u * v;
    a->directions = a->p_realloc(0, a->n_directions * 2 * sizeof(double));
    
    double* p = a->directions;
    size_t j;
    for (j = 0; j != v; ++j)
    {
        double theta = (0.4 * (j + .5)) * LAL_PI / 180; 
        size_t i;
        for (i = 0; i != u; ++i)
	{
	    double phi = (0.4 * (i + .5)) * LAL_PI / 180;
	    *(p++) = theta;
	    *(p++) = phi;
	}
    }
}

int analysis_identify_detector(const char* s)
{
    // truncate the string to first two letters
    char c[3];
    c[0] = s[0];
    c[1] = s[1];
    c[2] =   0;
    // compare with detector codes
    // TODO: we can probably replace this with iteration through
    //       LALCachedDetectors via LALDetector::frDetector.prefix
    if (strcmp(c, "T1") == 0) 
    {
    	return LAL_TAMA_300_DETECTOR;
    }
    else if (strcmp(c, "V1") == 0) 
    {
        return LAL_VIRGO_DETECTOR;
    }
    else if (strcmp(c, "G1") == 0) 
    {
        return LAL_GEO_600_DETECTOR;
    }
    else if (strcmp(c, "H2") == 0) 
    {
        return LAL_LHO_2K_DETECTOR;
    }
    else if (strcmp(c, "H1") == 0) 
    {
        return LAL_LHO_4K_DETECTOR;
    }
    else if (strcmp(c, "L1") == 0) 
    {
         return LAL_LLO_4K_DETECTOR;
    }
    else 
    {
        return -1;
    }
}

analysis* analysis_example(void)
{
    analysis* a = (analysis*) malloc(sizeof(analysis));
    
    analysis_default_construct(a);
    analysis_default_directions(a);

    double duration = 1; // seconds of data available
    RandomParams* rng = XLALCreateRandomParams(0);

    a->n_detectors = 3;

    a->detectors[0] = LAL_LHO_4K_DETECTOR;
    a->detectors[1] = LAL_LLO_4K_DETECTOR;
    a->detectors[2] = LAL_VIRGO_DETECTOR;
    
    a->rate = 512; // Hz
    
    a->min_t = 0.4; // seconds at earth barycenter
    a->max_t = 0.6;

    a->log_skymap = (double*) a->p_realloc(0, sizeof(double) * a->n_directions);

    size_t i;
    for (i = 0; i != a->n_detectors; ++i)
    {
        a->wSw[i] = 1.0; // white noise and unit template
	
	a->xSw_real[i] = (double*) a->p_realloc(0, sizeof(double) * a->rate * duration);
        a->xSw_imag[i] = (double*) a->p_realloc(0, sizeof(double) * a->rate * duration);
        
        size_t j;
	for (j = 0; j != (a->rate * duration); ++j)
	{
	    // make gaussian noise
	    a->xSw_real[i][j] = XLALUniformDeviate(rng) * sqrt(a->wSw[i]);
	    a->xSw_imag[i][j] = XLALUniformDeviate(rng) * sqrt(a->wSw[i]);
	}

        // allowed time ranges at each detector
	a->min_ts[i] = 0.49; // seconds from reference time
	a->max_ts[i] = 0.51;

    }

    analyze(a);

    // struct now contains pointers to all the input data used and the results

    return a;
}

void analyze(analysis* s)
{

    XLALSkymapPlanType plan;

    // Construct the network
    XLALSkymapPlanConstruct(s->rate, s->n_detectors, s->detectors, &plan);

    // Working array
    double* p_t = 0;

    size_t i;
    for (i = 0; i != s->n_directions; ++i) {
        XLALSkymapDirectionPropertiesType properties;
        XLALSkymapDirectionPropertiesConstruct(&plan, s->directions + (i * 2), &properties);

        double t_begin = s->min_t;
        double t_end = s->max_t;

        size_t j;
        for (j = 0; j != s->n_detectors; ++j) {
            t_begin = fmax(t_begin, s->min_ts[j] - properties.delay[j]);
            t_end = fmin(t_end, s->max_ts[j] - properties.delay[j]);
        }

        if (t_begin < t_end) {

            t_begin -= 0.0001;
            t_end   += 0.0001;

            XLALSkymapKernelType kernel;
            // strict version assumes no calibration error (compatible with LAL release)
            //XLALSkymapKernelConstruct(&plan, &properties, wSw, &kernel);
            // loose version allows for amplitude calibration error (requires LAL built from repository head as of 4/18/10)
            XLALSkymapUncertainKernelConstruct(&plan, &properties, s->wSw, s->calibration_error, &kernel);

            double old_p;
            double new_p = log(0);

            // take at least 10 samples
            double dt = fmin(s->delta_t, (t_end - t_begin) * 0.1);

            do {
                size_t p_t_size = (size_t) (ceil((t_end - t_begin) / dt));
                p_t = (double*) s->p_realloc(p_t, p_t_size * sizeof(double));
                double t = t_begin;
                for (j = 0; j != p_t_size; ++j)
                {
                      double log_p_real, log_p_imag;
                      XLALSkymapApply(&plan, &properties, &kernel, s->xSw_real, t, &log_p_real);
                      XLALSkymapApply(&plan, &properties, &kernel, s->xSw_imag, t, &log_p_imag);
                      p_t[j] = log_p_real + log_p_imag;
                      t += dt;
                }
                old_p = new_p;
                new_p = XLALSkymapLogTotalExp(&p_t[0], &p_t[0] + p_t_size) - log(p_t_size);
                dt = dt / 2;
            } while (fabs(new_p - old_p) > .1);
            s->log_skymap[i] = new_p + log(t_end - t_begin);
        }
        else {
            // the times of interest in each detector exclude this
            // direction
            s->log_skymap[i] = log(0);
        }
    }

    s->p_free(p_t);
    
}


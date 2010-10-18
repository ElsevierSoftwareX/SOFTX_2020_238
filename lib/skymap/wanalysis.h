// C headers

#include <stdlib.h>

// LAL headers

#include <lal/Skymap.h>

// Structure to hold analysis state

typedef struct
{

    // Number of detectors <= XLALSKYMAP_N

    size_t n_detectors;
    
    // Array holding LAL_DETECTOR codes to identify the instruments
    
    int detectors[XLALSKYMAP_N];

    // Sample rate (integer Hz) of timeseries
    
    int rate;
    
    // Matched filter normalization
    // Waveform * NoiseCovariance^{-1} * Waveform

    double wSw[XLALSKYMAP_N];
    
    // Time series unnormalized matched filter
    // Data * NoiseCovariance^{-1} * Waveform
    
    double* xSw_real[XLALSKYMAP_N]; 
    double* xSw_imag[XLALSKYMAP_N];
    
    // The waveform is assumed to be of the form A(t) e^{i phi(t)} with
    // a cosine-like (real) and sine-like (imaginary) component that each
    // have the same normalization term
    
    // Number of directions to compute for
    
    size_t n_directions;
    
    // Array of packed theta and phi values 
    // {theta[0], phi[0], theta[1], phi[1], ... }
    
    double* directions; // Directions to sample at
        
    // Timing information
    
    double min_t; // Earth Barycenter time limits
    double max_t;
    
    double delta_t; // Integration step size hint

    double min_ts[XLALSKYMAP_N]; // Individual instrument time limits
    double max_ts[XLALSKYMAP_N];

    // Array to store the allowed amplitude calibration error
    
    double calibration_error[XLALSKYMAP_N];
        
    // Output array (with n_directions elements)

    double* log_skymap;
    
    // Memory management hooks

    void* (*p_realloc)(void*, size_t);
    void (*p_free)(void*);
    
    
} analysis;

void analysis_default_construct(analysis* a);
void analysis_default_directions(analysis* a);

// Perform the analysis

void analyze(analysis* s);

// Convert between detector strings and LALDetectector identifiers

int analysis_identify_detector(const char* c);


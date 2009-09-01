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


/* standard includes */
#include <stdio.h>
#include <math.h>
#include <complex.h>

/* glib/gstreamer includes */
#include <glib.h>

/* gsl includes */
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

/* LAL Includes */

#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/Date.h>
#include <lal/Sequence.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/LALNoiseModels.h>
#include <lal/Units.h>
#include <lal/LALComplex.h>
#include <lal/Window.h>
#include <lal/FindChirp.h>
#include <lal/AVFactories.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/FindChirpTD.h>
#include <lal/FindChirp.h>
#include <lal/LALError.h>
#include <lal/LALStdio.h>
#include <lal/TimeFreqFFT.h>
#include <lal/RealFFT.h>
#include <lal/LALInspiral.h>

/* gstlal includes */
#include "gstlal.h"
#include "low_latency_inspiral_functions.h"
#include "gstlal_whiten.h"

#define TEMPLATE_DURATION 128	/* seconds */
#define ATEMPS 101

#define LAL_CALL( function, statusptr ) \
  ((function),lal_errhandler(statusptr,#function,__FILE__,__LINE__,rcsid))


static int SPAWaveform (InspiralTemplate *, REAL8, COMPLEX16FrequencySeries *);

static double time_to_freq(double M, double time)
  {
  /* This function gives the instantaneous frequency at a given time based
   * on the quadrupole approximation.  It is bound to be a bit off from other
   * template families so use it with caution 
   */
  double c3_8 = 3.0/8.0;
  double c5_256 = 5.0/256.0;
  double Mg = M*LAL_MTSUN_SI;
  return (1.0/(LAL_PI*Mg)) * (pow((c5_256)*(Mg/(time)),c3_8));
  }


static double freq_to_time(double M, double freq)
  {
  /* This function gives the time for a given frequency based
   * on the quadrupole approximation.  It is bound to be a bit off from other
   * template families so use it with caution 
   */
  double Mg = M*LAL_MTSUN_SI;
  return 5./256. * Mg / pow(LAL_PI*Mg*freq,8./3.);
  }


static void quadrupole_template(double M, double duration,
                          REAL8TimeSeries *template)
  {
  /* time (prior to coalescence) at which template goes through the Nyquist
   * frequency */
  double tstart = freq_to_time(M, 0.5 / template->deltaT);
  /* number of samples to compute */
  unsigned numsamps = round(duration / template->deltaT);
  unsigned i;

  M *= LAL_MTSUN_SI;

  if(numsamps > template->data->length)
    numsamps = template->data->length;

  memset(template->data->data, 0, (template->data->length - numsamps) * sizeof(*template->data->data));
  for (i=0; i< numsamps; i++)
    {
    /* t must equal -tstart in the last bin */
    double t = -duration + template->deltaT * (i + 1) - tstart;
    template->data->data[template->data->length - numsamps + i] = 4.0 * M * pow(5.0/256.0*(M/-t),0.25) * sin(-2.0 * pow(-t/(5.0*M),(5.0/8.0)));
    }
  }


static double calculate_real_time_shift(
			COMPLEX16FrequencySeries *template_reference,
			COMPLEX16FrequencySeries *template,
			COMPLEX16FrequencySeries *template_product, 
			COMPLEX16TimeSeries *convolution,
                        COMPLEX16FFTPlan *revplan,
			double deltaT,
			int column
 			) 
 {
   double real_max_conv;
   double im_max_conv;
   double real_tshift;
   double im_tshift;
   unsigned i;
   
   /* Calculates the product */ 

  for (i=0; i<template->data->length; i++)
    {
    /*Real part*/
    template_product->data->data[i].re = (template->data->data[i].re*template_reference->data->data[i].re) + (template->data->data[i].im*template_reference->data->data[i].im);
    /*Imaginary part*/
    template_product->data->data[i].im = (template->data->data[i].re*template_reference->data->data[i].im) - (template->data->data[i].im*template_reference->data->data[i].re);
    } 

  /* Inverse Fourier transform */
 
  XLALCOMPLEX16FreqTimeFFT(convolution, template_product, revplan); // Error message?


  /* Looks for the peak */  
  
  real_max_conv = fabs(convolution->data->data[0].re);
  im_max_conv = fabs(convolution->data->data[0].im);

  /* Real part */
  
  //char buffer[100];
  //sprintf(buffer, "real_convolution_%d.txt", column);
  //FILE *realfile = fopen(buffer, "w");
  for (i=0; i<convolution->data->length; i++)
    {
    /*if(realfile)
      {
      fprintf(realfile, "%u %e\n", i, convolution->data->data[i].re);
      //fprintf(stderr, "%u %e\n", i, convolution->data->data[i].re);
      }*/

    if(fabs(convolution->data->data[i].re) > real_max_conv) 
      {
      real_max_conv=fabs(convolution->data->data[i].re);
      real_tshift = 128-i*deltaT;
      }
    }
  fprintf(stderr, "Real time shift %e\n", real_tshift);
  //fflush(realfile);
  //fclose(realfile);

  /* Imaginary part */
  
  /*char imbuffer[100];
  sprintf(imbuffer, "im_convolution_%d.txt", column);
  FILE *imfile = fopen(imbuffer, "w");*/
  for (i=0; i<convolution->data->length; i++)
    {
    /*if(imfile)
      {
      fprintf(imfile, "%u %e\n", i, convolution->data->data[i].re);
      //fprintf(stderr, "%u %e\n", i, convolution->data->data[i].re);
      }*/

    if(fabs(convolution->data->data[i].im) > im_max_conv)
      {
      im_max_conv=fabs(convolution->data->data[i].im);
      im_tshift = 128-i*deltaT;
      }
    }
  fprintf(stderr, "Imag. time shift %e\n", im_tshift);
  //fflush(imfile);
  //fclose(imfile);

  return real_tshift;
 }



int generate_autocorrelation_bank(
			gsl_matrix *A,			
			COMPLEX16FrequencySeries *template,
			COMPLEX16FrequencySeries *template_product, 
			COMPLEX16TimeSeries *autocorrelation,
                        COMPLEX16FFTPlan *revplan,
			int U_column,
			int autocorr_numsamps,
			double base_sample_rate
 			) 
 {
   double real_max;
   double im_max;
   int im_peak;
   int real_peak;
   unsigned i;
   gsl_vector_view col;
   gsl_vector_view autocorr;
  
   /* Calculates the product the autocorrelation function*/ 

  for (i=0; i<template->data->length; i++)
    {
    complex double z = XLALCOMPLEX16Abs2(template->data->data[i]) * cexp(-I*LAL_TWOPI* (template->f0 + i*template->deltaF) * (autocorr_numsamps-1)/2 / base_sample_rate);
    LAL_SET_COMPLEX(&template_product->data->data[i], creal(z), cimag(z));
    } 


  /* Inverse Fourier transform */
 
  if(XLALCOMPLEX16FreqTimeFFT(autocorrelation, template_product, revplan)) 
    return -1;
  
  /*
   * Looks for the peak 
   */  
  
  real_max = fabs(autocorrelation->data->data[0].re);
  im_max = fabs(autocorrelation->data->data[0].im);


  /* Real part */
  
  for (i=0; i<autocorrelation->data->length; i++)
    {
    if(fabs(autocorrelation->data->data[i].re) > real_max) 
      {
      real_max=fabs(autocorrelation->data->data[i].re);
      real_peak = i;
      }
    }
  fprintf(stderr, "Real peak %e\n", real_max);

  if(real_max!=1) // Normalizes
  {
   for (i=0; i<autocorrelation->data->length; i++)
     {
      autocorrelation->data->data[i].re = autocorrelation->data->data[i].re/real_max;  
     }
  }


 /* Imaginary part */
  for (i=0; i<autocorrelation->data->length; i++)
    {
    if(fabs(autocorrelation->data->data[i].im) > im_max)
      {
      im_max=fabs(autocorrelation->data->data[i].im);
      im_peak = i;
      }
    }
  
  if(im_max!=1) // Normalizes
  {
   for (i=0; i<autocorrelation->data->length; i++)
     {
      autocorrelation->data->data[i].im = autocorrelation->data->data[i].im/im_max;  
     }
  }

  /*
   * Creates the autocorrelation matrix
   */

  /* Real part only */

  col = gsl_matrix_column(A, U_column);
  autocorr = gsl_vector_view_array_with_stride((double*) autocorrelation->data->data, 2, col.vector.size); // assuming the peak is in t=0
  
  fprintf(stderr, "Autocorrelation completed successfully!\n");

  if (gsl_vector_memcpy(&col.vector, &autocorr.vector))
   {
   fprintf(stderr, "create autocorrelation matrix FAILED\n");
   return -1;
   }

  return 0;
}



int create_template_from_sngl_inspiral(
                       InspiralTemplate *bankRow,
                       gsl_matrix *U, 
		       gsl_matrix *A,
                       gsl_vector *chifacs,
                       int fsamp,
                       int downsampfac, 
                       double t_end,
                       double t_total_duration,
		       int autocorr_numsamps,
		       double tshift,
                       int U_column,
                       COMPLEX16TimeSeries *template_out,
		       COMPLEX16TimeSeries *convolution,
		       COMPLEX16TimeSeries *autocorrelation,
		       COMPLEX16TimeSeries *short_autocorr,
		       COMPLEX16FrequencySeries *template_product,
                       COMPLEX16FrequencySeries *fft_template,
		       COMPLEX16FrequencySeries *fft_template_full,
		       COMPLEX16FrequencySeries *fft_template_full_reference,
                       REAL8FFTPlan *fwdplan,
                       COMPLEX16FFTPlan *revplan,
                       REAL8FrequencySeries *psd
                       )

  {
  unsigned i;
  int t_total_length = round(t_total_duration * fsamp);	/* length of the template */
  double norm;
  gsl_vector_view col;
  gsl_vector_view tmplt;
  fprintf(stderr, "fsamp %d downsampfac %d t_end %e t_total_duration %e\n", fsamp, downsampfac,t_end,t_total_duration);
  SPAWaveform (bankRow, template_out->deltaT, fft_template);
 
  /*
   * Whiten the template.
   */

  if(!XLALWhitenCOMPLEX16FrequencySeries(fft_template,psd)) {
    fprintf(stderr, "create_template_from_sngl_inspiral XLALWhitenCOMPLEX16FrequencySeries(fft_template,psd) FAILED\n");
    return -1;
    }
  /* compute the quadrature phases now we need a complex frequency series that
   * is twice as large.  We'll store the negative frequency components that'll
   * give the sine and cosine phase */
  fft_template->data->data[0].im = 0;
  fft_template->data->data[0].re = 0;
  fft_template->data->data[fft_template->data->length-1].im = 0;

  for (i = 0; i < fft_template->data->length; i++)
    {
    /* conjugate times 2 */
    fft_template_full->data->data[fft_template->data->length - 1 - i].re = 2.0 * fft_template->data->data[i].re;
    fft_template_full->data->data[fft_template->data->length - 1 - i].im = -2.0 * fft_template->data->data[i].im;
    }

  memset(&fft_template_full->data->data[fft_template->data->length], 0, (fft_template_full->data->length - fft_template->data->length) * sizeof(*fft_template_full->data->data));


  if(XLALCOMPLEX16FreqTimeFFT(template_out, fft_template_full, revplan)) {
    fprintf(stderr, "create_template_from_sngl_inspiral XLALCOMPLEX16FreqTimeFFT(template_out, fft_template_full, revplan) FAILED\n");
    return -1;
    }

 /* MIREIA */

  bankRow->order = LAL_PNORDER_TWO; //bankRow???

  if(tshift!=0)
  {
    //tshift = calculate_real_time_shift(fft_template_full_reference, fft_template_full, template_product, convolution, revplan, template_out->deltaT, U_column); // deltaT??

    /*Boundaries*/
    //if(gstlal_spa_chirp_time(bankRow->mass1+bankRow->mass2, bankRow->eta, 150.0, bankRow->order)>(t_end-tshift)) fprintf(stderr, "ERROR!! f>f_nyq !!! Tshift %e is too big!!\n", tshift);
      //else fprintf(stderr, "COOL! Tshift inside the function %e \n", tshift);

    /*Shifts the template*/
    //XLALResizeCOMPLEX16TimeSeries(template_out, (int) round(tshift*fsamp), template_out->data->length);
  }


  /*
   * Normalize the template.
   *
   * The normalization is (s|s) = \vec{s} \cdot \vec{s} = 2
   *
   * The right-hand side equalling 2 yields an "SNR" that this consistent
   * with the inspiral definition, wherein each of the real and imaginary
   * components has a mean square of 1, so the mean square of the complex
   * sample is 2.
   *
   * Multiplying the vector by \sqrt{2 / \sum s^{2}} makes the dot product
   * of the vector with itself equal to 2.
   *
   * Because the sample rate will be adjusted (samples removed from the
   * time series) the normalization factor must be scaled up by a factor of
   * \sqrt{down-sample factor} so that the remaining samples alone will
   * have a dot product with themselves equal to 2.
   */

  norm = sqrt(2 * downsampfac / XLALCOMPLEX16SequenceSumSquares(template_out->data, template_out->data->length - t_total_length, t_total_length));
	
	

  /* Prints the real part of the template!!*/ 

  /*char buffer[50];
  sprintf(buffer, "conv_negative_unshifted_whitened_template_%d.txt", U_column);

  FILE *file = fopen(buffer, "w");
  if(file)
  {
    for (i=0; i<template_out->data->length; i++)
    {
    fprintf(file, "%e %e\n", i*(double)template_out->deltaT, template_out->data->data[i].re);
    }
    fflush(file);
    fclose(file);
  }*/



 /* Autocorrelation
  * */ 
  fprintf(stderr, "doing autocorrelation\n");

  if(generate_autocorrelation_bank(A, fft_template_full, template_product, autocorrelation, revplan, U_column, autocorr_numsamps, fsamp)<0)
  	{
	fprintf(stderr, "autocorrelation FAILED\n");
	return -1;
	}

  /*
   * Extract a piece of the template.
   */

  /* Real part */
  /* there are twice as many waveforms as templates hence the multiplying 
   * U_colum by 2*/
  col = gsl_matrix_column(U,2*U_column);
  tmplt = gsl_vector_view_array_with_stride((double *) (template_out->data->data + template_out->data->length - (int) round(t_end * fsamp)), 2*downsampfac, col.vector.size);
  if (gsl_vector_memcpy(&col.vector, &tmplt.vector)) {
    fprintf(stderr, "create_template_from_sngl_inspiral() gsl_vector_memcpy(&col.vector, &tmplt.vector) FAILED\n");
    return -1;
    }

  if (gsl_vector_scale(&col.vector, norm)) {
    fprintf(stderr, "create_template_from_sngl_inspiral() gsl_vector_scale(&col.vector, norm) FAILED\n"); 
    return -1;
    }


  /* Imaginary part */
  col = gsl_matrix_column(U,2*U_column + 1);
  tmplt = gsl_vector_view_array_with_stride((double *) (template_out->data->data + template_out->data->length - (int) round(t_end * fsamp)) + 1, 2*downsampfac, col.vector.size);

  if (gsl_vector_memcpy(&col.vector, &tmplt.vector)) {
    fprintf(stderr, "create_template_from_sngl_inspiral() gsl_vector_memcpy(&col.vector, &tmplt.vector) FAILED\n");
    return -1;
    }

  if (gsl_vector_scale(&col.vector, norm)) {
    fprintf(stderr, "create_template_from_sngl_inspiral() gsl_vector_scale(&col.vector, norm) FAILED\n");
    return -1;
    }


  /*
   * Compute the \Chi^2 factor.
   */

  gsl_vector_set(chifacs,U_column,pow(gsl_blas_dnrm2(&col.vector),2.0));

  return 0;
  }



int compute_time_frequency_boundaries_from_bank(char * bank_name,
                                                double min_subtemplate_samples,
						double base_sample_rate,
						double f_lower,
						gsl_vector **sample_rates,
						gsl_vector **start_times,
                                                gsl_vector **stop_times,
						int verbose)
  {
  InspiralTemplate *bankHead     = NULL;
  int numtemps = InspiralTmpltBankFromLIGOLw( &bankHead, bank_name,-1,-1);  
  double maxMass = 0;
  double minMass = 0;
  double minChirpMass = 0;
  double minFreq = 0;
  double Nyquist = base_sample_rate / 2.0;
  double base_time = 0;
  double duration = 0;
  double freq = 0;
  double time = 0;
  double sampleRate = 0;
  double prev_time = 0;
  int veclength = 0;
  
  if (verbose) fprintf(stderr, "Read %d templates from file %s\n",numtemps, bank_name);

  /* Compute the minimum and maximum masses as well as the minimum chirpmass 
   * The Masses will be used to determine frequency boundaries.  However the 
   * chirp mass will be used to determine the duration */
  maxMass = bankHead->mass1+bankHead->mass2;
  minChirpMass = bankHead->chirpMass;
  while (bankHead)
    {
    if ( (bankHead->mass1 + bankHead->mass2) > maxMass)
          maxMass = bankHead->mass1 + bankHead->mass2;
    if ( (bankHead->mass1 + bankHead->mass2) < minMass)
              minMass = bankHead->mass1 + bankHead->mass2;
    if (bankHead->chirpMass < minChirpMass) minChirpMass = bankHead->chirpMass;
    bankHead = bankHead->next;
    }

  /* This assumes that the maximum integration point is light ring */
  minFreq = 1.0 / (pow(3.0,1.5) * minMass) / LAL_PI / LAL_MTSUN_SI;
  /* if the lowest termination frequency exceeds nyquist then we should set */
  /* it to Nyquist */
  if (minFreq > Nyquist) minFreq = Nyquist;
  if (verbose) fprintf(stderr,"Lowest LR frequency in the bank = %f minChirp = %f\n",minFreq,minChirpMass);

 /* We need to start at time defined by the first frequency this could be a
   * long time before coalescence if the nyquist frequency is low compared to 
   * the light ring frequency
   */
  base_time = freq_to_time(minChirpMass,minFreq);
  duration = freq_to_time(minChirpMass,f_lower) - base_time;
  sampleRate = 2.0 * (int) round(pow(2.0,floor(log(minFreq)/log(2.0))));
  freq = sampleRate;
  time = 0;
  prev_time = 0;
  if (verbose) fprintf(stderr, "sampleRate is %f base time is %f duration is %f\n",sampleRate,base_time,duration-base_time);
  
  while (freq > f_lower)
    {
    time+=min_subtemplate_samples/sampleRate;
    if (verbose) fprintf(stderr, "Sample rate is %f interval is [%f, %f)\n",sampleRate,prev_time,time);
    prev_time = time;
    freq = time_to_freq(minChirpMass,time+base_time);
    sampleRate = 2.0 * (pow(2.0,ceil(log(freq)/log(2.0))));
    veclength++;
    }

  /* Allocate the return vectors */
  *start_times = gsl_vector_calloc(veclength);
  *stop_times = gsl_vector_calloc(veclength);
  *sample_rates = gsl_vector_calloc(veclength);

  /* populate the vectors */
  veclength = 0;
  base_time = freq_to_time(minChirpMass,minFreq);
  duration = freq_to_time(minChirpMass,f_lower) - base_time;
  sampleRate = 2.0 * (int) floor(pow(2.0,floor(log(minFreq)/log(2.0))));
  freq = sampleRate;
  time = 0;
  prev_time = 0;
  while (freq > f_lower)
    {
    time+=min_subtemplate_samples/sampleRate;
    gsl_vector_set(*start_times,veclength,prev_time);
    gsl_vector_set(*stop_times,veclength,time);
    gsl_vector_set(*sample_rates,veclength,sampleRate);  
    prev_time = time;
    freq = time_to_freq(minChirpMass,time+base_time);
    sampleRate = 2.0 * (pow(2.0,ceil(log(freq)/log(2.0))));
    veclength++;
    }

  return 0;
  }


int gstlal_gsl_linalg_SV_decomp_mod(
  gsl_matrix **U,
  gsl_matrix **V,
  gsl_vector **S)
{
  int result;
  int transposed;
  gsl_vector *work_space;
  gsl_matrix *work_space_matrix;

  if((*U)->size1 < (*U)->size2)
  {
    if(not_gsl_matrix_transpose(U))
      return -1;
    transposed = 1;
  }
  else
    transposed = 0;

  *S = gsl_vector_calloc((*U)->size2);
  *V = gsl_matrix_calloc((*U)->size2, (*U)->size2);

  work_space = gsl_vector_calloc((*U)->size2);
  work_space_matrix = gsl_matrix_calloc((*U)->size2, (*U)->size2);
  /* Alternate GSL SVD functions */
  /*return gsl_linalg_SV_decomp(U, V, S, work_space);*/
  /*return gsl_linalg_SV_decomp_jacobi(U, V, S);*/
  result = gsl_linalg_SV_decomp_mod(*U, work_space_matrix, *V, *S, work_space);
  gsl_matrix_free(work_space_matrix);
  gsl_vector_free(work_space);

  if(transposed)
  {
    gsl_matrix *tmp = *U;
    *U = *V;
    *V = tmp;
  }

  return result;
}


int generate_bank(
                      gsl_matrix **U, 
                      gsl_vector **chifacs,
		      gsl_matrix **A,
                      const char *xml_bank_filename,
		      const char *reference_psd_filename,
                      int base_sample_rate,
                      int down_samp_fac, 
                      double t_start,
                      double t_end, 
                      double t_total_duration, 
	              int verbose)
{
  InspiralTemplate *bankRef = NULL, *bankRow, *bankHead = NULL;
  int numtemps = InspiralTmpltBankFromLIGOLw( &bankHead, xml_bank_filename,-1,-1);
  double minChirpMass;
  double jreference=0;
  double tshift=1;
  size_t j;
  size_t numsamps = round((t_end - t_start) * base_sample_rate / down_samp_fac);
  size_t autocorr_numsamps = ATEMPS;
  size_t full_numsamps = base_sample_rate*TEMPLATE_DURATION;
  COMPLEX16TimeSeries *template_out;
  COMPLEX16TimeSeries *convolution=NULL;
  COMPLEX16TimeSeries *autocorrelation=NULL;
  COMPLEX16TimeSeries *short_autocorr=NULL; // WE DON'T NEED THIS!!!! DELETEEE! FIXME FIXME
  COMPLEX16FrequencySeries *fft_template;
  COMPLEX16FrequencySeries *fft_template_full;
  COMPLEX16FrequencySeries *fft_template_full_reference=NULL;
  COMPLEX16FrequencySeries *somethingelse=NULL;
  COMPLEX16TimeSeries *template_reference=NULL;
  COMPLEX16FrequencySeries *template_product=NULL;

  REAL8FrequencySeries *psd;
  REAL8FFTPlan *fwdplan;
  COMPLEX16FFTPlan *revplan;
  
  if (numtemps <= 0) {
    fprintf(stderr, "FAILED generate_bank_svd() numtemps <=0\n");
    exit(1);
    }

  if (verbose) fprintf(stderr,"read %d templates\n", numtemps);
  
  /* There are twice as many waveforms as templates */
  *U = gsl_matrix_calloc(numsamps, 2 * numtemps);
  *A = gsl_matrix_calloc(autocorr_numsamps, numtemps);

  /* I have just computed chifacs for one of the quadratures...it should be
   * redundant */
  *chifacs = gsl_vector_calloc(numtemps);

  /*fprintf(stderr,"U = %zd,%zd V = %zd,%zd S = %zd\n",(*U)->size1,(*U)->size2,(*V)->size1,(*V)->size2,(*S)->size);*/

  g_mutex_lock(gstlal_fftw_lock);
  fwdplan = XLALCreateForwardREAL8FFTPlan(full_numsamps, 0);
  if (!fwdplan)
    {
    fprintf(stderr, "FAILED Generating the forward plan failed\n");
    exit(1);
    }
  revplan = XLALCreateReverseCOMPLEX16FFTPlan(full_numsamps, 0);
  if (!revplan)
    {
    fprintf(stderr, "FAILED Generating the reverse plan failed\n");
    exit(1);
    }
  g_mutex_unlock(gstlal_fftw_lock);


  /* create workspace vectors for the templates */
  template_out = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  convolution = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  autocorrelation = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  short_autocorr = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, autocorr_numsamps); 
  template_reference = XLALCreateCOMPLEX16TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  fft_template = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 1.0 / TEMPLATE_DURATION, &lalDimensionlessUnit, full_numsamps / 2 + 1);
  fft_template_full = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 1.0 / TEMPLATE_DURATION, &lalDimensionlessUnit, full_numsamps);
  fft_template_full_reference = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 1.0 / TEMPLATE_DURATION, &lalDimensionlessUnit, full_numsamps);
  template_product = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 1.0 / TEMPLATE_DURATION, &lalDimensionlessUnit, full_numsamps);
  
  
  /* get the reference psd */
  psd = gstlal_get_reference_psd(reference_psd_filename, template_out->f0, 1.0/TEMPLATE_DURATION, fft_template->data->length);

  if (!template_out || !convolution || !template_reference || !fft_template || !fft_template_full || !fft_template_full_reference || !template_product || !psd){
    fprintf(stderr, "FAILED Allocating template or reading psd failed\n");
    exit(1);
    }
    

  if (verbose) fprintf(stderr, "template_out->data->length %d fft_template->data->length %d fft_template_full->data->length %d \n",template_out->data->length, fft_template->data->length, fft_template_full->data->length);

  /* "fix" the templates in the bank */
  for(bankRow = bankHead; bankRow; bankRow = bankRow->next)
    {
    bankRow->fFinal = 0.95 * (base_sample_rate / 2.0 - 1); /*95% of Nyquist*/
    bankRow->fLower = 25.0;
    bankRow->tSampling = base_sample_rate;
    bankRow->fCutoff = bankRow->fFinal;
    /*bankRow->order = LAL_PNORDER_THREE_POINT_FIVE;*/
    bankRow->order = LAL_PNORDER_TWO;
    bankRow->signalAmplitude = 1.0;
    }

  /* MIREIA */ 
  

  REAL8 minMass, mineta;
  LALPNOrder minorder;

  if(bankHead) {
    minChirpMass = bankHead->chirpMass;
    minMass = bankHead->mass1+bankHead->mass2;
    mineta = bankHead->eta;
    minorder = bankHead->order;
    bankRef = bankHead;

    for(bankRow = bankHead, j=0; bankRow; bankRow = bankRow->next, j++)
      if(bankRow->chirpMass < minChirpMass) 
        {
	minChirpMass = bankRow->chirpMass;
	minMass = bankRow->mass1+bankRow->mass2;
	mineta = bankRow->eta;
	minorder = bankRow->order;
	bankRef = bankRow;
	jreference = j;
	}
   }
  

  
  /* create the reference template */

  if(create_template_from_sngl_inspiral(bankRef, *U, *A, *chifacs, base_sample_rate, down_samp_fac, t_end, t_total_duration, autocorr_numsamps, 0, jreference, template_reference, convolution, autocorrelation, short_autocorr, template_product, fft_template, fft_template_full_reference, somethingelse, fwdplan, revplan, psd) < 0)
    {
    exit(1);
    }
  if (verbose)
    {
    fprintf(stderr, "Referece Template: M_chirp=%e\n", bankRef->chirpMass);
    }

  /* create the templates in the bank */
  for(bankRow = bankHead, j = 0; bankRow; bankRow = bankRow->next, j++)
    {

    /* MIREIA */
    
    //double tshift = (gstlal_spa_chirp_time(minMass, mineta, 150.0, minorder))-(gstlal_spa_chirp_time((REAL8) bankRow->mass1+bankRow->mass2, (REAL8) bankRow->eta, 150.0, bankRow->order));
    
    if(j!=jreference)
     { 
       if(create_template_from_sngl_inspiral(bankRow, *U, *A, *chifacs, base_sample_rate, down_samp_fac,t_end, t_total_duration, autocorr_numsamps, tshift, j, template_out, convolution, autocorrelation, short_autocorr, template_product, fft_template, fft_template_full, fft_template_full_reference, fwdplan, revplan, psd) < 0)
         {
          exit(1);
         }
       if (verbose)
         {
          fprintf(stderr, "template %zd M_chirp=%e\n",j, bankRow->chirpMass);
          //fprintf(stderr,"   Tshift (real) %g, Total mass %g\n " , tshift, bankRow->mass1+bankRow->mass2);
         }
      }
    }

  /* Destroy plans */
  g_mutex_lock(gstlal_fftw_lock);
  XLALDestroyREAL8FFTPlan(fwdplan);
  XLALDestroyCOMPLEX16FFTPlan(revplan);
  g_mutex_unlock(gstlal_fftw_lock);

  /* Destroy time/freq series */
  XLALDestroyCOMPLEX16FrequencySeries(fft_template);
  XLALDestroyCOMPLEX16FrequencySeries(fft_template_full);
  XLALDestroyCOMPLEX16FrequencySeries(fft_template_full_reference);
  XLALDestroyCOMPLEX16FrequencySeries(template_product);
  XLALDestroyCOMPLEX16TimeSeries(template_out);
  XLALDestroyCOMPLEX16TimeSeries(convolution);
  XLALDestroyCOMPLEX16TimeSeries(template_reference);
  XLALDestroyCOMPLEX16TimeSeries(autocorrelation);
  XLALDestroyCOMPLEX16TimeSeries(short_autocorr);

  /* free the template list */
  while(bankHead)
    {
    InspiralTemplate *next = bankHead->next;
    XLALFree(bankHead);
    bankHead = next;
    }

  /* done */
  return 0;
}

int generate_bank_and_svd(
                      gsl_matrix **U, 
                      gsl_vector **S, 
		      gsl_matrix **V,
                      gsl_vector **chifacs,
		      gsl_matrix **A,
                      const char *xml_bank_filename,
		      const char *reference_psd_filename,
                      int base_sample_rate,
                      int down_samp_fac, 
                      double t_start,
                      double t_end, 
                      double t_total_duration, 
                      double tolerance,
	              int verbose)
{
  size_t i, j;
  int result = generate_bank(U, chifacs, A, xml_bank_filename, reference_psd_filename, base_sample_rate, down_samp_fac, t_start, t_end, t_total_duration, verbose);
  if(result)
    return result;

  if (verbose) fprintf(stderr,"Doing the SVD \n");
  if(gstlal_gsl_linalg_SV_decomp_mod(U, V, S))
    {
    fprintf(stderr,"FAILED could not do SVD \n");
        exit(1); 
    }

  /* FIXME: check the templates to see that the snr-weighted tolerance is correct */
  tolerance = 1. - (1. - tolerance)/gsl_vector_max(*chifacs);
  trim_matrix(U,V,S,tolerance);
  for (i = 0; i < (*S)->size; i++)
    for (j = 0; j < (*V)->size1; j++)
      gsl_matrix_set(*V,j,i,gsl_vector_get(*S,i)*gsl_matrix_get(*V,j,i));

  if ( not_gsl_matrix_transpose(U) || not_gsl_matrix_transpose(V) ) {
    fprintf(stderr, "FAILED matrix transpose failed\n");
    exit(1);
    }

  if(verbose) fprintf(stderr, "%.16g s -- %.16g s: %zd orthogonal templates, V is %zdx%zd, U is %zdx%zd\n\n", t_start, t_end, (*U)->size1, (*V)->size1, (*V)->size2, (*U)->size1, (*U)->size2);

  return 0;
}


int not_gsl_matrix_transpose(gsl_matrix **m)
{
  gsl_matrix *new = gsl_matrix_calloc((*m)->size2, (*m)->size1);
  if(!new) {
    fprintf(stderr, "not_gsl_matrix_transpose() transpose failed to allocate memory\n");
    return -1;
  }
  gsl_matrix_transpose_memcpy(new, *m);
  gsl_matrix_free(*m);
  *m = new;
  return 0;
}

int trim_matrix(gsl_matrix **U, gsl_matrix **V, gsl_vector **S, 
                        double tolerance)
  {
  double sumb = 0;
  double cumsumb = 0;
  int maxb = 0;
  size_t i = 0;
  double tol = tolerance; /*1- (1-tolerance)/2.0;*/
  for (i = 0; i < (*S)->size; i++) 
    {
    sumb+= gsl_vector_get(*S,i) * gsl_vector_get(*S,i);
    }
  /*sumb = gsl_vector_get(*S,0);*/
  for (i = 0; i < (*S)->size; i++)
    {
    cumsumb += gsl_vector_get(*S,i) * gsl_vector_get(*S,i) ;
    if (i>1 && sqrt(cumsumb / sumb) > tol) break;
    }
  maxb = i;/* (*S)->size;*/
  if (not_gsl_matrix_chop(U,(*U)->size1,maxb)) return 1;
  if (not_gsl_matrix_chop(V,(*V)->size1,maxb)) return 1;
  if (not_gsl_vector_chop(S,maxb)) return 1;
  return 0;
  }

/*FIXME this is terrible and needs to be made more efficient!!!!!!!*/
 int not_gsl_matrix_chop(gsl_matrix **M, size_t m, size_t n)
  {
  gsl_matrix *tmp = (*M);
  gsl_matrix *newM = NULL;
  size_t i = 0; 
  size_t j = 0;
  
  if ( (*M)->size1 < m ) return 1;
  if ( (*M)->size2 < n ) return 1;
  newM = gsl_matrix_calloc(m,n);

  for (i=0; i<m; i++)
    {
    for (j=0; j<n; j++)
      {
      gsl_matrix_set(newM,i,j,gsl_matrix_get(*M,i,j));
      }
    }
  *M = newM;
  gsl_matrix_free(tmp);
  return 0;
  }

/*FIXME this is terrible and needs to be made more efficient!!!!!!!*/
int not_gsl_vector_chop(gsl_vector **V, size_t m)
  {

  gsl_vector *tmp = (*V);
  gsl_vector *newV = NULL;
  size_t i = 0;

  if ( (*V)->size < m ) return 1;
  newV = gsl_vector_calloc(m);
  for (i=0; i<m; i++)
    {
    gsl_vector_set(newV,i,gsl_vector_get(*V,i));
    }
  *V = newV;
  gsl_vector_free(tmp);
  return 0;
  }


static int SPAWaveform (
    InspiralTemplate           *tmplt,
    REAL8			deltaT,
    COMPLEX16FrequencySeries   *signal 
    )
/* </lalVerbatim> */
{
  UINT4         numPoints  = 0;
  REAL8         deltaF     = 0.0;
  REAL8         m          = 0.0;
  REAL8         eta        = 0.0;
  REAL8         mu         = 0.0;
  REAL8         tNorm      = 0.0;
  COMPLEX16    *expPsi     = NULL;
  REAL8         x1         = 0.0;
  REAL8         psi        = 0.0;
  REAL8         psi0       = 0.0;
  INT4          k          = 0;
  INT4          kmin       = 0;
  INT4          kmax       = 0;

  REAL8         distNorm;
  const REAL8   cannonDist = 1.0; /* Mpc */

  /* pn constants */
  REAL8 c0, c10, c15, c20, c25, c25Log, c30, c30Log, c35, c40P; 
  REAL8 x;

  /* chebychev coefficents for expansion of sin and cos */
  const REAL8 s2 = -0.16605;
  const REAL8 s4 =  0.00761;
  const REAL8 c2 = -0.49670;
  const REAL8 c4 =  0.03705;
  /*FILE *FP = fopen("template.txt","w");*/
  /*
   *
   * check that the arguments are reasonable
   *
   */
  
  /* set up pointers */
  expPsi = signal->data->data;
  numPoints = 2 * (signal->data->length - 1);

  /* set the waveform approximant */
  tmplt->approximant = tmplt->approximant;

  /* set the pN order of the template */
  tmplt->order = tmplt->order;

  /* zero output */
  memset( expPsi, 0, signal->data->length * sizeof(*expPsi) );

  /* parameters */
  deltaF = signal->deltaF;
  m      = (REAL8) tmplt->totalMass;
  eta    = (REAL8) tmplt->eta;
  mu     = (REAL8) tmplt->mu;

  /* template dependent normalisation */
  distNorm = 2.0 * LAL_MRSUN_SI / (cannonDist * 1.0e6 * LAL_PC_SI);

  tNorm = sqrt( (5.0*mu) / 96.0 ) *
    pow( m / (LAL_PI*LAL_PI) , 1.0/3.0 ) *
    pow( LAL_MTSUN_SI / (REAL8) deltaT, -1.0/6.0 );
  tNorm *= tNorm;
  tNorm *= distNorm * distNorm;

  /* Initialize all PN phase coeffs to zero. */
  c0 = c10 = c15 = c20 = c25 = c25Log = 0.;
  c30 = c30Log = c35 = c40P = 0.;

  /* Switch on PN order, set the appropriate phase coeffs for that order */
  switch( tmplt->order )
  {
    case LAL_PNORDER_PSEUDO_FOUR:
      c40P = 3923.0;
    case LAL_PNORDER_THREE_POINT_FIVE:
      c35 = LAL_PI*(77096675.0/254016.0 + eta*378515.0/1512.0 
            - eta*eta*74045.0/756.0);
    case LAL_PNORDER_THREE:
      c30 = 11583231236531.0/4694215680.0 - LAL_GAMMA*6848.0/21.0 
            - LAL_PI*LAL_PI*640.0/3.0 + eta*(LAL_PI*LAL_PI*2255.0/12.0 
            - 15737765635.0/3048192.0) + eta*eta*76055.0/1728.0 
            - eta*eta*eta*127825.0/1296.0 - 6848.0*log(4.0)/21.0;
      c30Log = -6848.0/21.0;
    case LAL_PNORDER_TWO_POINT_FIVE:
      c25 = LAL_PI*38645.0/756.0 - LAL_PI*eta*65.0/9.0;
      c25Log = 3*c25;
    case LAL_PNORDER_TWO:
      c20 = 15293365.0/508032.0 + eta*(27145.0/504.0 + eta*3085.0/72.0);
      c15 = -16*LAL_PI;
      c10 = 3715.0/756.0 + eta*55.0/9.0;
      c0  = 3.0/(eta*128.0);
      break;
    default:
      fprintf(stderr, "different LAL_PNORDER");
      break;
  }

  /* x1 */
  x1 = pow( LAL_PI * m * LAL_MTSUN_SI * deltaF, -1.0/3.0 );

  /* frequency cutoffs */
  kmin = tmplt->fLower / deltaF > 1 ? tmplt->fLower / deltaF : 1;
  kmax = tmplt->fFinal / deltaF < numPoints/2 ? 
    tmplt->fFinal / deltaF : numPoints/2;

  /* compute psi0: used in range reduction */

  /* This formula works for any PN order, because */
  /* higher order coeffs will be set to zero.     */

  x = x1 * pow((REAL8) kmin, -1.0/3.0);
  psi = c0 * ( x * ( c20 + x * ( c15 + x * (c10 + x * x ) ) ) 
                + c25 - c25Log * log(x) + (1.0/x) 
                * ( c30 - c30Log * log(x) + (1.0/x) * ( c35 - (1.0/x) 
                * c40P * log(x) ) ) );
  psi0 = -2 * LAL_PI * ( floor ( 0.5 * psi / LAL_PI ) );

  /* Chirp Time */
  /* This formula works for any PN order, because */
  /* higher order coeffs will be set to zero.     */
  for ( k = kmin; k < kmax ; ++k )
    {
    REAL8 x = x1 * pow((REAL8) k,- 1.0/3.0);
    REAL8 psi = c0 * ( x * ( c20 + x * ( c15 + x * (c10 + x * x ) ) ) 
                  + c25 - c25Log * log(x) + (1.0/x) * ( c30 - c30Log * log(x) 
                  + (1.0/x) * ( c35 - (1.0/x) * c40P * log(x) ) ) );
    REAL8 psi1 = psi + psi0;
    REAL8 psi2;

    /* range reduction of psi1 */
    while ( psi1 < -LAL_PI )
    {
      psi1 += 2 * LAL_PI;
      psi0 += 2 * LAL_PI;
    }
    while ( psi1 > LAL_PI )
    {
      psi1 -= 2 * LAL_PI;
      psi0 -= 2 * LAL_PI;
    }

    /* compute approximate sine and cosine of psi1 */
    if ( psi1 < -LAL_PI/2 )
    {
      psi1 = -LAL_PI - psi1;
      psi2 = psi1 * psi1;
      /* XXX minus sign added because of new sign convention for fft */
      expPsi[k].im = - psi1 * ( 1 + psi2 * ( s2 + psi2 * s4 ) );
      expPsi[k].re = -1 - psi2 * ( c2 + psi2 * c4 );
    }
    else if ( psi1 > LAL_PI/2 )
    {
      psi1 = LAL_PI - psi1;
      psi2 = psi1 * psi1;
      /* XXX minus sign added because of new sign convention for fft */
      expPsi[k].im = - psi1 * ( 1 + psi2 * ( s2 + psi2 * s4 ) );
      expPsi[k].re = -1 - psi2 * ( c2 + psi2 * c4 );
    }
    else
    {
      psi2 = psi1 * psi1;
      /* XXX minus sign added because of new sign convention for fft */
      expPsi[k].im = - psi1 * ( 1 + psi2 * ( s2 + psi2 * s4 ) );
      expPsi[k].re = 1 + psi2 * ( c2 + psi2 * c4 );
    }

    /* put in the first order amplitude factor */
    expPsi[k].re *= pow(k, -7.0/6.0);
    expPsi[k].im *= pow(k, -7.0/6.0);
    /*fprintf(FP,"%d %e %e\n", k, expPsi[k].re, expPsi[k].im);*/
  }
  /*fclose(FP);*/

  tmplt->tC = gstlal_spa_chirp_time((REAL8) tmplt->totalMass, (REAL8) tmplt->eta, (REAL8) tmplt->fLower, tmplt->order);

  return 0;
}


double gstlal_spa_chirp_time(REAL8 m, REAL8 eta, REAL8 fLower, LALPNOrder order)
{

  /* variables used to compute chirp time */
  REAL8 c0T, c2T, c3T, c4T, c5T, c6T, c6LogT, c7T;
  REAL8 xT, x2T, x3T, x4T, x5T, x6T, x7T, x8T;

  /*
   *
   * compute the length of the stationary phase chirp
   *
   */

  /* This formula works for any PN order, because */
  /* higher order coeffs will be set to zero.     */


  /* Initialize all PN chirp time coeffs to zero. */
  c0T = c2T = c3T = c4T = c5T = c6T = c6LogT = c7T = 0.;

 /* Switch on PN order, set the chirp time coeffs for that order */
  switch( order )
  {
    case LAL_PNORDER_PSEUDO_FOUR:
    case LAL_PNORDER_THREE_POINT_FIVE:
      c7T = LAL_PI*(14809.0*eta*eta - 75703.0*eta/756.0 - 15419335.0/127008.0);
    case LAL_PNORDER_THREE:
      c6T = LAL_GAMMA*6848.0/105.0 - 10052469856691.0/23471078400.0
            + LAL_PI*LAL_PI*128.0/3.0 + eta*( 3147553127.0/3048192.0
            - LAL_PI*LAL_PI*451.0/12.0 ) - eta*eta*15211.0/1728.0
            + eta*eta*eta*25565.0/1296.0 + log(4.0)*6848.0/105.0;
      c6LogT = 6848.0/105.0;
    case LAL_PNORDER_TWO_POINT_FIVE:
      c5T = 13.0*LAL_PI*eta/3.0 - 7729.0/252.0;
    case LAL_PNORDER_TWO:
      c4T = 3058673.0/508032.0 + eta * (5429.0/504.0 + eta * 617.0/72.0);
      c3T = -32.0 * LAL_PI / 5.0;
      c2T = 743.0/252.0 + eta * 11.0/3.0;
      c0T = 5.0 * m * LAL_MTSUN_SI / (256.0 * eta);
      break;
    default:
      fprintf(stderr, "ERROR!!!\n");
      break;
  }

  /* This is the PN parameter v evaluated at the lower freq. cutoff */
  xT  = pow( LAL_PI * m * LAL_MTSUN_SI * fLower, 1.0/3.0);
  x2T = xT * xT;
  x3T = xT * x2T;
  x4T = x2T * x2T;
  x5T = x2T * x3T;
  x6T = x3T * x3T;
  x7T = x3T * x4T;
  x8T = x4T * x4T;

  /* Computes the chirp time as tC = t(v_low)    */
  /* tC = t(v_low) - t(v_upper) would be more    */
  /* correct, but the difference is negligble.   */

  /* This formula works for any PN order, because */
  /* higher order coeffs will be set to zero.     */

  return c0T * ( 1 + c2T*x2T + c3T*x3T + c4T*x4T + c5T*x5T
              + ( c6T + c6LogT*log(xT) )*x6T + c7T*x7T ) / x8T;
}


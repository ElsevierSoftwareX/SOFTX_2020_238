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
#include <lal/LALInspiral.h>
#include <lal/FindChirp.h>
#include <lal/AVFactories.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/FindChirpTD.h>
#include <lal/LALError.h>
#include <lal/LALStdio.h>

/* gstlal includes */
#include "gstlal.h"
#include "low_latency_inspiral_functions.h"
#include "gstlal_whiten.h"

#define TEMPLATE_DURATION 128	/* seconds */

#define LAL_CALL( function, statusptr ) \
  ((function),lal_errhandler(statusptr,#function,__FILE__,__LINE__,rcsid))

static int create_template_from_sngl_inspiral(
                       InspiralTemplate *bankRow,
                       gsl_matrix *U, 
                       gsl_vector *chifacs,
                       int fsamp,
                       int downsampfac, 
                       double t_end,
                       double t_total_duration, 
                       int U_column,
                       FindChirpFilterInput *fcFilterInput,
                       FindChirpTmpltParams *fcTmpltParams,
                       REAL8TimeSeries *template,
                       COMPLEX16FrequencySeries *fft_template,
                       REAL8FFTPlan *fwdplan,
                       REAL8FFTPlan *revplan,
                       REAL8FrequencySeries *psd
                       )

  {
  unsigned i;
  int t_total_length = floor(t_total_duration * fsamp + 0.5);	/* length of the template */
  double norm;
  gsl_vector_view col;
  gsl_vector_view tmplt;
  LALStatus status;

  memset(&status, 0, sizeof(status));
 
  LALFindChirpTDTemplate( &status, fcFilterInput->fcTmplt,
                  bankRow, fcTmpltParams );

  for (i=0; i< template->data->length; i++)
    template->data->data[i] = (REAL8) fcTmpltParams->xfacVec->data[i];

  /*
   * Whiten the template.
   */

  XLALREAL8TimeFreqFFT(fft_template,template,fwdplan);
  XLALWhitenCOMPLEX16FrequencySeries(fft_template,psd);
  XLALREAL8FreqTimeFFT(template,fft_template,revplan);

  /*
   * Normalize the template.  If s is the template and n is a stationary
   * noise process of independent samples, s is normalized so that
   *
   *	< (n|s)^2 > = < n^2 >
   *
   * that is, s acts as a mean-square preserving filter.  That condition is
   * equivalent to
   *
   *	(s|s) = (1/T) \sum s^2 \Delta t = T / \Delta t = N
   */

  norm = t_total_length / sqrt(XLALREAL8SequenceSumSquares(template->data, template->data->length - t_total_length, t_total_length));

  /*
   * Extract a piece of the template.  The change in sample rate
   * necessitates an adjustment to the normalization:
   *
   *	(s|s) --> (s|s) \Delta t / \Delta t'
   *
   * "Huh?  \sqrt{8 / 0.98}?"  No, my friend, don't ask questions you don't
   * want to hear the answer to.
   */

  col = gsl_matrix_column(U, U_column);
  tmplt = gsl_vector_view_array_with_stride(template->data->data + template->data->length - (int) floor(t_end * fsamp + 0.5), downsampfac, col.vector.size);
  gsl_vector_memcpy(&col.vector, &tmplt.vector);
  gsl_vector_scale(&col.vector, norm * sqrt(8.0 / 0.99148));

  /*
   * Compute the \Xi^2 factor.
   */

  gsl_vector_set(chifacs,U_column,gsl_blas_dnrm2(&col.vector));

  return 0;
  }


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
  /* This function gives the instantaneous frequency at a given time based
   * on the quadrupole approximation.  It is bound to be a bit off from other
   * template families so use it with caution 
   */
  double Mg = M*LAL_MTSUN_SI;
  return 5./256. * Mg / pow(LAL_PI*Mg*freq,8./3.);
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
  sampleRate = 2.0 * floor(pow(2.0,floor(log(minFreq)/log(2.0))));
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
  sampleRate = 2.0 * floor(pow(2.0,floor(log(minFreq)/log(2.0))));
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

/* FIXME: this is a place holder and needs to be implemented rigorously with  
 * lal functions */
int generate_bank_svd(
                      gsl_matrix **U, 
                      gsl_vector **S, 
		      gsl_matrix **V,
                      gsl_vector **chifacs,
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
  InspiralTemplate *bankRow, *bankHead = NULL;
  int numtemps = InspiralTmpltBankFromLIGOLw( &bankHead, xml_bank_filename,-1,-1);
  size_t i, j;
  size_t numsamps = floor((t_end - t_start) * base_sample_rate / down_samp_fac + 0.5);
  size_t full_numsamps = base_sample_rate*TEMPLATE_DURATION;
  gsl_vector *work_space = gsl_vector_calloc(numtemps);
  gsl_matrix *work_space_matrix = gsl_matrix_calloc(numtemps,numtemps);
  REAL8TimeSeries *template;
  COMPLEX16FrequencySeries *fft_template;
  REAL8FrequencySeries *psd;
  FindChirpFilterInput *fcFilterInput  = NULL;
  FindChirpTmpltParams *fcTmpltParams  = NULL;
  FindChirpInitParams fcInitParams;
  LALStatus status;
  REAL8FFTPlan *fwdplan;
  REAL8FFTPlan *revplan;

  if (verbose) fprintf(stderr,"read %d templates\n", numtemps);
  
  memset(&status, 0, sizeof(status));
  memset(&fcInitParams, 0, sizeof(fcInitParams));

  *U = gsl_matrix_calloc(numsamps,numtemps);
  *S = gsl_vector_calloc(numtemps);
  *V = gsl_matrix_calloc(numtemps,numtemps);
  *chifacs = gsl_vector_calloc(numtemps);

  fprintf(stderr,"U = %zd,%zd V = %zd,%zd S = %zd\n",(*U)->size1,(*U)->size2,(*V)->size1,(*V)->size2,(*S)->size);

  g_mutex_lock(gstlal_fftw_lock);
  fwdplan = XLALCreateForwardREAL8FFTPlan(full_numsamps, 1);
  revplan = XLALCreateReverseREAL8FFTPlan(full_numsamps, 1);
  g_mutex_unlock(gstlal_fftw_lock);

  template = XLALCreateREAL8TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0, 1.0 / base_sample_rate, &lalStrainUnit, full_numsamps);
  fft_template = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0,0}, 0, 0, &lalDimensionlessUnit, template->data->length / 2 + 1);
  psd = gstlal_get_reference_psd(reference_psd_filename, template->f0, 1.0/TEMPLATE_DURATION, fft_template->data->length);

  fcInitParams.numPoints      = full_numsamps;
  fcInitParams.numSegments    = 1;
  fcInitParams.numChisqBins   = 0;
  fcInitParams.createRhosqVec = 0;
  fcInitParams.ovrlap         = 0;
  fcInitParams.approximant    = EOB;
  fcInitParams.order          = threePointFivePN;
  fcInitParams.createCVec     = 0;
  bankHead->order = threePointFivePN;

  if (verbose) fprintf(stderr,"LALFindChirpTemplateInit() ...\n");
  LALFindChirpTemplateInit( &status, &fcTmpltParams, &fcInitParams );
  if (verbose) fprintf(stderr,"LALFindChirpTemplateInit() done\n");
  fcTmpltParams->deltaT = 1.0 / base_sample_rate;
  fcTmpltParams->fLow = 25; 

  if (verbose) fprintf(stderr,"chirpmass = %f, flow = %f\n\n",bankHead->chirpMass,fcTmpltParams->fLow);

  fcTmpltParams->reverseChirpBank = 0;
  fcTmpltParams->taperTmplt = 1;

  /*fcTmpltParams->order = order;*/

  /* Create Template - to be replaced by a LAL template generation call */
  if (verbose) fprintf(stderr,"LALCreateFindChirpInput() ...\n");
  LALCreateFindChirpInput( &status, &fcFilterInput, &fcInitParams );
  if (verbose) fprintf(stderr,"LALCreateFindChirpInput() done\n");

  fprintf(stderr, "LALFindChirpTDTemplate() tmplate is %p\n", fcFilterInput->fcTmplt);

  fprintf(stderr, "bankHead order is %d",bankHead->order);

  fcTmpltParams->order = threePointFivePN;
  fcTmpltParams->approximant = EOB;
  fcTmpltParams->dynRange = pow(2,63);


  if (verbose) fprintf(stderr, "allocated matrices...\n");
  /* create the templates in the bank */
  for(bankRow = bankHead, j = 0; bankRow; bankRow = bankRow->next, j++)
    {
    /* this sets the cut off frequency in my version of lal only */
    /* Sathya is supposed to fix this */

    bankRow->fFinal = base_sample_rate / 2.0 - 1; /*nyquist*/

    create_template_from_sngl_inspiral(bankRow, *U, *chifacs, base_sample_rate,down_samp_fac,t_end, t_total_duration, j, fcFilterInput, fcTmpltParams, template, fft_template, fwdplan, revplan, psd);
    if (verbose) fprintf(stderr, "template %zd M_chirp=%e\n",j, bankRow->chirpMass);
    }

  /* SET THIS IN create_template_.. gsl_vector_set(*chifacs,i,sqrt(tmpltpower));*/
  if (verbose)     fprintf(stderr,"Doing the SVD \n");

  /*if(gsl_linalg_SV_decomp(*U,*V, *S, work_space))*/
  /*if(gsl_linalg_SV_decomp_jacobi(*U, *V, *S))*/
  if(gsl_linalg_SV_decomp_mod(*U, work_space_matrix, *V, *S, work_space))
    {
    fprintf(stderr,"could not do SVD \n");
    return 1; 
    }
  trim_matrix(U,V,S,tolerance);
  for (i = 0; i < (*S)->size; i++)
    for (j = 0; j < (*V)->size1; j++)
      gsl_matrix_set(*V,j,i,gsl_vector_get(*S,i)*gsl_matrix_get(*V,j,i));

  not_gsl_matrix_transpose(U);
  not_gsl_matrix_transpose(V);
  if(verbose) fprintf(stderr, "%.16g s -- %.16g s: %zd orthogonal templates, V is %zdx%zd, U is %zdx%zd\n\n", t_start, t_end, (*U)->size1, (*V)->size1, (*V)->size2, (*U)->size1, (*U)->size2);


  gsl_vector_free(work_space);
  gsl_matrix_free(work_space_matrix);
  g_mutex_lock(gstlal_fftw_lock);
  XLALDestroyREAL8FFTPlan(fwdplan);
  XLALDestroyREAL8FFTPlan(revplan);
  g_mutex_unlock(gstlal_fftw_lock);
  LALFindChirpTemplateFinalize( &status, &fcTmpltParams );
  XLALDestroyCOMPLEX16FrequencySeries(fft_template);
  XLALDestroyREAL8TimeSeries(template);
  LALDestroyFindChirpInput(&status,&fcFilterInput);
  while(bankHead)
    {
    InspiralTemplate *next = bankHead->next;
    XLALFree(bankHead);
    bankHead = next;
    }
  return 0;
  }


void not_gsl_matrix_transpose(gsl_matrix **m)
{
  gsl_matrix *new = gsl_matrix_calloc((*m)->size2, (*m)->size1);
  if(new)
    gsl_matrix_transpose_memcpy(new, *m);
  gsl_matrix_free(*m);
  *m = new;
}

double normalize_template(double M, double ts, double duration,
                                int fsamp)

  {
  int numsamps = fsamp*duration;
  double tmpltpower = 0;
  double h = 0;
  int i = 0;
  double dt = 1.0/fsamp;
  for (i=0; i< numsamps; i++)
    {
    h = 4.0 * M * pow(5.0/256.0*(M/(-ts+dt*i)),0.25) 
      * sin(-2.0/2.0/M_PI * pow((-ts+dt*i)/(5.0*M),(5.0/8.0)));
    tmpltpower+=h*h*dt;
    }
  return sqrt(tmpltpower);
   
  }

 int trim_matrix(gsl_matrix **U, gsl_matrix **V, gsl_vector **S, 
                        double tolerance)
  {
  double sumb = 0;
  double cumsumb = 0;
  int maxb = 0;
  size_t i = 0;
  /*for (i = 0; i < (*S)->size; i++) 
    {
    sumb+= gsl_vector_get(*S,i);
    fprintf(stderr, "S(%d) = %f",i,gsl_vector_get(*S,i));
    }*/
  sumb = gsl_vector_get(*S,0);
  for (i = 0; i < (*S)->size; i++)
    {
    cumsumb = 1-gsl_vector_get(*S,i)/sumb;
    if ((cumsumb*cumsumb) > tolerance) break;
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
  /*FILE *FP = NULL;*/
  gsl_matrix *tmp = (*M);
  gsl_matrix *newM = NULL;
  size_t i = 0; 
  size_t j = 0;
  
  if ( (*M)->size1 < m ) return 1;
  if ( (*M)->size2 < n ) return 1;
  /*FP = fopen("svd.dat","w");*/
  newM = gsl_matrix_calloc(m,n);

  for (i=0; i<m; i++)
    {
    for (j=0; j<n; j++)
      {
      gsl_matrix_set(newM,i,j,gsl_matrix_get(*M,i,j));
      /*fprintf(FP,"%e\n",gsl_matrix_get(*M,i,j));*/
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

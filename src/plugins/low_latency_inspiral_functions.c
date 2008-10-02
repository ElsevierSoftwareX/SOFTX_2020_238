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

/* gsl includes */
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

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

#define LAL_CALL( function, statusptr ) \
  ((function),lal_errhandler(statusptr,#function,__FILE__,__LINE__,rcsid))

/* FIXME: this is a place holder and needs to be implemented rigorously with  
 * lal functions */
int generate_bank_svd(gsl_matrix **U, gsl_vector **S, gsl_matrix **V,
                           gsl_vector **chifacs,
                           char * bank_name, int base_sample_rate,
                           int down_samp_fac, double t_start,
                           double t_end, double tmax, double tolerance,
			   int verbose)
  {
  InspiralTemplate *bankHead     = NULL;
  int numtemps = InspiralTmpltBankFromLIGOLw( &bankHead, bank_name,-1,-1);
  size_t i = 0;
  size_t j = 0;
  int svd_err_code = 0;
  size_t numsamps = floor((double) (t_end-t_start) * base_sample_rate 
               / down_samp_fac);
  gsl_vector *work_space = gsl_vector_calloc(numtemps);
  gsl_matrix *work_space_matrix = gsl_matrix_calloc(numtemps,numtemps);
  *U = gsl_matrix_calloc(numsamps,numtemps);
  *S = gsl_vector_calloc(numtemps);
  *V = gsl_matrix_calloc(numtemps,numtemps);
  *chifacs = gsl_vector_calloc(numtemps);
  
  if (verbose) fprintf(stderr,"read in %d templates bank_head %p\n",numtemps,bankHead);

  if (verbose) fprintf(stderr, "allocated matrices...\n");
  /* create the templates in the bank */
  j = 0;
  while(bankHead)
    {
    create_template_from_sngl_inspiral(bankHead, *U, *chifacs, tmax, base_sample_rate,down_samp_fac,t_start,t_end,j);
    if (verbose) fprintf(stderr, "M_chirp=%e",bankHead->chirpMass);
    bankHead = bankHead->next;
    j++;
    }
  j = 0;

  /* SET THIS IN create_template_.. gsl_vector_set(*chifacs,i,sqrt(tmpltpower));*/
    
  svd_err_code = gsl_linalg_SV_decomp_mod(*U, work_space_matrix, 
                                             *V, *S, work_space);
  /*svd_err_code = gsl_linalg_SV_decomp(*U,*V, *S, work_space);*/
  /*svd_err_code = gsl_linalg_SV_decomp_jacobi(*U, *V, *S);*/
  if ( svd_err_code ) 
    {
    fprintf(stderr,"could not do SVD \n");
    return 1; 
    }
  trim_matrix(U,V,S,tolerance);
  for (i = 0; i < (*S)->size; i++)
    {
    for (j = 0; j < (*V)->size1; j++)
      {
      gsl_matrix_set(*V,j,i,gsl_vector_get(*S,i)*gsl_matrix_get(*V,j,i));
      }
    }
  not_gsl_matrix_transpose(U);
  not_gsl_matrix_transpose(V);
  if(verbose) fprintf(stderr, "%.16g s -- %.16g s: %d orthogonal templates, V is %dx%d, U is %dx%d\n\n", t_start, t_end, (int) (*U)->size1, (int) (*V)->size1, (int) (*V)->size2, (int) (*U)->size1, (int) (*U)->size2);
  gsl_vector_free(work_space);
  gsl_matrix_free(work_space_matrix);
  return 0;
  }


static double time_to_freq(double M, double time)
  {
  double c3_8 = 3.0/8.0;
  double c5_256 = 5.0/256.0;
  double c = 299792458;
  double G = 6.67428e-11;
  double Msol = 1.98893e30;
  double Mg = M*Msol*G/c/c/c;
  return (1.0/(M_PI*Mg)) * (pow((c5_256)*(Mg/(-time)),c3_8));
  }

int create_template_from_sngl_inspiral(InspiralTemplate *bankHead,
                       gsl_matrix *U, gsl_vector *chifacs, 
		       double duration, int fsamp, 
		       int downsampfac, double t_start, double t_end, 
		       int U_column)

  {
  int numsamps = fsamp*duration;
  int i = 0;
  int counter = 0;
  double dt = 1.0/fsamp;
  double chinorm = 0;
  REAL8TimeSeries *template = NULL;
  COMPLEX16FrequencySeries *fft_template = NULL;
  REAL8FrequencySeries *psd = NULL;
  FindChirpFilterInput         *fcFilterInput  = NULL;
  FindChirpTmpltParams         *fcTmpltParams  = NULL;
  FindChirpInitParams          *fcInitParams   = NULL;
  LALStatus *status = NULL;
  status = (LALStatus *) calloc(1,sizeof(LALStatus));
  /*fcFilterInput = (FindChirpFilterInput *) calloc(1,sizeof(FindChirpFilterInput));
 *   fcFilterInput->fcTmplt = (FindChirpTemplate *) calloc(1,sizeof(FindChirpTemplate));*/
  printf("creating fft plans \n");
  REAL8FFTPlan *fwdplan = XLALCreateForwardREAL8FFTPlan(numsamps, 0);
  REAL8FFTPlan *revplan = XLALCreateReverseREAL8FFTPlan(numsamps, 0);

  printf("Reading psd \n");
  psd = gstlal_get_reference_psd("reference_psd.txt", 0, 1.0/duration, numsamps / 2 + 1);

  if ( ! ( fcInitParams = (FindChirpInitParams *)
        LALCalloc( 1, sizeof(FindChirpInitParams) ) ) )
  {
    fprintf( stderr, "could not allocate memory for findchirp init params\n" );
    exit( 1 );
  }
  /* maybe remove these?*/
  fcInitParams->numPoints      = numsamps;
  fcInitParams->numSegments    = 1;
  fcInitParams->numChisqBins   = 0;
  fcInitParams->createRhosqVec = 0;
  fcInitParams->ovrlap         = 0;
  /* FIXME PPN has a hard coded max frequency that wont work !!! EOB for now*/
  fcInitParams->approximant    = EOB;
  fcInitParams->order          = twoPN;
  fcInitParams->createCVec     = 0;
  bankHead->order = threePointFivePN;
  printf("LALFindChirpTemplateInit()\n");
  LALFindChirpTemplateInit( status, &fcTmpltParams,
        fcInitParams );

  fcTmpltParams->deltaT = dt;
  /* FIXME This needs to be calculated based on the mass range of the bank.
 *      We want all the templates to fit in the allotted durations */
  fprintf(stderr,"chirpmass = %f, flow = %f\n\n",bankHead->chirpMass,time_to_freq(bankHead->chirpMass,duration/2.0));
  fcTmpltParams->fLow = 25; /*floor(time_to_freq(bankHead->chirpMass,duration/2.0));*/
  fcTmpltParams->reverseChirpBank = 0;
  fcTmpltParams->taperTmplt = 1;

  /*fcTmpltParams->order = order;*/

  printf("XLALCreateREAL8TimeSeriest() dt is %f\n",dt);
  template = XLALCreateREAL8TimeSeries(NULL, &(LIGOTimeGPS) {0,0}, 0.0,
                                       dt, &lalStrainUnit, numsamps);

  printf("XLALCreateCOMPLEX16FrequencySeries()\n");
  fft_template = XLALCreateCOMPLEX16FrequencySeries(NULL,
                      &(LIGOTimeGPS) {0,0}, 0, 0, &lalDimensionlessUnit,
                      numsamps / 2 + 1);

  /* Create Template - to be replaced by a LAL template generation call */
  printf("LALCreateFindChirpInput()\n");

  LALCreateFindChirpInput( status, &fcFilterInput, fcInitParams );

  printf("LALFindChirpTDTemplate() tmplate is %p \n", fcFilterInput->fcTmplt);

  printf("status is %p\n bankHead order is %d",status,bankHead->order);

  fcTmpltParams->order = threePointFivePN;
  fcTmpltParams->approximant = EOB;
  fcTmpltParams->dynRange = pow(2,63);
  LALFindChirpTDTemplate( status, fcFilterInput->fcTmplt,
                  bankHead, fcTmpltParams );

  /*CHECKSTATUSPTR(status); */
  /*printf("Copying template\n");
  FILE *FP = fopen("tmp.txt","w");*/
  for (i=0; i< numsamps; i++)
    {
    /*fprintf(FP,"%f\n",fcTmpltParams->xfacVec->data[i]);*/
    template->data->data[i] = (REAL8) fcTmpltParams->xfacVec->data[i];
    /*fft_template->data->data[i].im = (REAL8) fcFilterInput->fcTmplt->data->data[i].im;
    fft_template->data->data[i].re = (REAL8) fcFilterInput->fcTmplt->data->data[i].re;*/
    }
  XLALREAL8TimeFreqFFT(fft_template,template,fwdplan);
  XLALWhitenCOMPLEX16FrequencySeries(fft_template,psd);
  XLALREAL8FreqTimeFFT(template,fft_template,revplan);

  /*FP = fopen("tmpwh.txt","w");
  for (i=0; i< numsamps; i++)
    {
    fprintf(FP,"%f\n",template->data->data[i]);
    }
  fclose(FP);*/

  /* Actually return the peice of the template */

  /* Deallocate everything */
  /* FIXME DONT FORGET TO DESTROY LALCreateFindChirpInput */
  counter = 0;
  /*FP = fopen("tmpshort.txt","w");*/
  for (i=numsamps - t_end*fsamp; i< numsamps-t_start*fsamp; i+=downsampfac)
    {
    /*fprintf(FP,"%f\n",template->data->data[i]);*/
    chinorm+= template->data->data[i] * template->data->data[i] * dt*downsampfac;
    /*gsl_matrix_set(U,counter,U_column,template->data->data[i]);*/  
    counter++;
    }
  /*gsl_vector_set(chifacs,U_row,sqrt(chinorm)) */

  /*fclose(FP);*/
  XLALDestroyREAL8FFTPlan(fwdplan);
  XLALDestroyREAL8FFTPlan(revplan);
  LALFindChirpTemplateFinalize( status, &fcTmpltParams );
  LALFree( fcInitParams );
  XLALDestroyCOMPLEX16FrequencySeries(fft_template);
  XLALDestroyREAL8TimeSeries(template);
  LALDestroyFindChirpInput(status,&fcFilterInput);
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
    printf("S(%d) = %f",i,gsl_vector_get(*S,i));
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

#include "low_latency_inspiral_functions.h"
#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <math.h>
#include <gsl/gsl_linalg.h>

/* FIXME: this is a place holder and needs to be implemented rigorously with  
 * lal functions */
int generate_bank_svd(gsl_matrix **U, gsl_vector **S, gsl_matrix **V,
                           gsl_vector **chifacs,
                           double chirp_mass_start, int base_sample_rate,
                           int down_samp_fac, int numtemps, double t_start,
                           double t_end, double tmax, double tolerance,
			   int verbose)
  {
  FILE *FP = NULL;
  double c = 299792458;
  double G = 6.67428e-11;
  double Msol = 1.98893e30;
  double M = chirp_mass_start;
  double Mg = M*Msol*G/c/c/c;
  double ny_freq = 0.5*base_sample_rate;
  double T = 0;
  int i = 0;
  int j = 0;
  int svd_err_code = 0;
  int numsamps = floor((double) (t_end-t_start) * base_sample_rate 
               / down_samp_fac);
  double dt = (double) down_samp_fac/base_sample_rate;
  double tmpltpower = 0;
  double h=0;
  double norm = 0;
  double maxFreq = 0;
  gsl_vector *work_space = gsl_vector_calloc(numtemps);
  if (verbose) FP = fopen("tmpbankdata.dat","w");
  *U = gsl_matrix_calloc(numsamps,numtemps);
  *S = gsl_vector_calloc(numtemps);
  *V = gsl_matrix_calloc(numtemps,numtemps);
  *chifacs = gsl_vector_calloc(numtemps);
  if (verbose) printf("allocated matrices...\n");
  /* create the templates in the bank */
  for (i=0;i<numtemps;i++)
    {
    if (verbose) printf("template number %d...\n",i);
    /* increment the mass */
    M = chirp_mass_start + 0.0001*i*M/0.7;
    Mg = M*Msol*G/c/c/c;
    T = -1.0/( pow(M_PI * Mg * ny_freq, 8.0/3.0) / (5.0/256.0*Mg) )
      - t_start;
    /* FIXME We should check that the frequency at this time fits within the */
    /* downsampled rate!!! 						     */
    maxFreq = (1.0/(M_PI*Mg)) * (pow((5.0/256.0)*(Mg/(-T)),3.0/8.0));
    if (verbose) printf("T %e nyfreq %e chirpm %e max freq %e\n",T,ny_freq,M,maxFreq);

    if (maxFreq > ((double) (ny_freq/down_samp_fac+1.0/base_sample_rate)) )
      {
      fprintf(stderr,
              "cannot generate template segment at requested sample rate\n");
      return 1;
      }
    h = 0;
    norm = normalize_template(Mg, T, tmax, base_sample_rate);
    for(j =numsamps-1; j>=0; j--)
      {
      h = 4.0*Mg*pow(5.0/256.0*(Mg/(-T+dt*j)),0.25)
        * sin(-2.0/2.0/M_PI* pow((-T+dt*j)/(5.0*Mg),(5.0/8.0)));
      tmpltpower+=h*h;		 
      gsl_matrix_set(*U,j,i,h/norm);
      /*if (verbose) fprintf(FP,"%e\n",h/norm);*/
      }
    gsl_vector_set(*chifacs,i,sqrt(tmpltpower));
    }
  /*gsl_matrix_fprintf(FP,*U,"%f");*/
  if (FP) fclose(FP);
  svd_err_code = gsl_linalg_SV_decomp_jacobi(*U, *V, *S);
  if ( svd_err_code ) 
    {
    fprintf(stderr,"could not do SVD \n");
    return 1; 
    }
  trim_matrix(U,V,S,tolerance);
  if (verbose) fprintf(stderr,"sub template number = %d\n",(*U)->size2);
  gsl_vector_free(work_space);
  return 0;
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
    tmpltpower+=h*h;
    }
  return sqrt(tmpltpower);
   
  }

 int trim_matrix(gsl_matrix **U, gsl_matrix **V, gsl_vector **S, 
                        double tolerance)
  {
  double sumb = 0;
  double cumsumb = 0;
  int maxb = 0;
  int i = 0;
  for (i = 0; i < (*S)->size; i++) 
    {
    sumb+= gsl_vector_get(*S,i);
    }

  for (i = 0; i < (*S)->size; i++)
    {
    cumsumb += gsl_vector_get(*S,i) / sumb;
    if (cumsumb >= tolerance) break;
    }
  if ( (i < 5) && (10 < (*S)->size))
  maxb = 10;
  else maxb = i;
  if (!not_gsl_matrix_chop(U,(*U)->size1,maxb)) return 1;
  if (!not_gsl_matrix_chop(V,maxb,maxb)) return 1;
  if (!not_gsl_vector_chop(S,maxb)) return 1;

  }

/*FIXME this is terrible and needs to be made more efficient!!!!!!!*/
 int not_gsl_matrix_chop(gsl_matrix **M, size_t m, size_t n)
  {

  gsl_matrix *tmp = (*M);
  gsl_matrix *newM = NULL;
  int i = 0; 
  int j = 0;

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
  int i = 0;

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

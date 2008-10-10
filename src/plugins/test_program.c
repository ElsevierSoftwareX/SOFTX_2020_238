#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <lal/LALInspiral.h>
#include <lal/FindChirp.h>
#include <lal/LIGOLwXMLRead.h>
#include "low_latency_inspiral_functions.h"

int main()
  {
  gsl_vector *start;
  gsl_vector *stop;
  gsl_vector *sample;
  FILE *FP = NULL;
  FP = fopen("gsl_vecs.txt","w");
  lalDebugLevel = LALINFO | LALWARNING | LALERROR | LALNMEMDBG | LALNMEMPAD | LALNMEMTRK;

  compute_time_frequency_boundaries_from_bank(
    "H1-TMPLTBANK_01_1.080-874000000-2048.xml",
                                                2048,
                                                2048,
                                                40,
						&sample,
						&start,
						&stop,
                                                1);

  gsl_vector_fprintf(FP,sample,"%f");
  gsl_vector_fprintf(FP,start,"%f");
  gsl_vector_fprintf(FP,stop,"%f");
  /*InspiralTemplate             *bankHead     = NULL;
  int numTmplts = InspiralTmpltBankFromLIGOLw( &bankHead, "Bank.xml",-1,-1);
  printf("read in %d templates bank_head %p\n",numTmplts,bankHead);*/
  }

#if 0
int main()
{
  int tmpDur = 128.0;
  int noiseDur = 128.0;
  int base_sample_rate = 2048;
  int down_samp_fac = 16;
  double t_start = 13;
  double t_end = 29;
  double tmax = 128.0;
  double tolerance = 0.97;
  int vrb = 1;
  int i = 0;
  int j = 0;
/*  double numsamps = (t_end-t_start)*base_sample_rate/down_samp_fac;*/
  FILE *FP = NULL;
  gsl_matrix *U = NULL;
  gsl_vector *S = NULL;
  gsl_matrix *V = NULL;
  gsl_matrix *bank = NULL;
  gsl_vector *chifacs = NULL;
  generate_bank_svd(&U,&S,&V,&chifacs,"../../examples/H1-TMPLTBANK_09_1.207-874000000-2048.xml",base_sample_rate,
                    down_samp_fac,t_start,t_end,tmax,tolerance,vrb);

  fprintf(stderr,"U = %zd,%zd V = %zd,%zd S = %zd\n",U->size1,U->size2,V->size1,V->size2,S->size);

  /*gsl_matrix_transpose(V);*/
  /*for (i = 0; i < S->size; i++)
    {
    for (j = 0; j < V->size1; j++)
      {
      gsl_matrix_set(V,j,i,gsl_vector_get(S,i)*gsl_matrix_get(V,j,i));
      }
    }  THIS IS BEING DONE BY THE generate_bank_svd NOW!!!*/
  
  /*FP = fopen("svd.dat","w");
  gsl_matrix_fprintf(FP,U,"%f");
  fclose(FP);*/
  /*bank = gsl_matrix_calloc(U->size2,V->size2);
  if (!bank) fprintf(stderr,"could not allocate bank of size %d %d\n",U->size2,V->size2);
  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,V,U,0.0,bank);
  FP = fopen("reconstructed_bank.dat","w");
  for (j = 0; j < bank->size2; j++)
    {
    fprintf(FP,"%e %e\n", j * down_samp_fac / (double) base_sample_rate, gsl_matrix_get(bank,1,j));
    }
  */  
  fclose(FP);
  
  return 0;
}  


#endif

#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <lal/LALInspiral.h>
#include <lal/FindChirp.h>
#include <lal/LIGOLwXMLRead.h>
#include "low_latency_inspiral_functions.h"

#if 0 
int main()
  {
  lalDebugLevel = LALINFO | LALWARNING | LALERROR | LALNMEMDBG | LALNMEMPAD | LALNMEMTRK;
  InspiralTemplate             *bankHead     = NULL;
  int numTmplts = InspiralTmpltBankFromLIGOLw( &bankHead, "Bank.xml",-1,-1);
  printf("read in %d templates bank_head %p\n",numTmplts,bankHead);
  /*create_template_from_sngl_inspiral(bankHead, NULL, NULL, 128, 2048,2,0,1,0);*/
  }
#endif

int main()
{
  int tmpDur = 128.0;
  int noiseDur = 128.0;
  int base_sample_rate = 2048;
  int down_samp_fac = 1;
  double t_start = 0;
  double t_end = 1;
  double tmax = 128.0;
  double tolerance = 0.97;
  int vrb = 1;
  int i = 0;
  int j = 0;
/*  double numsamps = (t_end-t_start)*base_sample_rate/down_samp_fac;
  FILE *FP = NULL;*/
  gsl_matrix *U = NULL;
  gsl_vector *S = NULL;
  gsl_matrix *V = NULL;
  gsl_vector *chifacs = NULL;
  generate_bank_svd(&U,&S,&V,&chifacs,"Bank.xml.bk","reference_psd.txt",base_sample_rate,
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
  /*
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,V,U,0.0,bank);
  FP = fopen("reconstructed_bank.dat","w");
  for (j = 0; j < bank->size2; j++)
    {
    fprintf(FP,"%e %e\n", j * down_samp_fac / (double) base_sample_rate, gsl_matrix_get(bank,1,j));
    }
  fclose(FP);
  */
  return 0;
}  




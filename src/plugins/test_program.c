#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

int main()
{
  int tmpDur = 64;
  int noiseDur = 90;
  double chirpmass_start = 1.0;
  int base_sample_rate = 2048;
  int down_samp_fac = 1;
  int numtemps = 50;
  double t_start = 0;
  double t_end = 1;
  double tmax = 64;
  double tolerance = 0.99;
  int vrb = 1;
  int i = 0;
  int j = 0;
  double numsamps = (t_end-t_start)*base_sample_rate/down_samp_fac;
  FILE *FP = NULL;
  gsl_matrix *U = NULL;
  gsl_vector *S = NULL;
  gsl_matrix *V = NULL;
  gsl_vector *chifacs = NULL;
  gsl_matrix *bank = gsl_matrix_calloc(numtemps,numsamps);
  generate_bank_svd(&U,&S,&V,&chifacs,chirpmass_start,base_sample_rate,
                    down_samp_fac,numtemps,t_start,t_end,tmax,tolerance,vrb);
  printf("size U %d %d\n",U->size1,U->size2);
  printf("U = %d,%d V = %d,%d S = %d\n",U->size1,U->size2,V->size1,V->size2,S->size);

  /*gsl_matrix_transpose(V);*/
  /*for (i = 0; i < S->size; i++)
    {
    for (j = 0; j < V->size1; j++)
      {
      gsl_matrix_set(V,j,i,gsl_vector_get(S,i)*gsl_matrix_get(V,j,i));
      }
    }  THIS IS BEING DONE BY THE generate_bank_svd NOW!!!*/
  printf("U = %d,%d V = %d,%d S = %d, bank %d,%d\n",U->size1,U->size2,V->size1,V->size2,S->size,bank->size1,bank->size2);
  
  /*FP = fopen("svd.dat","w");
  gsl_matrix_fprintf(FP,U,"%f");
  fclose(FP);*/
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,V,U,0.0,bank);
  FP = fopen("reconstructed_bank.dat","w");
  for (j = 0; j < bank->size2; j++)
    {
    fprintf(FP,"%e %e\n", j * down_samp_fac / (double) base_sample_rate, gsl_matrix_get(bank,1,j));
    }
  fclose(FP);
  return 0;
}  

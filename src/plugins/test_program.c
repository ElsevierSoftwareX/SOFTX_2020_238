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
  int numtemps = 200;
  double t_start = 0;
  double t_end = 1;
  double tmax = 64;
  double tolerance = 0.99999999999999;
  int vrb = 1;
  int i = 0;
  int j = 0;
  double numsamps = (t_end-t_start)*base_sample_rate/down_samp_fac;
  FILE *FP = NULL;
  gsl_matrix *U = NULL;
  gsl_vector *S = NULL;
  gsl_matrix *V = NULL;
  gsl_vector *chifacs = NULL;
  gsl_matrix *bank = gsl_matrix_calloc(numsamps,numtemps);
  generate_bank_svd(&U,&S,&V,&chifacs,chirpmass_start,base_sample_rate,
                    down_samp_fac,numtemps,t_start,t_end,tmax,tolerance,vrb);
  printf("size U %d %d\n",U->size1,U->size2);
  /*gsl_matrix_transpose(V);*/
  for (i = 0; i < S->size; i++)
    {
    for (j = 0; j < V->size2; j++)
      {
      gsl_matrix_set(V,i,j,gsl_vector_get(S,i));
      }
    }
  printf("U = %d,%d V = %d,%d bank %d,%d\n",U->size1,U->size2,V->size1,V->size2,bank->size1,bank->size2);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,U,V,0.0,bank);
  FP = fopen("reconstructed_bank.dat","w");
  j = bank->size2-1;
  for (i = 0; i < bank->size1; i++)
    {
    fprintf(FP,"%e\n",gsl_matrix_get(bank,i,j));
    }
  fclose(FP);

}  

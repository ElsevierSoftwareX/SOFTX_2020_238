/*
 *         Copyright (C) 2015 Yichun Li(buckfryspj@gmail.com), Yan Wang (yan.wang@ligo.org) 
 *                         
 *                                 This code converts pdf (probability density function) of SNR and chi-squared to cdf (cumulative density function).
 *                                  
 *                                  */



///////////////////////////////
//void ssvkernel(gsl_vector * x, gsl_vector * tin, gsl_vector * y_hist_result,gsl_matrix * result)
//
//input:
//x: one-dimensional sample data vector
//tin: points at which estimation was computed (must be in ascending order)
//
//output:
//y_hist_result: one-dimensional histogram
//result: a L*L matrix. result(i,j) = count(Bin_i)/h(x_j)*K[x-point(Bin_i)/h(x_j)]
//(each variable's meaning: see document of 'Shimazaki’s method' part.).
//

//int main()
//
//input:
//two file's names
//
//output:
//result: two-dimensional probability density
//pc: pdf and cdf
///////////////////////////////


#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>

//#define MAX(a,b) (a>b)?(a):(b)
//#define MIN(a,b) (a<b)?(a):(b)
//#define PI 3.14159265358979323846264338
//#define REAL(z,i) ((z)[2*(i)])
//#define IMAG(z,i) ((z)[2*(i)+1])

#define XN 320
#define YN 300


typedef struct {
        int xn, yn;       // can be changed to lal/LALInspiral.h convesion if required in the future
	double xtick[XN], ytick[YN];
	double xspac[XN], yspac[YN];
	//long double xyspac[xn][yn];
	long double pdf[XN][YN];
	long double cdf[XN][YN];  // can be changed to variable length if required in the future
	} PdfCdf;
				

void pdf2cdf(PdfCdf *pc);

void pdf2cdf_sharpcut(PdfCdf *pc);

void gsl_matrix_xmul(gsl_vector * x1, gsl_vector * x2, gsl_matrix * result);
double gsl_matrix_sum(gsl_matrix * x);
double gsl_vector_sum(gsl_vector * x);
long gsl_vector_long_sum(gsl_vector_long * x);
void Gauss(gsl_vector * x, double w, gsl_vector * y);
void Boxcar(gsl_vector * x, gsl_vector * w, gsl_vector * y);
double CostFunction(gsl_vector* y_hist, double N, gsl_vector * t, double dt,
		gsl_matrix * optws, gsl_vector * WIN, char * WinFunc, double g,
		gsl_matrix * yv, gsl_vector * optwp);
void fftkernelWin(gsl_vector * data, double w, char * WinFunc, gsl_vector * y);
void fftkernel(gsl_vector * data, double w, gsl_vector * y);
void gsl_vector_histc(gsl_vector * x, gsl_vector * edges, gsl_vector* result);
void gsl_vector_long_to_double(gsl_vector_long *in, gsl_vector *out);
void gsl_matrix_long_to_double(gsl_matrix_long *in, gsl_matrix *out);

void gsl_vector_linspace(double min_val, double max_val, size_t num,
		gsl_vector* result);
double gsl_vector_mindiff(gsl_vector * x); //x must be in ascending order
void ssvkernel_from_hist(gsl_vector * y_hist_input, gsl_vector * tin, gsl_matrix * result);
void ssvkernel(gsl_vector * x, gsl_vector * tin, gsl_vector * y_hist_result,gsl_matrix * result);

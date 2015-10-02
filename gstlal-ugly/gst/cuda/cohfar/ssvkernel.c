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
#include "ssvkernel.h"

#define MAX(a,b) (a>b)?(a):(b)
#define MIN(a,b) (a<b)?(a):(b)
#define PI 3.14159265358979323846264338
#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])


void pdf2cdf(PdfCdf *pc) {

	int i,j,ii,jj;
	long double pdfSum=0;

	for ( i=1; i < pc->xn; i++) {
		pc->xspac[i-1] = pc->xtick[i] - pc->xtick[i-1];
	}
	pc->xspac[pc->xn - 1] = pc->xspac[pc->xn-2];

	for ( i=1; i < pc->yn; i++) {
		pc->yspac[i-1] = pc->ytick[i] - pc->ytick[i-1];
	}
	pc->yspac[pc->yn - 1] = pc->yspac[pc->yn-2];

	pdfSum = 0;
	for ( i=0; i < pc->xn; i++) {
		for ( j=0; j < pc->yn; j++) {
			pc->pdf[i][j] *= pc->xspac[i]*pc->yspac[j];
			pdfSum += pc->pdf[i][j];
		}
	}

	for ( i=0; i < pc->xn; i++) {
		for ( j=0; j < pc->yn; j++) {
			pc->pdf[i][j] = pc->pdf[i][j]/pdfSum; 
		}
	}

	for ( i=0; i < pc->xn; i++) {
		for ( j=0; j < pc->yn; j++) {
			pc->cdf[i][j] = 0;
			for ( ii=0; ii < pc->xn; ii++) {
				for ( jj=0; jj < pc->yn; jj++) {
					if (pc->pdf[i][j]>=pc->pdf[ii][jj]){
						pc->cdf[i][j] += pc->pdf[ii][jj];
					}
				}
			
			}
		}
	}

}


void pdf2cdf_sharpcut(PdfCdf *pc) {

	int i,j,ii,jj;
	long double pdfSum=0;

	for ( i=1; i < pc->xn; i++) {
		pc->xspac[i-1] = pc->xtick[i] - pc->xtick[i-1];
	}
	pc->xspac[pc->xn - 1] = pc->xspac[pc->xn-2];

	for ( i=1; i < pc->yn; i++) {
		pc->yspac[i-1] = pc->ytick[i] - pc->ytick[i-1];
	}
	pc->yspac[pc->yn - 1] = pc->yspac[pc->yn-2];

	pdfSum = 0;
	for ( i=0; i < pc->xn; i++) {
		for ( j=0; j < pc->yn; j++) {
			pc->pdf[i][j] *= pc->xspac[i]*pc->yspac[j];
			pdfSum += pc->pdf[i][j];
		}
	}

	for ( i=0; i < pc->xn; i++) {
		for ( j=0; j < pc->yn; j++) {
			pc->pdf[i][j] = pc->pdf[i][j]/pdfSum; 
		}
	}

	for ( i=0; i < pc->xn; i++) {
		for ( j=0; j < pc->yn; j++) {
			pc->cdf[i][j] = 0;
			for ( ii=i; ii < pc->xn; ii++) {
				for ( jj=0; jj <= j; jj++) {
					pc->cdf[i][j] += pc->pdf[ii][jj];
				}
			
			}
		}
	}

}


void gsl_matrix_xmul(gsl_vector * x1, gsl_vector * x2, gsl_matrix * result){
	size_t i,j;
	for(i=0;i<x1->size;i++){
		for(j=0;j<x2->size;j++){
			gsl_matrix_set(result,i,j,gsl_vector_get(x1,i)*gsl_vector_get(x2,j));
		}
	}
}
double gsl_matrix_sum(gsl_matrix * x){
	double result =0 ;
	size_t i,j;
	for(i=0;i<x->size1;i++){
		for(j=0;j<x->size2;j++)
			result+=gsl_matrix_get(x,i,j);
	}
	return result;
}
double gsl_vector_sum(gsl_vector * x) {
	double result = 0;
	size_t i = 0;
	for (i = 0; i < x->size; i++) {
		result += gsl_vector_get(x, i);
	}
	return result;
}
long gsl_vector_long_sum(gsl_vector_long * x) {
	long result = 0;
	size_t i = 0;
	for (i = 0; i < x->size; i++) {
		result += gsl_vector_long_get(x, i);
	}
	return result;
}
void Gauss(gsl_vector * x, double w, gsl_vector * y) {
	double temp = 1 / sqrt(2 * PI) / w;
	size_t i;
	for (i = 0; i < x->size; i++) {
		double temp2 = gsl_vector_get(x, i);
		temp2 = -temp2 * temp2 / 2 / (w * w);
		gsl_vector_set(y, i, temp * exp(temp2));
	}
}
void Boxcar(gsl_vector * x, gsl_vector * w, gsl_vector * y) {
	gsl_vector_memcpy(y, w);
	gsl_vector_scale(y, sqrt(12));
	size_t i;
	for (i = 0; i < x->size; i++) {
		double t1 = gsl_vector_get(x, i);
		double t2 = gsl_vector_get(y, i);
		if (fabs(t1) > t2 / 2)
			gsl_vector_set(y, i, 0);
		else
			gsl_vector_set(y, i, 1 / gsl_vector_get(y, i));
	}
}
double CostFunction(gsl_vector* y_hist, double N, gsl_vector * t, double dt,
		gsl_matrix * optws, gsl_vector * WIN, char * WinFunc, double g,
		gsl_matrix * yv, gsl_vector * optwp) {
	//Selecting w/W = g bandwidth
	size_t L = y_hist->size;
	gsl_vector * optwv = gsl_vector_alloc(L);
	size_t k, i;
	gsl_vector * gs = gsl_vector_alloc(optws->size1);
	for (k = 0; k < L; k++) {
		gsl_matrix_get_col(gs, optws, k);
		gsl_vector_div(gs, WIN);
		if (g > gsl_vector_max(gs)) {
			gsl_vector_set(optwv, k, gsl_vector_min(WIN));
		} else {
			if (g < gsl_vector_min(gs)) {
				gsl_vector_set(optwv, k, gsl_vector_max(WIN));
			} else {
				for (i = 0; i < gs->size; i++) {
					if (gsl_vector_get(gs, gs->size - 1 - i) >= g) {
						gsl_vector_set(optwv, k,
								g * gsl_vector_get(WIN, gs->size - 1 - i));
						break;
					}
				}
			}
		}
	}


	//Nadaraya-Watson kernel regression
	gsl_vector * Z = gsl_vector_alloc(t->size);
	gsl_vector * temp_t = gsl_vector_alloc(t->size);
	gsl_vector * temp_optwv = gsl_vector_alloc(t->size);
	gsl_vector_memcpy(temp_optwv, optwv);
	gsl_vector_scale(temp_optwv, 1 / g);
	for (k = 0; k < L; k++) {
		gsl_vector_memcpy(temp_t, t);
		gsl_vector_scale(temp_t, -1);
		gsl_vector_add_constant(temp_t, gsl_vector_get(t, k));
		if (strcmp(WinFunc, "Boxcar") == 0) {
			Boxcar(temp_t, temp_optwv, Z);
			double temp_sum = gsl_vector_sum(Z);
			gsl_vector_mul(Z, optwv);
			gsl_vector_set(optwp, k, gsl_vector_sum(Z) / temp_sum);
		}
	}
	//Baloon estimator (speed optimized)
	//////////////////////////////////
//	size_t idx = 0;
//	for (i = 0; i < y_hist->size; i++) {
//		if (gsl_vector_get(y_hist, i) != 0) {
//			idx++;
//		}
//	}
//
//	gsl_vector * y_hist_nz = gsl_vector_alloc(idx);
//	gsl_vector * t_nz = gsl_vector_alloc(idx);
//	idx = 0;
//	for (i = 0; i < y_hist->size; i++) {
//		if (gsl_vector_get(y_hist, i) != 0) {
//			gsl_vector_set(y_hist_nz, idx, gsl_vector_get(y_hist, i));
//			gsl_vector_set(t_nz, idx, gsl_vector_get(t, i));
//			idx++;
//		}
//	}
//
//	gsl_vector * temp_t_nz = gsl_vector_alloc(t_nz->size);
//	gsl_vector * temp_t_nz2 = gsl_vector_alloc(t_nz->size);
//
//	for (k = 0; k < L; k++) {
//		gsl_vector_memcpy(temp_t_nz, t_nz);
//		gsl_vector_scale(temp_t_nz, -1);
//		gsl_vector_add_constant(temp_t_nz, gsl_vector_get(t, k));
//		Gauss(temp_t_nz, gsl_vector_get(optwp, k), temp_t_nz2);
//		gsl_vector_mul(temp_t_nz2, y_hist_nz);
//		gsl_vector_set(yv, k, gsl_vector_sum(temp_t_nz2) * dt);
//	}
//	gsl_vector_scale(yv, N / (gsl_vector_sum(yv) * dt)); //rate


	for (k = 0; k < L; k++) {
		gsl_vector_memcpy(temp_t, t);
		gsl_vector_scale(temp_t, -1);
		gsl_vector_add_constant(temp_t,gsl_vector_get(t,k));
		Gauss(temp_t,gsl_vector_get(optwp,k),temp_t);
		gsl_vector_mul(temp_t,y_hist);
		gsl_vector_scale(temp_t,dt);
		gsl_matrix_set_col(yv,k,temp_t);
	}
	gsl_matrix_scale(yv,N/(gsl_matrix_sum(yv)*dt));
///////////////////////////////////
	//Cost function of the estimated density
	double Cg = 0;
	double temp = 2 / sqrt(2 * PI);
	///////////////////////////////////
//	for (i = 0; i < yv->size; i++) {
//		double t1 = gsl_vector_get(yv, i);
//		double t2 = gsl_vector_get(y_hist, i);
//		double t3 = gsl_vector_get(optwp, i);
//		Cg += t1 * t1 - 2 * t1 * t2 + temp / t3 * t2;
//	}
	gsl_vector * yv_col = gsl_vector_alloc(yv->size1);
	for (i = 0; i < yv->size2; i++) {
		gsl_matrix_get_col(yv_col, yv, i);
		double t1 = gsl_vector_sum(yv_col);
		double t2 = gsl_vector_get(y_hist, i);
		double t3 = gsl_vector_get(optwp, i);
		Cg += t1 * t1 - 2 * t1 * t2 + temp / t3 * t2;
	}

	//////////////////////////////////
	Cg *= dt;
	gsl_vector_free(yv_col);
	gsl_vector_free(optwv);
	gsl_vector_free(gs);
	gsl_vector_free(Z);
	gsl_vector_free(temp_t);
	gsl_vector_free(temp_optwv);
//	gsl_vector_free(y_hist_nz);
//	gsl_vector_free(t_nz);
//	gsl_vector_free(temp_t_nz);
//	gsl_vector_free(temp_t_nz2);
	return Cg;

}
void fftkernelWin(gsl_vector * data, double w, char * WinFunc, gsl_vector * y) {
	size_t L = data->size;
	double Lmax = L + 3 * w;

	size_t i, n = pow(2, ceil(log(Lmax) / log(2)));
	double X[n], data2[2 * n];
	for (i = 0; i < n; i++) {
		if (i < L)
			X[i] = gsl_vector_get(data, i);
		else
			X[i] = 0;
	}
	gsl_fft_real_radix2_transform(X, 1, n);
	double f[n], t[n];
	for (i = 0; i < n / 2 + 1; i++) {
		f[i] = -(double)i / (double) n;
		t[i] = f[i] * 2 * PI;
	}
	for (; i < n; i++) {
		f[i] = (double)(n - i) / (double) n;
		t[i] = f[i] * 2 * PI;
	}
	double K[n];
	if (strcmp(WinFunc, "Boxcar") == 0) {
		//Boxcar
//		printf("Boxcar\n");
		double a = sqrt(12) * w;
		for (i = 0; i < n; i++) {
			K[i] = 2 * sin(a * t[i] / 2) / (a * t[i]);
		}
		K[0] = 1;
	} else if (strcmp(WinFunc, "Laplace") == 0) {
		//Laplace
		for (i = 0; i < n; i++) {
			double temp = w * 2 * PI * f[i];
			K[i] = 1 / (temp * temp / 2 + 1);
		}
	} else if (strcmp(WinFunc, "cauchy")) {
		//Cauchy
		for (i = 0; i < n; i++) {
			double temp = fabs(2 * PI * f[i]);
			K[i] = exp(-w * temp);
		}
	} else {
		//Gauss
		double temp = w * 2 * PI;
		for (i = 0; i < n; i++) {
			double temp2 = temp * f[i];
			K[i] = exp(-0.5 * temp2 * temp2);
		}
	}


	gsl_fft_halfcomplex_radix2_unpack(X, data2, 1, n);
	for (i = 0; i < n; i++) {
		REAL(data2, i) = REAL(data2, i) * K[i];
		IMAG(data2, i) = IMAG(data2, i) * K[i];
	}
	gsl_fft_complex_radix2_inverse(data2, 1, n);

	for (i = 0; i < L; i++) {
		gsl_vector_set(y, i, REAL(data2, i));
	}
}
void fftkernel(gsl_vector * data, double w, gsl_vector * y) {
	size_t L = data->size;
	double Lmax = L + 3 * w;
	size_t n = pow(2, ceil(log(Lmax) / log(2)));
	double X[n], data2[2 * n];
	size_t i;
	printf("n %d, L %d\n", n, L);
	for (i = 0; i < n; i++) {
		if (i < L)
			X[i] = gsl_vector_get(data, i);
		else
			X[i] = 0;
	}
	gsl_fft_real_radix2_transform(X, 1, n);

	double f[n], K[n];
	for (i = 0; i < n / 2 + 1; i++) {
		f[i] = -(double)i / (double) n;
	}
	for (; i < n; i++) {
		f[i] = (double)(n - i) / (double) n;
	}

	//Gauss
	for (i = 0; i < n; i++) {
		double temp = w * 2 * PI * f[i];
		K[i] = exp(-0.5 * temp * temp);
	}
	gsl_fft_halfcomplex_radix2_unpack(X, data2, 1, n);
	for (i = 0; i < n; i++) {
		REAL(data2, i) = REAL(data2, i) * K[i];
		IMAG(data2, i) = IMAG(data2, i) * K[i];
	}
	gsl_fft_complex_radix2_inverse(data2, 1, n);
	for (i = 0; i < L; i++) {
		gsl_vector_set(y, i, REAL(data2, i));
	}

}

double logexp(double x) {
	if (x < 1e2)
		return log(1 + exp(x));
	else
		return x;
}
void gsl_vector_logexp(gsl_vector *x)
{
	int i;
	for (i=0; i<x->size; i++)
		gsl_vector_set(x, i, logexp(gsl_vector_get(x, i)));
}
double ilogexp(double x) {
	if (x < 1e2)
		return log(exp(x) - 1);
	else
		return x;
}
void gsl_matrix_hist3(gsl_vector * x_dim1,gsl_vector * x_dim2, gsl_vector * edges_dim1,gsl_vector * edges_dim2,gsl_matrix * result){
	size_t i,j,l,m;
	double temp1,temp2;
	for (i = 0; i < x_dim1->size; i++) {
		temp1 = gsl_vector_get(x_dim1, i);
		temp2 = gsl_vector_get(x_dim2, i);
		for (l = 0; l < edges_dim1->size - 1; l++) {
			if (temp1 >= gsl_vector_get(edges_dim1, l)
					&& temp1 < gsl_vector_get(edges_dim1, l + 1)) {
				for (m = 0; m < edges_dim2->size - 1; m++) {
					if (temp2 >= gsl_vector_get(edges_dim2, m)
							&& temp2 < gsl_vector_get(edges_dim2, m + 1)) {
						gsl_matrix_set(result, l, m,
								gsl_matrix_get(result, l, m) + 1);
						break;
					}
				}
				if (temp2 == gsl_vector_get(edges_dim2, edges_dim2->size - 1)) {
					gsl_matrix_set(result, l, edges_dim2->size - 1,
							gsl_matrix_get(result, l, edges_dim2->size - 1)
									+ 1);
				}
				break;
			}

		}
		if (temp1 == gsl_vector_get(edges_dim1, edges_dim1->size - 1)) {
			for (m = 0; m < edges_dim2->size - 1; m++) {
				if (temp2 >= gsl_vector_get(edges_dim2, m)
						&& temp2 < gsl_vector_get(edges_dim2, m + 1)) {
					gsl_matrix_set(result, edges_dim1->size-1, m,
							gsl_matrix_get(result, edges_dim1->size-1, m) + 1);
					break;
				}
			}
			if (temp2 == gsl_vector_get(edges_dim2, edges_dim2->size - 1)) {
				gsl_matrix_set(result, edges_dim1->size-1, edges_dim2->size - 1,
						gsl_matrix_get(result, edges_dim1->size-1, edges_dim2->size - 1) + 1);
			}
		}
	}
}
void gsl_vector_histc(gsl_vector * x, gsl_vector * edges, gsl_vector* result) {
	size_t i, j;
	double temp;
	for (i = 0; i < x->size; i++) {
		temp = gsl_vector_get(x, i);
		for (j = 0; j < edges->size - 1; j++) {
			if (temp >= gsl_vector_get(edges, j)
					&& temp < gsl_vector_get(edges, j + 1)) {
				gsl_vector_set(result, j, gsl_vector_get(result, j) + 1);
				break;
			}
		}
		if (temp == gsl_vector_get(edges, edges->size - 1))
			gsl_vector_set(result, edges->size - 1,
					gsl_vector_get(result, edges->size - 1) + 1);
	}
}
void gsl_vector_long_to_double(gsl_vector_long *in, gsl_vector *out)
{
	int i;
	for (i=0; i<in->size; i++) 
		gsl_vector_set(out, i, (double)gsl_vector_long_get(in, i));
	
}
void gsl_matrix_long_to_double(gsl_matrix_long *in, gsl_matrix *out)
{
	int i, j;
	for (i=0; i<in->size1; i++) 
		for (j=0; j<in->size2; j++) 
			gsl_matrix_set(out, i, j, (double)gsl_matrix_long_get(in, i, j));
	
}

void gsl_vector_linspace(double min_val, double max_val, size_t num,
		gsl_vector* result) {
	if (num == 1) {
		gsl_vector_set(result, 0, max_val);
		return;
	}
	double interval = (max_val - min_val) / (num - 1);
	size_t i = 0;
	for (i = 0; i < num; i++) {
		gsl_vector_set(result, i, min_val + (double)i * interval);
	}
}
double gsl_vector_mindiff(gsl_vector * x) { //x must be in ascending order
	double min = gsl_vector_get(x, 1) - gsl_vector_get(x, 0);
	size_t i = 0;
	for (i = 2; i < x->size; i++) {
		double temp = gsl_vector_get(x, i) - gsl_vector_get(x, i - 1);
		if (temp < min) {
			min = temp;
		}
	}
	return min;
}

void ssvkernel_from_hist(gsl_vector * y_hist_input, gsl_vector * tin, gsl_matrix * result) {
	size_t M = 80;	//Number of bandwidths examined for optimization.

	char * WinFunc = "Boxcar";	//Window function ('Gauss','Laplace','Cauchy')
	// only "Boxcar" is available

	size_t nbs = 1 * 1e2;	//number of bootstrap samples

	double confidence = 0.95;

	size_t i, j;

	double max_tin = gsl_vector_max(tin);
	double min_tin = gsl_vector_min(tin);
	double T = max_tin - min_tin;
	double dt = gsl_vector_mindiff(tin); // equals bins1D->step

	///////////////////////////////////////////
	gsl_vector *y_hist = gsl_vector_alloc(y_hist_input->size);
	gsl_vector_memcpy(y_hist, y_hist_input);
	gsl_vector_scale(y_hist,1/dt);
	size_t L = y_hist->size;
	double N = gsl_vector_sum(y_hist) * dt;

	//Computing local MISEs and optimal bandwidths
	printf("computing local bandwidths....\n");

	//Window sizes
	gsl_vector * temp = gsl_vector_alloc(M);
	gsl_vector * WIN = gsl_vector_alloc(M);

	gsl_vector_linspace(ilogexp(5 * dt), ilogexp(T), M, temp);
	gsl_vector_linspace(ilogexp(5 * dt), ilogexp(T), M, WIN);


	gsl_vector_logexp(temp);
	gsl_vector_logexp(WIN);

	gsl_matrix * c = gsl_matrix_alloc(M, L);
	gsl_vector * yh = gsl_vector_alloc(L);

	double sqrt_temp = 2 / sqrt(2 * PI);
	for (j = 0; j < M; j++) {
		double w = gsl_vector_get(WIN, j);
		printf("j %d, w %f, dt %f\n", j, w, dt);
		fftkernel(y_hist, w / dt, yh);
		for (i = 0; i < L; i++) {
			double yh_val = gsl_vector_get(yh, i);
			double y_hist_val = gsl_vector_get(y_hist, i);
			double temp = yh_val * yh_val - 2 * yh_val * y_hist_val
					+ sqrt_temp / w * y_hist_val;
			gsl_matrix_set(c, j, i, temp);
		}
	}

	gsl_matrix * optws = gsl_matrix_alloc(M, L);
	gsl_matrix * C_local = gsl_matrix_alloc(M, L);

	gsl_vector * c_row = gsl_vector_alloc(L);
	gsl_vector * C_local_row = gsl_vector_alloc(L);
	gsl_vector * C_local_col = gsl_vector_alloc(M);

	for (i = 0; i < M; i++) {
		double Win = gsl_vector_get(WIN, i);
		for (j = 0; j < M; j++) {
			//computing local cost funtion
			gsl_matrix_get_row(c_row, c, j);
			fftkernelWin(c_row, Win / dt, WinFunc, C_local_row);
			gsl_matrix_set_row(C_local, j, C_local_row);
		}
		for (j = 0; j < L; j++) {
			gsl_matrix_get_col(C_local_col, C_local, j);//find optw at t=1....L
			gsl_matrix_set(optws, i, j,
					gsl_vector_get(WIN, gsl_vector_min_index(C_local_col)));
		}
	}

	//Golden section search of the stiffness parameter of variable bandwidths.
	//Selecting a bandwidth w/W = g.
	printf("adapting local bandwidths....\n");

	double tol = 1e-5, a = 1e-12, b = 1;
	double phi = (sqrt(5) + 1) / 2;

	double c1 = (phi - 1) * a + (2 - phi) * b;
	double c2 = (2 - phi) * a + (phi - 1) * b;

	////////////////////////////////////
//	gsl_vector * yv = gsl_vector_alloc(L);

	gsl_matrix * yv = gsl_matrix_alloc(L,L);
	/////////////////////////////////////
	gsl_vector *optwp = gsl_vector_alloc(L);

	double f1 = CostFunction(y_hist, N, tin, dt, optws, WIN, WinFunc, c1, yv,
			optwp);
	double f2 = CostFunction(y_hist, N, tin, dt, optws, WIN, WinFunc, c2, yv,
			optwp);
//	printf("f1 = %.16lf, f2=%.16lf \n",f1,f2);
	size_t k = 1;
	while ((fabs(b - a) > tol * (fabs(c1) + fabs(c2))) && k < 30) {
		if (f1 < f2) {
			b = c2;
			c2 = c1;
			c1 = (phi - 1) * a + (2 - phi) * b;
			f2 = f1;
			f1 = CostFunction(y_hist, N, tin, dt, optws, WIN, WinFunc, c1, yv,
					optwp);
//				printf("K=%lu,f1:%.16lf\n",k,f1);
		} else {
			a = c1;
			c1 = c2;
			c2 = (2 - phi) * a + (phi - 1) * b;
			f1 = f2;
			f2 = CostFunction(y_hist, N, tin, dt, optws, WIN, WinFunc, c2, yv,
					optwp);
//				printf("K=%lu,f2:%.16lf\n",k,f2);
		}
		k++;
	}
	///////////////////////////////////
//	gsl_vector_scale(yv, 1 / (gsl_vector_sum(yv) * dt));

	///////////////////////////////////

	printf("optimization completed\n");
	gsl_matrix_memcpy(result,yv);
}
	

void ssvkernel(gsl_vector * x, gsl_vector * tin, gsl_vector * y_hist_result,gsl_matrix * result) {
	size_t M = 80;	//Number of bandwidths examined for optimization.

	char * WinFunc = "Boxcar";	//Window function ('Gauss','Laplace','Cauchy')
	// only "Boxcar" is available

	size_t nbs = 1 * 1e2;	//number of bootstrap samples

	double confidence = 0.95;

	size_t i, j;

	double max_tin = gsl_vector_max(tin);
	double min_tin = gsl_vector_min(tin);
	double T = max_tin - min_tin;

	size_t number = 0;

	//set x_ab variable
	for (i = 0; i < x->size; i++) {
		if (gsl_vector_get(x, i) <= max_tin
				&& gsl_vector_get(x, i >= min_tin)) {
			number++;
		}
	}
	gsl_vector * x_ab = gsl_vector_alloc(number);
	gsl_vector * temp_x_ab = gsl_vector_alloc(number);

	number = 0;
	for (i = 0; i < x->size; i++) {
		if (gsl_vector_get(x, i) <= max_tin
				&& gsl_vector_get(x, i) >= min_tin) {
			gsl_vector_set(x_ab, number, gsl_vector_get(x, i));
			number++;
		}
	}
	//finished
	gsl_vector_memcpy(temp_x_ab,x_ab);
	gsl_sort_vector(temp_x_ab);
	number = 0;
	double dt_samp = 0;
	for (i = 0; i < temp_x_ab->size - 1; i++) {
		double diff = gsl_vector_get(temp_x_ab, i + 1)
				- gsl_vector_get(temp_x_ab, i);
		if (diff != 0) {
			if (number == 0 || diff < dt_samp) {
				dt_samp = diff;
				number++;
			}
		}
	}

	//set t variable
	gsl_vector * t;
	////////////////////////////////////
//	if (dt_samp > gsl_vector_mindiff(tin)) {
//		size_t temp = MIN(ceil(T / dt_samp), 1e3);
//		t = gsl_vector_alloc(temp);
//		gsl_vector_linspace(min_tin, max_tin, temp, t);
//	} else {
		t = gsl_vector_alloc(tin->size);
		gsl_vector_memcpy(t, tin);
//	}
	/////////////////////////////
	//finished
	double dt = gsl_vector_mindiff(t);

	//Create a finest histogram
	gsl_vector * t_dt2 = gsl_vector_alloc(t->size);
	gsl_vector_memcpy(t_dt2, t);
	gsl_vector_add_constant(t_dt2, -dt / 2);
	gsl_vector * y_hist = gsl_vector_alloc(t->size);
	///////////////////////////////////////////
	gsl_vector_histc(x_ab, t_dt2, y_hist);
	gsl_vector_memcpy(y_hist_result,y_hist);
	///////////////////////////////////////////
	gsl_vector_scale(y_hist,1/dt);
	size_t L = y_hist->size;
	double N = gsl_vector_sum(y_hist) * dt;

	//Computing local MISEs and optimal bandwidths
	printf("computing local bandwidths....\n");

	//Window sizes
	gsl_vector * temp = gsl_vector_alloc(M);
	gsl_vector * WIN = gsl_vector_alloc(M);

	gsl_vector_linspace(ilogexp(5 * dt), ilogexp(T), M, temp);
	gsl_vector_linspace(ilogexp(5 * dt), ilogexp(T), M, WIN);


	gsl_matrix * c = gsl_matrix_alloc(M, L);
	gsl_vector * yh = gsl_vector_alloc(L);

	double sqrt_temp = 2 / sqrt(2 * PI);
	for (j = 0; j < M; j++) {
		double w = gsl_vector_get(WIN, j);
		fftkernel(y_hist, w / dt, yh);
		for (i = 0; i < L; i++) {
			double yh_val = gsl_vector_get(yh, i);
			double y_hist_val = gsl_vector_get(y_hist, i);
			double temp = yh_val * yh_val - 2 * yh_val * y_hist_val
					+ sqrt_temp / w * y_hist_val;
			gsl_matrix_set(c, j, i, temp);
		}
	}

	gsl_matrix * optws = gsl_matrix_alloc(M, L);
	gsl_matrix * C_local = gsl_matrix_alloc(M, L);

	gsl_vector * c_row = gsl_vector_alloc(L);
	gsl_vector * C_local_row = gsl_vector_alloc(L);
	gsl_vector * C_local_col = gsl_vector_alloc(M);

	for (i = 0; i < M; i++) {
		double Win = gsl_vector_get(WIN, i);
		for (j = 0; j < M; j++) {
			//computing local cost funtion
			gsl_matrix_get_row(c_row, c, j);
			fftkernelWin(c_row, Win / dt, WinFunc, C_local_row);
			gsl_matrix_set_row(C_local, j, C_local_row);
		}
		for (j = 0; j < L; j++) {
			gsl_matrix_get_col(C_local_col, C_local, j);//find optw at t=1....L
			gsl_matrix_set(optws, i, j,
					gsl_vector_get(WIN, gsl_vector_min_index(C_local_col)));
		}
	}

	//Golden section search of the stiffness parameter of variable bandwidths.
	//Selecting a bandwidth w/W = g.
	printf("adapting local bandwidths....\n");

	double tol = 1e-5, a = 1e-12, b = 1;
	double phi = (sqrt(5) + 1) / 2;

	double c1 = (phi - 1) * a + (2 - phi) * b;
	double c2 = (2 - phi) * a + (phi - 1) * b;

	////////////////////////////////////
//	gsl_vector * yv = gsl_vector_alloc(L);

	gsl_matrix * yv = gsl_matrix_alloc(L,L);
	/////////////////////////////////////
	gsl_vector *optwp = gsl_vector_alloc(L);

	double f1 = CostFunction(y_hist, N, t, dt, optws, WIN, WinFunc, c1, yv,
			optwp);
	double f2 = CostFunction(y_hist, N, t, dt, optws, WIN, WinFunc, c2, yv,
			optwp);
//	printf("f1 = %.16lf, f2=%.16lf \n",f1,f2);
	size_t k = 1;
	while ((fabs(b - a) > tol * (fabs(c1) + fabs(c2))) && k < 30) {
		if (f1 < f2) {
			b = c2;
			c2 = c1;
			c1 = (phi - 1) * a + (2 - phi) * b;
			f2 = f1;
			f1 = CostFunction(y_hist, N, t, dt, optws, WIN, WinFunc, c1, yv,
					optwp);
//				printf("K=%lu,f1:%.16lf\n",k,f1);
		} else {
			a = c1;
			c1 = c2;
			c2 = (2 - phi) * a + (phi - 1) * b;
			f1 = f2;
			f2 = CostFunction(y_hist, N, t, dt, optws, WIN, WinFunc, c2, yv,
					optwp);
//				printf("K=%lu,f2:%.16lf\n",k,f2);
		}
		k++;
	}
	///////////////////////////////////
//	gsl_vector_scale(yv, 1 / (gsl_vector_sum(yv) * dt));

	///////////////////////////////////

	printf("optimization completed\n");
	gsl_matrix_memcpy(result,yv);
	//Bootstrap Confidence Interval
//	printf("computing bootsrap confidence intervals....\n");
//	gsl_matrix * yb = gsl_matrix_alloc(nbs, tin->size);
//
//	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
//	gsl_vector * y_histb = gsl_vector_alloc(t->size);
//	gsl_vector * temp_t = gsl_vector_alloc(t->size);
//	gsl_vector * yb_buf = gsl_vector_alloc(L);
//	gsl_interp_accel *acc = gsl_interp_accel_alloc();
//	gsl_interp * linear = gsl_interp_alloc(gsl_interp_linear, t->size);	//linear interpolation
//
//	for (i = 0; i < nbs; i++) {
//		size_t Nb = gsl_ran_poisson(r, N);
//		gsl_vector * xb = gsl_vector_alloc(Nb);
//		for (j = 0; j < Nb; j++) {
//			gsl_vector_set(xb, j,
//					gsl_vector_get(x_ab, floor(gsl_ran_flat(r, 0, N))));
//		}
//
//		gsl_vector_histc(xb, t_dt2, y_histb);
//		for (k = 0; k < L; k++) {
//			gsl_vector_memcpy(temp_t, t);
//			gsl_vector_scale(temp_t, -1);
//			gsl_vector_add_constant(temp_t, gsl_vector_get(t, k));
//			Gauss(temp_t, gsl_vector_get(optwp, k), temp_t);
//			gsl_vector_mul(temp_t, y_histb);
//			gsl_vector_set(yb_buf, k, gsl_vector_sum(temp_t) / Nb);
//		}
//		gsl_vector_scale(yb_buf, 1/(gsl_vector_sum(yb_buf) * dt));
//		gsl_interp_init(linear, t->data, yb_buf->data, t->size);
//		for (j = 0; j < tin->size; j++) {
//			gsl_matrix_set(yb, i, j,
//					gsl_interp_eval(linear, t->data, yb_buf->data,
//							gsl_vector_get(tin, j), acc));
//		}
//		gsl_vector_free(xb);
//	}
//	gsl_vector * yb_col = gsl_vector_alloc(yb->size1);

//	for (i = 0; i < lower_bound->size; i++) {
//		gsl_matrix_get_col(yb_col, yb, i);
//		gsl_sort_vector(yb_col);
//		gsl_vector_set(lower_bound, i,
//				gsl_vector_get(yb_col, floor((1 - confidence) * nbs)));
//		gsl_vector_set(upper_bound, i,
//				gsl_vector_get(yb_col, floor((confidence) * nbs)));
//	}
//	gsl_interp_init(linear, t->data, yv->data, t->size);
//	for (i = 0; i < tin->size; i++) {
//		gsl_vector_set(result, i,
//				gsl_interp_eval(linear, t->data, yv->data,
//						gsl_vector_get(tin, i), acc));
//	}
//	gsl_interp_accel_free(acc);
//	gsl_interp_free(linear);
//	gsl_rng_free(r);
//	gsl_vector_free(temp_t);
	gsl_vector_free(temp_x_ab);
//	gsl_vector_free(yb_buf);
	gsl_vector_free(x_ab);
	gsl_vector_free(t);
	gsl_vector_free(t_dt2);
	gsl_vector_free(y_hist);
	gsl_vector_free(temp);
	gsl_vector_free(WIN);
	gsl_vector_free(yh);
	gsl_vector_free(c_row);
	gsl_vector_free(C_local_row);
	gsl_vector_free(C_local_col);
	///////////////////////////////////
//	gsl_vector_free(yv);

	gsl_matrix_free(yv);
	///////////////////////////////////
	gsl_vector_free(optwp);
//	gsl_vector_free(y_histb);
//	gsl_vector_free(yb_col);
	gsl_matrix_free(c);
	gsl_matrix_free(optws);
	gsl_matrix_free(C_local);
//	gsl_matrix_free(yb);
}
#if 0
int main() {
	int MAX_LINE = 100;
	size_t num_line = 0; //number of line in file
	size_t i,j,k,l,m;
	char buf[MAX_LINE];
	FILE *fp1,*fp2;
	if ((fp1 = fopen("/home/wangyan/Code/new_pdfcdf/AdaptiveKDE/data/data.txt", "r")) == NULL) { /* open file of data in dimension1*/
		printf("read file error dimen1\n");
		return 0;
	}
	if ((fp2 = fopen("/home/wangyan/Code/new_pdfcdf/AdaptiveKDE/data/data2.txt", "r")) == NULL) { /* open file of data in dimension2*/
		printf("read file error dimen1\n");
		return 0;
	}
	while (fgets(buf, MAX_LINE, fp1) != NULL) { // count number of lines in file.
		num_line++;								//each file's number of lines must be the same
	}
	rewind(fp1);

	//data_dim1 and data_dim2 contain the data of two dimensions
	gsl_vector * data_dim1 = gsl_vector_alloc(num_line);
	gsl_vector * data_dim2 = gsl_vector_alloc(num_line);
	i = 0;
	while (fgets(buf, MAX_LINE, fp1) != NULL) { //read the data
		gsl_vector_set(data_dim1, i, atoi(buf));
		fgets(buf, MAX_LINE, fp2) ;
		gsl_vector_set(data_dim2, i, atoi(buf));
		i++;
	}

	//tin_dim1 and tin_dim2 contains points at which estimations are computed
	size_t num_bin1 = XN, num_bin2 = YN; //each bin's size
	gsl_vector * tin_dim1 =  gsl_vector_alloc(num_bin1);
	gsl_vector * tin_dim2 =  gsl_vector_alloc(num_bin2);
    //bin of each dimension
    //
    	double tin_dim1_max = gsl_vector_max(data_dim1) + 0.5;  // linspace in power (i.e. logspace)
	double tin_dim1_min = gsl_vector_min(data_dim1) - 0.5;
	double tin_dim2_max = gsl_vector_max(data_dim2) + 0.5;
	double tin_dim2_min = gsl_vector_min(data_dim2) - 0.5;

	gsl_vector_linspace(tin_dim1_min, tin_dim1_max, num_bin1, tin_dim1);
	gsl_vector_linspace(tin_dim2_min, tin_dim2_max, num_bin2, tin_dim2);
	gsl_vector * y_hist_result_dim1 = gsl_vector_alloc(num_bin1);
	gsl_vector * y_hist_result_dim2 = gsl_vector_alloc(num_bin2);
    //histogram of each dimension
	gsl_matrix * result_dim1  = gsl_matrix_alloc(tin_dim1->size,tin_dim1->size);
	gsl_matrix * result_dim2  = gsl_matrix_alloc(tin_dim2->size,tin_dim2->size);

	//Compute temporary result of each dimension, equal to the 'y1' and 'y2' in matlab code 'test.m';
	ssvkernel(data_dim2,tin_dim2,y_hist_result_dim2,result_dim2);
	ssvkernel(data_dim1,tin_dim1,y_hist_result_dim1,result_dim1);

	//two-dimensional histogram
	gsl_vector * temp_tin_dim1 =  gsl_vector_alloc(num_bin1);
	gsl_vector * temp_tin_dim2 =  gsl_vector_alloc(num_bin2);
	gsl_vector_memcpy(temp_tin_dim1,tin_dim1);
	gsl_vector_add_constant(temp_tin_dim1,-gsl_vector_mindiff(tin_dim1)/2);
	gsl_vector_memcpy(temp_tin_dim2,tin_dim2);
	gsl_vector_add_constant(temp_tin_dim2,-gsl_vector_mindiff(tin_dim2)/2);
	gsl_matrix * histogram = gsl_matrix_alloc(tin_dim1->size,tin_dim2->size);
	gsl_matrix_hist3(data_dim1,data_dim2,temp_tin_dim1,temp_tin_dim2,histogram);

	//Compute the 'scale' variable in matlab code 'test.m'
	for(i=0;i<histogram->size1;i++){
		for(j=0;j<histogram->size2;j++){
			double temp = gsl_matrix_get(histogram,i,j);
			temp = temp/(gsl_vector_get(y_hist_result_dim1,i)*gsl_vector_get(y_hist_result_dim2,j));
			if(isnan(temp))
				gsl_matrix_set(histogram,i,j,0);
			else
				gsl_matrix_set(histogram,i,j,temp);
		}
	}

	//compute the two-dimensional estimation
	gsl_matrix * result = gsl_matrix_alloc(tin_dim1->size,tin_dim2->size);//final result
	gsl_matrix * temp_matrix = gsl_matrix_alloc(tin_dim1->size,tin_dim2->size);
	for(i=0;i<tin_dim1->size;i++){
		for(j=0;j<tin_dim2->size;j++){
			gsl_matrix_get_col(y_hist_result_dim1,result_dim1,i);
			gsl_matrix_get_col(y_hist_result_dim2,result_dim2,j);
			gsl_matrix_xmul(y_hist_result_dim1,y_hist_result_dim2,temp_matrix);
			gsl_matrix_mul_elements(temp_matrix,histogram);
			gsl_matrix_set(result,i,j,gsl_matrix_sum(temp_matrix)/(double)data_dim1->size);
		}
	}

	PdfCdf *pc = (PdfCdf *) malloc(sizeof(PdfCdf));
	pc->xn = (int) XN;
	pc->yn = (int) YN;

	for ( i=0; i<tin_dim1->size; i++) {
		pc->xtick[i] = gsl_vector_get(tin_dim1, i);
	}

	for ( j=0; j<tin_dim2->size; j++) {
		pc->ytick[j] = gsl_vector_get(tin_dim2, j);
	}

	for ( i=0; i<tin_dim1->size; i++) {
		for ( j=0; j<tin_dim2->size; j++) {
		
			pc->pdf[i][j] = gsl_matrix_get(result, i, j);

		}
	}

	pdf2cdf(pc);
	//pdf2cdf_sharpcut(pc);


/*	printf("printing PDF \n");
	for ( i=0; i<pc->xn; i++) {
		for ( j=0; j<pc->yn; j++) {
		
			printf("%Lf \t", pc->pdf[i][j]);

		}
		printf("\n");
	}
*/


	printf("printing CDF \n");
	for ( i=0; i<pc->xn; i++) {
		for ( j=0; j<pc->yn; j++) {
		
			printf("%Lf \t", pc->cdf[i][j]) ;

		}
		printf("\n");
	}

	free(pc);
	
	fclose(fp1);
	fclose(fp2);
	gsl_vector_free(y_hist_result_dim1);
	gsl_vector_free(y_hist_result_dim2);
	gsl_vector_free(data_dim1);
	gsl_vector_free(data_dim2);
	gsl_vector_free(tin_dim1);
	gsl_vector_free(tin_dim2);
	gsl_vector_free(temp_tin_dim1);
	gsl_vector_free(temp_tin_dim2);
	gsl_matrix_free(result_dim1);
	gsl_matrix_free(result_dim2);
	gsl_matrix_free(result);
	gsl_matrix_free(histogram);
	gsl_matrix_free(temp_matrix);
}
#endif

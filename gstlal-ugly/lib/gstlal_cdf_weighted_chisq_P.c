/*
 * Copyright (C) Robert Davies
 * with modifications
 * Copyright (C) 2010  Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * This code is adapted from Robert Davies' "qf" source code
 *
 *	http://www.robertnz.net/ftp/qf.tar.gz
 *
 * implementing the algorithm described in
 *
 * (1973) Numerical inversion of a characteristic function.  Biometrika  60
 * 415-417
 *
 * and
 *
 * (1980) The distribution of a linear combination of chi-squared random
 * variables.  Applied Statistics  29 323-333
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <stdlib.h>
#include <math.h>
#include <setjmp.h>


#include <gstlal_cdf_weighted_chisq_P.h>


#define TRUE  1
#define FALSE 0
#define PI 3.1415926535897932384626433832795028841968
#define LOG2_OVER_8 .0866433975699931636771540151822720710094	/* ln(2) / 8 */


/*
 * ============================================================================
 *
 *                             Internal Functions
 *
 * ============================================================================
 */


/*
 * count number of calls to errbd, truncation, cfe
 */


static void counter(jmp_buf env, int *count, int lim)
{
	if(++*count > lim && lim >= 0)
		longjmp(env, 1);
}


/*
 * if (first) log(1 + x) ; else  log(1 + x) - x
 */


static double log1(double x, int first)
{
	if(fabs(x) > 0.1) {
		return first ? log(1 + x) : (log(1 + x) - x);
	} else {
		double s, s1, term, y, k;
		y = x / (2 + x);
		term = 2 * pow(y, 3);
		k = 3;
		s = (first ? 2 : -x) * y;
		y = pow(y, 2);
		for(s1 = s + term / k; s1 != s; s1 = s + term / k) {
			k += 2;
			term *= y;
			s = s1;
		}
		return s;
	}
}


/*
 * find bound on tail probability using mgf, cutoff point returned to *xconst
 */


static double errbd(double u, double *xconst, jmp_buf env, int *count, int lim, double var, const double *A, const double *noncent, const int *dof, int N)
{
	double sum;
	int j;

	counter(env, count, lim);

	*xconst = u * var;
	sum = u * u * var;
	u *= 2;
	for(j = 0; j < N; j++) {
		double x = u * A[j];
		double y = (noncent[j] / (1 - x) + dof[j]) / (1 - x);
		*xconst += A[j] * y;
		sum += pow(x, 2) * y + dof[j] * log1(-x, FALSE);
	}

	return exp(-sum / 2);
}


/*
 * find ctff so that p(qf > ctff) < acc if upn > 0, p(qf < ctff) < acc otherwise
 */


static double ctff(double acc, double upn, jmp_buf env, int *count, int lim, double var, const double *A, const double *noncent, const int *dof, int N, double Amin, double Amax, double mean)
{
	double u, xconst, c2;
	double c1 = mean;
	double u1 = 0;
	double rb = 2 * ((upn > 0) ? Amax : Amin);

	for(; errbd(u = upn / (1 + upn * rb), &c2, env, count, lim, var, A, noncent, dof, N) > acc; upn *= 2) {
		u1 = upn;
		c1 = c2;
	}

	for(u = (c1 - mean) / (c2 - mean); u < 0.9; u = (c1 - mean) / (c2 - mean)) {
		u = (u1 + upn) / 2;
		if(errbd(u / (1 + u * rb), &xconst, env, count, lim, var, A, noncent, dof, N) > acc) {
			u1 = u;
			c1 = xconst;
		} else {
			upn = u;
			c2 = xconst;
		}
	}

	return c2;
}


/*
 * bound integration error due to truncation at u
 */


static double truncation(double u, double var_plus_tausq, jmp_buf env, int *count, int lim, const double *A, const double *noncent, const int *dof, int N)
{
	double sum1, sum2, prod1, prod2, prod3, x, y, err1, err2;
	int j, s;

	counter(env, count, lim);
	sum1 = 0;
	prod2 = 0;
	prod3 = 0;
	s = 0;
	sum2 = var_plus_tausq * pow(u, 2);
	prod1 = 2 * sum2;
	u *= 2;
	for(j = 0; j < N; j++) {
		x = pow(u * A[j], 2);
		sum1 += noncent[j] * x / (1 + x);
		if(x > 1) {
			prod2 += dof[j] * log(x);
			prod3 += dof[j] * log1(x, TRUE);
			s += dof[j];
		} else
			prod1 += dof[j] * log1(x, TRUE);
	}
	sum1 /= 2;
	prod2 += prod1;
	prod3 += prod1;
	x = exp(-sum1 -  prod2 / 4) / PI;
	y = exp(-sum1 - prod3 / 4) / PI;
	err1 = (s == 0) ? 1 : x * 2 / s;
	err2 = (prod3 > 1) ? 2.5 * y : 1;
	if(err2 < err1)
		err1 = err2;
	x = sum2 / 2;
	err2 = (x <= y) ? 1 : y / x;
	return (err1 < err2) ? err1 : err2;
}


/*
 * find u such that truncation(u) <= acc and truncation(u / ~1.09) > acc
 */


static double findu(double u, double var, double acc, jmp_buf env, int *count, int lim, const double *A, const double *noncent, const int *dof, int N)
{
	static const double factors[] = {
		4,	/* 4^{1} */
		2,	/* 4^{0.5} */
		1.41421356237309504880,	/* 4^{.25} */
		1.18920711500272106671, /* 4^{.125} */
		1.09050773266525765920, /* 4^{.0625} */
		0	/* end */
	};
	int i;

	/* if u is too small, increase by factors of 4 until it's large
	 * enough;  otherwise try decreasing by factors of 4 as long as
	 * it's still big enough */
	if(truncation(u, var, env, count, lim, A, noncent, dof, N) > acc) {
		do
			u *= factors[0];
		while(truncation(u, var, env, count, lim, A, noncent, dof, N) > acc);
	} else while(truncation(u / factors[0], var, env, count, lim, A, noncent, dof, N) <= acc)
		u /= factors[0];

	/* <-- u is not less than, and not more than a factor of 4 bigger
	 * than the desired value */

	/* remove successively smaller factors from u as long as it remains
	 * large enough */
	for(i = 1; factors[i]; i++)
		if(truncation(u / factors[i], var, env, count, lim, A, noncent, dof, N) <= acc)
			u /= factors[i];

	return u;
}


/*
 * carry out integration with (nterm + 1) terms, at stepsize delta_u.  if
 * !mainx multiply integrand by 1-exp(-0.5*tausq*u^2)
 */


static double integrate(int nterm, double delta_u, double var, double tausq, int mainx, const double *A, const double *noncent, const int *dof, double c, int N, double *absolute_sum, const int *index)
{
	double inpi = delta_u / PI;
	double integral = 0;
	int k, j;

	for(k = nterm; k >= 0; k--) {
		double u = (k + 0.5) * delta_u;
		double sum1 = -2 * u * c;
		double sum2 = fabs(sum1);
		double sum3 = -var * pow(u, 2);
		double x;
		for(j = 0; j < N; j++) {
			double y;
			double z;
			x = 2 * u * A[index[j]];
			y = pow(x, 2);
			sum3 -= 0.5 * dof[index[j]] * log1(y, TRUE);
			y = noncent[index[j]] * x / (1 + y);
			z = dof[index[j]] * atan(x) + y;
			sum1 += z;
			sum2 += fabs(z);
			sum3 -= x * y;
		}
		x = inpi * exp(sum3 / 2) / u;
		if(!mainx)
			x *= 1 - exp(-0.5 * tausq * pow(u, 2));
		sum1 = sin(sum1 / 2) * x;
		sum2 *= x / 2;
		integral += sum1;
		*absolute_sum += sum2;
	}

	return integral;
}


/*
 * construct a look-up table giving the indexes of the elements of A sorted
 * in increasing order by absolute value
 */


static int compare_sort_data(const void *a, const void *b)
{
	double diff = fabs(**(const double **) a) - fabs(**(const double **) b);
	return diff > 0 ? +1 : diff < 0 ? -1 : 0;
}

static int *order(const double *A, int N)
{
	int *index = malloc(N * sizeof(*index));
	const double **sort_data = malloc(N * sizeof(*sort_data));
	int j;

	if(!sort_data || !index) {
		free(sort_data);
		free(index);
		return NULL;
	}

	for(j = 0; j < N; j++)
		sort_data[j] = &A[j];

	qsort(sort_data, N, sizeof(*sort_data), compare_sort_data);

	for(j = 0; j < N; j++)
		index[j] = sort_data[j] - A;

	free(sort_data);
	return index;
}


/*
 * coef of tausq in error when convergence factor of exp(-0.5*tausq*u^2) is
 * used when df is evaluated at x
 */


static double cfe(double x, jmp_buf env, int *count, int lim, const double *A, const double *noncent, const int *dof, int N, int *index)
{
	double axl, sxl, sum;
	int j;
	counter(env, count, lim);
	axl = fabs(x);
	sxl = (x > 0) ? +1 : -1;
	sum = 0;
	for(j = 0; j < N; j++) {
		int i = index[j];
		if(A[i] * sxl > 0) {
			double absA = fabs(A[i]);
			double axl1 = axl - absA * (dof[i] + noncent[i]);
			double axl2 = absA / LOG2_OVER_8;
			if(axl1 <= axl2) {
				if(axl > axl2)
					axl = axl2;
				sum = (axl - axl1) / absA;
				for(j++; j < N; j++)
					sum += dof[index[j]] + noncent[index[j]];
				break;
			}
			axl = axl1;
		}
	}

	/* double precision floats overflow at ~2^{1024} */
	if(sum >= 4096)
		return nan("");
	return pow(2, sum / 4) / (PI * pow(axl, 2));
}


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * Compute the cummulative distribution function for a linear combination
 * of non-central chi-squared random variables.  Returns the value of the
 * cumulative distribution function at c or NaN on failure.
 */


double gstlal_cdf_weighted_chisq_P(
	/* coefficient of j-th \chi^{2} variable */
	const double *A,
	/* non-centrality parameter of the j-th \chi^{2} variable */
	const double *noncent,
	/* degrees of freedom of the j-th \chi^{2} variable */
	const int *dof,
	/* number of \chi^{2} variables */
	int N,
	/* variance of zero-mean normal variable */
	double var,
	/* point at which distribution is to be evaluated */
	double c,
	/* maximum number of terms in integration;  < 0 --> no limit */
	int lim,
	/* maximum error */
	double accuracy,
	/* output: if not NULL will contain diagnostic information */
	struct gstlal_cdf_weighted_chisq_P_trace *trace,
	/* output: if not NULL will contain reason for failure
	 *	1      required accuracy NOT achieved
	 *	2      round-off error possibly significant
	 *	3      invalid parameters
	 *	4      unable to locate integration parameters
	 *	5      out of memory
	 */
	int *fault
)
{
	int j;
	int count = 0;
	int nterm_limit = lim;
	double xnt;
	double utx;
	double delta_u;
	double x;
	double Amax;
	double Amin;
	double mean;	/* mean of the distribution */
	double stddev;	/* standard deviation of the distribution */
	double integral;
	double absolute_sum;
	int *index;
	double result = nan("");
	struct gstlal_cdf_weighted_chisq_P_trace trace_local = GSTLAL_CDF_WEIGHTED_CHISQ_P_TRACE_INITIALIZER;
	int fault_local = 0;
	jmp_buf env;

	/*
	 * construct look-up table for sorted coefficients
	 */

	index = order(A, N);
	if(!index) {
		fault_local = 5;
		goto done;
	}

	/*
	 * check input
	 */

	if(setjmp(env)) {
		fault_local = 4;
		goto done;
	}

	if(N < 0) {
		fault_local = 3;
		goto done;
	}
	for(j = 0; j < N; j++)
		if(dof[j] < 0 || noncent[j] < 0) {
			fault_local = 3;
			goto done;
		}

	/*
	 * find mean & std. dev. of distribution, and max and min of A.
	 * iterate over A in increasing order of coefficient magnitude to
	 * help with numerical accuracy
	 */

	stddev = var;
	Amin = Amax = (N == 0) ? 0 : A[0];
	mean = 0;
	for(j = 0; j < N; j++) {
		stddev += pow(A[index[j]], 2) * (2 * dof[index[j]] + 4 * noncent[index[j]]);
		mean += A[index[j]] * (dof[index[j]] + noncent[index[j]]);
		if(A[j] > Amax)
			Amax = A[j];
		else if(A[j] < Amin)
			Amin = A[j];
	}
	stddev = sqrt(stddev);

	/*
	 * check that parameter values are valid
	 */

	if(Amin == 0 && Amax == 0 && var == 0) {
		fault_local = 3;
		goto done;
	}

	/*
	 * special case:  Dirac \delta
	 */

	if(stddev == 0) {
		result = (c > 0) ? 1 : 0;
		goto done;
	}

	/*
	 * truncation point with no convergence factor.  use 16/stddev as
	 * initial guess for u
	 */

	utx = findu(16 / stddev, var, accuracy / 2, env, &count, lim, A, noncent, dof, N);

	/*
	 * does convergence factor help?
	 */

	if(c != 0 && N > 0 && fabs(A[index[N-1]]) > 0.07 * stddev) {
		double tausq = accuracy / 4 / cfe(c, env, &count, lim, A, noncent, dof, N, index);
		if(!isnan(tausq) && truncation(utx, var + tausq, env, &count, lim, A, noncent, dof, N) < accuracy / 5) {
			var += tausq;
			utx = findu(utx, var, accuracy / 4, env, &count, lim, A, noncent, dof, N);
			trace_local.init_convergence_factor_sd = sqrt(tausq);
		}
	}
	trace_local.truncation_point = utx;
	accuracy /= 2;

	/*
	 * find RANGE of distribution, quit if outside this
	 */

	integral = absolute_sum = 0;
	while(TRUE) {
		double xntm;

		/*
		 * find integration interval
		 */

		double d1 = ctff(accuracy, +4.5 / stddev, env, &count, lim, var, A, noncent, dof, N, Amin, Amax, mean) - c;
		double d2 = c - ctff(accuracy, -4.5 / stddev, env, &count, lim, var, A, noncent, dof, N, Amin, Amax, mean);

		if(d1 < 0) {
			result = 1;
			goto done;
		}
		if(d2 < 0) {
			result = 0;
			goto done;
		}

		delta_u = 2 * PI / ((d1 > d2) ? d1 : d2);

		/*
		 * calculate number of terms required for main and
		 * auxillary integrations
		 */

		xnt = utx / delta_u;
		xntm = 3 / sqrt(accuracy);
		if(xnt <= xntm * 1.5)
			break;
		if(nterm_limit <= 0 && lim >= 0) {
			fault_local = 1;
			goto done;
		}

		/*
		 * parameters for auxillary integration
		 */

		{
		int nterms = round(xntm);
		double delta_u = utx / nterms;
		double tausq;

		x = 2 * PI / delta_u;
		if(x <= fabs(c))
			break;

		/*
		 * calculate convergence factor
		 */

		tausq = .33 * accuracy / (1.1 * (cfe(c - x, env, &count, lim, A, noncent, dof, N, index) + cfe(c + x, env, &count, lim, A, noncent, dof, N, index)));
		if(isnan(tausq))
			break;
		accuracy *= .67;

		/*
		 * auxillary integration
		 */

		integral += integrate(nterms, delta_u, var, tausq, FALSE, A, noncent, dof, c, N, &absolute_sum, index);
		nterm_limit -= nterms;
		var += tausq;
		trace_local.number_of_integrations++;
		trace_local.number_of_terms += nterms + 1;
		}

		/*
		 * find truncation point with new convergence factor
		 */

		utx = findu(utx, var, accuracy / 4, env, &count, lim, A, noncent, dof, N);
		accuracy *= 0.75;
	}

	/*
	 * main integration
	 */

	trace_local.integration_interval = delta_u;

	{
	int nterms = round(xnt);

	if(nterm_limit < nterms && lim >= 0) {
		fault_local = 1;
		goto done;
	}
	integral += integrate(nterms, delta_u, var, 0, TRUE, A, noncent, dof, c, N, &absolute_sum, index);
	trace_local.number_of_integrations++;
	trace_local.number_of_terms += nterms + 1;
	}

	trace_local.absolute_sum = absolute_sum;
	result = 0.5 - integral;

	/*
	 * test whether round-off error could be significant allow for
	 * radix 8 or 16 machines
	 */

	x = absolute_sum + accuracy / 10;
	for(j = 8; j; j /= 2)
		if(j * x == j * absolute_sum) {
			fault_local = 2;
			break;
		}

done:
	trace_local.cycles = count;
	if(trace)
		*trace = trace_local;
	if(fault)
		*fault = fault_local;
	free(index);
	return result;
}

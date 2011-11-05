/*
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


#ifndef __GSTLAL_CDF_WEIGHTED_CHISQ_P_H__
#define __GSTLAL_CDF_WEIGHTED_CHISQ_P_H__


struct gstlal_cdf_weighted_chisq_P_trace {
	/* absolute sum */
	double absolute_sum;
	/* total number of integration terms */
	int number_of_terms;
	/* number of integrations */
	int number_of_integrations;
	/* integration interval in final integration */
	double integration_interval;
	/* truncation point in initial integration */
	double truncation_point;
	/* s.d. of initial convergence factor */
	double init_convergence_factor_sd;
	/* cycles to locate integration parameters */
	int cycles;
};


#define GSTLAL_CDF_WEIGHTED_CHISQ_P_TRACE_INITIALIZER ((struct gstlal_cdf_weighted_chisq_P_trace) {0.0, 0, 0, 0.0, 0.0, 0.0, 0})


double gstlal_cdf_weighted_chisq_P(const double *, const double *, const int *, int, double, double, int, double, struct gstlal_cdf_weighted_chisq_P_trace *, int *);


#endif /* __GSTLAL_CDF_WEIGHTED_CHISQ_P_H__ */

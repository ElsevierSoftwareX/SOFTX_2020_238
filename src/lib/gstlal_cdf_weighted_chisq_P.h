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

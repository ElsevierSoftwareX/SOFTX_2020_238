\page gstlalinspiralprofileXeon_E5-2640v3_SL7_icc_page Profiling of gstlal_inspiral on Xeon E5-2640v3 with SL7 using icc

\section Template per core throughput

	real	18m23.089s
	user	540m25.100s
	sys	42m4.062s


\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 25600 \f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 1103 \f$
 - \f$N_c = 16 \f$


This gives a template per core throughput of 7,250

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 2.601e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	5242049  28.0392  libgstaudioresample.so   resampler_basic_direct_single
	2434765  13.0233  libpython2.7.so.1.0      /usr/lib64/libpython2.7.so.1.0
	2170160  11.6080  no-vmlinux               /no-vmlinux
	1302951   6.9694  libmkl_avx2.so           mkl_blas_avx2_sgemm_mscale
	1177956   6.3008  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1040987   5.5681  libmkl_avx2.so           mkl_blas_avx2_xsgemv
	840025    4.4932  libgstaudioresample.so   resample_float_resampler_process_float
	710738    3.8017  libintlc.so.5            __intel_ssse3_rep_memcpy
	365344    1.9542  libfftw3f.so.3.3.2       /usr/lib64/libfftw3f.so.3.3.2
	335005    1.7919  libmkl_avx2.so           anonymous symbol from section .text
	275648    1.4744  libgstaudioresample.so   resampler_basic_direct_double
	265048    1.4177  libgobject-2.0.so.0.3600.3 /usr/lib64/libgobject-2.0.so.0.3600.3
	223539    1.1957  libgstaudioresample.so   resampler_basic_direct_double
	199289    1.0660  libgsl.so.0.16.0         gsl_sf_sinc_e
	177757    0.9508  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	144983    0.7755  libm-2.17.so             __sin_avx
	138904    0.7430  libmkl_avx2.so           anonymous symbol from section .text
	117843    0.6303  libglib-2.0.so.0.3600.3  /usr/lib64/libglib-2.0.so.0.3600.3
	67397     0.3605  libc-2.17.so             msort_with_tmp.part.0
	66266     0.3545  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	46947     0.2511  multiarray.so            /usr/lib64/python2.7/site-packages/numpy/core/multiarray.so
	46427     0.2483  libintlc.so.5            __intel_new_memset
	43418     0.2322  libm-2.17.so             __ieee754_log_avx
	42826     0.2291  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)

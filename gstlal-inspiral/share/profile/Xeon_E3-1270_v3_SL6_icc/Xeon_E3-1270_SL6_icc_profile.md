\page gstlalinspiralprofileXeon_E3-1270_SL6_icc_page Profiling of gstlal_inspiral on Xeon E5-1270 with SL6 using icc

NOTE HYPERTHREADING WAS ENABLED BUT THE PHYSICAL CORES WERE USED FOR THE CALCULATION

\section Template per core throughput

	real    14m6.680s
	user    106m43.951s
	sys     5m41.636s


\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 6400\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 846s\f$
 - \f$N_c = 4 \f$


This gives a template per core throughput of 9450

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 3491.9 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	7671647  30.2273  libgstaudioresample.so   resampler_basic_direct_single
	3514057  13.8458  libpython2.6.so.1.0      /usr/lib64/libpython2.6.so.1.0
	1493646   5.8852  no-vmlinux               /no-vmlinux
	1341091   5.2841  orcexec.oWZitd (deleted) /tmp/orcexec.oWZitd (deleted)
	1299549   5.1204  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1193354   4.7020  libmkl_avx2.so           mkl_blas_avx2_xsgemv
	1153135   4.5435  libgstaudioresample.so   resample_float_resampler_process_float
	1073590   4.2301  libintlc.so.5            __intel_ssse3_rep_memcpy
	937053    3.6921  libmkl_avx2.so           mkl_blas_avx2_sgemm_mscale
	890793    3.5098  libfftw3f.so.3.2.3       /usr/lib64/libfftw3f.so.3.2.3
	671214    2.6447  libmkl_avx2.so           anonymous symbol from section .text
	434964    1.7138  libgstaudioresample.so   resampler_basic_direct_double
	321321    1.2660  libgstaudioresample.so   resampler_basic_direct_double
	273192    1.0764  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	232797    0.9173  libglib-2.0.so.0.2800.8  /lib64/libglib-2.0.so.0.2800.8
	225229    0.8874  libgsl.so.0.16.0         gsl_sf_sinc_e
	200489    0.7900  libgobject-2.0.so.0.2800.8 /lib64/libgobject-2.0.so.0.2800.8
	195498    0.7703  libm-2.12.so             sin
	195209    0.7691  libmkl_avx2.so           anonymous symbol from section .text
	122775    0.4837  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	110000    0.4334  libc-2.12.so             msort_with_tmp
	75646     0.2981  libc-2.12.so             memcpy
	60215     0.2373  libintlc.so.5            __intel_new_memset
	59432     0.2342  libgstlal.so             filter
	55290     0.2178  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)

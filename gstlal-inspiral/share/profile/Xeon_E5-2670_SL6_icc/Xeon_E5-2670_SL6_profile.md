\page gstlalinspiralprofileXeon_E5-2670_SL6_icc_page Profiling of gstlal_inspiral on Xeon E5-2670 with SL6 using icc

NOTE HYPERTHREADING WAS ENABLED BUT THE PHYSICAL CORES WERE USED FOR THE CALCULATION

\section Template per core throughput

	real 22:12.13
	user 38735.71
	sys 2776.41

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 25600\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 1332s\f$
 - \f$N_c = 16 \f$


This gives a template per core throughput of 6006

\section Profile results

	CPU: Intel Sandy Bridge microarchitecture, speed 2600.03 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	9170406  28.8839  libgstaudioresample.so   resampler_basic_direct_single
	5045721  15.8924  libpython2.6.so.1.0      /usr/lib64/libpython2.6.so.1.0
	2647054   8.3374  no-vmlinux               /no-vmlinux
	2012778   6.3396  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1719451   5.4157  libmkl_avx.so            mkl_blas_avx_xsgemv
	1384209   4.3598  libmkl_avx.so            mkl_blas_avx_sgemm_mscale
	1149518   3.6206  libgstaudioresample.so   resample_float_resampler_process_float
	1038193   3.2700  libfftw3f.so.3.2.3       /usr/lib64/libfftw3f.so.3.2.3
	944067    2.9735  orcexec.8xMUJL (deleted) /usr1/channa/orcexec.8xMUJL (deleted)
	809187    2.5487  libintlc.so.5            __intel_ssse3_rep_memcpy
	646881    2.0375  libmkl_avx.so            anonymous symbol from section .text
	578408    1.8218  libgstaudioresample.so   resampler_basic_direct_double
	395629    1.2461  libgstaudioresample.so   resampler_basic_direct_double
	371571    1.1703  libmkl_avx.so            anonymous symbol from section .text
	288164    0.9076  libgsl.so.0.16.0         gsl_sf_sinc_e
	273165    0.8604  libglib-2.0.so.0.2800.8  /lib64/libglib-2.0.so.0.2800.8
	271035    0.8537  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	248560    0.7829  libgobject-2.0.so.0.2800.8 /lib64/libgobject-2.0.so.0.2800.8
	212862    0.6704  libm-2.12.so             sin
	116937    0.3683  libc-2.12.so             msort_with_tmp
	112638    0.3548  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	103436    0.3258  libc-2.12.so             memcpy
	93767     0.2953  libc-2.12.so             vfprintf
	93163     0.2934  multiarray.so            /usr/lib64/python2.6/site-packages/numpy/core/multiarray.so
	72267     0.2276  libmkl_avx.so            anonymous symbol from section .text
	70177     0.2210  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)


\page gstlalinspiralprofileXeon_E5-1660v3_SL7_icc_page Profiling of gstlal_inspiral on Xeon E5-2640v3 with SL7 using icc

\section Template per core throughput

	real    13m4.790s
	user    193m3.408s
	sys     11m35.404s


\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 12800 \f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 784 \f$
 - \f$N_c = 8 \f$


This gives a template per core throughput of 10,200

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 3.001e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	6203537  32.9191  libgstaudioresample.so   resampler_basic_direct_single
	2716103  14.4130  libpython2.7.so.1.0      /usr/lib64/libpython2.7.so.1.0
	1522030   8.0767  no-vmlinux               /no-vmlinux
	1195225   6.3425  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1096438   5.8183  libmkl_avx2.so           mkl_blas_avx2_xsgemv
	909846    4.8281  libgstaudioresample.so   resample_float_resampler_process_float
	621760    3.2994  libintlc.so.5            __intel_ssse3_rep_memcpy
	619638    3.2881  libmkl_avx2.so           mkl_blas_avx2_sgemm_mscale
	406578    2.1575  libfftw3f.so.3.3.2       /usr/lib64/libfftw3f.so.3.3.2
	318704    1.6912  libmkl_avx2.so           anonymous symbol from section .text
	314497    1.6689  libgstaudioresample.so   resampler_basic_direct_double
	265224    1.4074  libgobject-2.0.so.0.3600.3 /usr/lib64/libgobject-2.0.so.0.3600.3
	248555    1.3190  libgstaudioresample.so   resampler_basic_direct_double
	211855    1.1242  libgsl.so.0.16.0         gsl_sf_sinc_e
	172819    0.9171  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	166632    0.8842  libmkl_avx2.so           anonymous symbol from section .text
	160510    0.8517  libm-2.17.so             __sin_avx
	104440    0.5542  libglib-2.0.so.0.3600.3  /usr/lib64/libglib-2.0.so.0.3600.3
	78964     0.4190  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	72901     0.3868  libc-2.17.so             msort_with_tmp.part.0
	50279     0.2668  multiarray.so            /usr/lib64/python2.7/site-packages/numpy/core/multiarray.so
	45296     0.2404  libm-2.17.so             __ieee754_log_avx
	41782     0.2217  libc-2.17.so             _int_malloc
	41751     0.2216  libintlc.so.5            __intel_new_memset
	39575     0.2100  libmkl_avx2.so           anonymous symbol from section .text
	39063     0.2073  libmkl_avx2.so           anonymous symbol from section .text
	37718     0.2002  libc-2.17.so             __memcpy_ssse3_back
	36684     0.1947  libgstlal.so             filter
	36428     0.1933  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)


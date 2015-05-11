\page gstlalinspiralprofileATOM-C2758_SL7_icc_page Profiling of gstlal_inspiral on ATOM C2758 with SL7 using icc

\section Template per core throughput

	real	32m14.606s
	user	241m11.619s
	sys	12m16.181s

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 6400\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 1934 \f$
 - \f$N_c = 8 \f$


This gives a template per core throughput of 2070

\section Profile results

	CPU: Intel Architectural Perfmon, speed 2.4e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	13636511 30.0088  libgstaudioresample.so   resampler_basic_direct_single
	9400893  20.6878  libpython2.7.so.1.0      /usr/lib64/libpython2.7.so.1.0
	3598680   7.9193  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	3280882   7.2200  no-vmlinux               /no-vmlinux
	2630299   5.7883  libmkl_mc3.so            mkl_blas_mc3_xsgemv
	1473931   3.2436  libgstaudioresample.so   resample_float_resampler_process_float
	1404113   3.0899  libmkl_mc3.so            LN8_M4_Kgas_1
	1136214   2.5004  libintlc.so.5            __intel_ssse3_rep_memcpy
	905304    1.9922  libmkl_mc3.so            mkl_blas_mc3_sgemm_mscale
	832465    1.8319  libgstaudioresample.so   resampler_basic_direct_double
	719882    1.5842  libfftw3f.so.3.3.2       /usr/lib64/libfftw3f.so.3.3.2
	488786    1.0756  libgstaudioresample.so   resampler_basic_direct_double
	451695    0.9940  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	450879    0.9922  libmkl_mc3.so            LN8_M4_WRAPUPgas_1
	449108    0.9883  libgsl.so.0.16.0         gsl_sf_sinc_e
	371581    0.8177  libgobject-2.0.so.0.3600.3 /usr/lib64/libgobject-2.0.so.0.3600.3
	291698    0.6419  libm-2.17.so             __sin_sse2
	245295    0.5398  multiarray.so            /usr/lib64/python2.7/site-packages/numpy/core/multiarray.so
	183280    0.4033  libglib-2.0.so.0.3600.3  /usr/lib64/libglib-2.0.so.0.3600.3
	147457    0.3245  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	143452    0.3157  libmkl_mc3.so            LN8_M4_LOOPgas_1
	124934    0.2749  libc-2.17.so             msort_with_tmp.part.0
	118373    0.2605  libc-2.17.so             __memcpy_ssse3
	114984    0.2530  libc-2.17.so             _int_malloc
	114156    0.2512  libgstlal.so             filter
	110904    0.2441  libc-2.17.so             vfprintf
	93961     0.2068  libc-2.17.so             __strlen_sse42
	77656     0.1709  libc-2.17.so             __memcmp_sse4_1
	76354     0.1680  libc-2.17.so             _int_free
	75991     0.1672  liblal.so.8.0.0          lanczos_cost
	75129     0.1653  libm-2.17.so             __ieee754_log_sse2
	70330     0.1548  libc-2.17.so             __memset_sse2
	66646     0.1467  libgsl.so.0.16.0         gsl_matrix_float_set_col
	66496     0.1463  libgstaudioresample.so   resample_double_resampler_process_float
	62998     0.1386  libfftw3.so.3.3.2        /usr/lib64/libfftw3.so.3.3.2
	61871     0.1362  libintlc.so.5            __intel_new_memset
	57494     0.1265  libc-2.17.so             malloc
	49491     0.1089  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)


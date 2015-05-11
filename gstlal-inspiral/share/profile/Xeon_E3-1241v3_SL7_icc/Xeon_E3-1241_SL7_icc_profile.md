\page gstlalinspiralprofileXeon_E3-1241_SL7_icc_profile_page Profiling of gstlal_inspiral on Xeon E3-1241 on SL7 with icc

\section Template per core throughput

real    13m52.817s
user    102m37.976s
sys     5m41.440s

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 6400\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 832s\f$
 - \f$N_c = 4 \f$

This gives a template per core throughput of 9615

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 3.501e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	5286501  33.2081  libgstaudioresample.so   resampler_basic_direct_single
	2150454  13.5084  libpython2.7.so.1.0      /usr/lib64/libpython2.7.so.1.0
	1022892   6.4255  no-vmlinux               /no-vmlinux
	890681    5.5950  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	830927    5.2196  libmkl_avx2.so           mkl_blas_avx2_xsgemv
	764252    4.8008  libgstaudioresample.so   resample_float_resampler_process_float
	763307    4.7948  libintlc.so.5            __intel_ssse3_rep_memcpy
	663895    4.1704  libmkl_avx2.so           mkl_blas_avx2_sgemm_mscale
	448979    2.8203  libmkl_avx2.so           anonymous symbol from section .text
	316660    1.9892  libfftw3f.so.3.3.2       /usr/lib64/libfftw3f.so.3.3.2
	286939    1.8025  libgstaudioresample.so   resampler_basic_direct_double
	213897    1.3436  libgobject-2.0.so.0.3600.3 /usr/lib64/libgobject-2.0.so.0.3600.3
	213883    1.3435  libgstaudioresample.so   resampler_basic_direct_double
	181923    1.1428  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	153194    0.9623  libgsl.so.0.16.0         gsl_sf_sinc_e
	141381    0.8881  libmkl_avx2.so           anonymous symbol from section .text
	123473    0.7756  libm-2.17.so             __sin_avx
	97889     0.6149  libglib-2.0.so.0.3600.3  /usr/lib64/libglib-2.0.so.0.3600.3
	80256     0.5041  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	56793     0.3568  libc-2.17.so             msort_with_tmp.part.0
	41547     0.2610  libintlc.so.5            __intel_new_memset
	36491     0.2292  libgstlal.so             filter
	36070     0.2266  libc-2.17.so             _int_malloc
	33689     0.2116  libmkl_avx2.so           anonymous symbol from section .text
	33357     0.2095  libmkl_avx2.so           anonymous symbol from section .text
	32727     0.2056  libm-2.17.so             __ieee754_log_avx
	32334     0.2031  libmkl_avx2.so           anonymous symbol from section .text
	30583     0.1921  libc-2.17.so             __memcpy_ssse3_back
	29850     0.1875  libgstaudioresample.so   resample_double_resampler_process_float
	29611     0.1860  multiarray.so            /usr/lib64/python2.7/site-packages/numpy/core/multiarray.so
	29554     0.1856  liblal.so.8.0.0          XLALPSDRegressorAdd
	28346     0.1781  libmkl_avx2.so           mkl_blas_avx2_sgemm_scopy_right_opt
	28252     0.1775  libfftw3.so.3.3.2        /usr/lib64/libfftw3.so.3.3.2
	27534     0.1730  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)

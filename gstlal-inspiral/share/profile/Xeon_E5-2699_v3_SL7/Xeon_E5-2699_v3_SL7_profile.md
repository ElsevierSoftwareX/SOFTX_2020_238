\page gstlalinspiralprofileXeon_E5-2699_v3_SL7_page Profiling of gstlal_inspiral on Xeon E5-2699 v3 with SL7

\section Template per core throughput

real	17m46.379s
user	1116m53.900s
sys	115m5.609s

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 28800\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 1066s\f$
 - \f$N_c = 36 \f$


This gives a template per core throughput of 3750

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 2.3e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               app name                 symbol name
	4861892  22.2303  libgstaudioresample.so   python2.7                resampler_basic_direct_single
	2499629  11.4292  no-vmlinux               python2.7                /no-vmlinux
	2425239  11.0890  libpython2.7.so.1.0      python2.7                /usr/lib64/libpython2.7.so.1.0
	1354790   6.1946  libsatlas.so             python2.7                ATL_sgezero
	1250073   5.7158  libsatlas.so             python2.7                ATL_sdot_xp1yp1aXbX
	1213938   5.5506  libgstlal.so.0.0.0       python2.7                gstlal_float_complex_peak_over_window
	1188018   5.4320  libsatlas.so             python2.7                ATL_sJIK0x0x6TN6x6x0_a1_bX
	1055861   4.8278  libc-2.17.so             python2.7                __memcpy_ssse3_back
	779098    3.5623  libgstaudioresample.so   python2.7                resample_float_resampler_process_float
	486816    2.2259  libsatlas.so             python2.7                LOOPM
	464683    2.1247  libgstaudioresample.so   python2.7                resampler_basic_direct_double
	323786    1.4805  libfftw3f.so.3.3.2       python2.7                /usr/lib64/libfftw3f.so.3.3.2
	300532    1.3741  libsatlas.so             python2.7                MNLOOP
	294156    1.3450  libgstaudioresample.so   python2.7                resampler_basic_direct_double
	264564    1.2097  libgobject-2.0.so.0.3600.3 python2.7                /usr/lib64/libgobject-2.0.so.0.3600.3
	264382    1.2088  libgstlal.so.0.0.0       python2.7                gstlal_autocorrelation_chi2_float
	263587    1.2052  libgsl.so.0.16.0         python2.7                gsl_sf_sinc_e
	238993    1.0928  libgstlal.so             python2.7                __mulsc3
	160556    0.7341  libm-2.17.so             python2.7                __sin_avx
	147298    0.6735  libsatlas.so             python2.7                ATL_sJIK0x0x7TN7x7x0_a1_bX
	141814    0.6484  libgstlal.so             python2.7                filter
	137907    0.6306  libglib-2.0.so.0.3600.3  python2.7                /usr/lib64/libglib-2.0.so.0.3600.3
	117623    0.5378  libsatlas.so             python2.7                LOOPM
	72958     0.3336  libgstlal.so.0.0.0       python2.7                __muldc3
	66780     0.3053  libc-2.17.so             python2.7                msort_with_tmp.part.0
	61624     0.2818  multiarray.so            python2.7                /usr/lib64/python2.7/site-packages/numpy/core/multiarray.so
	59493     0.2720  libgstlal.so.0.0.0       python2.7                gstlal_float_complex_series_around_peak
	58579     0.2678  libc-2.17.so             python2.7                __memset_sse2
	50245     0.2297  libframecppcmn.so.4.0.2  python2.7                FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)


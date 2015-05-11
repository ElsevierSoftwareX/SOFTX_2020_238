\page gstlalinspiralprofileXeon_E3-1231_Deb_7_8_profile_page Profiling of gstlal_inspiral on Xeon E3-1231 Debian 7.8

\section Template per core throughput

real    14m8.017s
user    111m4.676s
sys     1m23.336s


\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 6400\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 848s\f$
 - \f$N_c = 4 \f$

This gives a template per core throughput of 9430

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 3392.19 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	6842990  28.3126  libgstaudioresample.so   resampler_basic_direct_single
	4676655  19.3495  libsatlas.so             ATL_sdot_xp1yp1aXbX
	2392788   9.9001  python2.7                /usr/bin/python2.7
	1522303   6.2985  libsatlas.so             ATL_sJIK0x0x0TN0x0x0_a1_bX
	1188239   4.9163  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	904170    3.7410  libgstaudioresample.so   resample_float_resampler_process_float
	739616    3.0601  libsatlas.so             ATL_sgezero
	729009    3.0162  libc-2.13.so             __memcpy_ssse3
	541084    2.2387  no-vmlinux               /no-vmlinux
	467388    1.9338  libgstaudioresample.so   resampler_basic_direct_double
	403639    1.6700  libsatlas.so             MNLOOP
	367107    1.5189  libfftw3f.so.3.3.2       /usr/lib/x86_64-linux-gnu/libfftw3f.so.3.3.2
	297067    1.2291  libgstaudioresample.so   resampler_basic_direct_double
	280099    1.1589  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	272192    1.1262  libgstlal.so             __mulsc3
	267530    1.1069  libgobject-2.0.so.0.3200.4 /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0.3200.4
	256813    1.0626  libgsl.so.0.16.0         gsl_sf_sinc_e
	171561    0.7098  libm-2.13.so             sin
	117570    0.4864  libgstlal.so             filter
	98292     0.4067  libc-2.13.so             msort_with_tmp
	97219     0.4022  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	97140     0.4019  libgstlal.so.0.0.0       __muldc3
	77699     0.3215  libglib-2.0.so.0.3200.4  /lib/x86_64-linux-gnu/libglib-2.0.so.0.3200.4
	59652     0.2468  libc-2.13.so             _int_malloc
	55924     0.2314  libc-2.13.so             __memset_sse2
	45636     0.1888  libc-2.13.so             __strcmp_sse42
	45296     0.1874  libm-2.13.so             __ieee754_log
	42560     0.1761  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)


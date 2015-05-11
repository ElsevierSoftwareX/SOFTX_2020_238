\page gstlalinspiralprofileXeon_E3-1271_Ubuntu-14_04_page Profiling of gstlal_inspiral on Xeon E3-1271 Ubuntu-14-04

NOTE HYPERTHREADING WAS ENABLED - BUT CORE COUNT IS PHYSICAL NOT HT CORE

\section Template per core throughput

	real	14m39.067s
	user	108m24.972s
	sys	4m25.770s

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 6400\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 880s\f$
 - \f$N_c = 4 \f$

This gives a template per core throughput of 9090

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 3591.53 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	6977399  25.5787  libgstaudioresample.so   resampler_basic_direct_single
	5486270  20.1123  libsatlas.so             ATL_sdot_xp1yp1aXbX
	2413226   8.8467  python2.7                /usr/bin/python2.7
	1695809   6.2167  libsatlas.so             ATL_sJIK0x0x6TN6x6x0_a1_bX
	1316330   4.8256  no-vmlinux               /no-vmlinux
	1272496   4.6649  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1018295   3.7330  libc-2.19.so             __memcpy_sse2_unaligned
	901533    3.3050  libgstaudioresample.so   resample_float_resampler_process_interleaved_float
	826133    3.0286  libsatlas.so             ATL_sgezero
	519883    1.9059  libgstaudioresample.so   resampler_basic_direct_double
	433554    1.5894  libsatlas.so             MNLOOP
	393087    1.4410  libfftw3f.so.3.3.2       /usr/lib/x86_64-linux-gnu/libfftw3f.so.3.3.2
	341608    1.2523  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	329496    1.2079  libgstaudioresample.so   resampler_basic_direct_double
	300864    1.1029  libgobject-2.0.so.0.4002.0 /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0.4002.0
	273952    1.0043  libgsl.so.0.16.0         gsl_sf_sinc_e
	273827    1.0038  libgstlal.so             __mulsc3
	216301    0.7929  libsatlas.so             ATL_sJIK0x0x7TN7x7x0_a1_bX
	180066    0.6601  libm-2.19.so             __sin_avx
	146374    0.5366  libgstlal.so             filter
	118326    0.4338  libglib-2.0.so.0.4002.0  /lib/x86_64-linux-gnu/libglib-2.0.so.0.4002.0
	113369    0.4156  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	103736    0.3803  libgstlal.so.0.0.0       __muldc3
	88605     0.3248  libc-2.19.so             msort_with_tmp.part.0
	75771     0.2778  libc-2.19.so             memset
	61772     0.2265  libc-2.19.so             __GI___strcmp_ssse3
	55476     0.2034  libsatlas.so             ATL_sJIK0x0x0NN0x0x0_aX_bX
	52776     0.1935  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)

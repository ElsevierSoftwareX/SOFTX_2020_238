\page gstlalinspiralprofileXeon_E3-1271_Ubuntu-14_04_page Profiling of gstlal_inspiral on Xeon E3-1271 Ubuntu-14-04

\section Overview

This page benchmarks the following:
	- gstlal-inspiral-0.3.2
	- gstlal-0.7.1

The software dependencies stack can be configured from the following makefile
	- share/profile/Xeon_E3-1271_Ubuntu-14.04/Makefile.ligosoftware_ubuntu14.04

You can find the evironment script to source here
	- share/profile/Xeon_E3-1271_Ubuntu-14.04/profilerc

The tarball with the complete software stack is here
	- share/profile/profile.tar.gz

## NOTE

You should make the throughput target have as many unique output files as you do cores on the box.

\section Template per core throughput

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where \f$ N_t 3200 \, T_d = 5000s \, T_w = 740s \, N_c = 4 \f$

This gives a template per core throughput of 5400

\section Profile results

		CPU: Intel Haswell microarchitecture, speed 3592.06 MHz (estimated)
		Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
		samples  %        image name               symbol name
		15531684 34.4201  libgstaudioresample.so   resampler_basic_direct_single
		5479207  12.1426  libsatlas.so             ATL_sJIK0x0x6TN6x6x0_a1_bX
		3127639   6.9312  python2.7                /usr/bin/python2.7
		3112012   6.8966  no-vmlinux               /no-vmlinux
		2796048   6.1964  libgstaudioresample.so   resample_float_resampler_process_float
		2655439   5.8848  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
		2270538   5.0318  libc-2.19.so             __memcpy_sse2_unaligned
		1843783   4.0860  libsatlas.so             ATL_sdot_xp1yp1aXbX
		1480874   3.2818  libsatlas.so             ATL_sgezero
		677865    1.5022  libgstaudioresample.so   resampler_basic_direct_double
		531186    1.1772  libfftw3f.so.3.3.2       /usr/lib/x86_64-linux-gnu/libfftw3f.so.3.3.2
		517002    1.1457  libgobject-2.0.so.0.4002.0 /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0.4002.0
		514616    1.1405  libgstaudioresample.so   resampler_basic_direct_double
		423129    0.9377  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
		415519    0.9208  libgsl.so.0.16.0         gsl_sf_sinc_e
		380025    0.8422  libgstlal.so             __mulsc3
		261234    0.5789  libm-2.19.so             __sin_avx
		257092    0.5697  libsatlas.so             ATL_sJIK0x0x10TN10x10x0_a1_bX
		206760    0.4582  libgstlal.so             filter
		187944    0.4165  libc-2.19.so             msort_with_tmp.part.0
		142118    0.3150  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
		122196    0.2708  libglib-2.0.so.0.4002.0  /lib/x86_64-linux-gnu/libglib-2.0.so.0.4002.0
		106639    0.2363  libgstlal.so.0.0.0       __muldc3
		104447    0.2315  liblal.so.8.0.0          XLALPSDRegressorAdd
		97513     0.2161  libfftw3.so.3.3.2        /usr/lib/x86_64-linux-gnu/libfftw3.so.3.3.2
		92897     0.2059  libc-2.19.so             memset
		91138     0.2020  libm-2.19.so             __ieee754_log_avx
		81718     0.1811  libc-2.19.so             __GI___strcmp_ssse3
		57757     0.1280  libgstreamer-0.10.so.0.30.0 gst_util_uint64_scale_int_round
		56014     0.1241  libsatlas.so             ATL_scol2blk_a1
		54394     0.1205  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)
		50427     0.1118  liblal.so.8.0.0          lanczos_cost
		50019     0.1108  libc-2.19.so             _int_malloc
		45062     0.0999  libc-2.19.so             __memcpy_sse2


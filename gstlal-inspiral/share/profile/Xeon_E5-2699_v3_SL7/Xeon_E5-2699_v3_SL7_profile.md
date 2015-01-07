\page gstlalinspiralprofileXeon_E5-2699_v3_SL7_page Profiling of gstlal_inspiral on Xeon E5-2699 v3 with SL7

\section Overview

This page benchmarks the following:
	- gstlal-inspiral-0.3.2
	- gstlal-0.7.1

The software dependencies stack can be configured from the following makefile
	- share/profile/Xeon_E5-2699_v3_SL7/Makefile.ligosoftware

You can find the evironment script to source here
	- share/profile/Xeon_E5-2699_v3_SL7/profilerc

The tarball with the complete software stack is here
	- share/profile/profile.tar.gz

## NOTE

You should make the throughput target have as many unique output files as you do cores on the box. With this test we used 20, which saturated the box

\section Template per core throughput

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where \f$ N_t 16000 \, T_d = 5000s \, T_w = 1110s \, N_c = 32 \f$

This gives a template per core throughput of 2000

\section Profile results

		CPU: Intel Haswell microarchitecture, speed 3.6e+06 MHz (estimated)
		Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
		samples  %        image name               symbol name
		5358437  27.2559  libgstaudioresample.so   resampler_basic_direct_single
		2618492  13.3191  no-vmlinux               /no-vmlinux
		2201920  11.2002  libsatlas.so             ATL_sJIK0x0x6TN6x6x0_a1_bX
		1615109   8.2153  libpython2.7.so.1.0      /usr/lib64/libpython2.7.so.1.0
		1374697   6.9925  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
		1083775   5.5127  libgstaudioresample.so   resample_float_resampler_process_float
		941186    4.7874  libsatlas.so             ATL_sgezero
		878910    4.4706  libc-2.17.so             __memcpy_ssse3_back
		348623    1.7733  libgstaudioresample.so   resampler_basic_direct_double
		279610    1.4222  libfftw3f.so.3.3.2       /usr/lib64/libfftw3f.so.3.3.2
		272669    1.3869  libgstaudioresample.so   resampler_basic_direct_double
		268275    1.3646  libsatlas.so             _b::ATL_smvtk()
		251060    1.2770  libgstlal.so             __mulsc3
		248221    1.2626  libgobject-2.0.so.0.3600.3 /usr/lib64/libgobject-2.0.so.0.3600.3
		229735    1.1686  libgsl.so.0.16.0         gsl_sf_sinc_e
		141558    0.7200  libgstlal.so             filter
		133169    0.6774  libsatlas.so             ATL_sJIK0x0x10TN10x10x0_a1_bX
		133053    0.6768  libm-2.17.so             __sin_avx
		119700    0.6089  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
		77554     0.3945  libc-2.17.so             msort_with_tmp.part.0
		70425     0.3582  libglib-2.0.so.0.3600.3  /usr/lib64/libglib-2.0.so.0.3600.3
		44893     0.2284  libgstreamer-0.10.so.0.30.0 gst_util_uint64_scale_int_round
		43147     0.2195  libm-2.17.so             __ieee754_log_avx
		41871     0.2130  libgstlal.so.0.0.0       __muldc3
		41114     0.2091  libfftw3.so.3.3.2        /usr/lib64/libfftw3.so.3.3.2
		36102     0.1836  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
		28900     0.1470  libc-2.17.so             __memset_sse2
		28277     0.1438  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)
		28043     0.1426  multiarray.so            /usr/lib64/python2.7/site-packages/numpy/core/multiarray.so
		27288     0.1388  libc-2.17.so             _int_malloc
		27064     0.1377  liblal.so.8.0.0          lanczos_cost
		24750     0.1259  libc-2.17.so             malloc
		23065     0.1173  libgsl.so.0.16.0         gsl_matrix_float_set_col
		21554     0.1096  libc-2.17.so             _int_free
		19556     0.0995  libsatlas.so             ATL_scol2blk_a1


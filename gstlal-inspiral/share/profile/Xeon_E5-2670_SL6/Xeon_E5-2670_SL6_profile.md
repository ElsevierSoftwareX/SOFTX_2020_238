\page gstlalinspiralprofileXeon_E5-2670_SL6_page Profiling of gstlal_inspiral on Xeon E5-2670 with SL6

NOTE HYPERTHREADING WAS ENABLED BUT THE PHYSICAL CORES WERE USED FOR THE CALCULATION

\section Template per core throughput

	real	24m10.477s
	user	713m18.617s
	sys	42m0.273s

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 25600\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 1450s\f$
 - \f$N_c = 16 \f$


This gives a template per core throughput of 5517

\section Profile results

	CPU: Intel Sandy Bridge microarchitecture, speed 2600.03 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	8049420  22.2829  libgstaudioresample.so   resampler_basic_direct_single
	5190557  14.3688  libpython2.6.so.1.0      /usr/lib64/libpython2.6.so.1.0
	2917303   8.0758  libsatlas.so             ATL_sJIK0x0x0TN0x0x0_a1_bX
	2659314   7.3617  no-vmlinux               /no-vmlinux
	2192244   6.0687  libsatlas.so             ATL_sdot_xp1yp1aXbX
	2008971   5.5613  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1498171   4.1473  libc-2.12.so             memcpy
	1481497   4.1012  libsatlas.so             ATL_sgezero
	1162254   3.2174  libgstaudioresample.so   resample_float_resampler_process_float
	1028489   2.8471  libfftw3f.so.3.2.3       /usr/lib64/libfftw3f.so.3.2.3
	927193    2.5667  orcexec.e3gHiT (deleted) /usr1/channa/orcexec.e3gHiT (deleted)
	658775    1.8237  libsatlas.so             LOOPM
	565681    1.5659  libgstaudioresample.so   resampler_basic_direct_double
	503457    1.3937  libsatlas.so             MNLOOP
	447667    1.2393  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	419251    1.1606  libsatlas.so             LOOPM
	373590    1.0342  libgstaudioresample.so   resampler_basic_direct_double
	309622    0.8571  libgsl.so.0.16.0         gsl_sf_sinc_e
	308525    0.8541  libgstlal.so             __mulsc3
	269774    0.7468  libglib-2.0.so.0.2800.8  /lib64/libglib-2.0.so.0.2800.8
	245464    0.6795  libgobject-2.0.so.0.2800.8 /lib64/libgobject-2.0.so.0.2800.8
	221539    0.6133  libm-2.12.so             sin
	141174    0.3908  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	139212    0.3854  libgstlal.so             filter
	115045    0.3185  libc-2.12.so             msort_with_tmp
	108877    0.3014  libgstlal.so.0.0.0       __muldc3
	95563     0.2645  libc-2.12.so             vfprintf
	93156     0.2579  multiarray.so            /usr/lib64/python2.6/site-packages/numpy/core/multiarray.so
	82688     0.2289  libc-2.12.so             __GI_memset
	74944     0.2075  libc-2.12.so             _int_malloc
	71015     0.1966  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)
	   

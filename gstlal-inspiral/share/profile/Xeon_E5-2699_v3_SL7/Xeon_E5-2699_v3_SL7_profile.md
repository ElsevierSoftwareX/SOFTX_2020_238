\page gstlalinspiralprofileXeon_E5-2699_v3_SL7_page Profiling of gstlal_inspiral on Xeon E5-2699 v3 with SL7

NOTE HYPERTHREADING WAS DISABLED

\section Template per core throughput

real	18m3.855s
user	581m14.715s
sys	53m18.239s

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 25600\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 1080s\f$
 - \f$N_c = 36 \f$


This gives a template per core throughput of 3292

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 3.6e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	4244686  21.8100  libgstaudioresample.so   resampler_basic_direct_single
	2917686  14.9916  libsatlas.so             ATL_sdot_xp1yp1aXbX
	2183866  11.2211  libpython2.7.so.1.0      /usr/lib64/libpython2.7.so.1.0
	1867220   9.5941  no-vmlinux               /no-vmlinux
	1157875   5.9494  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1061519   5.4543  libsatlas.so             ATL_sJIK0x0x6TN6x6x0_a1_bX
	850428    4.3697  libsatlas.so             ATL_sgezero
	673701    3.4616  libgstaudioresample.so   resample_float_resampler_process_float
	556696    2.8604  libc-2.17.so             __memcpy_ssse3_back
	459010    2.3585  libgstaudioresample.so   resampler_basic_direct_double
	289121    1.4856  libgstaudioresample.so   resampler_basic_direct_double
	281215    1.4449  libsatlas.so             MNLOOP
	280961    1.4436  libfftw3f.so.3.3.2       /usr/lib64/libfftw3f.so.3.3.2
	262811    1.3504  libgsl.so.0.16.0         gsl_sf_sinc_e
	236152    1.2134  libgobject-2.0.so.0.3600.3 /usr/lib64/libgobject-2.0.so.0.3600.3
	222553    1.1435  libgstlal.so             __mulsc3
	157215    0.8078  libm-2.17.so             __sin_avx
	145103    0.7456  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	135857    0.6981  libsatlas.so             ATL_sJIK0x0x7TN7x7x0_a1_bX
	126348    0.6492  libgstlal.so             filter
	97579     0.5014  libglib-2.0.so.0.3600.3  /usr/lib64/libglib-2.0.so.0.3600.3
	65910     0.3387  libgstlal.so.0.0.0       __muldc3
	64958     0.3338  libc-2.17.so             msort_with_tmp.part.0
	51519     0.2647  multiarray.so            /usr/lib64/python2.7/site-packages/numpy/core/multiarray.so
	43756     0.2248  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	43405     0.2230  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)

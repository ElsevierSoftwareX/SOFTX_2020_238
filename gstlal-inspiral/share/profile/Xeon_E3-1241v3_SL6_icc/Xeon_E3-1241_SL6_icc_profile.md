\page gstlalinspiralprofileXeon_E3-1241_SL6_icc_profile_page Profiling of gstlal_inspiral on Xeon E3-1241 on SL6 with icc

\section nospeedstep Template per core throughput with speedstep disabled

real	14m8.741s
user	106m37.968s
sys	5m50.815s


\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 6400\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 848s\f$
 - \f$N_c = 4 \f$

This gives a template per core throughput of 9430 

\section speedstep Template per core throughput with speedstep enabled

real	14m18.363s
user	107m18.137s
sys	6m2.936s

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 6400\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 858s\f$
 - \f$N_c = 4 \f$

This gives a template per core throughput of 9320

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 3.501e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	7175913  30.3174  libgstaudioresample.so   resampler_basic_direct_single
	3240136  13.6892  libpython2.6.so.1.0      /usr/lib64/libpython2.6.so.1.0
	1458729   6.1630  no-vmlinux               /no-vmlinux
	1216184   5.1382  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1199115   5.0661  orcexec.oXRIJh (deleted) /usr1/channa/orcexec.oXRIJh (deleted)
	1092156   4.6142  libgstaudioresample.so   resample_float_resampler_process_float
	1091065   4.6096  libmkl_avx2.so           mkl_blas_avx2_xsgemv
	985162    4.1622  libintlc.so.5            __intel_ssse3_rep_memcpy
	831745    3.5140  libmkl_avx2.so           mkl_blas_avx2_sgemm_mscale
	794797    3.3579  libfftw3f.so.3.2.3       /usr/lib64/libfftw3f.so.3.2.3
	608315    2.5701  libmkl_avx2.so           anonymous symbol from section .text
	467527    1.9752  libgstaudioresample.so   resampler_basic_direct_double
	305694    1.2915  libgstaudioresample.so   resampler_basic_direct_double
	258486    1.0921  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	247233    1.0445  libgsl.so.0.16.0         gsl_sf_sinc_e
	212476    0.8977  libglib-2.0.so.0.2800.8  /lib64/libglib-2.0.so.0.2800.8
	186640    0.7885  libgobject-2.0.so.0.2800.8 /lib64/libgobject-2.0.so.0.2800.8
	180393    0.7621  libm-2.12.so             sin
	179306    0.7575  libmkl_avx2.so           anonymous symbol from section .text
	111054    0.4692  libgstlal.so.0.0.0       gstlal_float_complex_series_around_peak
	96278     0.4068  libc-2.12.so             msort_with_tmp
	70065     0.2960  libc-2.12.so             memcpy
	60925     0.2574  libgstlal.so             filter
	60308     0.2548  libc-2.12.so             vfprintf
	55109     0.2328  libintlc.so.5            __intel_new_memset
	53388     0.2256  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)


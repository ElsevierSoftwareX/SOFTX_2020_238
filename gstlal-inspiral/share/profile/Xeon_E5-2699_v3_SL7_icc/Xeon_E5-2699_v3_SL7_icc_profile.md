\page gstlalinspiralprofileXeon_E5-2699_v3_SL7_icc_page Profiling of gstlal_inspiral on Xeon E5-2699 v3 with SL7 with icc

\section Template per core throughput

real	21m37.542s
user	1133m8.688s
sys	181m1.017s

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where

 - \f$ N_t = 28800\f$
 - \f$T_d = 5000s\f$
 - \f$T_w = 1297s\f$
 - \f$N_c = 36 \f$


This gives a template per core throughput of 3084

\section Profile results

	CPU: Intel Haswell microarchitecture, speed 2.3e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	1590361  22.6175  libgstaudioresample.so   resampler_basic_direct_single
	1058955  15.0600  no-vmlinux               /no-vmlinux
	778014   11.0646  libpython2.7.so.1.0      /usr/lib64/libpython2.7.so.1.0
	715842   10.1804  libmkl_avx2.so           mkl_blas_avx2_sgemm_mscale
	422131    6.0034  libmkl_avx2.so           mkl_blas_avx2_xsgemv
	355095    5.0500  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	306262    4.3555  libintlc.so.5            __intel_ssse3_rep_memcpy
	243654    3.4651  libgstaudioresample.so   resample_float_resampler_process_float
	185801    2.6424  libfftw3f.so.3.3.2       /usr/lib64/libfftw3f.so.3.3.2
	147906    2.1035  libgstaudioresample.so   resampler_basic_direct_double
	110711    1.5745  libgobject-2.0.so.0.3600.3 /usr/lib64/libgobject-2.0.so.0.3600.3
	95148     1.3532  libgstaudioresample.so   resampler_basic_direct_double
	86793     1.2343  libmkl_avx2.so           anonymous symbol from section .text
	76571     1.0890  libgsl.so.0.16.0         gsl_sf_sinc_e
	74562     1.0604  libglib-2.0.so.0.3600.3  /usr/lib64/libglib-2.0.so.0.3600.3
	49069     0.6978  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	41210     0.5861  libm-2.17.so             __sin_avx
	39076     0.5557  libmkl_avx2.so           anonymous symbol from section .text
	29821     0.4241  libgstlal.so             filter
	22776     0.3239  libpthread-2.17.so       pthread_mutex_lock
	22316     0.3174  libgstreamer-0.10.so.0.30.0 gst_util_uint64_scale_int_round
	20787     0.2956  libc-2.17.so             __memcpy_ssse3_back
	20631     0.2934  libc-2.17.so             _int_malloc
	20023     0.2848  libc-2.17.so             msort_with_tmp.part.0
	16382     0.2330  libc-2.17.so             malloc
	16228     0.2308  libframecppcmn.so.4.0.2  FrameCPP::Common::CheckSumCRC::calc(void const*, unsigned int)


\page gstlalo1profiling Profiling of O1 with Modulefiles

\section overview Overview

This page is documentation for the current profiling efforts with modulefiles in the O1 low-latency analysis using optimized software.

\section usingsoftware Using the Software

The optimized software is built with the intention to be available for all users without the user needing to build the software themselves

\subsection atcit At CIT

The build at CIT was made with [this makefile](https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/profile/Makefile.ligosoftware_icc) on ldas-pcdev12.

Currently, this build is only supported on cluster nodes and headnodes with the Haswell architecture.

-# To use the software you need to point modules where it can find user-created modulefiles
 - $ module use /home/gstlalcbc/modules/modulefiles
-# Then you set the environment variables by loading the modulefile
 - $ module load gstlal-opt-haswell

\subsection atuwm At UWM

The build at UWM was made with [this makefile](https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/profile/Makefile.ligosoftware_gcc) at UWM on pcdev3.

This build currently supports cluster nodes and headnodes with either the Westmere and Sandy Bridge architecture.

-# To use the software you need to point modules where it can find user-created-modulefiles
 - $ module use /home/gstlalcbc/modules/modulefiles
-# Then you set the environment variables by loading the appropriate modulefile
 - $ module load gstlal-opt-westmere
 - $ module load gstlal-opt-sandybridge

\section procedure Procedure for Profiling

-# Load an environment with the steps outlined above
-# Profiling is done by unpacking [this tarball](https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/profile/O1_ll_profiling.tar.gz)
	-# $ make
		-# produces profile.txt which contains a list of functional calls by gstlal_inspiral ordered by computational cost
	-# $ time make -j throughput
		-# the real time, in seconds, is used for the throughput calculation

\section throughput Throughput Calculations

The throughput calculation is computed with the formula

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

where

 - \f$N_t\f$ is the number of templates
	- found by
		-# $ ligolw_print -t sngl_inspiral -c mass1 PATH_TO_SVD_BANK | wc
		-# Multiply the output by the number of concurrent jobs run
		-# Multiply by the number of subbanks each job processes
 - \f$T_d\f$ is the duration time
	- found by subtracting the GPS_START_TIME from the GPS_END_TIME in the Makefile from the profile tarball
 - \f$T_w\f$ is the wall time
	- $ time make -j throughput
	- convert to seconds
 - \f$N_c\f$ is the number of physical cores
	- $ vim \\proc\\cpuinfo

\subsubsection citthroughput At CIT

<!--
	real    99m44.055s
	user    626m31.034s
	sys     26m3.783s

- \f$N_t = 3110 * 4 * 2 = 24880\f$
- \f$T_d = 5000\f$
- \f$T_w = 99 * 60 + 44 = 5984\f$
- \f$N_c = 4\f$

-\f$T_{100\%} = 5197\f$
-->

\subsubsection uwmthroughput AT UWM
<!--
	real    9m12.648s
	user    94m20.670s
	sys     5m56.106s

- \f$N_t = 3110 * 2 * 2 = 12440\f$
- \f$T_d = 1000\f$
- \f$T_w = 9 * 60 + 12 = 552\f$
- \f$N_c = 16\f$

-\f$T_{100\%}\f$ = 1408
-->
\subsection costlycalls Top 15 Computationally Expensive Functional Calls

\subsubsection compcostcit At CIT
<!--
	CPU: Intel Haswell microarchitecture, speed 3.501e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	52390889 24.8012  libgstaudioresample.so   resampler_basic_direct_single
	25892897 12.2574  orcexec.lx7WBM (deleted) /usr1/ryan.everett/orcexec.lx7WBM (deleted)
	20105958  9.5179  libmkl_avx2.so           mkl_blas_avx2_sgemm_mscale
	17693511  8.3759  libpython2.6.so.1.0      /usr/lib64/libpython2.6.so.1.0
	13208748  6.2528  libgstaudioresample.so   resample_float_resampler_process_float
	12026827  5.6933  libintlc.so.5            __intel_ssse3_rep_memcpy
	11036189  5.2244  libmkl_avx2.so           anonymous symbol from section .text
	10049763  4.7574  no-vmlinux               /no-vmlinux
	5126960   2.4270  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	3936254   1.8634  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	3836809   1.8163  libglib-2.0.so.0.2800.8  /lib64/libglib-2.0.so.0.2800.8
	2076018   0.9828  libmkl_avx2.so           anonymous symbol from section .text
	1960202   0.9279  libgsl.so.0.16.0         gsl_sf_sinc_e
	1842011   0.8720  libgobject-2.0.so.0.2800.8 /lib64/libgobject-2.0.so.0.2800.8
	1802666   0.8534  libm-2.12.so             sin

-->
\subsubsection compcostuwm At UWM
<!--
	CPU: Intel Sandy Bridge microarchitecture, speed 2.001e+06 MHz (estimated)
	Counted CPU_CLK_UNHALTED events (Clock cycles when not halted) with a unit mask of 0x00 (No unit mask) count 100000
	samples  %        image name               symbol name
	21159409 25.0838  libsatlas.so             ATL_sJIK0x0x0TN0x0x0_a1_bX
	18653446 22.1131  libgstaudioresample.so   resampler_basic_direct_single
	8310294   9.8516  libsatlas.so             ATL_sgezero
	4726643   5.6033  libgstaudioresample.so   resample_float_resampler_process_float
	4051552   4.8030  orcexec.EcBml4 (deleted) /localscratch/ryan.everett/orcexec.EcBml4 (deleted)
	3838193   4.5501  no-vmlinux               /no-vmlinux
	2661546   3.1552  libc-2.13.so             __memcpy_ssse3
	1449773   1.7187  libgstlal.so.0.0.0       gstlal_float_complex_peak_over_window
	1373665   1.6284  libfftw3f.so.3.3.2       /usr/lib/x86_64-linux-gnu/libfftw3f.so.3.3.2
	1079079   1.2792  python2.7                PyEval_EvalFrameEx
	998281    1.1834  libgstlal.so             __mulsc3
	907842    1.0762  libgsl.so.0.16.0         gsl_sf_sinc_e
	883677    1.0476  libgstlal.so.0.0.0       gstlal_autocorrelation_chi2_float
	797285    0.9452  libgobject-2.0.so.0.3200.4 /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0.3200.4
	625273    0.7412  libm-2.13.so             sin
-->

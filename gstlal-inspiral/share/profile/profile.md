\page gstlalinspiralprofilepage Profiling of gstlal_inspiral

Basic instructions

	$ wget https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/profile/gcc_profile.tar.gz
	$ tar -zxvf gcc_profile.tar.gz	
	$ wget https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/profile/profile.tar.gz
	$ tar -zxvf profile.tar.gz
	$ source optimalrc
	$ make -f Makefile.ligosoftware

After potentially several hours the software dependencies will hopefully compile successfully.  Now you are ready to conduct the profiling and timing tests.  To profile do:

	$ cd profile
	$ make

To time do (after potentially adjusting the number of parallel jobs):

	$ time make -j throughput

To compute the template per core throughput

\f$ \mathcal{T}_{100 \%} = N_t * T_d / T_w / N_c \f$

Where \f$ N_t = \f$ the number of parallel templates (800 times the number of parallel jobs that run with -j)  \f$ T_d = 5000s \, T_w =\f$ the number "real" seconds \f$ N_c = \f$ the number of cores on the machine

-# \ref gstlalinspiralprofileXeon_E3-1270_SL6_icc_page
-# \ref gstlalinspiralprofileXeon_E3-1231_Deb_7_8_profile_page
-# \ref gstlalinspiralprofileXeon_E3-1271_Ubuntu-14_04_page
-# \ref gstlalinspiralprofileXeon_E5-2670_SL6_icc_page
-# \ref gstlalinspiralprofileXeon_E5-2670_SL6_page
-# \ref gstlalinspiralprofileXeon_E5-2699_v3_SL7_page
-# \ref gstlalinspiralprofileXeon_E5-2699_v3_SL7_icc_page
-# \ref gstlalinspiralprofileNVIDIA_K4000_page


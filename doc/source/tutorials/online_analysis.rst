Running an online compact binary coalescence analysis
========================================================================

Prerequisites
-------------

 - Fully functional gstlal, gstlal-ugly, gstlal-inspiral installation
 - Condor managed computing resource using the LIGO Data Grid configuration
   with dedicated nodes to support online jobs (which run indefinitely)
 - Network streaming gravitational wave data
 - Optional, but recommended, accounts for LVAlert and a robot certificate to
   authtenticate uploades to the GRavitational-wave Candidate Event DataBase
   (GraceDB).

Introduction
------------

This tutorial will will help you to setup a real-time gravitational wave search
for merging neutron stars and black holes.  

The online analysis has a somewhat involved setup procedure.  This
documentation covers all of it. The steps are:

 1. Generate template banks for the target search area
 2. Decompose the template waveforms using the SVD in chirpmass and chi bins
 3. Setup and run the actual online analysis.

You can expect the setup (steps 1 and 2) to take several days.  Furthermore,
the analsysis (step 3) requires at minimum several days of burn-in time until
it learns the noise statistics of the data before it should be allowed to
submit candidate gravitational waves.  Plan accordingly.

**Note, this tutorial will assume a specific directory structure and certain
configuration files.  These can and should be changed by the user.  This
tutorial should be considered to be a guide, not cut-and-paste instructions.**

Generate template banks for the target search area
--------------------------------------------------

This tutorial will describe the steps relative to the root directory on the CIT
cluster::

 /home/gstlalcbc/observing/3/online/sept_opa

While not necessary, it is best to organize the analysis into distinct
sub-directories.  We will do that for this tutorial::

 mkdir -p sept_opa/banks/bns sept_opa/banks/nsbh sept_opa/banks/bbh sept_opa/banks/imbh


Making the BNS bank
^^^^^^^^^^^^^^^^^^^


Go into the bns directory and get the example configuration file from gitlab::

 cd sept_opa/banks/bns
 wget https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/sbank_bns.ini

You will also need an **appropriate** PSD for the data you intend to analyze.
Here is an example file, but it is important you use an appropriate one::

 wget https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/H1L1V1-REFERENCE_PSD-1186624818-687900.xml.gz

**NOTE you will need to modify the content for your code installation and
desired parameter space - this is simply an example file.  You can see
lalapps_cbc_sbank --help for more information** 

Next generate the condor dag by running lalapps_cbc_sbank_pipe::

 lalapps_cbc_sbank_pipe --config-file sbank_bns.ini --user-tag GSTLAL_BNS

Submit it to condor::

 condor_submit_dag GSTLAL_BNS.dag 

You can monitor the progress by doing::

 tail -f GSTLAL_BNS.dag.dagman.out

You need to wait for the BNS bank to finish before moving on to the SVD
decomposition step for the BNS bank, however the other banks (NSBH, BBH, IMBH)
can be generated simultaneously.


Making the NSBH, BBH, and IMBH banks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


You can repeat the above procedure for generating the NSBH, BBH and IMBH banks.
You will need to change the sbank configuration file (.ini). Examples can be
found here:

 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/sbank_nsbh.ini
 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/sbank_bbh.ini
 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/sbank_imbh.ini

You can generate all of these banks in parallel.

Decompose the template waveforms using the SVD in chirpmass and chi bins
------------------------------------------------------------------------


In order to remain organized we will make new directories for the svd
decomposed template banks.  First go to the projects root directory, e.g.::

 cd /home/gstlalcbc/observing/3/online/

Then make new directories for the bank::

 mkdir -p sept_opa/svd/bns sept_opa/svd/nsbh sept_opa/svd/bbh sept_opa/svd/imbh


Decomposing the BNS bank
^^^^^^^^^^^^^^^^^^^^^^^^


Go into the bns svd sub directory::

 cd sept_opa/svd/bns

Get the config file example::

 wget https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/Makefile.bns_svd

**NOTE this file is provided as an example. You will in general have to suit it
to the spcifics of your environment and the search you plan to conduct**

Then run make to generate an SVD dag::

 make -f Makefile.bns_svd 

Submit it::

 condor_submit_dag bank.dag

You have to wait for this dag to finish before starting the actual analysis.  


Decomposing the NSBH, BBH and IMBH banks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


You can repeat the above procedure for the NSBH, BBH and IMBH banks.  You
should modify these example files to suit your needs, but here are example make
files.

 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/Makefile.nsbh_svd
 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/Makefile.bbh_svd
 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/Makefile.imbh_svd


Combining the SVD bank caches into a single cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to move to the next step, one must combine the cache files after all of the SVD jobs have finished::

 cd /home/gstlalcbc/observing/3/online/svd

Then combine the cache files with::

 cat bns/H1_bank.cache bbh/H1_bank.cache nsbh/H1_bank.cache imbh/H1_bank.cache > H1_bank.cache
 cat bns/L1_bank.cache bbh/L1_bank.cache nsbh/L1_bank.cache imbh/L1_bank.cache > L1_bank.cache
 cat bns/V1_bank.cache bbh/V1_bank.cache nsbh/V1_bank.cache imbh/V1_bank.cache > V1_bank.cache


Setup and run the actual online analysis
----------------------------------------

You need to make a directory for the analysis results, e.g.,::

 cd /home/gstlalcbc/observing/3/online/
 mkdir trigs
 cd trigs

Then get an example Makefile::

 wget https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/Makefile.online_analysis

Modify the example Makefile to your needs.  **NOTE when starting an analysis from scratch it is important to have the --gracedb-far-threshold = 1**

Run make::

 make -f Makefile.online_analysis

And submit the condor dag::

 condor_submit_dag trigger_pipe.dag


Basic LIGO/ALIGO colored Gaussian noise on the command line
-----------------------------------------------------------


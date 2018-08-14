Documentation for starting an online compact binary coalescence analysis
========================================================================

Prerequisites
-------------

 - Fully functional gstlal, gstlal-ugly, gstlal-inspiral installation
 - Condor managed computing resource using the LIGO Data Grid configuraiton with dedicated nodes to support online jobs (which run indefinitely)
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

You will also need an **appropriate** PSD for the data you intend to analyze.  Here is an example file, but it is important you use an appropriate one::

 wget https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/H1L1V1-REFERENCE_PSD-1186624818-687900.xml.gz

**NOTE you will need to modify the content for your code installation and desired parameter space - this is simply an example file.  You can see lalapps_cbc_sbank --help for more information** 

Next generate the condor dag by running lalapps_cbc_sbank_pipe::

 lalapps_cbc_sbank_pipe --config-file sbank_bns.ini --user-tag GSTLAL_BNS

Submit it to condor::

 condor_submit_dag GSTLAL_BNS.dag 

You can monitor the progress by doing::

 tail -f GSTLAL_BNS.dag.dagman.out


Making the NSBH, BBH, and IMBH banks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


You can repeat the above procedure for generating the NSBH, BBH and IMBH banks.  You will need to change the sbank configuration file (.ini). Examples can be found here:

 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/sbank_nsbh.ini
 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/sbank_bbh.ini
 - https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/share/O3/sept_opa/sbank_imbh.ini


Decompose the template waveforms using the SVD in chirpmass and chi bins
------------------------------------------------------------------------

Setup and run the actual online analysis
----------------------------------------

Basic LIGO/ALIGO colored Gaussian noise on the command line
-----------------------------------------------------------


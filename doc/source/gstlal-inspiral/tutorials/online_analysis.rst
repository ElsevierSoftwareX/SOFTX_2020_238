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
 cd sept_opa/banks/bns

 


Decompose the template waveforms using the SVD in chirpmass and chi bins
------------------------------------------------------------------------

Setup and run the actual online analysis
----------------------------------------

Basic LIGO/ALIGO colored Gaussian noise on the command line
-----------------------------------------------------------


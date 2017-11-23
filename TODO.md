General
=======

* Verify that all elements support discontinuities.
* Verify that all elements support gaps.


Plugins/elements to hand over to GStreamer
==========================================

* ``lal_gate`` element
* ``lal_reblock`` element
* ``lal_adder`` element: make the stock ``adder`` synchronous, or figure out how to use it
* ``lal_firbank`` element
* ``lal_matrixmixer`` element
* ``lal_nxydump`` element


Streaming with gstreamer
============================

* Figure out how to do live streaming through web (pcdev1 or ATLAS) of a OmegaGram  movie

## Immediate development tasks in no particular order

* revert reverts 2878f98fea97ad1148f1f847b7a2d3c827227dfb 10e1a0c7881eba52387706474ca659c46ee2efc1 8b2089258581050bf9e7c5b0b5b456b2da345bde 026505ac1dfe0ce6bd67347da47e69efa1c010ce and test and make them work.
* streamthinca/thinca.  ligolw_thinca requires sngl_inspiral objects with a .get_effective_snr() method.  we are now the only users of ligolw_thinca, and we don't use effective snr, so strip all that stuff out of ligolw_thinca, then move the remaining basic access methods to the C code in pylal so that ligolw_thinca can work with the bare pylal type, *then* remove the sngl_inspiral row object duplication loop from streamthinca so that we only ever work with the original triggers from the pipeline and never (possibly incomplete) copies of them.  this will fix the "mchirp=0" bug in the gracedb uploads.
* Audit all import statements:  there's a lot of useless imports, especially in gstlal-inspiral/bin/
* Remove the registry files and get url cron jobs and provide direct access to the running jobs from the outside
 * Note this involves firewall changes at CIT
 * Note we will need to make sure and cache results for at least ~5 minutes to prevent DOS
* Incorporate bank veto
* add to the gstlal summary page
  * population statement plot
  * plots showing likelihood distributions, signal and background pdfs, etc.
* inspiral DAGs:
  * gstlal_inspiral --svd-bank option:
    * is there a reason the instrument needs to be identified?  is the instrument not recorded in the file, if not why not, and would it suffice if it was?
* finish turning the appropriate bits of the Data class into a pipeline handler, and get the input and output segments recorded correctly
* complete the removal of deprecated data input and conditioning code from lloidparts.py
* block diagonal matrix support to reduce number of threads
* fix known gstreamer bugs:
  * audiofirfilter has "timestamp" bugs (?),
  * funnel won't work in time-ordered mode,
* framecpp demuxer and muxer cannot decode/encode FrHistory objects.  figure out how and fix.  (will be required for calibration work, not required for inspiral search)
* complete port to swig bindings (long term, divest ourselves of pylal).
* review readiness:  remove dead code, create tests for elements, create validation tests for applications.
* incorporate rate estimation code into pipeline (have student working on this)
* show Jolien how to add an element to the gstlaldebug plugin

## Summary page requests
* Fix error bars on count vs likelihood
* Make the pages less ugly and merge online/offline pages
* Add links to Makefile.
* Add links to segment list, segment xml, and injection xmls
* Report cluster and output directory
* Report CPU and clock time of run
* Report UTC, GPS, PT, CT times
* Report duty cycle
* Sec 1.4: left plot in hours, right plot of BNS horizon distance (Mpc)
* Representative PSD plot
* Example of what would be nice
* Inspiral range for different masses
* Reinstate template bank section w/ link to Makefile
* Sec 2: make dots bigger; link to inspinj file/Makefile
* Redmine Bug 2506
* Sec 6: 1d histogram of sngl raw SNR (CAT1,CAT2,CAT3?)
* Add color for FAR to missed/found plots
* split on even amount of chi[1]
* add plots with expected SNR and plot missed / found vs expected SNR
* add SNR threshold to sngls plotter
* Take SNR cutoff from process params table for SNR plots
* SNR time series
* Chisq time series
* Omega scans
* Spectrogram
* Omicron triggers
* Plots of triggers vs params (mchirp, eta)
* VT and expected VT based on noise curves (shared code?)
* Links to daily summary page
* Links to Alog
* Plot of segments and range near events
* List of DQ flags
* Documentation for each plot
* Injection channels plot
* Gating info, pipeline segment info (ie close to FFT boundary?)
* Audio files
* Horizon Distance plots on summary information tab
* Channel names on summary information tab
* Template bank plot on summary information tab
* Segment xml file on summary information tab
* Fix point sizes on injection parameters tab
* Add process params for injection files on injection parameters tab
* FIX RA/DEC and INC/cos(INC) units and tickmarks on injection parameters tab
* Remove M_sun from histogram plots on injection accuracy tab
* Make histograms on injection accuracy tab black and white
* Make histograms with rate.py (??) on injection accuracy tab (email Chad or Kipp for more info)
* Make plots both ways (with missed on top and vica versa) on missed found tab
* Investigate if the summary table is giving the wrong impression on missed found tab
* Indicate FAP as the color of the injection (make these separately from the plots that exist now) on missed found tab
* add sub-bank information to background plots
* Add pdf plot on background tab
* Ditch effective SNR plots on chi-squared tab
* Change spin intervals to chi on chi-squared tab
* Plot these (injections?) on top of background PDF (help to determine if the injections are on the expected area)
* Add total calendar time of search to summary page to supplement actual analyzed time
* goal is to have something like "Analyzed data from January 1 2008 to February 1 2008, total livetime was 90000 seconds"
* Add segment plot to summary page
* Add injection params to summary page
* Can get injection params from process_params table in injection files
* Add a dynamic plot section that allows user to select a time range over which to generate plot(s) specified by user
* Switch RA and DEC plots on injections tab to mollweide projection
* Insert collapsible text boxes into search subsection which describe the plots contained in that section Could possibly implement them as title string. Want to include equations in these descriptions as well (so that we can include e.g. the definition of symmetric mass ratio in symmetric mass ratio plots)
* Move effective distance plots above physical distance plots in missed found section
* Add fractional accuracy plots which use log axes
* Need to take the absolute value of the accuracy plot
* Consolidate injection summary tables in missed found section
* Improve bin selection for search sensitivity page plots
* Correct time shift label in money plots
* Compute correct error bars in IFAR plot (non-trivial)
* Make summary table in money plots section dynamic, allow user to select number of events to display
* Make background section dynamic, allowing users to select a region of parameter space to look at plots from. Include options to overlay injections and/or candidates on PDF plots
* Color code injection accuracy scatter plots by FAR
* Chisq vs time for triggers near event
* Originally intended for gracedb pages, but could be done for loudest events
* SNR vs time for triggers near event
* Originally intended for gracedb pages, but could be done for loudest events
* Upload XML file of singles near event
* Originally intended for gracedb pages, but could be done for loudest events
* Take SNR cutoff from process params table for SNR plots
* Strain time series plot
* Originally intended for gracedb pages, but could be done for loudest events
* Nearby missed injections
* Add padding to segments plot


## Completed tasks

* move headers into a gstlal/ sub-directory in the source tree so the same #include's work in and out of the source tree (e.g., simplifies moving elements between packages)
* merge fake data generation programs into one
* sub-sample interpolation of triggers (done 2013-06-06)
* Putting the lvalert listening scripts into the dag
* have offline dag split output across multiple directories (done 2013-06-06)
* inspiral DAGs: (Done 2014-02-15)
  * give jobs human-readable names
  * add data pre-staging and serial processing of banks in offline DAG
  * gstlal_inspiral --svd-bank option:
    * why use commas instead of providing the option multiple times? : provided multiple times now
* switch online code to framexmitsrc (after framexmitsrc is made to not jam up when data stops flowing) (done 2014-01-15)
* separate likelihood code from burst pipelines (done 2013-08-29)
* fix the numerator in the likelihood ratio ranking statistic to account
  for inter-instrument correlations in signal parameters (my student's work
  on implementing the rate estimation code might have turned up some code
  that'll do a good brute-force job of this) (done 2013-08-29)
* create a gstlal debug plugin
* create matrix mixer unit test with non-identity mix matrix
* send a pipeline to reviewers demoing the use of the whitener with music or other streams
* explain to Florent how to implement a whitener test using a different spectrum
* gstreamer bugs:
  * small gaps crash resampler,
* add to the gstlal summary page
  * histograms for injection accuracy
* inspiral DAGs:
  * try to get it to do something sane on 1 core?
  * calc_likelihood command lines are too long:  use caches of likelihood files
  * could the file list be provided in a cache file?
* fix known gstlal bugs:
  * compute strain in ER3 revealed bugs in something, can't remember what,
    has fallen off radar.
* fix service discovery to allow more than one job to be running on the same computer and remove need for URL dump files to be exchanged via filesystem.
* framecpp demuxer and muxer cannot decode/encode gaps using "DataValid" bit vector stream.  figure out how and fix.  (probably required to implement offline pipeline's data staging, maybe not)
* Add --min-instruments support to the DAG script (gstlal_inspiral and the create_prior programs) **Cody**
* Add --min-log-L support to the DAG script (gstlal_inspiral) **Cody**

## O2 Development

### Urgent
* Don't set a --singles-threshold on gstlal_inspiral jobs doing injection runs (since now the dag is patched to not delete singles, so we should watch out for injection jobs) - really just make sure that the injection jobs are sane. **Chad + Cody??**
* Get the O1 marg liklihood file converted (for now use one from prior dist stats?). This will be used to assign preliminary LRs **Kipp**
* Figure out plumbing to assign likelihood in offline and disentangle the LR assignment from FAP assignment from snapshotting, etc. **Chad and Cody**
* Different injection channels online **Duncan**
* subdirectory reorg (by GPS time etc) **Cody**
* Additional degrees to LR (phase/time/etc) **Sarah + Cody**
* Improve file merger / data reduction at the end **Cody**
* Allow reranking (mass dependence, coinc only analysis) ideally by fast running post processing dag **Surabhi/Sarah/Cody**

### Less urgent
* Upload segment lists with events; this might mean keeping track of "old" segments in the online jobs (for the previous 4 hours to cover boundary effects)
* We have a request to provide bottle routes for the snr history of the individual detectors in addition to the coinc snr.
* Write a program to plot gracedb events (now with SNR time series).
* lal_interpolator for faster resampling
* expectation value of chisq in noise (check this)
* scipy interpolator issues in rate.py (why is it broken?)
* bottle routes for max LR every second, single detector non-coinc SNR, FAR, what else??
* Write a data summary job that does the following:
  * measures and archives PSD every 8 seconds
  * computes horizon distances every 8 seconds
  * tracks h(t) nonstationarity
  * tracks analysis segments
  * include bottle routes and make a data aggregator for all of these things
* Some annotation to explain what is present in the online summary pages. 
  * units
  * What does "Status OK" mean?
  * provide detchar relevant summary of triggers
  * The "bank" tab - say chirp mass
  * when should we be concerned about RAM history
  * What is "state vector on-off gap"
* Patch itac to populate the SNR vectors on the triggers.  I have made all the changes needed to switch the code over to the new trigger object but the SNR vector is not being populated.  I don't know how. Somebody who does needs to take a look. **Chad**
 * When itac is patched, a gracedb upload needs to be performed ASAP to test the plumbing all the way through and to give Leo a sample file to look at for him to code against. **Chad**

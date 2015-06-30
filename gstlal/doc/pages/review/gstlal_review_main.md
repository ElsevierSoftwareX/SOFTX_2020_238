\page gstlal_review_main_page Review Page

\section links Links

- \ref gstlal_review_howto_page
- \ref gstlal_review_codes_page
- \ref gstlalmeetingspage
- \ref gstlalactionitemspage

\section Team Review Team 2014

- Reviewees: Chad, Kipp, full gstlal development team
- Reviewers: Jolien, Florent, Duncan Me, Sathya


<!---
\section action Action items

*NOTE: This list contains broad action times and not code specific actions.
Consult the \ref gstlal_review_codes_page for more details about code action
items.

- Write documentation for autochisq (paper in progress)
- Explore autocorrelation chisquared mismatch scaling with number of samples e.g., @f$ \nu + \epsilon(\nu) \delta^{2} @f$
- Make sure capsfilters and other similar elements are well documented within graphs (e.g. put in rates, etc)
- Add description of arrows in graphs
- Feature request for detchar - It would be helpful to have online instrument state that could be queried to know if an instrument will be down for an extended time

- figure out what the PE people want to present and when;  it should be
  related to this subject matter of this review meeting

- ranking statistic:
	- far.py
	- gstlal_inspiral_calc_likelihood
	- gstlal_marginalize_likelihood
	- gstlal_compute_far_from_snr_chisq_histograms

- bits of gstlal_inspiral:
	- pipeio.py
	- inspiral.py
	- httpinterface.py
	- hoftcache.py
	- track down ligolw_thinca validation from ihope
	- streamthinca.py

- elements:
	- whiten:
		- get plots into documentation
	- lal_cachesrc:
		- link missing from review status page
	- lal_drop:
		- write unit test
	- lal_simulation:
		- consider patching lal to remove start/stop parameters from
		  XML loading functions so that they just load everything
	- lal_adder
	- lal_firbank:
		- impulse tests of filtering code?
		- pick a PSD, generate template bank, inject an exact
		  template whitened with that PSD, confirm that SNR stream
		  is the autocorrelation recorded in the svd bank file with
		  the correct SNR for the injection
	- lal_itac

- get documentation generated and installed

- explain why approximating transition from signals invisible to the next most sensitive instrument to certainly visible by convolving hard edge with \chi distribution with hard edge at detection threshold is a good idea for joint SNR PDFs

- show histogram of horizon distance history

- Run the pipeline with Gaussian noise with the color expected in O1/O2/O3/aLIGO Design (no need to run on all, one or two will do) with BNS template waveforms with and without spin

\section completed_action Completed action items

- Verify that all the segments are being tracked in online mode via the Handler (this is coupled to the inspiral.Data class, so it will come up again there) & Generate plots of the various segment output and come up with sanity checks.
	- *Done:* (not pushed yet, but PR open https://bugs.ligo.org/redmine/issues/2051)

- Background estimations should have more informative plots e.g., smoothed likelihood functions
	- *Done:* see 33b29f8b653c1bb10fdec477e05644ed6b46da0d 

- Test delta function input to LLOID algorithm (e.g with and without SVD)
	- *Done:* see \gstlalmeeting20150112page

- Consider how to let the user change SNR threshold consistently (if at all).  Note this is tied to SNR bins in far.py
- *Chad this will not be done right now*: The SNR threshold is tied to many
   histogramming objects.  Currently the value is set at 4 which is at the
saturation point for Gaussian noise, e.g., we expect to get an SNR 4 trigger
about once per second.  A user can  change this only after the histograming
code is generalized. It will be left for a future feature request.  Things to consider:
   - Change the histogram boundaries
   - Change the composite detection statistic threshold
   - Change the peak time on lal_itac
   - Study the dependence of coincidence triggers on SNR threshold
- Test robustness of fixed bank (start by figuring out the right question!)
 - Sathya to contact Marcel Kehl to enquire about goals of testing constant template banks.
 - *Marcel: Done* His thesis makes conclusions that it is okay: https://dcc.ligo.org/LIGO-L1400140

- Add synopses for all programs in documentation
 - *Chad: Done*
- Document offline pipeline including graphs of workflows
 - *Chad: Done*, see \ref gstlalinspiralofflinesearchpage
- Test pipeline with control peak times set to different values
 - *Chad: Done* see \ref gstlalinspiralcontrolpeaktimestudypage

- put iterutils.randindex() test plot into minutes for 2014-11-05 telecon

- added check to svd_bank that the command line option matches snr_min in
  far.py

- added test of behaviour of sampler generators to glue's test suite

- expanded docstring for NDBins.volumes() so that this method is covered by
  pylal's test suite

- removed ImportError path for speed of light from snglcoinc.py

- see if P(instruments | signal) can be included in P(snr, ... |
  instruments):  yes it can be, like the SNR PDFs it depends only on the
  ratios of the horizon distances;  have not made this improvement yet but
  have put a FIXME in the code to remind me that this is how to achieve the
  tracking of the time dependence of the instrument combination factor

- see if the noise coinc rates can be recomputed on the fly by factoring
  out the rates:  yes, they can be;  have not made this improvement because
  it would require tinkering with the code a bit and we're about to do a
  release, but I have added a FIXME to remind me that this performance
  improvement is possible.

nightly build:
	- turned off SL6 until it works
	- got nightly build running on debian:
		- includes all gstlal-packages, all documentation, all unit tests

elements:
	- lal_statevector:
		- added warning messages if required-on/required-off have too many bits for width of input stream
		- generalized transform_caps() so that sink to src conversions are complete
		- added notifications for sample count properties
		- wrote unit test
	- lal_sumsquares:
		- wrote unit test
	- lal_togglecomplex:
		- wrote unit test
	- lal_cachesrc:
		- are the warnings and errors related to lack of data in
		  do_seek() correct?  i.e., are warnings and errors needed
		  for these conditions?  done:  lack of start time is no
		  longer an error, element seeks to start of cache in this
		  case
	- lal_gate:
		- removed 64-bit support for control stream:  not possible
		  to specify threshold to that precision
		- why not signal control_queue_head_changed on receipt of
		  NEW_SEGMENT?  not needed.
	- lal_statevector
		- statevector:  why the mask?  remove?  maybe safer to
		  remove.  removed
	- lal_segmentsrc:
		- wrote a unit test
	- lvshmsink/src:
		- wrote pass-through unit test
	- lal_cachesrc:
		- wrote a unit test


- Analysis Makefiles should be documented (e.g., parameters); Do we want them to be made more generic?
 - *Chad: Done*.  \ref gstlalinspiralofflinesearchpage
- Write joint likelihood ranking and FAP calculation (paper in progress)
 - *Kipp: Done* LIGO P1400175 http://arxiv.org/abs/1504.04632
-->

\section studies Studies

- \ref gstlal_inspiral_BNS_MDC_study
- \ref gstlalinspiralcontrolpeaktimestudypage
- \ref gstlalinspiralautochisqlenstudypage
- \ref gstlal_inspiral_impulse_response_study

\page gstlal_review_main_page Review Page

\section links Links

- \ref gstlal_review_howto_page
- \ref gstlal_review_codes_page
- \ref gstlalmeetingspage

\section Team Review Team 2014

- Reviewees: Chad, Kipp, full gstlal development team
- Reviewers: Jolien, Florent, Duncan Me, Sathya


\section action Action items

*NOTE: This list contains broad action times and not code specific actions.
Consult the \ref gstlal_review_codes_page for more details about code action
items.

- Test robustness of fixed bank (start by figuring out the right question!)
 - Sathya to contact Marcel Kehl to enquire about goals of testing constant template banks.
- Analysis Makefiles should be documented (e.g., parameters); Do we want them to be made more generic?
 - *Chad: This process has been started.  See, e.g., Makefile.triggers_example*
- Test delta function input to LLOID algorithm (e.g with and without SVD)
- Consider how to let the user change SNR threshold consistently (if at all).  Note this is tied to SNR bins in far.py
- Background estimations should have more informative plots e.g., smoothed likelihood functions
- Study the dependence of coincidence triggers on SNR threshold
- Write documentation for autochisq (paper in progress)
- Write joint likelihood ranking and FAP calculation (paper in progress)
- Explore autocorrelation chisquared mismatch scaling with number of samples e.g., @f$ \nu + \epsilon(\nu) \delta^{2} @f$
- Make sure capsfilters and other similar elements are well documented within graphs (e.g. put in rates, etc)
- Add description of arrows in graphs
- Verify that all the segments are being tracked in online mode via the Handler (this is coupled to the inspiral.Data class, so it will come up again there)
- Feature request for detchar - It would be helpful to have online instrument state that could be queried to know if an instrument will be down for an extended time

\section completed_action Completed action items
- Add synopses for all programs in documentation
 - *Chad: Done*
- Document offline pipeline including graphs of workflows
 - *Chad: Done*, see \ref gstlalinspiralofflinesearchpage
- Test pipeline with control peak times set to different values
 - *Chad: Done* see \ref gstlalinspiralcontrolpeaktimestudypage

\section studies Studies

- \ref gstlalinspiralcontrolpeaktimestudypage


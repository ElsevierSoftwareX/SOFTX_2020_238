\page gstlalmeeting20141022page Review Meeting October 22, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- Links for telecon:
 - https://ligo-vcs.phys.uwm.edu/cgit/lalsuite/tree/glue/glue/iterutils.py
 - https://ligo-vcs.phys.uwm.edu/cgit/lalsuite/tree/pylal/pylal/rate.py
 - https://ligo-vcs.phys.uwm.edu/cgit/lalsuite/tree/pylal/pylal/snglcoinc.py
 - https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/python/far.py

\section minutes minutes

In attendance: Jolien, Laleigh, Chad, Cody, Florent, Duncan Meacher, Sathya, Steve Privithera, Tom Dent,

1. Eq. (2) of the paper should treat the instrument combinations in a time-dependent way in a
future release.

2. There are two bits in the chi^2; noise dominated for low-SNR and systematics for the high SNR. Is this reflected in the measurements. (Looks like it is not perhaps becuase we never reach high enough SNRs).

3. Paper should state assumption about frequency sensitivity of the instruments.

4. Should consider doing Monte Carlo using c-code and parallelize the code too.

5. The probability distribution in Fig 4 could be computed using Rician instead of the intrinsic SNR in the least sensitive instrument.

6. PDFs are computed when (ratios of) horizon distance(s) changes by 20%. Is 20% the right number?

7. Chisquare in different instruments may not factor at very high SNR (see 2. above).  It is OK as long as SNR is < 100 (as shown in the plots) but this may not hold good for higher masses. (Almost definitely!)

8. Noise probabilities are computed once a week. JC suggests you could update p-values on the fly.

9. The code assumes that each template is equally likely (Section F of the paper). Is this correct? Does it not mean lower mass sources more favoured than high mass sources? Do we need to worry about this?

10. The extinction model doesn't account for the glitches correctly.

11. This statistic should be OK if it is used as a ranking statistic but it could be a problem if it is used as a rate estimator since numerator assumes Gaussian background.

12. JC: So I bet that the reason for the difference between the background and the zero-lag curves in the extinction model is because w is Lambda-dependent.

iterutils.py: randindex()

13. Need a test code to show that the distribution produced is the intended one. Please produce some histograms of the distribution and attach to review documentation.

14. Beware of the change in LAL constants.

snglcoinc.py

15. Interpolation: it is not unique and so how is the choice made?




rates.py:

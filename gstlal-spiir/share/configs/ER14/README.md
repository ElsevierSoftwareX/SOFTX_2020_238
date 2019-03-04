# ER14 configurations

All the bash scripts have documentation inside that is supposed to explain every setting.

## prod
 - uses the code of branch spiir-review-O3, tag spiir-ER14. Reviewers agreed triggers uploaded to GraceDB.
 - compared to tag spiir-ER13 used in ER13: correct minor bugs and minor improvement on skymap speed:
  1.  changed fixed time-shifting to cascade time-shifting for collecting background from different detectors
  2.  fixed the overflow of FAR by restricting it to machine precision, about 10-38
  3.  improved the speed of skymap generation by splitting detector response map into two parts
 - `gen_pipeline.sh`: generate a condor file and a couple of sub files to run on CIT
 - `2det_lowmass_prod_init.sh`: initialize parameters for lowmass pipeline using live LHO and LLO data.
 - `3det_lowmass_prod_init.sh`: initialize parameters for lowmass pipeline using live LHO, LLO, and Virgo data.
 - `2det_highmass_prod_init.sh`: initialize parameters for highmass pipeline using live LHO and LLO data.
 - `3det_highmass_prod_init.sh`: initialize parameters for highmass pipeline using live LHO, LLO, and Virgo data.

## fix
 - uses the code of branch spiir, tag spiir-fix-ER14. Reviewers agreed triggers uploaded to GraceDB-playground.
 - compared to branch spiir-review-O3 tag spiir-ER14 (above), a fix on the FAR estimation and interface updates accordingly:
  1.  changed the ranking statistic from `R = CDF(cohSNR, reducedChisq)` to 
 `(P(nullsnr|signal)*II P(SNR_nontrigger|SNR_trigger, signal))/ II P(SNR_I, chisq_I|noise)`. The background burn-in time need to be increased from a few hours to a few days to cover a 2N+3 dimension background distribution from 2 dimensions.
 - `gen_pipeline.sh`: generate a condor file and a couple of sub files to run on CIT
 - `3det_lowmass_fix_init.sh`: initialize parameters for lowmass pipeline using live LHO, LLO, and Virgo data.
 - `3det_highmass_fix_init.sh`: initialize parameters for highmass pipeline using live LHO, LLO, and Virgo data.

## online_monitor
 - `check_ER14_periodic.sh`: call `check_ER14_latency.sh` and `check_ER14_trigger.sh` every 5 mins.
 - `check_ER14_triggers.sh`: output GPS times and latencies for last 10 triggers of each pipeline
 - `check_ER14_latency.sh`: output GPS times and latencies for last 10 latency>10s triggers of each pipeline

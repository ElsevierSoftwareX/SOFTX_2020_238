#!/bin/bash


#
# This is a hard-coded script to do a once-off merge of the results of the
# 1st three blocks of O1 to obtain the "16 days as ranked" result.  This
# result merges the candidates from the three blocks into a single pile,
# recomputes the mapping from ln L to FAP and FAR based on the new total
# event count and livetime, then reassigns FAPs and FARs to the candidates.
# Their ranking statistics are left untouched.  Finally the minimal set of
# summary plots are produced, and the summary web-pages generated.  All of
# this is done in the current working directory except the final web pages
# which are written to a path in /home/gstlalcbc/public_html.
#
# It should be safe for any user *other than gstlalcbc* to run this script
# in any directory, but the final web page step will fail because you won't
# have write permission to the gstlalcbc account's public_html directory.
# You *should not* run this script as the gstlalcbc user unless you
# understand what you are doing and are prepared to overwrite the results
# pages.
#


cp -v \
	/home/gstlalcbc/observing/1/offline/1126051217-1127271617-run1/H1L1-ALL_LLOID-1126051217-1220400.sqlite \
	/home/gstlalcbc/observing/1/offline/1127271617-1128299417-run2-2/H1L1-ALL_LLOID-1127271617-1027800.sqlite \
	/home/gstlalcbc/observing/1/offline/1128299417-1129383017-run3-rerun/H1L1-ALL_LLOID-1128299417-1083600.sqlite \
	.

gstlal_inspiral_marginalize_likelihood --verbose --require-ranking-stat-data --output marginalized_likelihood.xml.gz \
	/home/gstlalcbc/observing/1/offline/1126051217-1127271617-run1/marginalized_likelihood.xml.gz \
	/home/gstlalcbc/observing/1/offline/1127271617-1128299417-run2-2/marginalized_likelihood.xml.gz \
	/home/gstlalcbc/observing/1/offline/1128299417-1129383017-run3-rerun/marginalized_likelihood.xml.gz

gstlal_compute_far_from_snr_chisq_histograms --verbose --force --background-bins-file marginalized_likelihood.xml.gz \
	--non-injection-db H1L1-ALL_LLOID-1126051217-1220400.sqlite \
	--non-injection-db H1L1-ALL_LLOID-1127271617-1027800.sqlite \
	--non-injection-db H1L1-ALL_LLOID-1128299417-1083600.sqlite

rm -rvf plots
mkdir plots

gstlal_inspiral_plotsummary --verbose --segments-name datasegments --user-tag 16DAYS_ASRANKED --output-dir plots \
	--likelihood-file post_marginalized_likelihood.xml.gz \
	H1L1-ALL_LLOID-1126051217-1220400.sqlite \
	H1L1-ALL_LLOID-1127271617-1027800.sqlite \
	H1L1-ALL_LLOID-1128299417-1083600.sqlite

gstlal_inspiral_plot_background --verbose --output-dir plots --user-tag 16DAYS_ASRANKED \
	post_marginalized_likelihood.xml.gz \
	--database H1L1-ALL_LLOID-1126051217-1220400.sqlite \
	--database H1L1-ALL_LLOID-1127271617-1027800.sqlite \
	--database H1L1-ALL_LLOID-1128299417-1083600.sqlite

rm -rvf /home/gstlalcbc/public_html/O1/gstlal/production_runs/1126051217-1129383017-16DAYS-ASRANKED/
mkdir -p /home/gstlalcbc/public_html/O1/gstlal/production_runs/1126051217-1129383017-16DAYS-ASRANKED/OPEN-BOX

gstlal_inspiral_summary_page --output-user-tag 16DAYS_ASRANKED --glob-path plots --webserver-dir /home/gstlalcbc/public_html/O1/gstlal/production_runs/1126051217-1129383017-16DAYS-ASRANKED/OPEN-BOX --title gstlal-1126051217-1129383017-16DAYS-ASRANKED-open-box --open-box

gstlal_inspiral_summary_page --output-user-tag 16DAYS_ASRANKED --glob-path plots --webserver-dir /home/gstlalcbc/public_html/O1/gstlal/production_runs/1126051217-1129383017-16DAYS-ASRANKED --title gstlal-1126051217-1129383017-16DAYS-ASRANKED-closed-box


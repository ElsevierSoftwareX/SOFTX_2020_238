#!/bin/sh

#
# create a working directory in which to build and run a dag and generate
# plots, then run this script in that working directory.  the script
# terminates part way through.  you will need to submit a dag at that
# point, then when the dag completes continue this script from the point
# where it exited.  I normally do this by just cutting and pasting the
# commands at the bottom of this script into a terminal and letting it go.
# It doesn't take long.
#
# NOTE:  the paths to source databases and cache files will need to be
# modified if this procedure is to be repeated on a different set of runs.
#


#
# clean up from a previous attempt
#

rm -rvf gstlal_inspiral_calc_rank_pdfs gstlal_inspiral_marginalize_likelihood logs plots recalc_pdfs.dag *.dag.* *.sqlite marginalized_likelihood.xml.gz post_marginalized_likelihood.xml.gz

#
# create required directories
#

mkdir plots/
mkdir logs/
mkdir gstlal_inspiral_calc_rank_pdfs/

#
# generate dag.  NOTE:  put gstlal_inspiral_recalc_rank_pdfs in your PATH
# and read the comments in gstlal_inspiral_recalc_rank_pdfs to see what you
# need to do to provide a suitable submit file.
#

gstlal_inspiral_recalc_rank_pdfs \
	/home/gstlalcbc/observing/1/offline/1126051217-1127271617-run1/gstlal_inspiral_calc_rank_pdfs/*.cache \
	/home/gstlalcbc/observing/1/offline/1127271617-1128299417-run2-2/gstlal_inspiral_calc_rank_pdfs/*.cache \
	/home/gstlalcbc/observing/1/offline/1128299417-1129383017-run3-rerun/gstlal_inspiral_calc_rank_pdfs/*.cache \

#
# run the dag.  continue from here when it's done
#

exit

#
# marginalize ranking statistic PDFs over template bank bin and run number
#

gstlal_inspiral_marginalize_likelihood --verbose --require-ranking-stat-pdf --output marginalized_likelihood.xml.gz gstlal_inspiral_calc_rank_pdfs/*.xml.gz

#
# collect the databases from the "16 days as ranked" result
#

cp -v ~gstlalcbc/observing/1/offline/1126051217-1129383017-16days-asranked/*.sqlite .

#
# assign new FAPs and FARs based on the new ranking statistic PDF (and
# segment lists)
#

gstlal_compute_far_from_snr_chisq_histograms --verbose --force --background-bins-file marginalized_likelihood.xml.gz \
	--non-injection-db H1L1-ALL_LLOID-1126051217-1220400.sqlite \
	--non-injection-db H1L1-ALL_LLOID-1127271617-1027800.sqlite \
	--non-injection-db H1L1-ALL_LLOID-1128299417-1083600.sqlite

#
# generate plots
#

gstlal_inspiral_plotsummary --verbose --segments-name datasegments --user-tag 16DAYS_ASRANKED_WZL --output-dir plots \
	--likelihood-file post_marginalized_likelihood.xml.gz \
	H1L1-ALL_LLOID-1126051217-1220400.sqlite \
	H1L1-ALL_LLOID-1127271617-1027800.sqlite \
	H1L1-ALL_LLOID-1128299417-1083600.sqlite

gstlal_inspiral_plot_background --verbose --output-dir plots --user-tag 16DAYS_ASRANKED_WZL --add-zerolag-to-background \
	--database H1L1-ALL_LLOID-1126051217-1220400.sqlite \
	--database H1L1-ALL_LLOID-1127271617-1027800.sqlite \
	--database H1L1-ALL_LLOID-1128299417-1083600.sqlite \
	post_marginalized_likelihood.xml.gz

#
# also generate plots from the original "16 days as ranked" to have
# versions made with the current version of plotsummary
#

gstlal_inspiral_plotsummary --verbose --segments-name datasegments --user-tag 16DAYS_ASRANKED --output-dir plots \
	--likelihood-file ~gstlalcbc/observing/1/offline/1126051217-1129383017-16days-asranked/post_marginalized_likelihood.xml.gz \
	~gstlalcbc/observing/1/offline/1126051217-1129383017-16days-asranked/H1L1-ALL_LLOID-1126051217-1220400.sqlite \
	~gstlalcbc/observing/1/offline/1126051217-1129383017-16days-asranked/H1L1-ALL_LLOID-1127271617-1027800.sqlite \
	~gstlalcbc/observing/1/offline/1126051217-1129383017-16days-asranked/H1L1-ALL_LLOID-1128299417-1083600.sqlite

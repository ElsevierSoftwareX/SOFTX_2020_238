EXTRA_DIST = \
	lvshmsinksrc_test_01.sh \
	plot_test \
	plots_test_01.py \
	ratefaker_test_01.py

if COND_GDS
GDS_TESTS = lvshmsinksrc_test_01.sh
else
GDS_TESTS =
endif
TESTS = $(GDS_TESTS)

clean-local :
	rm -f *.dump
	rm -f *.avi
	rm -f *.pyc

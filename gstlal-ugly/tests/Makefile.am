EXTRA_DIST = \
	framecpp_test_01.sh \
	lvshmsinksrc_test_01.sh \
	plot_test \
	plots_test_01.py \
	ratefaker_test_01.py

if COND_FRAMECPP
FRAMECPP_TESTS = framecpp_test_01.sh
else
FRAMECPP_TESTS =
endif

if COND_GDS
GDS_TESTS = lvshmsinksrc_test_01.sh
else
GDS_TESTS =
endif

TESTS = $(FRAMECPP_TESTS) $(GDS_TESTS)

clean-local :
	rm -f *.dump
	rm -f *.avi
	rm -f *.pyc

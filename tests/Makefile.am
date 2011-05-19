EXTRA_DIST = \
	firbank_test_01.py \
	fixtures.py \
	plots_test_01.py \
	resample_test_01.py \
	whiten_test_01.py \
	test_common.py \
	test_coinc.py \
	test_ceil_pow_2.py

check :
	python test_coinc.py
	python firbank_test_01.py
	python plots_test_01.py
	python resample_test_01.py
	python whiten_test_01.py

clean-local :
	rm -f *.dump
	rm -f *.avi
	rm -f {test_common,fixtures}.pyc

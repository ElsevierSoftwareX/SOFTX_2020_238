EXTRA_DIST = \
	fixtures.py \
	plots_test_01.py \
	test_coinc.py

check :
	python test_coinc.py
	python plots_test_01.py

clean-local :
	rm -f *.dump
	rm -f *.avi
	rm -f {test_common,fixtures}.pyc

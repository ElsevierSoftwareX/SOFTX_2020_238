if COND_DOXYGEN
all-local : html latex
endif

clean-local : clean-html clean-latex

clean-html :
	-rm -rvf html/

clean-latex :
	-rm -rvf latex/

html latex : Doxyfile
	$(DOXYGEN) $<

SUBDIRS = gobject

if COND_DOXYGEN
all-local: doxyfile.stamp

directory = $(top_srcdir)/docs/man/man3/

dist_man_MANS = $(directory)/gstlal_datasource.3
$(directory)/gstlal_datasource.3: doxyfile.stamp

doxyfile.stamp:
	$(DOXYGEN) Doxyfile
	echo Timestamp > doxyfile.stamp

CLEANFILES = doxyfile.stamp


clean-local : clean-html clean-latex clean-man

clean-man: 
	rm -rf $(top_srcdir)/docs/man

clean-html :
	-rm -rvf $(top_srcdir)/docs/html/

clean-latex :
	-rm -rvf $(top_srcdir)/docs/latex/
endif

SUBDIRS = gobject

if COND_DOXYGEN
all-local: doxyfile.stamp

directory = $(top_srcdir)/docs/man/man3/

dist_man_MANS = $(directory)/man_page_1.3 $(directory)/man_page_2.3
$(directory)/man_page_1.3: doxyfile.stamp
$(directory)/man_page_2.3: doxyfile.stamp

doxyfile.stamp:
	$(DOXYGEN) Doxyfile
	echo Timestamp > doxyfile.stamp

CLEANFILES = doxyfile.stamp


clean-local : clean-html clean-latex clean-man

clean-man: 
	rm -rf $(top_srcdir)/docs/man

clean-html :
	-rm -rvf html/

clean-latex :
	-rm -rvf latex/
endif

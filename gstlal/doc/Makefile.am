SUBDIRS = gobject

if COND_DOXYGEN
all-local: doxyfile.stamp

doxyfile.stamp:
	$(DOXYGEN) Doxyfile
	echo Timestamp > doxyfile.stamp

html/*.* html/search/*.* : doxyfile.stamp

# FIXME this doesn't feel right
# Will uninstall work??
nobase_pkgdoc_DATA = html/*.* html/search/*.*

CLEANFILES = doxyfile.stamp

clean-local : clean-local-html clean-local-latex clean-local-man

clean-local-man: 
	rm -rf man

clean-local-html :
	-rm -rf html

clean-local-latex :
	-rm -rf latex
endif

if COND_DOXYGEN
all-local : doxyfile.stamp

doxyfile.stamp :
	$(DOXYGEN) Doxyfile
	echo Timestamp > doxyfile.stamp

html/% html/search/% : doxyfile.stamp

# FIXME this doesn't feel right
# Will uninstall work??
nobase_pkgdoc_DATA = $(wildcard html/*.*) $(wildcard html/search/*.*)

CLEANFILES = doxyfile.stamp gstlalinspiraldoxy.tag

clean-local : clean-local-html clean-local-latex clean-local-man

clean-local-man :
	rm -rf man

clean-local-html :
	-rm -rf html

clean-local-latex :
	-rm -rf latex
endif

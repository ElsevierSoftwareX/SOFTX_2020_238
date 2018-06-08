%define gstreamername gstreamer1

Name: gstlal-spiir
Version: 1.0.0
Release: 1%{?dist}
Summary: GSTLAL SPIIR
License: GPL
Group: LSC Software/Data Analysis
Requires: avahi
Requires: avahi-glib
Requires: avahi-ui-tools
Requires: fftw >= 3
Requires: glue >= 1.54.1
Requires: glue-segments >= 1.54.1
Requires: gobject-introspection >= 1.30.0
Requires: gsl
Requires: %{gstreamername} >= 1.2.4
Requires: %{gstreamername}-plugins-bad-free
Requires: %{gstreamername}-plugins-base >= 1.2.4
Requires: %{gstreamername}-plugins-good >= 1.2.4
Requires: lal >= 6.18.0
Requires: lal-python >= 6.18.0
Requires: lalburst >= @MIN_LALBURST_VERSION@
Requires: lalmetaio >= 1.3.1
Requires: lalinspiral >= 1.7.7
Requires: lalsimulation >= @MIN_LALSIMULATION_VERSION@
Requires: numpy
Requires: orc >= @MIN_ORC_VERSION@
Requires: python >= 2.7
Requires: python-%{gstreamername}
Requires: scipy
BuildRequires: doxygen >= 1.8.3
BuildRequires: fftw-devel >= 3
BuildRequires: gobject-introspection-devel >= 1.30.0
BuildRequires: graphviz
BuildRequires: gsl-devel
BuildRequires: gtk-doc >= 1.11
BuildRequires: %{gstreamername}-devel >= 1.2.4
BuildRequires: %{gstreamername}-plugins-base-devel >= 1.2.4
BuildRequires: lal-devel >= 6.18.0
BuildRequires: lal-python >= 6.18.0
BuildRequires: lalburst-devel >= @MIN_LALBURST_VERSION@
BuildRequires: lalinspiral-devel >= 1.7.7
BuildRequires: lalmetaio-devel >= 1.3.1
BuildRequires: lalsimulation-devel >= @MIN_LALSIMULATION_VERSION@
BuildRequires: numpy
BuildRequires: orc >= @MIN_ORC_VERSION@
BuildRequires: python-devel >= 2.7
# needed for gstpythonplugin.c remove when we remove that plugin from gstlal
BuildRequires: pygobject3-devel
Source: gstlal-spiir-%{version}.tar.gz
URL: https://wiki.ligo.org/DASWG/GstLAL
Packager: Kipp Cannon <kipp.cannon@ligo.org>
BuildRoot: %{_tmppath}/%{name}-%{version}-root
%description
This package provides a variety of gstreamer elements for
gravitational-wave data analysis and some libraries to help write such
elements.  The code here sits on top of several other libraries, notably
the LIGO Algorithm Library (LAL), FFTW, the GNU Scientific Library (GSL),
and, of course, GStreamer.

This package contains the plugins and shared libraries required to run
gstlal-based applications.


%package devel
Summary: Files and documentation needed for compiling gstlal-based plugins and programs.
Group: LSC Software/Data Analysis
Requires: %{name} = %{version}
Requires: fftw-devel >= 3
Requires: gsl-devel
Requires: %{gstreamername}-devel >= 1.2.4
Requires: %{gstreamername}-plugins-base-devel >= 1.2.4
Requires: lal-devel >= 6.18.0
Requires: lalmetaio-devel >= 1.3.1
Requires: lalsimulation-devel >= @MIN_LALSIMULATION_VERSION@
Requires: lalburst-devel >= @MIN_LALBURST_VERSION@
Requires: lalinspiral-devel >= 1.7.7
Requires: python-devel >= 2.7
%description devel
This package contains the files needed for building gstlal-based plugins
and programs.


%prep
%setup -q -n %{name}-%{version}


%build
%configure --enable-gtk-doc
%{__make}


%install
# FIXME:  why doesn't % makeinstall macro work?
DESTDIR=${RPM_BUILD_ROOT} %{__make} install
# remove .so symlinks from libdir.  these are not included in the .rpm,
# they will be installed by ldconfig in the post-install script, except for
# the .so symlink which isn't created by ldconfig and gets shipped in the
# devel package
[ ${RPM_BUILD_ROOT} != "/" ] && find ${RPM_BUILD_ROOT}/%{_libdir} -name "*.so.*" -type l -delete
# don't distribute *.la files
[ ${RPM_BUILD_ROOT} != "/" ] && find ${RPM_BUILD_ROOT} -name "*.la" -type f -delete


%post
if test -d /usr/lib64 ; then
	ldconfig /usr/lib64
else
	ldconfig
fi


%postun
if test -d /usr/lib64 ; then
	ldconfig /usr/lib64
else
	ldconfig
fi


%clean
[ ${RPM_BUILD_ROOT} != "/" ] && rm -Rf ${RPM_BUILD_ROOT}
rm -Rf ${RPM_BUILD_DIR}/%{name}-%{version}


%files
%defattr(-,root,root)
%{_bindir}/*
%{_datadir}/gir-*/*
%{_datadir}/gstlal
%{_datadir}/gtk-doc/html/gstlal-*
%{_docdir}/gstlal-spiir
%{_libdir}/*.so.*
%{_libdir}/girepository-1.0/*
%{_libdir}/gstreamer-1.0/*.so
%{_libdir}/gstreamer-1.0/python/*
%{_prefix}/%{_lib}/python*/site-packages/gstlal

%files devel
%defattr(-,root,root)
%{_libdir}/*.a
%{_libdir}/*.so
%{_libdir}/pkgconfig/*
%{_libdir}/gstreamer-1.0/*.a
%{_includedir}/*

import os
from distutils.core import setup, Extension
import numpy

class PkgConfig(object):
	def __init__(self, names):
		def stripfirsttwo(string):
			return string[2:]
		self.libs = map(stripfirsttwo, os.popen("pkg-config --libs-only-l %s" % names).read().split())
		self.libdirs = map(stripfirsttwo, os.popen("pkg-config --libs-only-L %s" % names).read().split())
		self.incdirs = map(stripfirsttwo, os.popen("pkg-config --cflags-only-I %s" % names).read().split())
		self.extra_cflags = os.popen("pkg-config --cflags-only-other %s" % names).read().split()

lal_pkg_config = PkgConfig("lal")
lalsimulation_pkg_config = PkgConfig("lalsimulation")
lalinspiral_pkg_config = PkgConfig("lalinspiral")
waveModule= Extension('waveHandler',
		    include_dirs = lal_pkg_config.incdirs + lalsimulation_pkg_config.incdirs + lalinspiral_pkg_config.incdirs + [numpy.get_include()],
		    libraries = lal_pkg_config.libs + lalsimulation_pkg_config.libs + lalinspiral_pkg_config.libs,
		    library_dirs = lal_pkg_config.libdirs + lalsimulation_pkg_config.libdirs + lalinspiral_pkg_config.libdirs,
		    runtime_library_dirs = lal_pkg_config.libdirs + lalsimulation_pkg_config.libdirs + lalinspiral_pkg_config.libdirs,
		    sources=['waveHandler.c'])

setup(name='waveHandler',
    version='1.0',
    description = 'This is a handler for calling spin wave sources',
    ext_modules = [waveModule])

#/ivec/apps/gcc/4.4.7/python/2.6.8/include/python2.6/
#include_dirs=['/scratch/partner723/opt/lalsimulation/include/lal/',
#		  '/scratch/partner723/ibs/src/lalsuite/lalsimulation/include',
#		  '/scratch/partner723/ibs/src/lalsuite/lalsimulation/src',
#		  '/scratch/partner723/opt/lal/include',
#		  numpy.get_include()],
		    #include_dirs = lal_pkg_config.incdirs + lalsimulation_pkg_config.incdirs + lalinspiral_pkg_config.incdirs + ['/scratch/partner723/ibs/src/lalsuite/lalsimulation/src','/scratch/partner723/opt/lalsimulation/include/lal',numpy.get_include()],

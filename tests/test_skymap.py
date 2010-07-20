#!/usr/bin/env python
"""

Unit tests for lal_skymap.

"""
__author__       = "Leo Singer <leo.singer@ligo.org>"
__copyright__    = "Copyright 2010, Leo Singer"


import unittest
from fixtures import *
from gstlal.pipeutil import *


class TestSkymap(PipelineTestFixture):

	def runTest(self):
		pass # TODO

if __name__ == '__main__':
	suite = unittest.main()

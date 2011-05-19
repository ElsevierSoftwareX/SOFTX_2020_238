#!/usr/bin/env python
"""

Unit tests for gstlal.templates.ceil_pow_2.

"""
__author__       = "Leo Singer <leo.singer@ligo.org>"
__copyright__    = "Copyright 2010, Leo Singer"


import unittest
import math
from gstlal.templates import ceil_pow_2

class TestCeilPow2(unittest.TestCase):

	def test_negative(self):
		"""Test that negative numbers give nan."""
		self.assertTrue(math.isnan(ceil_pow_2(-100)))
		self.assertTrue(math.isnan(ceil_pow_2(-1)))
		self.assertTrue(math.isnan(ceil_pow_2(-0.1)))
		self.assertTrue(math.isnan(ceil_pow_2(-0.00000001)))

	def test_near_0_015625(self):
		"""Test values that are close to 1/64."""
		self.assertEqual(ceil_pow_2(0.015624), 0.015625)
		self.assertEqual(ceil_pow_2(0.015625), 0.015625)
		self.assertEqual(ceil_pow_2(0.0156251), 0.03125)
		self.assertEqual(ceil_pow_2(0.01562501), 0.03125)
		self.assertEqual(ceil_pow_2(0.015625001), 0.03125)
		self.assertEqual(ceil_pow_2(0.0156250001), 0.03125)
		self.assertEqual(ceil_pow_2(0.01562500001), 0.03125)
		self.assertEqual(ceil_pow_2(0.015625000001), 0.03125)
		self.assertEqual(ceil_pow_2(0.0156250000001), 0.03125)
		self.assertEqual(ceil_pow_2(0.01562500000001), 0.03125)
		self.assertEqual(ceil_pow_2(0.015625000000001), 0.03125)
		self.assertEqual(ceil_pow_2(0.0156250000000001), 0.03125)
		self.assertEqual(ceil_pow_2(0.01562500000000001), 0.03125)
		# 0.01562500000000001 == 0.0156250 in double precision
		self.assertEqual(ceil_pow_2(0.015625000000000001), 0.015625)
		self.assertEqual(ceil_pow_2(0.0156250000000000001), 0.015625)
		self.assertEqual(ceil_pow_2(0.01562500000000000001), 0.015625)

	def test_near_64(self):
		"""Test values that are close to 64."""
		self.assertEqual(ceil_pow_2(63.9), 64.)
		self.assertEqual(ceil_pow_2(64.), 64.)
		self.assertEqual(ceil_pow_2(64.1), 128.)
		self.assertEqual(ceil_pow_2(64.01), 128.)
		self.assertEqual(ceil_pow_2(64.001), 128.)
		self.assertEqual(ceil_pow_2(64.0001), 128.)
		self.assertEqual(ceil_pow_2(64.00001), 128.)
		self.assertEqual(ceil_pow_2(64.000001), 128.)
		self.assertEqual(ceil_pow_2(64.0000001), 128.)
		self.assertEqual(ceil_pow_2(64.00000001), 128.)
		self.assertEqual(ceil_pow_2(64.000000001), 128.)
		self.assertEqual(ceil_pow_2(64.0000000001), 128.)
		self.assertEqual(ceil_pow_2(64.00000000001), 128.)
		self.assertEqual(ceil_pow_2(64.000000000001), 128.)
		self.assertEqual(ceil_pow_2(64.0000000000001), 128.)
		self.assertEqual(ceil_pow_2(64.00000000000001), 128.)
		# 64.000000000000001 == 64.0 in double precision
		self.assertEqual(ceil_pow_2(64.000000000000001), 64.)
		self.assertEqual(ceil_pow_2(64.0000000000000001), 64.)
		self.assertEqual(ceil_pow_2(64.00000000000000001), 64.)

if __name__ == '__main__':
	suite = unittest.main()

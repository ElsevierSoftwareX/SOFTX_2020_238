#!/usr/bin/env python

import doctest
import random
import unittest
import sys
from glue import segments
from gstlal.stats import trigger_rate

def random_coalesced_list(n):
	def r():
		return random.uniform(1. / 128., 127. / 128.)
	l = trigger_rate.ratebinlist([None] * n)
	x = r()
	count = r()
	l[0] = trigger_rate.ratebin(x, x + count, count = count)
	x = l[0][1] + r()
	for i in range(1, n):
		count = r()
		l[i] = trigger_rate.ratebin(x, x + count, count = count)
		x = l[i][1] + r()
	return l

class test_segmentlist(unittest.TestCase):
	algebra_repeats = 8000
	algebra_listlength = 200

	def test__and__(self):
		for i in range(self.algebra_repeats):
			a = random_coalesced_list(random.randint(1, self.algebra_listlength))
			b = random_coalesced_list(random.randint(1, self.algebra_listlength))
			c = a & b
		try:
			self.assertEqual(c.segmentlist(), a.segmentlist() & b.segmentlist())
			self.assertEqual(c, a - (a - b))
			self.assertEqual(c, b - (b - a))
			self.assertAlmostEqual(c.count, float(abs(c)) * 2, places = 12)
		except AssertionError as e:
			raise AssertionError(str(e) + "\na = " + str(a) + "\nb = " + str(b))


suite = unittest.TestSuite()
suite.addTest(unittest.makeSuite(test_segmentlist))
if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
	sys.exit(1)

failures = doctest.testmod(trigger_rate)[0]
if failures:
	sys.exit(bool(failures))

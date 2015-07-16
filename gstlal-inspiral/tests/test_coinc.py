#!/usr/bin/env python
"""

Unit tests for lal_coinc.

"""
__author__       = "Leo Singer <leo.singer@ligo.org>"
__copyright__    = "Copyright 2010, Leo Singer"


import unittest
from fixtures import *
from gstlal.pipeutil import *

import gstlal.pipeparts as pipeparts
from pylal.xlal.datatypes.snglinspiraltable import SnglInspiralTable
from glue.ligolw import lsctables
from glue.ligolw import utils
from itertools import groupby

# FIXME Python < 2.5 compatibility
try:
	any
except NameError:
	from glue.iterutils import any


class CoincTestFixture(PipelineTestFixture):
	
	def single_detector_new_buffer(self, elem, user_data):
		sngls = SnglInspiralTable.from_buffer(elem.get_property("last-buffer"))
		for sngl in sngls:
			if sngl.mass1 in self.triggers_by_mass1:
				triggers = self.triggers_by_mass1[sngl.mass1]
			else:
				triggers = list()
				self.triggers_by_mass1[sngl.mass1] = triggers
			triggers.append((sngl.end_time * gst.SECOND + sngl.end_time_ns, sngl.ifo))

	def coinc_new_buffer(self, elem, user_data):
		sngls = SnglInspiralTable.from_buffer(elem.get_property("last-buffer"))
		for coinc in zip(*[sngls[i::len(self.ifos)] for i in range(len(self.ifos))]):
			mass1 = coinc[0].mass1
			trigger_records = frozenset(
				(gst.SECOND * sngl.end_time + sngl.end_time_ns, sngl.ifo)
				for sngl in coinc if sngl.ifo != '')
			if mass1 not in self.coincs:
				self.coincs[mass1] = set()
			self.coincs[mass1].add(trigger_records)

	def setUp(self):
		super(CoincTestFixture, self).setUp()

		self.dt = 100 * gst.MSECOND
		xml_location = '../examples/banks/1-split_bank-H1-TMPLTBANK_DATAFIND-871157768-2048.xml.gz'

		# Add coincidence element
		coinc = gst.element_factory_make('lal_coinc')
		coinc.set_property("dt", self.dt)
		self.pipeline.add(coinc)

		for i_ifo, ifo in enumerate(self.ifos):

			# Add trigger source element
			triggersrc = gst.element_factory_make('lal_faketriggersrc')
			triggersrc.set_property('instrument', ifo)
			triggersrc.set_property('xml-location', xml_location)
			self.pipeline.add(triggersrc)

			# Add tee element
			tee = gst.element_factory_make('tee')
			self.pipeline.add(tee)
			triggersrc.link(tee)

			# Add appsink element
			appsink = pipeparts.mkappsink(self.pipeline, tee)
			appsink.connect_after('new-buffer', self.single_detector_new_buffer, ifo)

			# Link to coinc element
			tee.link_pads('src%d', coinc, 'sink%d' % i_ifo)

		# Add final appsink
		appsink = pipeparts.mkappsink(self.pipeline, coinc)
		appsink.connect_after('new-buffer', self.coinc_new_buffer, None)

		self.triggers_by_mass1 = dict()
		self.coincs = dict()


class TestTripleCoinc(CoincTestFixture):

	def setUp(self):
		self.ifos = ('H1','L1','V1')
		super(TestTripleCoinc, self).setUp()

	def runTest(self):
		"""Check that lal_coinc finds double and triple coincidences from lal_faketriggersrc."""

		# Start pipeline
		self.pipeline.set_state(gst.STATE_PLAYING)

		# Start main loop
		self.mainloop.run()

		# Sort triggers by time
		for triggers in self.triggers_by_mass1.itervalues():
			triggers.sort()

		# Compute doubles
		doubles_by_mass1 = dict(
			(mass1, [frozenset(double) for double in zip(triggers[:-1], triggers[1:]) if double[-1][0] - double[0][0] <= self.dt])
			for mass1, triggers in self.triggers_by_mass1.iteritems())

		# Compute triples
		triples_by_mass1 = dict(
			(mass1, set(frozenset(triple) for triple in zip(triggers[:-2], triggers[1:-1], triggers[2:]) if triple[-1][0] - triple[0][0] <= self.dt))
			for mass1, triggers in self.triggers_by_mass1.iteritems())

		# Prune doubles that are also part of triples
		for mass1, doubles in doubles_by_mass1.items():
			if mass1 in triples_by_mass1:
				triples = triples_by_mass1[mass1]
				doubles_by_mass1[mass1] = frozenset(filter((lambda x: not(any(x.issubset(y) for y in triples))), doubles))

		# Build set of all coincs
		coincs = triples_by_mass1
		for mass1, doubles in doubles_by_mass1.iteritems():
			if mass1 in coincs:
				coincs[mass1] |= doubles
			else:
				coincs[mass1] = doubles

		for mass1, coinclist in coincs.iteritems():
			self.assertTrue(mass1 in self.coincs, "expected to find triggers with mass1=%f, but none found" % mass1)
			for coinc in coinclist:
				self.assertTrue(coinc in self.coincs[mass1], "coincidence " + str((mass1, coinc)) + " not found")

		for mass1, coinclist in self.coincs.iteritems():
			self.assertTrue(mass1 in coincs, "found triggers with mass1=%f, but should have found none" % mass1)
			for coinc in coinclist:
				self.assertTrue(coinc in coincs[mass1], "unexpected coincidence " + str((mass1, coinc)) + " found")


class TestDoubleCoinc(CoincTestFixture):

	def setUp(self):
		self.ifos = ('H1','L1')
		super(TestDoubleCoinc, self).setUp()

	def runTest(self):
		"""Check that lal_coinc finds double and triple coincidences from lal_faketriggersrc."""

		# Start pipeline
		self.pipeline.set_state(gst.STATE_PLAYING)

		# Start main loop
		self.mainloop.run()

		# Sort triggers by time
		for triggers in self.triggers_by_mass1.itervalues():
			triggers.sort()

		# Compute doubles
		doubles_by_mass1 = dict(
			(mass1, [frozenset(double) for double in zip(triggers[:-1], triggers[1:]) if double[-1][0] - double[0][0] <= self.dt])
			for mass1, triggers in self.triggers_by_mass1.iteritems())

		# Build set of all coincs
		coincs = doubles_by_mass1

		for mass1, coinclist in coincs.iteritems():
			self.assertTrue(mass1 in self.coincs, "expected to find triggers with mass1=%f, but none found" % mass1)
			for coinc in coinclist:
				self.assertTrue(coinc in self.coincs[mass1], "coincidence " + str((mass1, coinc)) + " not found")

		for mass1, coinclist in self.coincs.iteritems():
			self.assertTrue(mass1 in coincs, "found triggers with mass1=%f, but should have found none" % mass1)
			for coinc in coinclist:
				self.assertTrue(coinc in coincs[mass1], "unexpected coincidence " + str((mass1, coinc)) + " found")

if __name__ == '__main__':
	suite = unittest.main()

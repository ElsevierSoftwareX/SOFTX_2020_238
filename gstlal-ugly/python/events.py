#!/usr/bin/env python3

__author__ = "Patrick Godwin (patrick.godwin@ligo.org)"
__description__ = "a module for storing event processing utilities"

#-------------------------------------------------
### imports

import logging
import signal
import sys
import time
import timeit

try:
	from confluent_kafka import Producer, Consumer, KafkaError
except ImportError:
	raise ImportError('confluent_kafka is required for this module')


#-------------------------------------------------
### classes

class EventProcessor(object):
	"""Base class for processing events via Kafka.

	Parameters
	----------
	kafka_server :	`str`
		the host:port combination to connect to the Kafka broker
	input_topic : `str`
		the name of the input topic
	process_cadence : `float`
		maximum rate at which data is processed, defaults to 0.1s
	request_timeout : `float`
		timeout for requesting messages from a topic, defaults to 0.2s
	num_messages : `int`
		max number of messages to process per cadence, defaults to 10
	tag : `str`
		a nickname for the instance, defaults to 'default'

	"""
	_name = 'processor'

	def __init__(
		self,
		process_cadence=0.1,
		request_timeout=0.2,
		num_messages=10,
		kafka_server=None,
		input_topic=None,
		tag='default'
	):
		assert kafka_server, 'kafka_server needs to be set'
		assert input_topic, 'input_topic needs to be set'
		if isinstance(input_topic, str):
			input_topic = [input_topic]

		### processing settings
		self.process_cadence = process_cadence
		self.request_timeout = request_timeout
		self.num_messages = num_messages
		self.is_running = False

		### kafka settings
		self.kafka_settings = {
			'bootstrap.servers': kafka_server,
			'group.id': '-'.join([self._name, tag])
		}
		self.producer = Producer(self.kafka_settings)
		self.consumer = Consumer(self.kafka_settings)
		self.consumer.subscribe([topic for topic in input_topic])

		### signal handler
		for sig in [signal.SIGINT, signal.SIGTERM]:
			signal.signal(sig, self.catch)


	def fetch(self):
		"""Fetch for messages from a topic and processes them.

		"""
		messages = self.consumer.consume(
			num_messages=self.num_messages,
			timeout=self.request_timeout
		)

		for message in messages:
			### only add to queue if no errors in receiving data
			if message and not message.error():
				self.ingest(message)


	def process(self):
		"""Processes events at the specified cadence.

		"""
		while self.is_running:
			start = timeit.default_timer()
			self.fetch()
			self.handle()
			elapsed = timeit.default_timer() - start
			time.sleep(max(self.process_cadence - elapsed, 0))


	def start(self):
		"""Starts the event loop.

		"""
		logging.info('starting {}...'.format(self._name.replace('_', ' ')))
		self.is_running = True
		self.process()


	def stop(self):
		"""Stops the event loop.

		"""
		logging.info('shutting down {}...'.format(self._name.replace('_', ' ')))
		self.finish()
		self.is_running = False


	def catch(self, signum, frame):
		"""Shuts down the event processor gracefully before exiting.

		"""
		logging.info("SIG {:d} received, attempting graceful shutdown...".format(signum))
		self.stop()
		sys.exit(0)


	def ingest(self, message):
		"""Ingests a single event.

		NOTE: Derived classes need to implement this.
		"""
		return NotImplementedError


	def handle(self):
		"""Handles ingested events.

		NOTE: Derived classes need to implement this.
		"""
		return NotImplementedError


	def finish(self):
		"""Finish remaining events when stopped and/or shutting down.

		NOTE: Derived classes may implement this if desired.
		"""
		pass


#-------------------------------------------------
### utilities

def append_args(parser):
	"""Append event processing specific options to an ArgumentParser instance.

	"""
	group = parser.add_argument_group("Event processing options")
	group.add_argument("--tag", metavar = "string", default = "default",
		help = "Sets the name of the tag used. Default = 'default'")
	group.add_argument("--processing-cadence", type = float, default = 0.1,
		help = "Rate at which the event uploader acquires and processes data. Default = 0.1 seconds.")
	group.add_argument("--request-timeout", type = float, default = 0.2,
		help = "Timeout for requesting messages from a topic. Default = 0.2 seconds.")
	group.add_argument("--kafka-server", metavar = "string",
		help = "Sets the server url that the kafka topic is hosted on. Required.")
	group.add_argument("--input-topic", metavar = "string", action = "append",
		help = "Sets the input kafka topic. Required.")

	return parser

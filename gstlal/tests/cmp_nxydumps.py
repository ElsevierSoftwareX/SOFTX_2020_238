#!/usr/bin/env python


from lal import LIGOTimeGPS


default_timestamp_fuzz = 1e-9	# seconds
default_sample_fuzz = 1e-15	# relative


def compare(filename1, filename2, timestamp_fuzz = default_timestamp_fuzz, sample_fuzz = default_sample_fuzz):
	timestamp_fuzz = LIGOTimeGPS(timestamp_fuzz)
	try:
		for n, (line1, line2) in enumerate(zip(open(filename1), open(filename2)), start = 1):
			line1 = line1.strip().split()
			line2 = line2.strip().split()
			line1 = [LIGOTimeGPS(line1[0])] +  map(float, line1[1:])
			line2 = [LIGOTimeGPS(line2[0])] +  map(float, line2[1:])

			assert abs(line1[0] - line2[0]) <= timestamp_fuzz
			for val1, val2 in zip(line1[1:], line2[1:]):
				assert abs(val1 - val2) / max(val1, val2) <= sample_fuzz
	except AssertionError as e:
		raise AssertionError("line %d: %s" % (n, str(e)))


if __name__ == "__main__":
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("--timestamp-fuzz", metavar = "seconds", type = "float", default = default_timestamp_fuzz)
	parser.add_option("--sample-fuzz", metavar = "fraction", type = "float", default = default_sample_fuzz)
	options, (filename1, filename2) = parser.parse_args()

	try:
		compare(filename1, filename2, timestamp_fuzz = options.timestamp_fuzz, sample_fuzz = options.sample_fuzz)
	except AssertionError as e:
		raise type(e)("%s <--> %s: %s" % (filename1, filename2, str(e)))

#!/usr/bin/env python


from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS


def compare(filename1, filename2):
	try:
		for n, (line1, line2) in enumerate(zip(open(filename1), open(filename2))):
			line1 = line1.strip().split()
			line2 = line2.strip().split()
			line1 = [LIGOTimeGPS(line1[0])] +  map(float, line1[1:])
			line2 = [LIGOTimeGPS(line2[0])] +  map(float, line2[1:])

			assert abs(line1[0] - line2[0]) <= options.timestamp_fuzz
			for val1, val2 in zip(line1[1:], line2[1:]):
				assert abs(val1 - val2) / max(val1, val2) <= options.sample_fuzz
	except AssertionError, e:
		raise AssertionError("line %d: %s" % (n + 1, str(e)))


if __name__ == "__main__":
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("--timestamp-fuzz", metavar = "seconds", type = "float", default = 1.0e-9)
	parser.add_option("--sample-fuzz", metavar = "fraction", type = "float", default = 1e-15)
	options, (filename1, filename2) = parser.parse_args()
	options.timestamp_fuzz = LIGOTimeGPS(options.timestamp_fuzz)

	try:
		compare(filename1, filename2)
	except AssertionError, e:
		raise type(e)("%s <--> %s: %s" % (filename1, filename2, str(e)))

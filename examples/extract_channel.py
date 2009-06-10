import sys

n = tuple(map(int, sys.argv[1:]))
for i, line in enumerate(sys.stdin):
	i += 1	# call the first line #1
	if not i % 1000:
		print >>sys.stderr, "line %d\r" % i,
	line = line.strip().split()
	print line[0], " ".join(map(line[1:].__getitem__, n))
print >>sys.stderr, "line %d" % i

import sys

n = (0,) + tuple(int(a) + 1 for a in sys.argv[1:])
for i, line in enumerate(sys.stdin):
	i += 1	# call the first line #1
	if not i % 1000:
		print >>sys.stderr, "line %d\r" % i,
	line = line.strip().split()
	print " ".join(line[c] for c in n)
print >>sys.stderr, "line %d" % i

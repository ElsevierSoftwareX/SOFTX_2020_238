import sys

n = tuple(map(int, sys.argv[1:]))
for line in sys.stdin:
	line = line.strip().split()
	print line[0], " ".join(map(line[1:].__getitem__, n))

import sys

n = map(int, sys.argv[1:])
if 0 not in n:
	n = [0] + n
n = tuple(n)
for line in sys.stdin:
	line = line.strip().split()
	print " ".join(map(line.__getitem__, n))

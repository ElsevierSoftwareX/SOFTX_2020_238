import itertools
import sys

snrout = file("snr2.txt", "w")
chisqout = file("chisquare2.txt", "w")
xiout = file("xi.txt", "w")

max_phase_snr = 0
max_snr = 0
for snrline, chisqline in itertools.izip(file("snr_L1.txt"), file("chisquare_L1.txt")):
	snrline = snrline.strip().split()
	chisqline = chisqline.strip().split()
	snr = tuple(map(float, snrline[1:]))
	chisq = tuple(map(float, chisqline[1:]))
	snr2 = tuple((a**2 + b**2)**.5 for (a, b) in itertools.izip(snr[0::2], snr[1::2]))
	chisq2 = tuple(a + b for (a, b) in itertools.izip(chisq[0::2], chisq[1::2]))
	xi = tuple(a / (1.0 + 0.1 * b**2) for (a, b) in itertools.izip(chisq2, snr2))
	for i, s in enumerate(snr):
		if abs(s) > max_phase_snr:
			max_phase_snr, max_phase_channel = abs(s), i // 2
	for i, s in enumerate(snr2):
		if s > max_snr:
			max_snr, max_channel = s, i
	print >>snrout, snrline[0], " ".join("%.16g" % x for x in snr2)
	print >>chisqout, chisqline[0], " ".join("%.16g" % x for x in chisq2)
	print >>xiout, snrline[0], " ".join("%.16g" % x for x in xi)

print >>sys.stderr, "max |snr| in a phase was %.16g in channel %d" % (max_phase_snr, max_phase_channel)
print >>sys.stderr, "max |snr| was %.16g in channel %d" % (max_snr, max_channel)

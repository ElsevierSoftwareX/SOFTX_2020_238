import sys
import numpy
import numpy.random
import lal
from gstlal.stats import inspiral_extrinsics

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

TPDPDF = inspiral_extrinsics.InspiralExtrinsics()

R = {"H":lal.CachedDetectors[lal.LHO_4K_DETECTOR].response,"L":lal.CachedDetectors[lal.LLO_4K_DETECTOR].response,"V":lal.CachedDetectors[lal.VIRGO_DETECTOR].response}
X = {"H":lal.CachedDetectors[lal.LHO_4K_DETECTOR].location,"L":lal.CachedDetectors[lal.LLO_4K_DETECTOR].location,"V":lal.CachedDetectors[lal.VIRGO_DETECTOR].location}

def random_source(R = R, X = X, epoch = lal.LIGOTimeGPS(0), gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(0))):

	D = 1
	cosiota = numpy.random.uniform(-1.0,1.0)
	ra = numpy.random.uniform(0.0,2.0*numpy.pi)
	dec = numpy.arcsin(numpy.random.uniform(-1.0,1.0))
	psi = numpy.random.uniform(0.0,2.0*numpy.pi)
	hplus = 0.5 * (1.0 + cosiota**2)
	hcross = cosiota

	F = dict((ifo, dict(zip(("+","x"), lal.ComputeDetAMResponse(R[ifo], ra, dec, psi, gmst)))) for ifo in R)

	# FINDCHIRP Eq. (3.3b)
	phi = dict((ifo, -numpy.arctan2(F[ifo]["x"] * hcross, F[ifo]["+"] * hplus)) for ifo in F)

	# FINDCHIRP Eq. (3.3c)
	Deff = dict((ifo, D / ((F[ifo]["+"] * hplus)**2 + (F[ifo]["x"] * hcross)**2)**0.5) for ifo in F)

	T = dict((ifo, lal.TimeDelayFromEarthCenter(X[ifo], ra, dec, epoch)) for ifo in F)

	return T, Deff, phi


ndraws = 1000000
delta_phi_HV = []
delta_t_HV = []
i = 0

while i < ndraws:
	t,d,p = random_source()
	# Use EXACTLY the same covariance matrix assumptions as the gstlal
	# code.  It could be a bad assumption, but we are testing the method
	# not the assumptions by doing this.
	dpHV = p["H"] - p["V"] + numpy.random.randn() * inspiral_extrinsics.TimePhaseSNR.sigma["phase"]
	dtHV = t["H"] - t["V"] + numpy.random.randn() * inspiral_extrinsics.TimePhaseSNR.sigma["time"]
	rdHV = d["H"] / d["V"] - 1.0 + numpy.random.randn() * inspiral_extrinsics.TimePhaseSNR.sigma["deff"]
	# only choose things with almost the same effective distance ratio for
	# this test to agree with setting the same horizon and snr below
	if 0.1 >= rdHV >= -0.1:
		delta_phi_HV.append(dpHV)
		delta_t_HV.append(dtHV)
		print i
		i += 1
		

num = 100
dphivec = numpy.linspace(-2 * numpy.pi, 2 * numpy.pi, num)
dtvec = numpy.linspace(-0.028, 0.028, num)
DPProb = numpy.zeros(num)
DTProb = numpy.zeros(num)

for j, dt in enumerate(dtvec):
	for i, dp in enumerate(dphivec):
		t = {"H1": 0, "V1": dt}
		phi = {"H1": 0, "V1": dp}
		snr = {"H1": 1., "V1": 1.}
		horizon = {"H1": 1., "V1": 1.}
		# signature is (time, phase, snr, horizon)
		p = TPDPDF.time_phase_snr(t, phi, snr, horizon)
		DPProb[i] += p
		DTProb[j] += p

pyplot.figure(figsize=(10,4))
pyplot.subplot(121)
pyplot.hist(delta_phi_HV, dphivec, normed = True)
pyplot.plot(dphivec, DPProb / numpy.sum(DPProb) / (dphivec[1] - dphivec[0]))
pyplot.xlabel("H phase - V phase")
pyplot.subplot(122)
pyplot.hist(delta_t_HV, dtvec, normed = True)
pyplot.plot(dtvec, DTProb / numpy.sum(DTProb) / (dtvec[1] - dtvec[0]))
pyplot.xlabel("H time - V time")
pyplot.savefig("HVtest.png")

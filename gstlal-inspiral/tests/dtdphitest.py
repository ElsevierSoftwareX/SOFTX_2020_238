import sys
import numpy
import numpy.random
import lal
from gstlal.stats import inspiral_extrinsics

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['lines.markersize'] = 2
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')
matplotlib.rcParams['legend.fontsize'] = 10
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


ndraws = 100000 #100000 maybe take 10min to iterate
delta_phi_HV = []
delta_t_HV = []
i = 0

# the following cov matrix elements were derived from running `gstlal_inspiral_compute_dtdphideff_cov_matrix --psd-xml share/O3/2019-05-09-H1L1V1psd_new.xml.gz --H-snr 5 --L-snr 7.0 --V-snr 2.25`
covmat = [[0.000001, 0.000610], [0.000610, 0.648023]]
sigmadd = 0.487371
while i < ndraws:
	t,d,p = random_source()
	# Use EXACTLY the same covariance matrix assumptions as the gstlal
	# code.  It could be a bad assumption, but we are testing the method
	# not the assumptions by doing this.
	mean = [t["H"] - t["V"], p["H"] - p["V"]]
	dtHV, dpHV = numpy.random.multivariate_normal(mean, covmat)
	rdHV = numpy.log(d["H"] / d["V"]) + numpy.random.randn() * sigmadd
	# only choose things with almost the same effective distance ratio for
	# this test to agree with setting the same horizon and snr below
	if numpy.log(1.1 + 0.1) >= rdHV >= numpy.log(1.1 - 0.1): # 1.1 here comes from Deff_V1 / Deff_H1 = (110/5) / (45/2.25)
		delta_phi_HV.append(dpHV)
		delta_t_HV.append(dtHV)
		print i
		i += 1


num1 = 100
num2 = 101
dtvec = numpy.linspace(-0.028, 0.028, num1)
dphivec = numpy.linspace(-2 * numpy.pi, 2 * numpy.pi, num2)
DTProb = numpy.zeros(num1)
DPProb = numpy.zeros(num2)
Prob = numpy.zeros((num1,num2))

for j, dt in enumerate(dtvec):
	for i, dp in enumerate(dphivec):
		t = {"H1": 0, "V1": dt}
		phi = {"H1": 0, "V1": dp}
		snr = {"H1": 5., "V1": 2.25} # our choice of characteristic SNR
		horizon = {"H1": 110., "V1": 45.} # typical horizon distance taken from the summary page on 2019 May 09.
		# signature is (time, phase, snr, horizon)
		p = TPDPDF.time_phase_snr(t, phi, snr, horizon)
		Prob[j,i] = p
		DPProb[i] += p
		DTProb[j] += p

pyplot.figure(figsize=(7.5,7.5))
pyplot.subplot(221)
pyplot.hist(delta_phi_HV, dphivec, normed = True, label = "Simulation")
pyplot.plot(dphivec, DPProb / numpy.sum(DPProb) / (dphivec[1] - dphivec[0]), label ="Direct Calculation")
pyplot.ylabel(r"""$P(\Delta\phi | s, D_H = D_V, \rho_H = \rho_V)$""")
pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
pyplot.subplot(223)
pyplot.pcolor(dphivec, dtvec, Prob)
pyplot.xlabel(r"""$\phi_H - \phi_V$""")
pyplot.ylabel(r"""$t_H - t_V$""")
pyplot.subplot(224)
pyplot.hist(delta_t_HV, dtvec, normed = True, orientation = "horizontal")
pyplot.plot(DTProb / numpy.sum(DTProb) / (dtvec[1] - dtvec[0]), dtvec)
pyplot.xlabel(r"""$P(\Delta t | s, D_H = D_V, \rho_H = \rho_V)$""")
pyplot.savefig("HVPDF.pdf")

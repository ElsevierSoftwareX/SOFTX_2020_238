import sys
import numpy
import numpy.random
import scipy
from scipy import interpolate as interp
from scipy import stats
import lal
from gstlal.stats import inspiral_extrinsics
import h5py

import pdb

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
R = {"H1":lal.CachedDetectors[lal.LHO_4K_DETECTOR].response,"L1":lal.CachedDetectors[lal.LLO_4K_DETECTOR].response,"V1":lal.CachedDetectors[lal.VIRGO_DETECTOR].response,"K1":lal.CachedDetectors[lal.KAGRA_DETECTOR].response}
X = {"H1":lal.CachedDetectors[lal.LHO_4K_DETECTOR].location,"L1":lal.CachedDetectors[lal.LLO_4K_DETECTOR].location,"V1":lal.CachedDetectors[lal.VIRGO_DETECTOR].location,"K1":lal.CachedDetectors[lal.KAGRA_DETECTOR].location}
refhorizon = {"H1": 110., "L1": 140., "V1": 45., "K1": 45.}

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

def dtdphi2prob(dt, dphi, ifo1, ifo2, refsnr, refhorizon=refhorizon):
	"""function that takes dt,dphi samples and return the probability (density) values derived from the new pdf

	:dt: the time delay between detector pair
	:dphi: the difference of the coalescence phase between detector pair
	:returns: dtdpipdf(dt, dphi) * snr^4 (assuming snr = {"H1": 5., "V1": 2.25} and horizon = {"H1": 110., "V1": 45.})

	"""

	# ifo1 += "1"
	# ifo2 += "1"
	t = {ifo1: 0, ifo2: dt}
	phi = {ifo1: 0, ifo2: dphi}
	snr = {ifo1: refsnr[ifo1], ifo2: refsnr[ifo2]} # our choice of characteristic SNR
	horizon = {ifo1: refhorizon[ifo1], ifo2: refhorizon[ifo2]} # typical horizon distance taken from the summary page on 2019 May 09.
	# signature is (time, phase, snr, horizon)
	p = TPDPDF.time_phase_snr(t, phi, snr, horizon) * (sum(s**2 for s in snr.values())**.5)**4
	return float(p)

pdf_fname = sys.argv[1]
ifo1, ifo2 = sorted(sys.argv[2].split(","))
combo = ifo1 + "," + ifo2

ndraws = 100000#100000 maybe take 10min to iterate
delta_phi_list = []
delta_t_list = []
prob_simulation = []
i = 0

# the following cov matrix elements were derived from running `gstlal_inspiral_compute_dtdphideff_cov_matrix --psd-xml share/O3/2019-05-09-H1L1V1psd_new.xml.gz --H-snr 5 --L-snr 7.0 --V-snr 4.0 --K-snr 4.0`
data = h5py.File(pdf_fname)
# pdb.set_trace()
refsnr = dict((ifo, data["SNR"][ifo].value) for ifo in data["SNR"])
transmat_dtdphi = numpy.array([[data["transtt"][combo].value, data["transtp"][combo].value], [data["transpt"][combo].value, data["transpp"][combo].value]])
covmat_dtdphi = numpy.linalg.inv(numpy.dot(transmat_dtdphi.T, transmat_dtdphi))
sigmadd = 1. / (data["transdd"][combo].value)**2
rd_slice = (refhorizon[ifo1] / refsnr[ifo1]) / (refhorizon[ifo2] / refsnr[ifo2])
while i < ndraws:
	t,d,p = random_source()
	# Use EXACTLY the same covariance matrix assumptions as the gstlal
	# code.  It could be a bad assumption, but we are testing the method
	# not the assumptions by doing this.
	mean = [t[ifo1] - t[ifo2], p[ifo1] - p[ifo2]]
	dt, dp = numpy.random.multivariate_normal(mean, covmat_dtdphi)
	rd = numpy.log(d[ifo1] / d[ifo2]) + numpy.random.randn() * numpy.sqrt(sigmadd)
	# only choose things with almost the same effective distance ratio for
	# this test to agree with setting the same horizon and snr below
	if numpy.log(rd_slice + 0.05) >= rd >= numpy.log(rd_slice - 0.05):
		delta_phi_list.append(dp)
		delta_t_list.append(dt)
		prob_simulation.append(dtdphi2prob(dt, dp, ifo1, ifo2, refsnr, refhorizon))
		i += 1
prob_simulation.sort()

num1 = 100
num2 = 101
dtvec = numpy.linspace(-0.028, 0.028, num1)
dphivec = numpy.linspace(-2 * numpy.pi, 2 * numpy.pi, num2)
DTProb = numpy.zeros(num1)
DPProb = numpy.zeros(num2)
Prob = numpy.zeros((num1,num2))

for j, dt in enumerate(dtvec):
	for i, dp in enumerate(dphivec):
		p = dtdphi2prob(dt, dp, ifo1, ifo2, refsnr, refhorizon)
		Prob[j,i] = p
		DPProb[i] += p
		DTProb[j] += p

pyplot.figure(figsize=(7.5,7.5))
pyplot.subplot(221)
pyplot.hist(delta_phi_list, dphivec, normed = True, label = "Simulation")
pyplot.plot(dphivec, DPProb / numpy.sum(DPProb) / (dphivec[1] - dphivec[0]), label ="Direct Calculation")
pyplot.ylabel(r"""$P(\Delta\phi | s, \{D_{%s}, D_{%s}\}, \{\rho_{%s}, \rho_{%s}\})$""" % (ifo1, ifo2, ifo1, ifo2))
pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
pyplot.subplot(223)
pyplot.pcolor(dphivec, dtvec, Prob)
pyplot.xlabel(r"""$\phi_{%s} - \phi_{%s}$""" % (ifo1, ifo2))
pyplot.ylabel(r"""$t_{%s} - t_{%s}$""" % (ifo1, ifo2))
pyplot.subplot(224)
pyplot.hist(delta_t_list, dtvec, normed = True, orientation = "horizontal")
pyplot.plot(DTProb / numpy.sum(DTProb) / (dtvec[1] - dtvec[0]), dtvec)
pyplot.xlabel(r"""$P(\Delta t | s, \{D_{%s}, D_{%s}\}, \{\rho_{%s}, \rho_{%s}\})$""" % (ifo1, ifo2, ifo1, ifo2))
pyplot.savefig("%s%sPDF.pdf" % (ifo1, ifo2))

if False:
	prob_grid = numpy.sort(Prob.flatten())
	percentiles_grid = numpy.cumsum(prob_grid)
	percentiles_grid /= percentiles_grid[-1]
	p_interp = interp.interp1d(prob_grid, percentiles_grid)
	# plot of the interpolate function of percentile to see if it is a good approximation
	fig = pyplot.figure()
	ax1 = fig.add_subplot(111)
	ax1.loglog(prob_grid, percentiles_grid, linestyle="None", color="r", marker=".")
	ax1.loglog(prob_grid, p_interp(prob_grid))
	pyplot.savefig("./percentile_interp_check.png")
	# make pp-plot from the interpolate function of percentiles
	try:
		percentiles_exp = sorted(p_interp(prob_simulation))
	except ValueError:
		prob_simulation = numpy.array(prob_simulation)
		percentiles_exp = numpy.empty(len(prob_simulation))
		mask_ok = [(prob_simulation < max(prob_grid)) & (prob_simulation > min(prob_grid))]
		mask_above = [prob_simulation > max(prob_grid)]
		mask_below = [prob_simulation < min(prob_grid)]
		percentiles_exp[mask_ok] = sorted(p_interp(prob_simulation[mask_ok]))
		percentiles_exp[mask_above] = 1.
		percentiles_exp[mask_below] = 0
	percentiles_inj = numpy.cumsum(numpy.ones(len(percentiles_exp))) / len(percentiles_exp)
	fig = pyplot.figure()
	ax2 = fig.add_subplot(111)
	ax2.plot([0,1], [0, 1], linestyle="--")
	ax2.plot(percentiles_exp, percentiles_inj, linestyle="-", color="r", label="simulation")
	CI = 0.9
	quant = stats.norm.ppf(CI * 0.5 + 0.5)
	err = quant * numpy.sqrt(percentiles_inj * (1 - percentiles_inj) / len(percentiles_inj))
	ax2.fill_between(percentiles_exp, percentiles_inj - err, percentiles_inj + err, facecolor='r', label="90$\%$ measurement uncertainty", alpha=.3)
	pyplot.legend(loc = "lower right")
	ax2.set_xlabel("Percentile computed from the pdf")
	ax2.set_ylabel("Percentile in the injection set")
	pyplot.savefig("./%s%s_pp_plot.pdf" % (ifo1, ifo2))

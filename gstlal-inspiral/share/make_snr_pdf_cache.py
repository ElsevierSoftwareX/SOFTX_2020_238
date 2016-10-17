from glue import iterutils
from glue.ligolw import ligolw
from glue.ligolw import utils as ligolw_utils
from glue.text_progress_bar import ProgressBar
from gstlal import far
from gstlal.stats import inspiral_extrinsics

def all_instrument_combos(instruments, min_number):
	for n in range(min_number, len(instruments) + 1):
		for combo in iterutils.choices(instruments, n):
			yield combo

diststats, _, segs = far.parse_likelihood_control_doc(ligolw_utils.load_filename("all_o1_likelihood.xml.gz", contenthandler = far.ThincaCoincParamsDistributions.LIGOLWContentHandler, verbose = True))
#diststats.finish(segs, verbose = True)
if set(diststats.horizon_history) != set(diststats.instruments):
	raise ValueError("horizon histories are for %s, need for %s" % (", ".join(sorted(diststats.horizon_history)), ", ".join(sorted(diststats.instruments)))

snrpdf = inspiral_extrinsics.SNRPDF(diststats.snr_min)
snrpdf.snr_joint_pdf_cache.clear()

all_horizon_distances = diststats.horizon_history.all()
with ProgressBar(max = len(all_horizon_distances)) as progressbar:
	for horizon_distances in all_horizon_distances:
		progressbar.increment(text = ",".join(sorted("%s=%.5g" % item for item in horizon_distances.items())))
		progressbar.show()
		for off_instruments in all_instrument_combos(horizon_distances, 0):
			dists = horizon_distances.copy()
			for instrument in off_instruments:
				dists[instrument] = 0.
			for instruments in all_instrument_combos(diststats.instruments, diststats.min_instruments):
				snrpdf.add_to_cache(instruments, dists, verbose = True)

xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW()).appendChild(snrpdf.to_xml())
ligolw_utils.write_filename(xmldoc, "inspiral_snr_pdf.xml.gz", gz = True, verbose = True)

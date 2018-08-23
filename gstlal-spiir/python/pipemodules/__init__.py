#
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = 'Qi Chu <qi.chu@ligo.org>'
__date__ = '$Date$'
__cvs_tag__ = '$Name$'
try:
  __version__ = __cvs_tag__.split('-')[1] + '.' + __cvs_tag__.split('-')[2][0:-2]
except IndexError:
  __version__ = ''

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

from gstlal import pipeio
from gstlal import pipeparts
from gstlal.pipemodules import pipe_macro
#
# SPIIR many instruments, many template banks
#

def mktimeshift(pipeline, src, shift):
	properties = {
		"shift": shift
	}

	return pipeparts.mkgeneric(pipeline, src, "control_timeshift", **properties)

def mkitac_spearman(pipeline, src, n, bank, autocorrelation_matrix = None, mask_matrix = None, snr_thresh = 0, sigmasq = None):
	properties = {
		"n": n,
		"bank_filename": bank,
		"snr_thresh": snr_thresh
	}
	if autocorrelation_matrix is not None:
		properties["autocorrelation_matrix"] = pipeio.repack_complex_array_to_real(autocorrelation_matrix)
	if mask_matrix is not None:
		properties["autocorrelation_mask"] = mask_matrix
	if sigmasq is not None:
		properties["sigmasq"] = sigmasq
	return pipeparts.mkgeneric(pipeline, src, "lal_itac_spearman", **properties)

def mkcudaiirbank(pipeline, src, a1, b0, delay, name = None):
 	properties = dict((name, value) for name, value in (("name", name), ("delay_matrix", delay)) if value is not None)
 	if a1 is not None:
 		properties["a1_matrix"] = pipeio.repack_complex_array_to_real(a1)
 	if b0 is not None:
 		properties["b0_matrix"] = pipeio.repack_complex_array_to_real(b0)
 	elem = pipeparts.mkgeneric(pipeline, src, "cuda_iirbank", **properties)
 	elem = pipeparts.mknofakedisconts(pipeline, elem)	# FIXME:  remove after basetransform behaviour fixed
 	return elem


def mkcudamultiratespiir(pipeline, src, bank_fname, gap_handle = 0, stream_id = 0, name = None):
	properties = dict((name, value) for name, value in (("name", name), ("bank_fname", bank_fname), ("gap_handle", gap_handle), ("stream_id", stream_id)) if value is not None)
	elem = pipeparts.mkgeneric(pipeline, src, "cuda_multiratespiir", **properties)
	return elem


def mkcudapostcoh(pipeline, snr, instrument, detrsp_fname, autocorrelation_fname, sngl_tmplt_fname, hist_trials = 1, snglsnr_thresh = 4.0, cohsnr_thresh = 5.0, output_skymap = 0, detrsp_refresh_interval = 0, trial_interval = 0.1, stream_id = 0):
	properties = dict((name, value) for name, value in zip(("detrsp-fname", "autocorrelation-fname", "sngl-tmplt-fname", "hist-trials", "snglsnr-thresh", "cohsnr_thresh", "output-skymap", "detrsp-refresh-interval", "trial-interval", "stream-id"), (detrsp_fname, autocorrelation_fname, sngl_tmplt_fname, hist_trials, snglsnr_thresh, cohsnr_thresh, output_skymap, detrsp_refresh_interval, trial_interval, stream_id)))
	if "name" in properties:
		elem = gst.element_factory_make("cuda_postcoh", properties.pop("name"))
	else:
		elem = gst.element_factory_make("cuda_postcoh")
	# make sure stream_id go first
	for name, value in properties.items():
		if name == "stream-id":
			elem.set_property(name.replace("_", "-"), value)
	for name, value in properties.items():
		if name != "stream-id":
			elem.set_property(name.replace("_", "-"), value)

	pipeline.add(elem)
	snr.link_pads(None, elem, instrument)
	return elem


def mkcohfar_accumbackground(pipeline, src, ifos= "H1L1", hist_trials = 1, snapshot_interval = 0, history_fname = None, output_prefix = None, output_name = None, source_type = pipe_macro.SOURCE_TYPE_BNS):
	properties = {
		"ifos": ifos,
		"snapshot_interval": snapshot_interval,
		"hist_trials": hist_trials,
		"source_type": source_type
	}
	if history_fname is not None:
		properties["history_fname"] = history_fname
	if output_prefix is not None:
		properties["output_prefix"] = output_prefix
	if output_name is not None:
		properties["output_name"] = output_name

	print "source type %d" % source_type
	if "name" in properties:
		elem = gst.element_factory_make("cohfar_accumbackground", properties.pop("name"))
	else:
		elem = gst.element_factory_make("cohfar_accumbackground")
	# make sure ifos go first
	for name, value in properties.items():
		if name == "ifos":
			elem.set_property(name.replace("_", "-"), value)
	for name, value in properties.items():
		if name is not "ifos":
			elem.set_property(name.replace("_", "-"), value)

	pipeline.add(elem)
	if isinstance(src, gst.Pad):
		src.get_parent_element().link_pads(src, elem, None)
	elif src is not None:
		src.link(elem)
	return elem

def mkcohfar_assignfar(pipeline, src, ifos= "H1L1", assignfar_refresh_interval = 14400, silent_time = 2147483647, input_fname = None):
	properties = {
		"ifos": ifos,
		"refresh_interval": assignfar_refresh_interval,
		"silent_time": silent_time,
	}
	if input_fname is not None:
		properties["input_fname"] = input_fname

	if "name" in properties:
		elem = gst.element_factory_make("cohfar_assignfar", properties.pop("name"))
	else:
		elem = gst.element_factory_make("cohfar_assignfar")
	# make sure ifos go first
	for name, value in properties.items():
		if name == "ifos":
			elem.set_property(name.replace("_", "-"), value)
	for name, value in properties.items():
		if name != "ifos":
			elem.set_property(name.replace("_", "-"), value)

	pipeline.add(elem)
	if isinstance(src, gst.Pad):
		src.get_parent_element().link_pads(src, elem, None)
	elif src is not None:
		src.link(elem)
	return elem


def mkpostcohfilesink(pipeline, postcoh, location = ".", compression = 1, snapshot_interval = 0):
	properties = dict((name, value) for name, value in zip(("location", "compression", "snapshot-interval", "sync", "async"), (location, compression, snapshot_interval, False, False)))
	if "name" in properties:
		elem = gst.element_factory_make("postcoh_filesink", properties.pop("name"))
	else:
		elem = gst.element_factory_make("postcoh_filesink")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	postcoh.link(elem)
	return elem



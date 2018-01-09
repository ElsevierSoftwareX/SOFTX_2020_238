# Copyright (C) 2017 Qi Chu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

## @file
# The python module to implement false alarm probability and false alarm rate
#
#
#

# need to update the name when the background_utils.h update
import numpy as np
import math
from scipy import interpolate
# FIXME remove this when the LDG upgrades scipy on the SL6 systems, Debian
# systems are already fine
try:
	from scipy.optimize import curve_fit
except ImportError:
	from gstlal.curve_fit import curve_fit

import logging
from glue import iterutils
from glue.ligolw import ligolw, lsctables, array, param, utils, types
from glue.ligolw.utils import segments as ligolw_segments
from gstlal.postcoh_table_def import PostcohInspiralTable
import pdb

Attributes = ligolw.sax.xmlreader.AttributesImpl


# FIXME:  require calling code to provide the content handler
class DefaultContentHandler(ligolw.LIGOLWContentHandler):
  pass
array.use_in(DefaultContentHandler)
param.use_in(DefaultContentHandler)
lsctables.use_in(DefaultContentHandler)

def get_livetime(options, on_instruments, verbose = False):

	# get the segment lists
	segments_xmldoc = utils.load_filename(options.segments_file, contenthandler = DefaultContentHandler, verbose = options.verbose)
	seglists = ligolw_segments.segmenttable_get_by_name(segments_xmldoc, options.segments_name).coalesce()
	if options.shrink_data_segments is not None:
		# Make sure all segments are at least long enough to be
		# contracted (e.g. if a segment is 32 seconds long but
		# the user is going to contract it by 17 seconds, then
		# the end time of the segment would come before the
		# start time of the segment. This is bad)
		for seglist in seglists.values():
			iterutils.inplace_filter(lambda s: abs(s) > 2*options.shrink_data_segments, seglist)
		seglists.contract(options.shrink_data_segments)
	instruments = set(seglists)
	
	# get the veto lists
	vetoes_name = options.vetoes_name
	if vetoes_name is not None:
		vetoes_xmldoc = utils.load_filename(options.vetoes_file, contenthandler = DefaultContentHandler, verbose = options.verbose)
		veto_segments = ligolw_segments.segmenttable_get_by_name(vetoes_xmldoc, vetoes_name).coalesce()
		if options.extend_veto_segments is not None:
			veto_segments.protract(options.extend_veto_segments)
	else:
		veto_segments = segments.segmentlistdict()
	seglists -= veto_segments


	# get the live time
	zerolag_livetime = float(abs(seglists.intersection(on_instruments) - seglists.union(instruments - on_instruments)))
	if verbose:
		print >>sys.stderr, "calculated zerolag livetimes %f " % zerolag_livetime


	return zerolag_livetime


def array_from_xml(filename, array_name, contenthandler = DefaultContentHandler, verbose = False):

  # Load document
  xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)

  for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_spiir_cohfar"):
    result = array.get_array(root, array_name).array 
  return result

def postcoh_table_from_xml(filename, contenthandler = DefaultContentHandler, verbose = False):
 
   # Load document
   xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)
 
   postcoh = PostcohInspiralTable.get_table(xmldoc)
   return postcoh


class RankingData(object):

  def __init__(self, stats_filename, ifos, hist_trials = 100, verbose = None):
    # xml_snr_min = 0.54
    # xml_snr_step = 0.0082
    # xml_snr_nbin = 299
    # xml_chisq_min = -1.2
    # xml_chisq_step = 0.0173
    # xml_chisq_nbin = 299
    # FIXME: bin prop hard-coded, check the gst/cuda/cohfar/background_stats.h for the values
    self.step_snr = (3.0-0.54)/299
    self.snr_min = 0.54 - self.step_snr/2.
    self.step_chisq = (4.0+1.2)/299
    self.chisq_min = -1.2 - self.step_chisq/2.
    self.nbin = 300
    step_rank = 30.0/299
    self.rank_bounds = np.linspace(-30.0-step_rank/2., 0+step_rank/2., 301)
    self.rank_centers = np.linspace(-30.0, 0., 300)
    rank_map_name = "background_rank:%s_rank_map:array" % ifos
    rank_pdf_name = "background_rank:%s_rank_pdf:array" % ifos
    rank_rates_name = "background_rank:%s_rank_rates:array" % ifos


    self.rank_map = array_from_xml(stats_filename, rank_map_name)
    self.rank_pdf = array_from_xml(stats_filename, rank_pdf_name)
    self.rank_rates = array_from_xml(stats_filename, rank_rates_name)
    self.back_nevent = self.rank_rates.sum()
    if verbose:
      logging.info("background events %d" % self.back_nevent)
      logging.info("calc fap from rates")

    #crates = np.zeros((300, 300));
    #for snr_idx in range(299, -1, -1):
    #  for chisq_idx in range(1,301):
    #    crates[chisq_idx-1, snr_idx] = rates[0:chisq_idx, snr_idx:300].sum()
    #crates[crates==0] = 1
    #cdf_rates = crates/float(back_nevent)
    #min_cdf = cdf_rates.min()
    
    
    #lgcdf_rates = np.log10(cdf_rates)
    #min_lgcdf = max(-7, lgcdf_rates.min())
    #max_lgcdf = lgcdf_rates.max()
    
    
    self.hist_trials = hist_trials
    #tick_lgcdf = np.linspace(min_lgcdf, max_lgcdf, num=nstep)
    
    #back_lgfap_rates = np.zeros(len(tick_lgcdf))
    #back_lgfap_kde = np.zeros(len(tick_lgcdf))
    #
    #self.far_kde = np.zeros((300, 300));
    
    #for snr_idx in range(299, -1, -1):
    #  for chisq_idx in range(1,301):
    #    lgfar_rates[chisq_idx-1, snr_idx] = np.log10(float(1 + sum(rates[np.where(lgcdf_rates < lgcdf_rates[chisq_idx-1, snr_idx])])) / (nslide * coinc_time))
      
    #self.far_kde = self.fap_kde * self.back_nevent / (self.hist_trials * self.livetime)
    
class FAPFAR(object):
  def __init__(self, ranking_stats, connection, livetime = None):
    self.livetime = livetime
    self.ranking_stats = ranking_stats
    self.connection = connection
    pdb.set_trace()
    # construct zerolag rate distribution
    self.zlag_rates = np.zeros(300)
    self.count_zlag_rates()

    self.nzlag = self.zlag_rates.sum()

    extinct_pdf = self.extinct(self.ranking_stats.rank_rates, self.ranking_stats.rank_pdf, self.zlag_rates, self.ranking_stats.rank_centers)
    drank = self.ranking_stats.rank_bounds[3] - self.ranking_stats.rank_bounds[2]
    self.min_rank = min(self.ranking_stats.rank_centers)
    self.max_rank = max(self.ranking_stats.rank_centers)

    # cumulative distribution function and its complement.
    # it's numerically better to recompute the ccdf by
    # reversing the array of weights than trying to subtract
    # the cdf from 1.
    weights = extinct_pdf * drank
    cdf = weights.cumsum()
    cdf /= cdf[-1]
    ccdf = weights[::-1].cumsum()[::-1]
    ccdf /= ccdf[0]

    # cdf boundary condition:  cdf = 1/e at the ranking
    # statistic threshold so that self.far_from_rank(threshold)
    # * livetime = observed count of events above threshold.
    # FIXME this doesn't actually work.
    # FIXME not doing it doesn't actually work.
    # ccdf *= 1. - 1. / math.e
    # cdf *= 1. - 1. / math.e
    # cdf += 1. / math.e

    # last checks that the CDF and CCDF are OK
    assert not np.isnan(cdf).any(), "Rank CDF contains NaNs"
    assert not np.isnan(ccdf).any(), "Rank CCDF contains NaNs"
    assert ((0. <= cdf) & (cdf <= 1.)).all(), "Rank CDF failed to be normalized"
    assert ((0. <= ccdf) & (ccdf <= 1.)).all(), "Rank CCDF failed to be normalized"
    assert (abs(1. - (cdf[:-1] + ccdf[1:])) < 1e-12).all(), "Rank CDF + CCDF != 1 (max error = %g)" % abs(1. - (cdf[:-1] + ccdf[1:])).max()

    # build interpolators
    self.cdf_interpolator = interpolate.interp1d(self.ranking_stats.rank_centers, cdf)
    self.ccdf_interpolator = interpolate.interp1d(self.ranking_stats.rank_centers, ccdf)

  def count_zlag_rates(self):
    # use the far field for tempory rank assignment
    self.connection.create_function("rank_from_features", 2, self.rank_from_features)
    cur = self.connection.cursor()
    cur.execute(""" UPDATE postcoh SET
  far =  rank_from_features(cohsnr, cmbchisq)
  """)
    # count the rate of rank in a given range
    self.connection.commit()
    try:
        import sqlite3
        use_sqlite3 = True
    except ImportError:
        use_sqlite3 = False
    for irank, rank_min in enumerate(self.ranking_stats.rank_bounds):
      if irank < len(self.ranking_stats.rank_bounds) -1:
        rank_max = self.ranking_stats.rank_bounds[irank+1]
        if use_sqlite3:
            cur.execute(""" SELECT COUNT(*) FROM 
            postcoh WHERE far > ? and far <= ?""", (rank_min, rank_max)
            )
            self.zlag_rates[irank] = cur.fetchone()[0]
        else:
            cur.execute(""" SELECT COUNT(*) FROM 
            postcoh WHERE far > %s and far <= %s""", (rank_min, rank_max)
            )
            self.zlag_rates[irank] = cur.fetchone()[0]

  def extinct(self, bgcounts_ba_array, bgpdf_ba_array, zlagcounts_ba_array, ranks):
    # Generate arrays of complementary cumulative counts
    # for background events (monte carlo, pre clustering)
    # and zero lag events (observed, post clustering)
    zero_lag_compcumcount = zlagcounts_ba_array.cumsum()
    bg_compcumcount = bgcounts_ba_array.cumsum()

    # Fit for the number of preclustered, independent coincs by
    # only considering the observed counts safely in the bulk of
    # the distribution.  Only do the fit above 10 counts and below
    # 10000, unless that can't be met and trigger a warning
    fit_min_rank = -30.
    fit_min_counts = max(10., self.nzlag / 10. + 1)
    fit_max_counts = max(10000., self.nzlag / 1.) # the +2 gaurantees that fit_max_counts > fit_min_counts
    rank_range = np.logical_and(ranks > fit_min_rank, np.logical_and(zero_lag_compcumcount < fit_max_counts, zero_lag_compcumcount > fit_min_counts))
    if fit_max_counts < 10000.:
      warnings.warn("There are less than 10000 coincidences, extinction effects on background may not be accurately calculated, which will decrease the accuracy of the combined instruments background estimation.")
    if zero_lag_compcumcount.compress(rank_range).size < 1:
      raise ValueError("not enough zero lag data to fit background")

    # Use curve fit to find the predicted total preclustering
    # count. First we need an interpolator of the counts
    obs_counts = interpolate.interp1d(ranks, bg_compcumcount)
    bg_pdf_interp = interpolate.interp1d(ranks, bgpdf_ba_array)


    def extincted_counts(x, N_ratio):
        # NOTE: the following does not work for mdc data
      #out = max(zero_lag_compcumcount) * (1. - np.exp(-obs_counts(x) * N_ratio))
      out = obs_counts(x) * N_ratio
      out[~np.isfinite(out)] = 0.
      return out

    def extincted_pdf(x, N_ratio):
        # NOTE: the following does not work for mdc data
      #out = np.exp(np.log(N_ratio) - obs_counts(x) * N_ratio + np.log(bg_pdf_interp(x)))
      out = bg_pdf_interp(x) * N_ratio
      out[~np.isfinite(out)] = 0.
      return out

    # Fit for the ratio of unclustered to clustered triggers.
    # Only fit N_ratio over the range of ranks decided above 
    precluster_normalization, precluster_covariance_matrix = curve_fit(
      extincted_counts,
      ranks[rank_range],
      zero_lag_compcumcount.compress(rank_range),
      sigma = zero_lag_compcumcount.compress(rank_range)**.5,
      p0 = 1e-4
    )

    N_ratio = precluster_normalization[0]

    return extincted_pdf(ranks, N_ratio)


  def rank_from_features(self, snr, chisq):
    lgsnr = np.log10(snr)
    lgchisq = np.log10(chisq)
    snr_idx = min(max(int((lgsnr - self.ranking_stats.snr_min)/self.ranking_stats.step_snr), 0), self.ranking_stats.nbin-1) 
    chisq_idx = min(max(int((lgchisq - self.ranking_stats.chisq_min)/self.ranking_stats.step_chisq), 0), self.ranking_stats.nbin-1) 
    return self.ranking_stats.rank_map[chisq_idx, snr_idx]
  
  def fap_from_features(self, snr, chisq):
    rank = rank_from_features(snr, chisq)
    # implements equation (B4) of Phys. Rev. D 88, 024025.
    # arXiv:1209.0718.  the return value is divided by T to
    # convert events/experiment to events/second.
    rank = max(self.min_rank, min(self.max_rank, rank))
    fap = float(self.ccdf_interpolator(rank))
    return fap_after_trials(fap, self.nzlag)
  

  def assign_fars_sql(self, connection, mul_factor = 1):
    # assign false-alarm rates
    # FIXME:  choose a function name more likely to be unique?
    connection.create_function("far_from_rank", 2, self.far_from_rank)
    cur = self.connection.cursor()
    cur.execute("""
UPDATE
  postcoh
SET
  far =  far_from_rank(far, ?)
""", mul_factor)

  # FIXME: see Kipp's code to adjust fap for clustered zerolag
  # the mul_factor is used to combine far from different sqls
  def far_from_rank(self, rank, mul_factor):
    # implements equation (B4) of Phys. Rev. D 88, 024025.
    # arXiv:1209.0718.  the return value is divided by T to
    # convert events/experiment to events/second.
    assert self.livetime is not None, "cannot compute FAR without livetime"
    rank = max(self.min_rank, min(self.max_rank, rank))
    # true-dismissal probability = 1 - single-event false-alarm
    # probability, the integral in equation (B4)
    # NOTE: the following does not work for mdc data
  
    #tdp = float(self.cdf_interpolator(rank))
    #try:
    #  log_tdp = math.log(tdp)
    #except ValueError:
    #  # TDP = 0 --> FAR = +inf
    #  return PosInf
    #if log_tdp >= -1e-9:
    #  # rare event:  avoid underflow by using log1p(-FAP)
    #  log_tdp = math.log1p(-float(self.ccdf_interpolator(rank)))
    #return self.nzlag * -log_tdp / self.livetime
    fap = float(self.cdf_interpolator(rank))
    return self.nzlag * fap / self.livetime * mul_factor


def count_above_ifar_xml(zerolag_fname_xml, tick_lgifar):
  
  zerolag_fname_list = zerolag_fname_xml.split(',')
  all_table = lsctables.New(PostcohInspiralTable)

  for ifname in zerolag_fname_list:
    table = postcoh_table_from_xml(ifname)
    all_table.extend(table)
    iterutils.inplace_filter(lambda row:row.is_background == 0, all_table)

  zerolag_lgifar = - np.log10(all_table.getColumnByName("far"))
  zerolag_lgcevent = np.zeros(len(tick_lgifar))
    
  for itick in range(0, len(tick_lgifar)):
    cevent = len(zerolag_lgifar[np.where(zerolag_lgifar > tick_lgifar[itick])])
    if cevent > 0:
      zerolag_lgcevent[itick] = np.log10(cevent)
  return zerolag_lgcevent
    
def count_above_ifar_sql(zerolag_connection, tick_lgifar):
  
  zerolag_lgcevent = np.zeros(len(tick_lgifar))
  cur = zerolag_connection.cursor()    
  for itick in range(0, len(tick_lgifar)):
    far_min = 10**(-tick_lgifar[itick])
    cur.execute(""" SELECT COUNT(*) FROM 
            postcoh WHERE far <= ?""", (far_min,)
    )
    nevent = cur.fetchone()[0]
    if nevent > 0:
      zerolag_lgcevent[itick] = np.log10(nevent)
  return zerolag_lgcevent
    










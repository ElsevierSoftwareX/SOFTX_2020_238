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
import logging
from glue.ligolw import ligolw, lsctables, array, param, utils, types
from gstlal.postcoh_table_def import PostcohInspiralTable


Attributes = ligolw.sax.xmlreader.AttributesImpl


# FIXME:  require calling code to provide the content handler
class DefaultContentHandler(ligolw.LIGOLWContentHandler):
	pass
array.use_in(DefaultContentHandler)
param.use_in(DefaultContentHandler)
lsctables.use_in(DefaultContentHandler)



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

  def __init__(self, stats_filename, hist_trials = 100, livetime = None, verbose = None):

    # xml_snr_min = 0.54
    # xml_snr_step = 0.0082
    # xml_snr_nbin = 299
    # xml_chisq_min = -1.2
    # xml_chisq_step = 0.0173
    # xml_chisq_nbin = 299
    self.snr_bins = (0.54, 0.0082, 299)
    self.chisq_bins = (-1.2, 0.0173, 299)
    ifos = stats_filename.split("/")[-1].split("_")[0]
    fap_kde_name = "background_fap:%s_lgsnr_lgchisq:array" % ifos
    rates_name = "background_rates:%s_histogram:array" % ifos


    self.fap_kde = array_from_xml(stats_filename, fap_kde_name)
    rates = array_from_xml(stats_filename, rates_name)
    self.back_nevent = rates.sum()
    if verbose:
      logging.info("background events %d" % self.back_nevent)
      logging.info("calc fap from rates")

    #crates = np.zeros((300, 300));
    #for snr_idx in range(299, -1, -1):
    #	for chisq_idx in range(1,301):
    #		crates[chisq_idx-1, snr_idx] = rates[0:chisq_idx, snr_idx:300].sum()
    #crates[crates==0] = 1
    #cdf_rates = crates/float(back_nevent)
    #min_cdf = cdf_rates.min()
    
    
    #lgcdf_rates = np.log10(cdf_rates)
    #min_lgcdf = max(-7, lgcdf_rates.min())
    #max_lgcdf = lgcdf_rates.max()
    
    
    self.livetime = livetime
    self.hist_trials = hist_trials
    nstep = 30
    nslide = 100
    #tick_lgcdf = np.linspace(min_lgcdf, max_lgcdf, num=nstep)
    
    #back_lgfap_rates = np.zeros(len(tick_lgcdf))
    #back_lgfap_kde = np.zeros(len(tick_lgcdf))
    #
    self.far_kde = np.zeros((300, 300));
    
    #for snr_idx in range(299, -1, -1):
    #  for chisq_idx in range(1,301):
    #    lgfar_rates[chisq_idx-1, snr_idx] = np.log10(float(1 + sum(rates[np.where(lgcdf_rates < lgcdf_rates[chisq_idx-1, snr_idx])])) / (nslide * coinc_time))
    	
    self.far_kde = self.fap_kde * self.back_nevent / (self.hist_trials * self.livetime)
    
class FAR(object):
  def __init__(self, livetime = None):
    self.livetime = livetime


  def set_ranking_stats(self, ranking_stats):
    self.ranking_stats = ranking_stats

  def far_from_snr_chisq(self, snr, chisq):
    lgsnr = np.log10(snr)
    lgchisq = np.log10(chisq)
    snr_idx = max(min((lgsnr - self.ranking_stats.snr_bins[0])/ self.ranking_stats.snr_bins[1], self.ranking_stats.snr_bins[2]), 0)
    chisq_idx = max(min((lgchisq - self.ranking_stats.chisq_bins[0] )/ self.ranking_stats.chisq_bins[1], self.ranking_stats.chisq_bins[2]), 0)
    return self.ranking_stats.far_kde[chisq_idx, snr_idx]
	

  def assign_fars_sql(self, connection):
    # assign false-alarm rates
    # FIXME:  choose a function name more likely to be unique?
    connection.create_function("far_from_snr_chisq", 2, self.far_from_snr_chisq)
    connection.cursor().execute("""
UPDATE
	postcoh
SET
	far =	far_from_snr_chisq(cohsnr, cmbchisq)
""")

  def count_above_ifar_xml(self, zerolag_fname_str, tick_lgifar):
  
    zerolag_fname_list = zerolag_fname_str.split(',')
    for ifname in zerolag_fname_list:
      table = postcoh_table_from_xml(zf)
      all_table.extend(table)
      iterutils.inplace_filter(lambda row:row.is_background == 0, all_table)

    zerolag_lgifar_kde = - np.log10(all_table.getColumnByName("far"))
    zerolag_lgcevent_kde = np.zeros(len(tick_lgifar))
    
    for itick in range(0, len(tick_lgifar)):
    	cevent_kde = len(zerolag_lgifar_kde[np.where(zerolag_lgifar_kde > tick_lgifar[itick])])
    	if cevent_kde > 0:
    		zerolag_lgcevent_kde[itick] = np.log10(cevent_kde)
    return zerolag_lgcevent_kde
    






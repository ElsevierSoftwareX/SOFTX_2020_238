\page gstlal_review_codes_page Code Review Status Page

\section pythontable Python programs and modules

Redundant entries are omitted

<table>
<tr><th> Program						</th><th> Sub programs or modules	</th><th> Lines	</th><th> Review status	</th><th> Stability </th></tr>
<tr><td> @ref gstlal_fake_frames				</td><td>				</td><td> 360	</td><td> \notreviewed: But not used for production analysis	</td><td> \stable </td></tr>
<tr><td>							</td><td> pipeparts/__init__.py		</td><td> 965	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> reference_psd.py		</td><td> 648	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> simplehandler.py		</td><td> 143	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> datasource.py			</td><td> 749	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> multirate_datasource.py	</td><td> 291	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> glue.segments			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> glue.ligolw*			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.datatypes		</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.series			</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td> lalapps_tmpltbank					</td><td>                               </td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_bank_splitter				</td><td>                               </td><td> 187	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.spawaveform		</td><td> 1244	</td><td> \notreviewed: only gsl SVD used, rest switched to LAL swig </td><td> \stable </td></tr>
<tr><td>							</td><td> glue.lal			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> templates.py			</td><td> 299	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> inspiral_pipe.py		</td><td> 279	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_psd_xml_from_asd_txt			</td><td>                               </td><td> 81	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.xlal*			</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td> ligolw_add						</td><td>                               </td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_svd_bank_pipe			</td><td>                               </td><td> 201	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> glue.iterutils                </td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> glue.pipeliene                </td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_svd_bank					</td><td>                               </td><td> 164	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> svd_bank.py			</td><td> 363	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> cbc_template_fir.py		</td><td> 443	</td><td> \reviewed with actions</td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_create_prior_diststats		</td><td>                               </td><td> 125	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> far.py			</td><td> 1714	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.inject			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.rate			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.snglcoinc		</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_marginalize_likelihood		</td><td>                               </td><td> 167	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_ll_inspiral_pipe				</td><td>                               </td><td> -	</td><td> \notreviewed	</td><td> \moddev </td></tr>
<tr><td> @ref gstlal_inspiral					</td><td>                               </td><td> 707	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> lloidparts.py			</td><td> 826	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> pipeio.py			</td><td> 239	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> simulation.py			</td><td> 72	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> inspiral.py			</td><td> 949	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> streamthinca.py		</td><td> 387	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.ligolw_thinca		</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td>							</td><td> httpinterface.py		</td><td> 110	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> hoftcache.py			</td><td> 110	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_llcbcsummary				</td><td>                               </td><td> 450	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_llcbcnode					</td><td>                               </td><td> 318	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_lvalert_psd_plotter		</td><td>                               </td><td> 240	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_ll_inspiral_get_urls			</td><td>                               </td><td> 30	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_followups_from_gracedb		</td><td>                               </td><td> 177	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_recompute_online_far_from_gracedb	</td><td>                    </td><td> 18	</td><td> \notreviewed	</td><td> \hidev </td></tr>
<tr><td> @ref gstlal_inspiral_recompute_online_far		</td><td>                    		</td><td> 92	</td><td> \notreviewed	</td><td> \hidev </td></tr>
<tr><td> @ref gstlal_inspiral_calc_likelihood			</td><td>                    		</td><td> 409	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>							</td><td> pylal.burca2			</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td> @ref gstlal_ll_inspiral_gracedb_threshold		</td><td>                               </td><td> 106	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> lvalert_listen						</td><td>                               </td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
</table>

\section offline offline specific codes

<table>
<tr><th> Program						</th><th> Sub programs or modules       </th><th> Lines </th><th> Review status </th><th> Stability </th></tr>
<tr><td> @ref gstlal_inspiral_pipe				</td><td>                               </td><td> 729   </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td>							</td><td> dagparts.py                   </td><td> 196   </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td> @ref gstlal_compute_far_from_snr_chisq_histograms	</td><td>                         </td><td> 249   </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_plot_background			</td><td>                               </td><td> 541   </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_plot_sensitivity			</td><td>                               </td><td> 587   </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_plotsummary			</td><td>                               </td><td> 1244  </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td> @ref gstlal_inspiral_summary_page			</td><td>                               </td><td> 344   </td><td> \notreviewed  </td><td> \stable </td></tr>
</table>

\section gsttable gstreamer elements

<table>
<tr><th> Element					</th><th> depenedencies		</th><th> # lines </th><th> Review status	</th><th> Stability	</th></tr>
<tr><td> \ref pipeparts.mkwhiten() lal_whiten		</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mktogglecomplex() lal_togglecomplex</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mksumsquares() lal_sumsquares	</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkstatevector() lal_statevector	</td><td>			</td><td> 	  </td><td> \reviewed 		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkinjections() lal_simulation 	</td><td>			</td><td> 	  </td><td> \notreviewed (with actions)	</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mksegmentsrc() lal_segmentsrc 	</td><td>			</td><td> 	  </td><td> \reviewed (with actions)</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkreblock() lal_reblock 	</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkpeak() lal_peak		</td><td>			</td><td> 	  </td><td> \notreviewed: Not used for production analysis	</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mknxydump() lal_nxydump		</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mknofakedisconts() lal_nofakedisconts</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable 	</td></tr>
<tr><td> \ref pipeparts.mkmatrixmixer() lal_matrixmixer	</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkgate() lal_gate		</td><td>			</td><td> 	  </td><td> \reviewed (with actions)</td><td> \stable       </td></tr>
<tr><td> \ref pipeparts.mkfirbank() lal_firbank		</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable 	</td></tr>
<tr><td> \ref pipeparts.mkdrop() lal_drop		</td><td>			</td><td> 	  </td><td> \reviewed (with actions)</td><td> \stable 	</td></tr>
<tr><td> \ref pipeparts.mkcachesrc() lal_cachesrc	</td><td>			</td><td> 	  </td><td> \reviewed (with actions)</td><td> \stable 	</td></tr>
<tr><td> \ref pipeparts.mkitac() lal_itac		</td><td>			</td><td> 	  </td><td> \reviewed (with actions)</td><td> \stable	</td></tr>
<tr><td> gstlal_autocorrelation				</td><td>			</td><td> 	  </td><td> \reviewed (with actions)</td><td> \stable	</td></tr>
<tr><td> gstlal_peakfinder				</td><td>			</td><td> 	  </td><td> \reviewed (with actions)</td><td> \stable	</td></tr>
<tr><td> framecpp_filesink				</td><td>			</td><td> 	  </td><td> \notreviewed	</td><td> \stable 	</td></tr>
<tr><td> framecpp_channelmux				</td><td>			</td><td> 	  </td><td> \notreviewed	</td><td> \stable 	</td></tr>
<tr><td> framecpp_channeldemux				</td><td>			</td><td> 	  </td><td> \notreviewed	</td><td> \stable	</td></tr>
<tr><td> gds_framexmitsrc				</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable 	</td></tr>
<tr><td> gds_lvshmsrc					</td><td>			</td><td> 	  </td><td> \reviewed (with actions)</td><td> \stable 	</td></tr>
</table>


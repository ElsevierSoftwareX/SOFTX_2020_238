
##################################################################
##################################################################
##################################################################
#
#  remember to set the environment before execute this file
#  e.g. source ~/.gstlal_er14rc to use spiir branch code
#
##################################################################
##################################################################


##################################################################
#  source the parameters from the other file
#  e.g. 3det_highmass_er14code_init.sh
##################################################################

source $1

##################################################################
##################################################################
##################################################################
#
#  NOTE: The rest would be the same for ER14/O3 settings
#  generate get_url_${user}.sub
#  to get latencies and SNRs from last 1000 triggers
#  of each job for online monitoring
#
##################################################################
##################################################################

args="000"
for (( i=1; i<${njob}; i++ )); do
    jobno=$( seq -f "%03g" ${i} ${i} )
    args="$args $jobno"
done

echo -e "universe = local
executable =$mylocation/bin/gstlal_periodic_get_urls" > get_url_${user}.sub
echo "arguments = \"$args\"" >> get_url_${user}.sub
echo -e "getenv = True
accounting_group_user  =  $submitter
accounting_group  =  $myaccgroup
environment = GST_REGISTRY_UPDATE=no
log = ${log_dir}/trigger_pipe_${user}.dag.log.bJCa3q
error = logs/get_url_${user}-\$(cluster)-\$(process).err
output = logs/get_url_${user}-\$(cluster)-\$(process).out
notification = never
queue 1" >> get_url_${user}.sub


##################################################################
#  generate update_map_${user}.sub
#  this is to update the detector reponse map 
#  to capture the movement of Earch every day for the coherent search
##################################################################
if (( ${ndet} == 2 )); then
	ifo_horizons=H1:${dhH},L1:${dhL} 
else
	ifo_horizons=H1:${dhH},L1:${dhL},V1:${dhV} 
fi
echo -e "universe = local
executable =$mylocation/bin/gstlal_periodic_postcoh_update_detrspmap" > update_map_${user}.sub
echo "arguments= \" --data-loc ${H1DataDir} --ifo-horizons ${ifo_horizons} --chealpix-order ${npix} --output-coh-coeff ${mymap} --output-prob-coeff ${mymap_prob} --period ${MapUpdate_T}\" " >> update_map_${user}.sub
echo -e "getenv = True
accounting_group_user  =  $submitter
accounting_group  =  $myaccgroup
environment = GST_REGISTRY_UPDATE=no
log = ${log_dir}/trigger_pipe_${user}.dag.log.bJCa3q
error = logs/update_map_${user}-\$(cluster)-\$(process).err
output = logs/update_map_${user}\$(cluster)-\$(process).out
notification = never
queue 1" >> update_map_${user}.sub


##################################################################
#  generate clean_skymap_${user}.sub
#  to clean up old skymaps when not used for graceDB
##################################################################

if (( ${ndet} == 2 )); then
	clean_place=H1L1_skymap
else
	clean_place=H1L1V1_skymap,H1V1_skymap,L1V1_skymap,H1L1_skymap
fi
echo -e "universe = local
executable = ${mylocation}/bin/gstlal_periodic_clean_skymap" > clean_skymap_${user}.sub
echo "arguments = \"--data-loc ${H1DataDir} --clean-days-ago 0.5 --period 1200 --skymap-loc $clean_place \" " >> clean_skymap_${user}.sub
echo -e "getenv = True
accounting_group_user  =  $submitter
accounting_group  =  $myaccgroup
environment = GST_REGISTRY_UPDATE=no
log = /usr1/${user}/trigger_pipe.dag.log.bJCa30
error = logs/clean_skymap_${user}-\$(cluster)-\$(process).err
output = logs/clean_skymap_${user}-\$(cluster)-\$(process).out
notification = never
queue 1" >> clean_skymap_${user}.sub

##################################################################
#  generate lvalert_listen_${user}.sub, lvalert.ini, lvalert.sh
#  to upload coherent SNR skymap and probility skymap and background plots
#  if a SPIIR event has been uploaded to graceDB
##################################################################

echo -e "#!/bin/bash                                                                         
cat <&0 | tee >(${mylocation}/bin/gstlal_inspiral_postcohspiir_lvalert_plotter --gracedb-service-url=${GraceDB_URL})">lvalert.sh

echo -e "universe = local
executable =/bin/lvalert_listen" > lvalert_listen_${user}.sub
echo "arguments= \" --resource ${myrundir}  --server=${lvalert_server} --dont-wait --username ${mylvuser} --netrc=${mylvcert} --config-file lvalert.ini --verbose\" " >> lvalert_listen_${user}.sub
echo -e "getenv = True
accounting_group_user  =  ${submitter}
accounting_group  = ${myaccgroup}
environment = GST_REGISTRY_UPDATE=no
log = ${log_dir}/trigger_pipe_${user}.dag.log.bJCa3q2
error = logs/lvalert_listen_${user}-\$(cluster)-\$(process).err
output = logs/lvalert_listen_${user}-\$(cluster)-\$(process).out
notification = never
queue 1" >> lvalert_listen_${user}.sub

echo -e "[${gracedbgroupL}_spiir_${searchtypeL}]
executable=./lvalert.sh
">lvalert.ini

##################################################################
#
#  generate gstlal_inspiral_postcohspiir_${user}.sub
#  please see documentation of the spiir-review-O3 at git.ligo.org page
#  for explanation of the options.
#
##################################################################


echo -e "universe = vanilla
executable = $mylocation/bin/gstlal_inspiral_postcohspiir_online" > gstlal_inspiral_postcohspiir_${user}.sub

echo -e "arguments = \"
--job-tag \$(macrojobtag)
--tmp-space _CONDOR_SCRATCH_DIR
--iir-bank \$(macroiirbank)
--data-source ${mydatasrc}
--request-data ${mytag} 
--track-psd
--psd-fft-length ${psd_len}
--channel-name ${mychannel}
--state-channel-name ${mystate}
--gpu-acc on
--ht-gate-threshold ${htgate_thres}
--shared-memory-partition ${mymem}    
--cuda-postcoh-snglsnr-thresh ${snr_thres}
--cuda-postcoh-hist-trials ${Nhist}
--cuda-postcoh-detrsp-fname ${mymap}
--cuda-postcoh-output-skymap ${SNRmap}
--check-time-stamp
--finalsink-output-prefix \$(macrooutprefix)
--finalsink-snapshot-interval ${ZeroLag_T}
--cohfar-accumbackground-snapshot-interval ${FAR_T}
--cohfar-accumbackground-output-prefix \$(macrostatsprefix)
--cohfar-accumbackground-ifo-sense $ifo_horizons
--cohfar-assignfar-input-fname \$(macrofarinput)
--finalsink-fapupdater-output-fname \$(macrolocfapoutput)
--cohfar-assignfar-silent-time ${FAR_silent}
--cohfar-assignfar-refresh-interval ${FAR_refresh}
--finalsink-cluster-window ${tcluster}
--finalsink-fapupdater-interval ${Tfapupdate}
--finalsink-fapupdater-collect-walltime ${wtime1},${wtime2},${wtime3}
--finalsink-far-factor $nfac
--finalsink-gracedb-far-thresh ${far_thres}
--finalsink-need-online-perform 1
--finalsink-gracedb-group ${GraceDB_Group}
--finalsink-gracedb-search ${SearchType}
--finalsink-gracedb-service-url ${GraceDB_URL} 
--cuda-postcoh-detrsp-refresh-interval ${Tmap}
--code-version ${version_spiir} 
--finalsink-singlefar-veto-thresh ${FAR_single_thres} 
--finalsink-superevent-thresh ${FAR_event_thres}
--fir-whitener ${newwhiten}\"
" | /bin/tr '\n' ' ' >>gstlal_inspiral_postcohspiir_${user}.sub

echo  "
want_graceful_removal = True
getenv = True
accounting_group_user  =  ${submitter}
accounting_group  =  ${myaccgroup}
Requirements  =  Target.Online_CBC_IIR_GPU_X2200 =?= True
kill_sig = 15
+Online_CBC_IIR_GPU_X2200 = True
+General_Use_AMD = True
request_cpus = Target.Cpus
request_gpus = Target.Gpus
request_memory = 13GB
log = ${log_dir}/trigger_pipe_${user}.dag.log.JxyP5
error = logs/\$(macronodename)-\$(cluster)-job\$(macrojobtag).err
output = logs/\$(macronodename)-\$(cluster)-job\$(macrojobtag).out
stream_output = True
notification = never
queue 1
" >>gstlal_inspiral_postcohspiir_${user}.sub


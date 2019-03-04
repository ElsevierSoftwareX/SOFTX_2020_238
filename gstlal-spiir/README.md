# SPIIR
This repository hosts the source code and examples for the SPIIR pipeline. Please contact Qi Chu (qi.chu@ligo.org) if having any questions.

## Dependencies:
General libraries:
```
gcc/6.4.0
cuda/9.2
lapack/3.6.1 or 3.8.0
gsl/1.16 or 2.4
fftw/3.3.7
framel/8.30
metaio/8.4.0
cfitsio/3.420
numpy/1.9.1 or 1.14.1
scipy/0.14.0 or 1.0.0
python/2.7.5 or 2.7.14
```

Gstreamer-related libraries:
```
glib/2.29.92
pygobject/2.21.3
gstreamer/0.10.36
gst-plugins-base/0.10.36
gst-plugins-good/0.10.31
gst-plugins-ugly/0.10.19
gst-plugins-bad/0.10.23
gst-python/0.10.22
```

LIGO libraries:
```
ldas-tools/2.4.2
lalsuite/6.15.0+
GraceDB client/2.2.0
```


## Instructions  for installation, trouble shooting, and an script of the SPIIR online pipeline to test on CIT:
 * [Installation and script](https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/spiir/review/instruction)

## References:
 * [SPIIR O3 review page and references](https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/spiir/review)
 * [SPIIR project page](https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/spiir)

## Explanation of options in the online pipeline:
This is an example of a SPIIR online pipeline using LHO, LLO, and Virgo O2replay streaming data:
```
gstlal_postcohspiir_inspiral_online 
        --job-tag pcdev11 
        --iir-bank  H1:/mnt/qfs3/joel.bosveld/online/banks/gstlal_iir_bank_split/iir_H1-GSTLAL_SPLIT_BANK_0000-a1-0-0.xml.gz,L1:/mnt/qfs3/joel.bosveld/online/banks/gstlal_iir_bank_split/iir_L1-GSTLAL_SPLIT_BANK_0000-a1-0-0.xml.gz,V1:/mnt/qfs3/joel.bosveld/online/banks/gstlal_iir_bank_split/iir_V1-GSTLAL_SPLIT_BANK_0000-a1-0-0.xml.gz
        --iir-bank H1:/mnt/qfs3/joel.bosveld/online/banks/gstlal_iir_bank_split/iir_H1-GSTLAL_SPLIT_BANK_0001-a1-0-0.xml.gz,L1:/mnt/qfs3/joel.bosveld/online/banks/gstlal_iir_bank_split/iir_L1-GSTLAL_SPLIT_BANK_0001-a1-0-0.xml.gz,V1:/mnt/qfs3/joel.bosveld/online/banks/gstlal_iir_bank_split/iir_V1-GSTLAL_SPLIT_BANK_0001-a1-0-0.xml.gz
        --gpu-acc on
        --data-source lvshm
        --request-data O2Replay_H1_L1_V1
        --shared-memory-partition V1=K1VIRGO_Data
        --shared-memory-partition=L1=K1LLO_Data
        --shared-memory-partition=H1=K1LHO_Data
        --shared-memory-block-size 500000
        --shared-memory-assumed-duration 1 
        --track-psd 
        --psd-fft-length 16 
        --fir-whitener 1
        --channel-name H1=GDS-CALIB_STRAIN_O2Replay
        --channel-name L1=GDS-CALIB_STRAIN_O2Replay
        --channel-name V1=Hrec_hoft_16384Hz_O2Replay
        --state-channel-name H1=GDS-CALIB_STATE_VECTOR
        --state-channel-name L1=GDS-CALIB_STATE_VECTOR
        --state-channel-name V1=DQ_ANALYSIS_STATE_VECTOR
        --state-vector-on-bits H1=482
        --state-vector-on-bits L1=482
        --state-vector-on-bits V1=482
        --state-vector-off-bits H1=0
        --state-vector-off-bits L1=0
        --state-vector-off-bits V1=0 
        --cohfar-accumbackground-output-prefix pcdev11/bank0_stats 
        --cohfar-accumbackground-output-prefix pcdev11/bank1_stats 
        --cohfar-accumbackground-snapshot-interval 200 
		--cohfar-accumbackground-ifo-sense H1:100,L1:140,V1:50
        --cohfar-assignfar-silent-time 500 
        --cohfar-assignfar-input-fname pcdev11/marginalized_1w.xml.gz,pcdev11/marginalized_1d.xml.gz,pcdev11/marginalized_2h.xml.gz
        --cohfar-assignfar-refresh-interval 200 
        --gpu-acc on  
        --ht-gate-threshold 15.0 
        --cuda-postcoh-snglsnr-thresh 4 
        --cuda-postcoh-hist-trials 100 
        --cuda-postcoh-detrsp-fname H1L1V1_detrsp_map.xml 
        --cuda-postcoh-detrsp-refresh-interval 86400
        --cuda-postcoh-output-skymap 7
        --check-time-stamp 
        --finalsink-fapupdater-collect-walltime 604800,86400,7200
        --finalsink-fapupdater-update-interval 1800
        --finalsink-output-prefix lzerolag 
        --finalsink-snapshot-interval 200 
        --finalsink-cluster-window 1 
        --finalsink-far-factor 2 
        --finalsink-singlefar-veto-thresh 0.5
        --finalsink-superevent-thresh 0.0001
        --finalsink-need-online-perform 1 
        --finalsink-gracedb-far-threshold 0.0001
        --finalsink-gracedb-service-url https://gracedb.ligo.org/api/
        --finalsink-gracedb-pipeline spiir 
        --finalsink-gracedb-group CBC 
        --finalsink-gracedb-search LowMass
        --code-version 0c34343fsafldk
        --verbose
```

 - `job-tag`: specify the relative location (folder) to store all the output files of this job.
 - `gpu-acc`: use GPU for acceleration. Default is `on`.
 - `data-source`: specify the type of data: `frames` for offline data, `lvshm`for online data from shared memory or `framexmit` from broadcasting ports.
 - `request-data`: request the streaming service to be on for this type of data.
 - `shared-memory-partition`: partition name for specific streaming data. R1xx for O2replay, X1xx for live detector data.
 - `iir-bank`: location for the SPIIR banks. can give multiple times to process multiple sets of banks.
 - `track-psd`: track the psd on the fly and use this psd for whitening.
 - `psd-fft-length`: psd length in seconds for whitening.
 - `fir-whitener`: set to 1 if we want to use FIR filters for whitening. Set to 0 if using FFT whitening. Default is 0.
 - `channel-name`: strain channel name.
 - `state-channel-name`: state channel name that used to check the quality of data.
 - `state-vector-on-bits`: data of the state channel need to be 1 on these bits for strain data to be used.
 - `state-vector-off-bits`: data of the state channel need to be 0 on these bits for strain data to be used.
 - `cohfar-accumbackground-snapshot-interval`: the various rates of background events, used to estimate FARs for zerolag events, will be snapshotted at multiples of the time defined here. If it's set to 0, the background rates will be saved after finishing processing the whole data stream.
 - `cohfar-accumbackground-output-prefix`: the output prefix for the background rates. The background output will be saved in the job-tag folder.
 - `cohfar-accumbackground-ifo-sense`: horizon distance for each detector.
 - `cohfar-assignfar-silent-time`: Do not assign FARs to zerolag events during this time from the start of the pipeline. This is to avoid unstable FAR assignment at the beginning due to insufficient background collection.
 - `cohfar-assignfar-input-fname`: All the zerolags will be assigned a FAR which is maximum of FARs from given files.
 - `cohfar-assignfar-refresh-interval`The FAR file or files will be refreshed at multiples of this interval.
 - `ht-gate-threshold`: the values of the whitened data in each detector will be compared with this threshold. If it is over this threshold, it will be reset to zero. This is to remove obvious high-amplitude glitches.
 - `cuda-postcoh-snglsnr-thresh`: threshold to pick single triggers from each detector to do coherent searches, default is 4.0.
 - `cuda-postcoh-hist-trials`: number of time-shifts to collect background histogram, usually set to 100. Each time-shift is 0.1 second apart.
 - `cuda-postcoh-detrsp-fname`: the detector response file, which contains the sampled U and arrival time difference matrices, used to generate coherent SNR and skymap.
 - `cuda-postcoh-detrsp-refresh-interval`: reread the detrsp-fname specified above at multiples of this interval. This is to capture the change of the location of Earth every single day.
 - `cuda-postcoh-output-skymap`: threshold to output skymap, normally set to 7. If it is set to 0, will not output skymaps at all.
 - `check-time-stamp`: check if the data is continously flowing in the pipeline.
 - `finalsink-output-prefix`: the prefix for the zerolag output files.
 - `finalsink-fapupdater-collect-walltime`: How long we collect backgrounds for FAR estimations. Usually three scales spanning 1week, 1day, or 2hours. 
 - `finalsink-fapupdater-update-interval`: A FAR mapping is derived from the collected background for each of different scales set above.
 - `finalsink-snapshot-interval`: the zerolags will be dumped at multiples of this interval.
 - `finalsink-cluster-window`: all the zerolags of a job will be clustered over a window given here first before dumped.
 - `finalsink-far-factor`: a factor that will be multiplied to the FAR of any zerolag of any job before uploading to the database. This is because FAR estimation of each job is consistent with itself. This number should be set to the number of pipelines that run at the same time.
 - `finalsink-singlefar-veto-thresh`: apply single FAR threshold on superevents defined below. Usually set to 0.5 Hz meaning that at least two detectors have the FARs < 0.5 Hz.
 - `finalsink-superevent-thresh`: the FAR thresh for spiir triggers that we apply singlefar veto.
 - `finalsink-need-online-perform`: set to 1 if tracking latencies and coherent SNRs of last 1000 coherent events, no matter what FARs are, of a job. Set to 0 for no tracking.
 - `finalsink-gracedb-far-threshold`: the threshold below which triggers will be submitted to appointed database.
 - `code-version`: the git commit hash used for this run.
 - `verbose`: printout verbose information

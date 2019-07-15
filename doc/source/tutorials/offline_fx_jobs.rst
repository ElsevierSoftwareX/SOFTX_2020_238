Running offline feature extraction jobs
####################################################################################################

An offline DAG is provided in /gstlal-burst/share/feature_extractor/Makefile.gstlal_feature_extractor_offline
in order to provide a convenient way to launch offline feature extraction jobs. A condensed list of
instructions for use is also provided within the Makefile itself.

For general use cases, the only configuration options that need to be changed are:

 * User/Accounting tags: GROUP_USER, ACCOUNTING_TAG
 * Analysis times: START, STOP
 * Data ingestion: IFO, CHANNEL_LIST
 * Waveform parameters: WAVEFORM, MISMATCH, QHIGH

Launching DAGs
====================================================================================================

In order to start up offline runs, you'll need an installation of gstlal. An installation Makefile that
includes Kafka dependencies are located at: gstlal/gstlal-burst/share/feature_extractor/Makefile.gstlal_idq_icc

To generate a DAG, making sure that the correct environment is sourced:

  $ make -f Makefile.gstlal_feature_extractor_offline

Then launch the DAG with:

  $ condor_submit_dag feature_extractor_pipe.dag

Configuration options
====================================================================================================

  Analysis times:
    * START: set the analysis gps start time
    * STOP: set the analysis gps stop time

  Data ingestion:
    * IFO: select the IFO for auxiliary channels to be ingested (H1/L1).
    * CHANNEL_LIST: a list of channels for the feature extractor to process. Provided
        lists for O1/O2 and H1/L1 lists are in gstlal/gstlal-burst/share/feature_extractor.
    * MAX_SERIAL_STREAMS: Maximum # of streams that a single gstlal_feature_extractor job will
        process at once. This is determined by sum_i(channel_i * # rates_i). Number of rates for a
        given channels is determined by log2(max_rate/min_rate) + 1.
    * MAX_PARALLEL_STREAMS: Maximum # of streams that a single job will run in the lifespan of a job.
        This is distinct from serial streams since when a job is first launched, it will cache
        auxiliary channel frames containing all channels that meet the criterion here, and then process
        each channel subset sequentially determined by the serial streams. This is to save on input I/O.
    * CONCURRENCY: determines the maximum # of concurrent reads from the same frame file. For most
        purposes, it will be set to 1. Use this at your own risk.

  Waveform parameters:
    * WAVEFORM: type of waveform used to perform matched filtering (sine_gaussian/half_sine_gaussian).
    * MISMATCH: maximum mismatch between templates (corresponding to Omicron's mismatch definition).
    * QHIGH: maximum value of Q

  Data transfer/saving:
    * OUTPATH: directory in which to save features.
    * SAVE_CADENCE: span of a typical dataset within an hdf5 file.
    * PERSIST_CADENCE: span of a typical hdf5 file.

Setting the number of streams (ADVANCED USAGE)
====================================================================================================

  NOTE: This won't have to be changed for almost all use cases, and the current configuration has been
    optimized to aim for short run times.

  Definition: Target number of streams (N_channels x N_rates_per_channel) that each cpu will process.

    * if max_serial_streams > max_parallel_streams, all jobs will be parallelized by channel
    * if max_parallel_streams > num_channels in channel list, all jobs will be processed serially,
        with processing driven by max_serial_streams.
    * any other combination will produce a mix of parallelization by channels and processing channels serially per job.

  Playing around with combinations of MAX_SERIAL_STREAMS, MAX_PARALLEL_STREAMS, CONCURRENCY, will entirely
  determine the structure of the offline DAG. Doing so will also change the memory usage for each job, and so you'll
  need to tread lightly. Changing CONCURRENCY in particular may cause I/O locks due to jobs fighting to read from the same
  frame file.

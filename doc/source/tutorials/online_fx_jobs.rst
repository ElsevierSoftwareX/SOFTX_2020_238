Running online feature extraction jobs
####################################################################################################

An online DAG is provided in /gstlal-burst/share/feature_extractor/Makefile.gstlal_feature_extractor_online
in order to provide a convenient way to launch online feature extraction jobs as well as auxiliary jobs as
needed (synchronizer/hdf5 file sinks). A condensed list of instructions for use is also provided within the Makefile itself.

There are four separate modes that can be used to launch online jobs:

  1. Auxiliary channel ingestion:

    a. Reading from framexmit protocol (DATA_SOURCE=framexmit).
       This mode is recommended when reading in live data from LHO/LLO.

    b. Reading from shared memory (DATA_SOURCE=lvshm).
       This mode is recommended for reading in data for O2 replay (e.g. UWM).

  2. Data transfer of features:

    a. Saving features directly to disk, e.g. no data transfer.
       This will save features to disk directly from the feature extractor,
       and saves features periodically via hdf5.

    b. Transfer of features via Kafka topics.
       This requires a Kafka/Zookeeper service to be running (can be existing LDG
       or your own). Features get transferred via Kafka from the feature extractor,
       parallel instances of the extractor get synchronized, and then sent downstream
       where it can be read by other processes (e.g. iDQ). In addition, an streaming
       hdf5 file sink is launched where it'll dump features periodically to disk.

Launching DAGs
====================================================================================================

In order to start up online runs, you'll need an installation of gstlal. An installation Makefile that
includes Kafka dependencies are located at: gstlal/gstlal-burst/share/feature_extractor/Makefile.gstlal_idq_icc

To run, making sure that the correct environment is sourced:

  $ make -f Makefile.gstlal_feature_extractor_online

Then launch the DAG with:

  $ condor_submit_dag feature_extractor_pipe.dag

Configuration options
====================================================================================================

  General:
    * TAG: sets the name used for logging purposes, Kafka topic naming, etc.

  Data ingestion:
    * IFO: select the IFO for auxiliary channels to be ingested.
    * CHANNEL_LIST: a list of channels for the feature extractor to process. Provided
        lists for O1/O2 and H1/L1 lists are in gstlal/gstlal-burst/share/feature_extractor.
    * DATA_SOURCE: Protocol for reading in auxiliary channels (framexmit/lvshm).
    * MAX_STREAMS: Maximum # of streams that a single gstlal_feature_extractor process will
        process. This is determined by sum_i(channel_i * # rates_i). Number of rates for a
        given channels is determined by log2(max_rate/min_rate) + 1.

  Waveform parameters:
    * WAVEFORM: type of waveform used to perform matched filtering (sine_gaussian/half_sine_gaussian).
    * MISMATCH: maximum mismatch between templates (corresponding to Omicron's mismatch definition).
    * QHIGH: maximum value of Q

  Data transfer/saving:
    * OUTPATH: directory in which to save features.
    * SAVE_FORMAT: determines whether to transfer features downstream or save directly (kafka/hdf5).
    * SAVE_CADENCE: span of a typical dataset within an hdf5 file.
    * PERSIST_CADENCE: span of a typical hdf5 file.

  Kafka options:
    * KAFKA_TOPIC: basename of topic for features generated from feature_extractor
    * KAFKA_SERVER: Kafka server address where Kafka is hosted. If features are run in same location,
        as in condor's local universe, setting localhost:port is fine. Otherwise you'll need to determine
        the IP address where your Kafka server is running (using 'ip addr show' or equivalent).
    * KAFKA_GROUP: group for which Kafka producers for feature_extractor jobs report to.

  Synchronizer/File sink options:
    * PROCESSING_CADENCE: cadence at which incoming features are processed, so as to limit polling
        of topics repeatedly, etc. Default value of 0.1s is fine.
    * REQUEST_TIMEOUT: timeout for waiting for a single poll from a Kafka consumer.
    * LATENCY_TIMEOUT: timeout for the feature synchronizer before older features are dropped. This
        is to prevent a single feature extractor job from holding up the online pipeline. This will
        also depend on the latency induced by the feature extractor, especially when using templates
        that have latencies associated with them such as Sine-Gaussians.

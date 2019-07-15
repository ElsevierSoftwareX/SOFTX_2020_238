.. _feature_extraction:

Feature Extraction
====================================================================================================

The `fxtools` module and related feature-based executables contain relevant libraries to identify
glitches in low-latency using auxiliary channel data.

`gstlal_feature_extractor` functions as a modeled search for data quality by applying matched filtering
on auxiliary channel timeseries using waveforms that model a large number of glitch classes. Its primary
purpose is to whiten incoming auxiliary channels and extract relevant features in low-latency.

There are two different modes of output `gstlal_feature_extractor` can function in:

  1. **Timeseries:** Production of regularly-spaced feature rows, containing the SNR, waveform parameters,
                     and the time of the loudest event in a sampling time interval.
  2. **ETG:** This produces output that resembles that of a traditional event trigger generator (ETG), in
              which only feature rows above an SNR threshold will be produced.

One useful feature in using a matched filter approach to detect glitches is the ability to switch between
different glitch templates or generate a heterogeneous bank of templates.. Currently, there are Sine-Gaussian
and half-Sine-Gaussian waveforms implemented for use in detecting glitches, but the feature extractor was
designed to be fairly modular and so it isn't difficult to design and add new waveforms for use.

Since the GstLAL feature extractor uses time-domain convolution to matched filter auxiliary channel timeseries
with glitch waveforms, this allows latencies to be much lower than in traditional ETGs. The latency upon writing
features to disk are O(5 s) in the current layout when using waveforms where the peak occurs at the edge of the
template (zero-latency templates). Otherwise, there is extra latency incurred due to the non-causal nature of
the waveform itself.

 .. graphviz::

    digraph llpipe {
     labeljust = "r";
     label="gstlal_feature_extractor"
     rankdir=LR;
     graph [fontname="Roman", fontsize=24];
     edge [ fontname="Roman", fontsize=10 ];
     node [fontname="Roman", shape=box, fontsize=11];


     subgraph clusterNodeN {

         style=rounded;
         label="gstreamer pipeline";
         labeljust = "r";
         fontsize = 14;

         H1L1src [label="H1(L1) data source:\n mkbasicmultisrc()", color=red4];

         Aux1 [label="Auxiliary channel 1", color=red4];
         Aux2 [label="Auxiliary channel 2", color=green4];
         AuxN [label="Auxiliary channel N", color=magenta4];

         Multirate1 [label="Auxiliary channel 1\nWhiten/Downsample", color=red4];
         Multirate2 [label="Auxiliary channel 2\nWhiten/Downsample", color=green4];
         MultirateN [label="Auxiliary channel N\nWhiten/Downsample", color=magenta4];

         FilterBankAux1Rate1 [label="Auxiliary Channel 1:\nGlitch Filter Bank", color=red4];
         FilterBankAux1Rate2 [label="Auxiliary Channel 1:\nGlitch Filter Bank", color=red4];
         FilterBankAux1RateN [label="Auxiliary Channel 1:\nGlitch Filter Bank", color=red4];
         FilterBankAux2Rate1 [label="Auxiliary Channel 2:\nGlitch Filter Bank", color=green4];
         FilterBankAux2Rate2 [label="Auxiliary Channel 2:\nGlitch Filter Bank", color=green4];
         FilterBankAux2RateN [label="Auxiliary Channel 2:\nGlitch Filter Bank", color=green4];
         FilterBankAuxNRate1 [label="Auxiliary Channel N:\nGlitch Filter Bank", color=magenta4];
         FilterBankAuxNRate2 [label="Auxiliary Channel N:\nGlitch Filter Bank", color=magenta4];
         FilterBankAuxNRateN [label="Auxiliary Channel N:\nGlitch Filter Bank", color=magenta4];

         TriggerAux1Rate1 [label="Auxiliary Channel 1:\nMax SNR Feature (N Hz)", color=red4];
         TriggerAux1Rate2 [label="Auxiliary Channel 1:\nMax SNR Feature (N Hz)", color=red4];
         TriggerAux1RateN [label="Auxiliary Channel 1:\nMax SNR Feature (N Hz)", color=red4];
         TriggerAux2Rate1 [label="Auxiliary Channel 2:\nMax SNR Feature (N Hz)", color=green4];
         TriggerAux2Rate2 [label="Auxiliary Channel 2:\nMax SNR Feature (N Hz)", color=green4];
         TriggerAux2RateN [label="Auxiliary Channel 2:\nMax SNR Feature (N Hz)", color=green4];
         TriggerAuxNRate1 [label="Auxiliary Channel N:\nMax SNR Feature (N Hz)", color=magenta4];
         TriggerAuxNRate2 [label="Auxiliary Channel N:\nMax SNR Feature (N Hz)", color=magenta4];
         TriggerAuxNRateN [label="Auxiliary Channel N:\nMax SNR Feature (N Hz)", color=magenta4];

         H1L1src -> Aux1;
         H1L1src -> Aux2;
         H1L1src -> AuxN;

         Aux1 -> Multirate1;
         Aux2 -> Multirate2;
         AuxN -> MultirateN;

         Multirate1 -> FilterBankAux1Rate1 [label="4096Hz"];
         Multirate2 -> FilterBankAux2Rate1 [label="4096Hz"];
         MultirateN -> FilterBankAuxNRate1 [label="4096Hz"];
         Multirate1 -> FilterBankAux1Rate2 [label="2048Hz"];
         Multirate2 -> FilterBankAux2Rate2 [label="2048Hz"];
         MultirateN -> FilterBankAuxNRate2 [label="2048Hz"];
         Multirate1 -> FilterBankAux1RateN [label="Nth-pow-of-2 Hz"];
         Multirate2 -> FilterBankAux2RateN [label="Nth-pow-of-2 Hz"];
         MultirateN -> FilterBankAuxNRateN [label="Nth-pow-of-2 Hz"];

         FilterBankAux1Rate1 -> TriggerAux1Rate1;
         FilterBankAux1Rate2 -> TriggerAux1Rate2;
         FilterBankAux1RateN -> TriggerAux1RateN;
         FilterBankAux2Rate1 -> TriggerAux2Rate1;
         FilterBankAux2Rate2 -> TriggerAux2Rate2;
         FilterBankAux2RateN -> TriggerAux2RateN;
         FilterBankAuxNRate1 -> TriggerAuxNRate1;
         FilterBankAuxNRate2 -> TriggerAuxNRate2;
         FilterBankAuxNRateN -> TriggerAuxNRateN;
     }


     Synchronize [label="Synchronize buffers by timestamp"];
     Extract [label="Extract features from buffer"];
     Save [label="Save triggers to disk"];
     Kafka [label="Push features to queue"];

     TriggerAux1Rate1 -> Synchronize;
     TriggerAux1Rate2 -> Synchronize;
     TriggerAux1RateN -> Synchronize;
     TriggerAux2Rate1 -> Synchronize;
     TriggerAux2Rate2 -> Synchronize;
     TriggerAux2RateN -> Synchronize;
     TriggerAuxNRate1 -> Synchronize;
     TriggerAuxNRate2 -> Synchronize;
     TriggerAuxNRateN -> Synchronize;

     Synchronize -> Extract;

     Extract -> Save [label="Option 1"];
     Extract -> Kafka [label="Option 2"];

    }


**Highlights:**

* Launch feature extractor jobs in online or offline mode:

  * Online: Using /shm or framexmit protocol
  * Offline: Read frames off disk

* Online/Offline DAGs available for launching jobs.

  * Offline DAG parallelizes by time, channels are processed sequentially by subsets to reduce I/O concurrency issues. There are options to allow flexibility in choosing this, however.

* On-the-fly PSD generation (or take in a prespecified PSD)

* Auxiliary channels to be processed can be specified in two ways:

  * Channel list .INI file, provided by DetChar. This provides ways to filter channels by safety and subsystem.
  * Channel list .txt file, one line per channel in the form H1:CHANNEL_NAME:2048.

* Configurable min/max frequency bands for aux channel processing in powers of two. The default here is 32 - 2048 Hz.

* Verbose latency output at various stages of the pipeline. If regular verbosity is specified, latencies are given only when files are written to disk.

* Various file transfer/saving options:

  * Disk: HDF5
  * Transfer: Kafka (used for low-latency implementation)

* Various waveform configuration options:

  * Waveform type (currently Sine-Gaussian and half-Sine-Gaussian only)
  * Specify parameter ranges (frequency, Q for Sine-Gaussian based)
  * Min mismatch between templates

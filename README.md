![gstlal](https://git.ligo.org/lscsoft/gstlal/raw/master/doc/source/_static/gstlal_small.png "gstlal")

GStreamer elements for gravitational-wave data analysis
==================================================================

[![pipeline status](https://git.ligo.org/lscsoft/gstlal/badges/master/pipeline.svg)](https://git.ligo.org/lscsoft/gstlal/commits/master)

|               | version  | source   |
| :------------:| :------: | :------: |
| `gstlal` | 1.5.1  | [gstlal-1.5.1.tar.gz](http://software.ligo.org/lscsoft/source/gstlal-1.5.1.tar.gz)  |
| `gstlal-ugly`     | 1.6.6  | [gstlal-ugly-1.6.6.tar.gz](http://software.ligo.org/lscsoft/source/gstlal-ugly-1.6.6.tar.gz)  |
| `gstlal-inspiral`  | 1.6.9 | [gstlal-inspiral-1.6.9.tar.gz](http://software.ligo.org/lscsoft/source/gstlal-inspiral-1.6.9.tar.gz)  |
| `gstlal-calibration`  |  1.2.11  | [gstlal-calibration-1.2.11.tar.gz](http://software.ligo.org/lscsoft/source/gstlal-calibration-1.2.11.tar.gz)  |
| `gstlal-burst`  |  0.2.0  | [gstlal-burst-0.2.0.tar.gz](http://software.ligo.org/lscsoft/source/gstlal-burst-0.2.0.tar.gz)  |

Full documentation is provided [here](https://lscsoft.docs.ligo.org/gstlal/).

**GstLAL** provides a suite of GStreamer elements that expose gravitational-wave data analysis tools from the LALSuite library for use in GStreamer signal-processing pipelines.

Examples include an element to add simulated gravitational waves to an h(t) stream, and a source element to provide the contents of .gwf frame files to a GStreamer pipeline.
Overview

The **GstLAL** software package is used for the following activities:

  * GstLAL: Provides core Gstreamer plugins for signal processing workflows with LIGO data and core python bindings for constructing such workflows.
  * GstLAL Calibration: Provides real-time calibration of LIGO control system data into strain data.
  * GstLAL Inspiral: Provides additional signal processing plugins that are specific for LIGO / Virgo searches for compact binaries as well as a substantial amount of python code for post-processing raw signal processing results into gravitational wave candidate lists. Several publications about the methodology and workflow exist, see publications
  * GstLAL Ugly: An incubator project for gradual inclusion in the other packages.

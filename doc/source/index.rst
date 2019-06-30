GstLAL documentation
================================

`GstLAL` provides a suite of GStreamer elements that expose gravitational-wave data analysis tools from the LALSuite library for use in GStreamer signal-processing pipelines.

Examples include an element to add simulated gravitational waves to an h(t) stream, and a source element to provide the contents of .gwf frame files to a GStreamer pipeline.

Overview
-------------------------

The GstLAL software package is used for the following activities:

- **GstLAL:** The package `GstLAL <http://software.ligo.org/lscsoft/source/gstlal-1.4.1.tar.gz>`_ provides core Gstreamer plugins for signal processing workflows with LIGO data and core python bindings for constructing such workflows.  

- **GstLAL Calibration:** The package `GstLAL Calibration <http://software.ligo.org/lscsoft/source/gstlal-calibration-1.2.4.tar.gz>`_ provides real-time calibration of LIGO control system data into strain data.

- **GstLAL Inspiral:** The package `GstLAL Inspiral <http://software.ligo.org/lscsoft/source/gstlal-inspiral-1.5.1.tar.gz>`_ provides additional signal processing plugins that are specific for LIGO / Virgo searches for compact binaries as well as a substantial amount of python code for post-processing raw signal processing results into gravitational wave candidate lists. Several publications about the methodology and workflow exist, see :ref:`publications`

- **GstLAL Ugly:** The package `GstLAL Inspiral <http://software.ligo.org/lscsoft/source/gstlal-inspiral-1.5.1.tar.gz>`_ is an incubator project for gradual inclusion in the other packages.


.. _welcome-contents:

Contents
-------------------------

.. toctree::
   :maxdepth: 2

   getting-started
   projects
   publications

Build/Test Results
-------------------------

Results pages for the `Offline Tutorial Test <https://git.ligo.org/lscsoft/gstlal/blob/master/gstlal-inspiral/tests/Makefile.offline_tutorial_test>`_ are generated automatically and are located here:

* `gstlal_offline_tutorial test dag <gstlal_offline_tutorial/1000000000-1000002048-test_dag-run_1/>`_
* `gstlal_offline_tutorial test dag lite <gstlal_offline_tutorial/1000000000-1000002048-test_dag-run_1_lite/>`_

.. _welcome-indices:

Indices and tables
-------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


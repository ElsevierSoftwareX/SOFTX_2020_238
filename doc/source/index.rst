GstLAL documentation
================================

`GstLAL` provides a suite of GStreamer elements that expose gravitational-wave data analysis tools from the LALSuite library for use in GStreamer signal-processing pipelines.

Examples include an element to add simulated gravitational waves to an h(t) stream, and a source element to provide the contents of .gwf frame files to a GStreamer pipeline.

Overview
-------------------------

The GstLAL software package is used for the following activities:

- ``gstlal`` provides core Gstreamer plugins for signal processing workflows with LIGO data and core python bindings for constructing such workflows.  

- ``gstlal-calibration`` provides real-time calibration of LIGO control system data into strain data.

- ``gstlal-inspiral`` provides additional signal processing plugins that are specific for LIGO / Virgo searches for compact binaries as well as a substantial amount of python code for post-processing raw signal processing results into gravitational wave candidate lists. Several publications about the methodology and workflow exist, see :ref:`publications`

- ``gstlal-burst`` provides additional signal processing plugins for use in astrophysical and noise transient burst searches.

- ``gstlal-ugly`` is an incubator project for gradual inclusion in the other packages.


.. toctree::
    :caption: Getting Started
    :maxdepth: 2

    installation
    quickstart
    tutorials/tutorials

.. toctree::
    :caption: User Guide
    :maxdepth: 2

    cbc_analysis
    feature_extraction
    fake_data
    psd_estimation
    publications

.. toctree::
    :caption: API Reference
    :maxdepth: 2

    executables
    api

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


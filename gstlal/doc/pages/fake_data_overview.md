@page fakedataoverviewpage

## Introduction

gstlal provides several tools for producing fake gravitational wave data for
the purpose of simulating various scenarios.  This page works through several
examples starting with simple command line data generation, working up to
Condor DAGs that can generate months of fake data suitable for MDC studies.

## Basic LIGO/ALIGO colored Gaussian noise on the command line

Consult gstlal_fake_frames for more details

Here is an example to generate fake initial LIGO data on the command line

	$ gstlal_fake_frames --data-source=LIGO --channel-name=H1=FAKE-STRAIN --frame-type=H1_FAKE --gps-start-time=900000000 --gps-end-time=900005000 --output-path=testframes  --verbose

Note the contents of the frame directory: 

	$ ls testframes/H-FAKE-900/
	H-FAKE-900000000-1792.gwf  H-FAKE-900001792-3208.gwf

Similarly, one can generate advanced LIGO data

	$ gstlal_fake_frames --data-source=AdvLIGO --channel-name=L1=FAKE-STRAIN --frame-type=L1_FAKE --gps-start-time=900000000 --gps-end-time=900005000 --output-path=testframes  --verbose

To verify that this dat makes sense we can try to measure the psd using gstlal_reference_psd

First we need a frame cache

	$ ls testframes/*/* | lalapps_path2cache > frame.cache

Then we can measure the PSDs of the data like this:

	$ gstlal_reference_psd --data-source frames --frame-cache frame.cache --gps-start-time=900000000 --gps-end-time=900005000 --channel-name=H1=FAKE-STRAIN --channel-name=L1=FAKE-STRAIN --write-psd psd.xml.gz --verbose

We can see how we did by plotting the resulting psd

	$ gstlal_plot_psd psd.png psd.xml.gz 

![PSDs](images/H1L1fakedataexamplepsd.png "PSD for LIGO and Advanced LIGO")

## Custom colored noise, i.e. simulate your own detector

Obviously it is not practical to code up every possible noise curve to simulate as a custom data source.  However, it is possible to provide your own custom noise curve as an ASCII file with frequency in one column and strain/Hz in the second.

Take for example the early virgo noise curve <a href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal/share/v1_early_asd.txt>here</a>.

	$ wget http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal/share/v1_early_asd.txt

First convert it to an xml file appropriate for reading with gstlal

	$ gstlal_psd_xml_from_asd_txt --instrument=V1 --output v1psd.xml.gz v1_early_asd.txt

Then we can color the data like this:

	$ gstlal_fake_frames --data-source=white --color-psd v1psd.xml.gz --channel-name=V1=FAKE-STRAIN --frame-type=V1_FAKE --gps-start-time=900000000 --gps-end-time=900005000 --output-path=testframes  --verbose

And repeat a similar validation process as above

	$ ls testframes/*/* | lalapps_path2cache > frame.cache
	$ gstlal_reference_psd --data-source frames --frame-cache frame.cache --gps-start-time=900000000 --gps-end-time=900005000 --channel-name=V1=FAKE-STRAIN --write-psd psd.xml.gz --verbose
	$ gstlal_plot_psd psd.png psd.xml.gz

![PSDs](images/V1fakedataexamplepsd.png "PSD for LIGO and Advanced LIGO")

## Recolored noise

This procedure is very similar to the above except that instead of using white noise to drive the fake frame generation, we start with real frame data and whiten it.

## Simulating signals on the command line

## Recoloring existing data with a HTCondor dag

<a href=@cgit/share/Makefile.2015recolored>Makefile.2015recolored</a> is a place to start

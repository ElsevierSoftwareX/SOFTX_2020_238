\page gstlaltelecons20140730page Telecon July 30, 2014

\ref gstlalteleconspage

[TOC]

\section agenda Agenda

- S6 replay MDC data and analysis at UWM
- Status of optimized libraries
- Status of IMBHB/BBH
- Status of BNS MDC
- Discussion of peak disk space requirements for offline dags
- AOB

\section minutes minutes

S6 replay MDC data and analysis at UWM

Cody: Install optimized dependencies on NEMO then get online anlysis and bank dag going. Modify dag to include dq channel option name. Waiting to get access to cluster nodes. Ran jobs from head node.

How many nodes are needed for minimal setup?
1 node with 16 cores
No simultaenous injections since just rerunning ER5

Nodes need to be configured to have no run-time limits
At CIT running on sandybridge cores, on 5 computers with 80 total cores and 40 condor slots.
Static slots are fine

Use CGI as output. Allowed at NEMO?
Each gstlal_inspiral job runs a webserver which are not accessible outside at CIT. We run a job on the headnode and downloads all the jobs every 5 minutes and stores data at a shared file system. Then parsed to make quasi-real-time plots

Chad Email Tom regarding the specifics of how to implement this at NEMO

How to interpret state vector, what to do with it? If answered, give update, if not then answer. Two state vectors: IFO state vector and Data Quality state vector. A state vector is broadcast at NEMO - dq state vector

Identify state vector that was used by MBTA to run online. Should be consisten with what is in S6 frame files.

Demonstarte online infrastructure works and do a benchmark comparison with ihope. What data did ihope analyize? The answer is known

Low-latency question: Are we ok with a situation where the segments analyzed are different from the ihope segments? Generate a state vector that matches ihop if not.

Run online and offline analysis with gstlal. On online state vector from S6 any bad behavior from gstlal? Significantly worse than offline? 

Currently in the state vector there is documentation on the calibrated data.
https://www.lsc-group.phys.uwm.edu/daswg/wiki/S6OnlineGroup/CalibratedData

IFO state vector equivalent to OCE master and information redundant in DQ vector

https://www.lsc-group.phys.uwm.edu/daswg/wiki/S6OnlineGroup/CalibratedData#The_Data_Quality_Vector_channel_definition

Demand Calibrated bit is on and injection bit is off. Gives data we want without hardware injections. 

Cody: Modify command line to give options for turning bits on/off

Update documentation (from url above)page if necessary. Names may have changed.


Status of optimized libraries

Cody successful in compiling. testnode.cgca.uwm.edu is where libraries are compiled. Sufficient yet not completely optimal for entire UWM cluster. 
Strange crashes from matplotlib. Don't know why yet. Specifically, errorbar plot. Removed LD_PRELOAD for atlas. Try to get gsl to link against atlas and not successful except manually hack gsl package config to point to atlas libraries. Chad testing dag with this setup.

Update coming soon. 

Need to build optimized library to a machine specific architecture.
Don't need to hack package config
set gsl_cflags gsl_libs when compiling gstlal and lalsuite and package config should use these when built

Chad: Point people to nonoptimized dependencies


Status of IMBHB/BBH

Continued using ER5 release for BBH mass space
https://ldas-jobs.phys.uwm.edu/~sophia.xiao/newSummaryPage/pipelineDay1_trapezoid_30Hz/gstlal-966384015-966470415_closed_box.html?gstlal-966384015-966470415_closed_box_time_analysis.html

There may be a bug in the actual timing

What is the actual CPU time on the box? Can it be monitored for a running condor job? Yes, the CPU time is in the condor logs
Ensure time being recorded is CPU time and not wall time.

Les: https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/IMBHsearchCBCstatus/HighmassDevelopment
Injections start at 25Hz. Improvements made with lower frequencies.
Tapering is used
Chad --> Les: Change injection xml files for the low frequency to be 10Hz instead of 25Hz

Discussion of peak disk space requirements for offline dags

Running Dags at UWM has hit disk quotas. Peak disk usage for offline DAGs can be large. Reason: process all gstlal_inspiral jobs produce unclustered output before they are ranked and then clustered. Once clustered and ranked the permanant storage is reduced to a reasonable amount. 
Ryan: reorganize DAG to process more efficiently

Does condor submit jobs depth or breadth first? Can change default globally and perhaps user level.

Scaling up to a million second DAG uses all 2.5GB disk quota. Possibly need TB.

What will 01 spin aligned need for 1 day analysis?

Tom: Give TB to shared account and upgrade user accounts to 4-5GB
Chad: send email to Tom with account names

Other users haven't had disk space issues.

Come up with a peak disk usage estimate

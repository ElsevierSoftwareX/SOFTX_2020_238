\page gstlalinspiralofflinesearchpage Offline search documentation

[TOC]

\section Introduction Introduction

Please see \ref gstlalinspirallowlatencysearchpage for background information.

\section Preliminaries Preliminaries

_NOTE: ANALYSIS BEST SUPPORTED AT UWM._

Running elsewhere reuquires dynamic Condor slots and modifcations to the gstlal_reference_psd, gstlal_inspiral and gstlal_inspiral_inj submit files.  We are working to standardize this on the LDG.

- Start by making a directory where you will run the analysis, e.g.,:

		$ mkdir /home/channa/test

\section makefiles Get example makefiles tailored to your application

- Get two makefiles to set up the analysis dag.  This example is for BNS analysis in early ALIGO.  One defines standard rules that should not need to be modified by the user, the other is use-case specific.  The examples 

 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.offline_analysis_rules>Makefile.offline_analysis_rules</a>
 
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.triggers_example>Makefile.triggers_example</a>

will need to be modified to your situation and there are comments in the files for further documentation.

\section making Making the workflow

To make the workflow you need to run "make", e.g.,

		$ make -f Makefile.triggers_example

The makefile will execute the following graph that culminates in an HTCondor DAG

@dotfile Makefile_offline_triggers.dot

To see the HTCondor DAG please see the documenation for \ref gstlal_inspiral_pipe

\section submittion Submitting the HTCondor workflow

The DAG can be submitted via:

		$ condor_submit_dag trigger_pipe.dag

When the workflow is finished you should see a web page output in the directory specified in the Makefile

\section review Review status


See \ref gstlalinspirallowlatencysearchpage here for the modules and programs used by both low and high latency searches. Redundant entries are ommitted here

<table>
<tr><th> Program                                </th><th> Sub programs or modules       </th><th> Lines </th><th> Review status </th><th> Stability </th></tr>
<tr><td> gstlal_inspiral_pipe                   </td><td>                               </td><td> 729   </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td>                                        </td><td> dagparts.py                   </td><td> 196   </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td> gstlal_compute_far_from_snr_chisq_histograms </td><td>				</td><td> 249   </td><td> \notreviewed  </td><td> \moddev </td></tr>
<tr><td> gstlal_inspiral_plot_background	</td><td>				</td><td> 541   </td><td> \notreviewed  </td><td> \moddev </td></tr>
<tr><td> gstlal_inspiral_plot_sensitivity	</td><td>				</td><td> 587   </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td> gstlal_inspiral_plotsummary		</td><td>				</td><td> 1244  </td><td> \notreviewed  </td><td> \stable </td></tr>
<tr><td> gstlal_inspiral_summary_page		</td><td>				</td><td> 344	</td><td> \notreviewed  </td><td> \stable </td></tr>
</table>

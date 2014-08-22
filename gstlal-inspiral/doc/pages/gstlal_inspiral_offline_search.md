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

\section specifics Specific Examples

The following examples use these hashes:

- gstlal hash: e5bae89d07267ed97ba4bceb2f54fbef75d3fd03 
	- Apply <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/patches/compute_far_from_snr_chisq_histogram.patch>compute_far_from_snr_chisq_histogram_patch</a>
	- Apply <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/patches/gstlal_inspiral_plot_background.patch>gstlal_inspiral_plot_background_patch</a>
	- Apply <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/patches/gstlal_inspiral_plotsummary.patch>gstlal_inspiral_plotsummary_patch</a>
	- Apply <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/patches/gstlal_inspiral_plot_sensitivity.patch>gstlal_inspiral_plot_sensitivity_patch</a>
	- Apply <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/patches/gstlal_inspiral_summary_page.patch>gstlal_inspiral_summary_page_patch</a>
	- Apply <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/patches/gstlal_inspiral_pipe.patch>gstlal_inspiral_pipe_patch</a>
	- Apply <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/patches/templates.patch>templates_patch</a>
- lalsuite hash: 38fdd56f2ec5c73a030f679f9de9fedd554dbfba  
	- Apply <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/patches/spawaveform.patch>spawaveforms_patch</a>

\subsection non_spinning_aligned_BNS_gaussian Non-Spinning BNS with Gaussian Noise

This example needs the Makefile.offline_analysis_rules (see above) and the following Makefile:

- <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/mdc/non_spinning_BNS_gaussian/Makefile.nonspinning_BNS_gaussian>Makefile.nonspinning_BNS_gaussian</a>

- We inject both MDC injection files simultaneously
	- <a href=https://sugar-jobs.phy.syr.edu/~jveitch/bns/mdc/spin/BNS-SpinMDC-ALIGNED.xml>BNS-SpinMDC-ALIGNED.xml</a>
	- <a href=https://sugar-jobs.phy.syr.edu/~jveitch/bns/mdc/spin/BNS-SpinMDC-ISOTROPIC.xml>BNS-SpinMDC-ISOTROPIC.xml</a>

- The end result of this workflow creates a webpage with the results of this analysis
	- <a href=https://ldas-jobs.phys.uwm.edu/~ryan.everett/mdc/non_spinning_BNS_gaussian/ALL_LLOID_COMBINED_closebox.html?ALL_LLOID_COMBINED_closebox_summary.html>Results</a>

\section making Making the workflow

To make the workflow you need to run "make", e.g.,

		$ make -f FILENAME 

The makefile will execute the following graph that culminates in an HTCondor DAG

@dotfile Makefile_nonspinning_BNS_gaussian.dot

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

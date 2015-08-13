\page gstlaltelecons20140924page Telecon Sept 24, 2014

\ref gstlalteleconspage

[TOC]

\section agenda Agenda

- Optimization efforts
- S6VSR3Replay

\subsection optimazationefforst Optimization efforts

\section minutes Minutes  

Chad: It’s computationally expensive to up-sample from 128 to 1024. It might sound counterintuitive but it might cost less to use higher sample rate in the second slice. We might have to add some code to control how the time-slice decomposition is done for high-mass signals, maybe simply just specifying minimum sample rate but probably something more than that to balance the cost of resampling versus the cost of filtering at higher sample rate.

Tjonnie: You can see my result at  https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/StellarMassBBH/BBH_gstlal/benchmarking. 
Chad: We might need to try re-slicing of the templates and the resampler_direct_single function (?) is called instead of what  has been called here.

Jolien: Try turning off sub-sampling
Chad: It’s a very good idea specially for masses above some certain limit.

Chad: Are you still doing the measurement that runs 16 parallel jobs, Tjonnie?
Tjonnie: I run 16 parallel jobs using all of the sub-banks.

Chad: Have you checked if you’re saturating the machine doing this? 100 percent CPU usage?
Tjonnie: yes. The average is about 200 percent CPU.

Chad: There is one patch on master which can be helpful and it can reduce the cost by maybe a factor of 2 when we have audio resampling sorted. I can send the patch around.

Jolien: It’s a good time to write a custom up sampler and what you need to do is hard coding bunch of matrices for different sampling ratios that you possibly want. 
Chad: It’s a good idea since everyone who produced a profile graph saw that audio resampling has the dominant cost. 

Chad: Down-sampling cost is very small compare to up-sampling cost.

Les has done same follow-up as Tjonnie’s and has got same results.

Chad: It would be good if Jolien and Les prepare a patch that changes the command line option for the auto-correlation length to be in time instead of samples. Stephen P. had worked on that before and has written a code.


\subsection s6vsr3replay S6VSR3Replay 

- Statues of data broadcasting at UWM: Patrick Brockill and Chris P. are working on cleaning up the gaps in the data that cause the broadcasting code to fail. 

- Laleh and Chris P. have got the code for doing the injections on the server and then piping them as a duplicate channel which has the injections in it. The signal set that they are going to start with is 1 to 3 solar masses injectionis, no spin and uniformly distributed in mass space.

\subsection aob AOB

\section attendance Attendance
Chad, Chris P., Cody, Duncan, Ian, Kent, Laleh (minutes), Larne, Les, Maddie, Patrick Brady, Ryan, Stephen P.,  Surabhi, Tjonnie, UWM

\page gstlal_inspiral_impulse_response_study Impulse Response Study

\section intro Introduction

- Delta function test: Inject delta-function and compare with time-reversed chirp
- Inject a chirp waveform into "slience" and compare with signal auto-correlation function

\section method Method

- Python program:


	dt = 1.0 / 4096.0
	for i in xrange(4096*256):
		if i == 4096*128:
			v = 1.0
		else:
			v = 0.0
	print '%.09f\t%g' % (i*dt, v)


- Convert to frame file with `lalfr-fmt`
- Change file name with


	gst-launch filesrc location=foo.gwf blocksize=8395941 ! framecpp_channeldemux .C01 ! .H1:C01 framecpp_channelmux frames-per-file=256 ! filesink location=bar.gwf


- Run this through `gstlal_inspiral`


\section results Results

- Response of gstlal using TaylorF2 to an impulse injected at t=128 s compared to time-reversed TaylorF2: Plot shows high frequency end of the waveform

\htmlonly
<a href="gstlal_impulse_response_01.png" target="_blank"><img src="gstlal_impulse_response_01.png" width="300px" height="300px"></a>
\endhtmlonly

- Same as above but 0.25 seconds later; the discontinuity occurs as a result of combining different time slices

\htmlonly
<a href="gstlal_impulse_response_02.png" target="_blank"><img src="gstlal_impulse_response_02.png" width="300px" height="300px"></a>
\endhtmlonly

- Zoomed out version of the same

\htmlonly
<a href="gstlal_impulse_response_03.png" target="_blank"><img src="gstlal_impulse_response_03.png" width="300px" height="300px"></a>
\endhtmlonly

- Same as above but a second later when another feature is seen: discontinuity could be due to a different sampling rate as well

\htmlonly
<a href="gstlal_impulse_response_04.png" target="_blank"><img src="gstlal_impulse_response_04.png" width="300px" height="300px"></a>
\endhtmlonly

- Figure shows how the response changes as one moves from one time-slice to the next

\htmlonly
<a href="gstlal_impulse_response_05.png" target="_blank"><img src="gstlal_impulse_response_05.png" width="300px" height="300px"></a>
\endhtmlonly

- Zoomed out version of the previous figure to show transition from one time-slice to the next

\htmlonly
<a href="gstlal_impulse_response_06.png" target="_blank"><img src="gstlal_impulse_response_06.png" width="300px" height="300px"></a>
\endhtmlonly

- Within a given time-slice there is very good agreement between svd bank response and the expected response

\htmlonly
<a href="gstlal_impulse_response_07.png" target="_blank"><img src="gstlal_impulse_response_07.png" width="300px" height="300px"></a>
\endhtmlonly

- Another time-slice and a new SVD

\htmlonly
<a href="gstlal_impulse_response_08.png" target="_blank"><img src="gstlal_impulse_response_08.png" width="300px" height="300px"></a>
\endhtmlonly

- Behaviour towards the end of the waveform

\htmlonly
<a href="gstlal_impulse_response_09.png" target="_blank"><img src="gstlal_impulse_response_09.png" width="300px" height="300px"></a>
\endhtmlonly


The image below shows the injection of a TaylorF2 into "silence":

- Comparison of SVD output with the auto-correlation function

\htmlonly
<a href="gstlal_impulse_response_10.png" target="_blank"><img src="gstlal_impulse_response_10.png" width="300px" height="300px"></a>
\endhtmlonly

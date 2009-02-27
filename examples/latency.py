#!/usr/bin/python
from pylab import *

file = open("out4.61.txt")
progdictx = {"progress_src":[], "progress_sumsquares":[], "progress_snr":[], "progress_chisquare":[]}
progdicty = {"progress_src":[], "progress_sumsquares":[], "progress_snr":[], "progress_chisquare":[]}

for line in file.readlines():
  l = line.split()
  if l and not "progress" in l[0]: continue
  if l == []: continue
  clock = l[1].replace('(','').replace(')','').split(':')
  clocksec = int(clock[0])*3600 + int(clock[1])*60 + int(clock[2])
  progdicty[l[0]].append(clocksec)
  progdictx[l[0]].append(int(l[2]))



minVec = [min(progdictx["progress_src"]), min(progdictx["progress_sumsquares"]), min(progdictx["progress_snr"]), min(progdictx["progress_chisquare"])]
minVal = min(minVec)

plot(array(progdictx["progress_src"])-minVal, progdicty["progress_src"])
hold(1)
plot(array(progdictx["progress_sumsquares"])-minVal, progdicty["progress_sumsquares"])
plot(array(progdictx["progress_snr"])-minVal, progdicty["progress_snr"])
plot(array(progdictx["progress_chisquare"])-minVal, progdicty["progress_chisquare"])
lat = abs(mean(array(progdicty["progress_src"][-5:])-array(progdicty["progress_snr"][-5:])))
runtime = mean( array(progdicty["progress_src"][-5:]) / array(progdicty["progress_snr"][-5:]) )
#plot([0,max(["progress_src"] ], [0 ])
xlabel('Analysis time')
ylabel('Clock time')
title("latency = " + str(lat) + " run time = " + str(runtime))
legend(['progress_src','progress_sumsquares','progress_snr','progress_chisquare'],loc="lower right")
grid(1)
hold(0)
show()

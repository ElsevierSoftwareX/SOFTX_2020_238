from math import *

def TDFlops(inTup):
  numTemps = inTup[0]
  points = inTup[1]
  fsamp = inTup[2]
  return numTemps * points * fsamp / 1024 / 1024

def FFTFlops(inTup, latency):
  numTemps = inTup[0]
  points = inTup[1]
  fsamp = inTup[2]
  fftfac = 10.0
  return fftfac * numTemps * (points + latency * fsamp) * log(points + latency * fsamp, 2) / latency / 1024 / 1024

bank = []
bank.append([43, 2048, 2048])
bank.append([70, 2048, 512])
bank.append([91, 2048, 256])
bank.append([143, 2048, 128])

print "TD Convolution"
flops = 0
for t in bank:
  flops += TDFlops(t)
print flops, " MFLOPS"

for lat in [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]:
  print "FFT Convolution "+str(lat)+" second latency"
  flops = 0
  for t in bank:
    flops += FFTFlops(t,lat)
  print flops, " MFLOPS"
  

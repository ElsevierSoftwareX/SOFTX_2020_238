#!/usr/bin/env python

import sys
import os
import optparse
import signal

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import simps
import datetime
from pylab import ion 
from collections import deque

import pygtk
pygtk.require('2.0')
import pygst
pygst.require('0.10')
import gobject
gobject.threads_init()
import gst

from gstlal import lloidparts
from gstlal import pipeparts
from gstlal.pipeutil import mkelem
from gstlal import pipeio
from pylal import lalconstants as lc


__prog__ = "hdist_plotting.py"
__version__ = "$Id: hdist_plotting.py,v 1.0 2011/11/11 bdaudert Exp$"

#
# =============================================================================
#
#                                     BOILERPLATE
#
# =============================================================================
#
class Handler(object):
    def __init__(self, mainloop, pipeline, verbose = False):
        self.mainloop = mainloop
        self.pipeline = pipeline
        self.verbose = verbose

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

    def on_message(self, bus, message):
        if message.type == gst.MESSAGE_EOS:
            self.pipeline.set_state(gst.STATE_NULL)
            self.mainloop.quit()
        elif message.type == gst.MESSAGE_ERROR:
            gerr, dbgmsg = message.parse_error()
            self.pipeline.set_state(gst.STATE_NULL)
            self.mainloop.quit()
            sys.exit("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg))

#
# =============================================================================
#
#                                     Functions
#
# =============================================================================
#
def parse_args():

    usage = """ %prog [options]
            """

    parser = optparse.OptionParser( usage )
    
    parser.add_option("-v","--version",action="store_true",default=False,\
    help="display version information and exit")
    parser.add_option("-d","--duration",action="store",type="float",\
    default=5, metavar="DURATION",help="duration of low latency data streaming in seconds")        
    parser.add_option("-r","--rate",action="store",type="float",\
    default=2048.0,metavar="RATE",help="sample rate")
    parser.add_option("--candle-mass-1",action="store",type="float",\
    default=1.4, metavar="M1",help="Candle Mass 1")
    parser.add_option("--candle-mass-2",action="store",type="float",\
    default=1.4, metavar="M1",help="Candle Mass 2")
    parser.add_option("-s","--snr",action="store",type="float",\
    default=8.0, metavar="SNR",help="Desired snr")
    parser.add_option("--f-min",action="store",type="float",\
    default=40.0, metavar="F_MIN",help="Start Frequency")
    parser.add_option("--f-max",action="store",type="float",\
    default=800.0, metavar="F_MAX",help="End Frequency")
    parser.add_option("--ifos",action="store",\
        default=["H1", "H2", "L1", "V1"], metavar="IFOS",help="List of ifos")

    (opts,args) = parser.parse_args()

    return opts, args


def compute_hdist(M1, M2, snr, rate, deltaF, f_min, f_max, PSD):
    mu = (M1 * M2) / (M1 + M2)
    k_min = int(f_min/deltaF)
    k_max = int(f_max/deltaF)
    N = rate/deltaF
    deltaT = 1.0/rate
    distnorm = 2.0*lc.LAL_MRSUN_SI/(lc.LAL_PC_SI*10**6)
    a = np.sqrt(5.0*mu/96.0)*((M1+M2)/lc.LAL_PI **2)**(1.0/3) \
    *(lc.LAL_MTSUN_SI/deltaT)**(-1.0/6.0)
    sigmaSq = 4.0*(deltaT/N)*distnorm*distnorm*a*a

    sigmaSqSum = np.dot(np.arange(k_min,k_max+1)**(-7./3),
    1./PSD[k_min-1:k_max])
    sigmaSqSum*= N**(7./3)
    h_dist = np.sqrt(sigmaSq*sigmaSqSum)/snr
    return h_dist    

def signal_handler(signal, frame, pipeline):
    print >>sys.stderr, "*** SIG %d attempting graceful shutdown... ***" \
     % (signal,)
    bus = pipeline.get_bus()
    bus.post(gst.message_new_eos(pipeline))



def make_pipline(ifos, fig, lines, times, h_dist):
    ion()
    def generate_update_callback(index):
        def update_plots(elem):
            buffer = elem.get_property("last-buffer")
            hour = float(datetime.datetime.now().strftime('%H'))
            min = float(datetime.datetime.now().strftime('%M'))
            t = round(hour + min / 60, 2)

            p = pipeio.array_from_audio_buffer(buffer)
            hd = compute_hdist(opts.candle_mass_1, opts.candle_mass_2, 
            opts.snr, opts.rate, deltaF, opts.f_min, opts.f_max, p)

            times[index].append(t) 
            h_dist[index].append(hd) 
            if index == 0:
                k = len(list(h_dist[index])) / (86400/4)
                for j in range(k,-1,-1):
                    lines[k-j].set_data(np.array(list(times[index])[j*86400/4:(j+1)*86400/4]), \
                    np.array(list(h_dist[index])[(j)*86400/4:(j+1)*86400/4]))
            else:
                lines[6+index].set_data(np.array(times[index]),np.array(h_dist[index]))
            
            lines[-1].set_data([t,t],[370,390])        
            fig.canvas.draw()
            png_file = ".%s.tmp.png" %index
            fig.canvas.print_png(png_file)
            os.rename(png_file, "/home/bdaudert/public_html/sensemon.png")

            Times = np.zeros(1)
            H_dist = np.zeros(1)
            for key, value in times.iteritems():
                Times = np.append(Times,value)
            for key, value in h_dist.iteritems():
                H_dist = np.append(H_dist,value)
            data_file = ".tmp.data.npz"
            np.savez(data_file, time=Times, horizon_distance=H_dist)
            #os.rename(data_file, "/home/bdaudert/public_html/h_dist_data.npz")

        return update_plots

    pipe = gst.Pipeline("NDSTest") 
  
    d_name = {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
    channel = "FAKE-STRAIN"
    for i, ifo in enumerate(ifos):
        src = mkelem("gds_lvshmsrc", {'shm_name': d_name[ifo]})
        dmx = mkelem("framecpp_channeldemux", {'do_file_checksum':True, 'skip_bad_files': True})
        aud = mkelem("audiochebband", {'lower-frequency': 40, 'upper-frequency': 2500})
        resamp = mkelem("audioresample")
        caps_filt = mkelem("capsfilter", {'caps': gst.Caps("audio/x-raw-float, rate=2048")})
        appsink = mkelem("appsink", {'caps': gst.Caps("audio/x-raw-float"), 'sync': False,
        'async': False, 'emit-signals': True, 'max-buffers': 1, 'drop': True})
        whiten = mkelem("lal_whiten", {'psd-mode': 0, 'zero-pad': 0, 'fft-length': 8,
        'median-samples': 7, 'average-samples': 128})
        fakesink = mkelem("fakesink", {'sync': False, 'async': False})
        
        pipe.add(src, dmx, aud, resamp, caps_filt, whiten, appsink, fakesink)
        gst.element_link_many(src, dmx)
        pipeparts.src_deferred_link(dmx, "%s:%s" % (ifo, channel), aud.get_pad("sink"))
        gst.element_link_many(aud,resamp, caps_filt, whiten, fakesink)
        whiten.link_pads("mean-psd", appsink,"sink")
        appsink.connect_after("new-buffer", generate_update_callback(i))
        deltaF = whiten.get_property("delta-f")
    
    return pipe

#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#

opts, args = parse_args()

colors = ['red', 'magenta', 'blueviolet', 'chartreuse', 'chocolate', 'aqua', \
         'cadetblue', 'blue', 'green','darkgoldenrod']
#Chose dicts to be able to use deque effectively         
times = {}
h_dist = {}
lines = []
fig = plt.figure()
ax1 = fig.add_subplot(111)


for i, ifo in enumerate(opts.ifos):
    if i == 0:
        times[i] = deque(maxlen = 7 * 86400/4)
        h_dist[i] = deque(maxlen = 7 * 86400/4)
        line, = ax1.plot([],[],'*-',color='%s' %colors[i], \
        markersize=1.5,lw=1,mec='%s' %colors[i],label="%s" %ifo)
        lines.append(line)
        for j in range(1,7):
            line, = ax1.plot([],[],'*-',color='%s' %colors[j], \
            markersize=1,lw=0.5,mec='%s' %colors[j],alpha=0.8,label="-%s" %j  )
            lines.append(line)
    else:
        times[i] = deque(maxlen = 86400/4)
        h_dist[i] = deque(maxlen = 86400/4)
        line, = ax1.plot([],[],'*-',color='%s' %colors[i+6], \
        markersize=1.5,lw=1,mec='%s' %colors[i+6],label="%s" %ifo)
        lines.append(line)

vline, = ax1.plot([],[],'-', c='black', lw=0.5)
lines.append(vline)

ax1.legend()
ax1.set_xlim(0,24)
ax1.set_ylim(370,390)
ax1.set_xticks(np.arange(25))
ax1.set_ylabel('Hour')
ax1.set_ylabel('Horizondistance')

pipeline =  make_pipline(opts.ifos, fig, lines, times, h_dist)

print "Setting state to PAUSED:", pipeline.set_state(gst.STATE_PAUSED)
print pipeline.get_state()

mainloop = gobject.MainLoop()
handler = lloidparts.LLOIDHandler(mainloop, pipeline)


print "Setting state to PLAYING:", pipeline.set_state(gst.STATE_PLAYING)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

mainloop.run()




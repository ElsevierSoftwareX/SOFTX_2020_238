#!/usr/bin/env python

from __future__ import division

__author__ = "Britta Daudert <britta.daudert@ligo.org>, Nickolas Fotopoulos <nickolas.fotopoulos@ligo.org>"

import bisect
import signal
import shutil
import sys
import tempfile
import time
import optparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime
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
from pylal import date

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
    
    parser.add_option("-r","--rate",action="store",type="float",\
        default=2048.0,metavar="RATE",help="sample rate")
    parser.add_option("--candle-mass-1",action="store",type="float",\
        default=1.4, metavar="M1",help="Candle Mass 1")
    parser.add_option("--candle-mass-2",action="store",type="float",\
        default=1.4, metavar="M2",help="Candle Mass 2")
    parser.add_option("-s","--snr",action="store",type="float",\
        default=8.0, metavar="SNR",help="Desired snr")
    parser.add_option("--f-min",action="store",type="float",\
        default=40.0, metavar="F_MIN",help="Start Frequency")
    parser.add_option("--f-max",action="store",type="float",\
        default=800.0, metavar="F_MAX",help="End Frequency")
    parser.add_option("--ifos",action="store",\
        default="H1,H2,L1,V1", metavar="IFOS",help="Comma-separated list of ifos")
    parser.add_option("--history-files", metavar="FNAME", default="", help="load history from comma-separated list of .npz files")
    parser.add_option("--figure-path", metavar="FNAME", help="file to which to save plots")
    # TODO: Set timezone that defines midnight for the hour computation
    # TODO: Need to add some way to specify the history length per IFO
    # TODO: Drop first N buffers (whitener settling time); default to 8 or 10?

    (opts,args) = parser.parse_args()

    if opts.figure_path is None:
        parser.error("--figure-path is required")

    opts.ifos = opts.ifos.split(",")  # turn into list
    opts.history_files = opts.history_files.split(",")  # turn into list

    return opts, args


def compute_hdist(M1, M2, snr, rate, deltaF, f_min, f_max, PSD):
    """
    Compute horizon distance.
    """
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

def timestamp2hour(timestamp):
    """
    Return the number of hours since midnight of the given nanosecond timestamp.

    FIXME: Make relative to local time of main IFO; use pytz
    """
    dt_obj = datetime.datetime(*date.XLALGPSToUTC(date.XLALINT8NSToGPS(timestamp))[:6])
    delta = (dt_obj - dt_obj.replace(hour=0, minute=0, second=0, microsecond=0))
    return (delta.microseconds + (delta.seconds + delta.days * 24 * 3600) * 1e6) / 1e6 / 3600

def atomic_print_figure(fig, ifo, dest):
    """
    Multiple threads are writing files. Protect them by making the final plot update atomic.
    """
    tf = tempfile.NamedTemporaryFile(prefix="tmp_hdist_%s" % ifo, suffix=".png",
        delete=False)  # do first write to scratch space, as NFS sometimes interferes
    fig.canvas.print_png(tf.file)
    tf.close()
    # mkstemp() ignores umask, creates all files accessible
    # only by owner;  we should respect umask.  note that
    # os.umask() sets it, too, so we have to set it back after
    # we know what it is
    umsk = os.umask(0777)
    os.umask(umsk)
    os.chmod(tf.name, 0666 & ~umsk)
    hidden_dest = ("/.%s." % ifo).join(os.path.split(dest))  # same volume as dest, but hidden
    shutil.move(tf.name, hidden_dest)  # leaves target in bad state during transfer
    os.rename(hidden_dest, dest)  # atomic operation on same volume

def signal_handler(signal, frame, pipeline):
    print >>sys.stderr, "*** SIG %d attempting graceful shutdown... ***" \
     % (signal,)
    bus = pipeline.get_bus()
    bus.post(gst.message_new_eos(pipeline))

def make_pipline(ifos, fig, lines, times, h_dist):
    plt.ion()
    def generate_update_callback(opts, ifo, deltaF):
        ifo_times = times[ifo]
        ifo_h_dist = h_dist[ifo]
        ifo_lines = lines[ifo]
        now_line = lines["now"]

        def update(elem):
            # grab PSD
            buffer = elem.get_property("last-buffer")
            psd = pipeio.array_from_audio_buffer(buffer).squeeze()

            # compute horizon distance
            hd = compute_hdist(opts.candle_mass_1, opts.candle_mass_2, 
                opts.snr, opts.rate, deltaF, opts.f_min, opts.f_max, psd)

            # update history
            ifo_times.append(buffer.timestamp)  # nanoseconds
            ifo_h_dist.append(hd) 

            # update plot
            t = timestamp2hour(buffer.timestamp)
            tmp_ifo_times = np.array(ifo_times)  # FIXME: can't slice deques! Do something smarter.
            tmp_h_dist = np.array(ifo_h_dist)  # FIXME: can't slice deques! Do something smarter.
            for i, line in enumerate(ifo_lines):  # each line is a day
                low_ind = tmp_ifo_times.searchsorted(buffer.timestamp - (i + 1) * 86400 * 1e9)
                high_ind = tmp_ifo_times.searchsorted(buffer.timestamp - i * 86400 * 1e9)
                line.set_data(map(timestamp2hour, tmp_ifo_times[low_ind:high_ind]), tmp_h_dist[low_ind:high_ind])
            now_line[0].set_data([t, t], [0, 10000])  # hopefully 10 Gpc is safe

            # write plot; be very paranoid about race conditions
            fig.canvas.draw()
            atomic_print_figure(fig, ifo, opts.figure_path)

            # save history to disk
            data_file = "history_%s.npz" % ifo
            np.savez(data_file, **{"%s_times" % ifo: ifo_times, "%s_horizon_distance" % ifo: ifo_h_dist})
            shutil.move(data_file, "/home/nvf/public_html/%s_h_dist_data.npz" % ifo)
        return update

    pipe = gst.Pipeline("NDSTest") 
  
    d_name = {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
    channel = "FAKE-STRAIN"
    for ifo in ifos:
        src = mkelem("gds_lvshmsrc", {'shm_name': d_name[ifo]})
        dmx = mkelem("framecpp_channeldemux", {'do_file_checksum':True, 'skip_bad_files': True})
        aud = mkelem("audiochebband", {'lower-frequency': 8, 'upper-frequency': 2500})
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

        # hook updater to appsink
        deltaF = whiten.get_property("delta-f")
        appsink.connect_after("new-buffer", generate_update_callback(opts, ifo, deltaF))
    
    return pipe

#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#

opts, args = parse_args()

color_list = ['red', 'magenta', 'blueviolet', 'chartreuse', 'chocolate', 'aqua', \
         'cadetblue', 'blue', 'green','darkgoldenrod']

# Set up storage for the history of (time, horizon distance) plus the plot's line collections
times = {}
h_dist = {}
lines = {}  # each IFO can have multiple traces, so this dict maps to a list of lines
fig = plt.figure()
ax1 = fig.add_subplot(111)

for ifo, color in zip(opts.ifos, color_list):
    if ifo == "H1":  # FIXME: Unhardcode primary IFO
        history_len = 7
    else:
        history_len = 1
    times[ifo] = deque(maxlen = history_len * 86400/4)  # FIXME: unhardcode 4
    h_dist[ifo] = deque(maxlen = history_len * 86400/4)  # FIXME: unhardcode 4

    # load history
    for fname in opts.history_files:
        if os.path.basename(fname).startswith(ifo):
            npz = np.load(fname)
            times[ifo].extend(npz["%s_times" % ifo])
            h_dist[ifo].extend(npz["%s_horizon_distance" % ifo])

    ifo_lines = lines.setdefault(ifo, [])
    line, = ax1.plot([], [], '*-', color=color, markersize=1.5, lw=1, mec=color, label=ifo)
    ifo_lines.append(line)
    for j in range(1, history_len):  # add marker-less lines for additional history
        line, = ax1.plot([], [], '-', color=color, markersize=1, lw=0.5, mec=color,
            alpha=1. - j / history_len, label="_nolegend_"  )
        ifo_lines.append(line)

vline, = ax1.plot([], [], '-', c='black', lw=0.5)
lines["now"] = [vline]

ax1.grid(True)
ax1.legend()
ax1.set_xlim((0, 24))
ax1.set_ylim((0, 500))
ax1.set_xticks(range(25))
ax1.set_xlabel('UTC hour')  # FIXME: Make relative to local time of main IFO
ax1.set_ylabel('Horizon distance (Mpc)')

pipeline = make_pipline(opts.ifos, fig, lines, times, h_dist)

print "Setting state to PAUSED:", pipeline.set_state(gst.STATE_PAUSED)
print pipeline.get_state()

mainloop = gobject.MainLoop()
handler = lloidparts.LLOIDHandler(mainloop, pipeline)


print "Setting state to PLAYING:", pipeline.set_state(gst.STATE_PLAYING)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

mainloop.run()




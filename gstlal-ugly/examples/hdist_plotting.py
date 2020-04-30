#!/usr/bin/env python3

from __future__ import division

__author__ = "Britta Daudert <britta.daudert@ligo.org>, Nickolas Fotopoulos <nickolas.fotopoulos@ligo.org>"

import threading as thrd
import bisect
import signal
import shutil
import sys
import tempfile
import time
import optparse
import os
import datetime
from collections import deque
from itertools import izip

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


import pygtk
pygtk.require('2.0')
import pygst
pygst.require('0.10')
import gobject
gobject.threads_init()
import gst

from gstlal import simplehandler
from gstlal import pipeparts
from gstlal.pipeutil import mkelem
from gstlal import pipeio
from pylal import lalconstants as lc
from pylal import date
from pylal.inject import cached_detector, XLALComputeDetAMResponse
from pylal.xlal.constants import LAL_PI, LAL_TWOPI, LAL_PI_2, LAL_PI_4


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
#                    Network Response Plotting Functions
#
# =============================================================================
#
def polaz2lonlat(theta, phi):
    """
    Convert (polar, azimuthal) angles in radians to (longitude, latitude)
    in radians.
    """
    return phi, LAL_PI_2 - theta

def polaz2xyz(pol, az, mag=1):
    """
    Go from polar to cartesian coordinates.
    """
    x = mag * np.sin(pol) * np.cos(az)
    y = mag * np.sin(pol) * np.sin(az)
    z = mag * np.cos(pol)
    return x, y, z
    
def network_response(pol, az, network, horizons=None):
    """
    Find the sqrt of the sum of squares of F+ and Fx for each detector
    in a network, optionally weighted by the horizon distances. Network
    should be a list of detector objects.
    Ref: Schutz, CQG 2011, Equation 14
    """        
    longs, lats = polaz2lonlat(pol, az)
    F_RSS_tot = 0
    for det, horiz in izip(network, horizons):
        fp, fc = np.array(
            [XLALComputeDetAMResponse(det.response, lon, lat, 0, 0) for
             lon, lat in izip(longs.flat, lats.flat)]).T
        F_RSS_tot += horiz**2 * (fp**2 + fc**2)
        return np.sqrt(F_RSS_tot).reshape(pol.shape)

def get_cartesian_coordinates(arm_alt, arm_az, vert_lat, vert_lon):
    """
    Get the cartesian coordinates of a given detector arm.
    arm_alt, arm_az are the altitude and azimuth (clockwise from North) of the
    end station.
    vert_lat, vert_lon are the latitude and longitude of the corner station.
    """             
    c = np.cos
    s = np.sin
    uNorth = c(arm_alt) * c(arm_az);
    uRho = -s(vert_lat) * uNorth + c(vert_lat) * s(arm_alt)
    return np.array((c(vert_lon) * uRho - s(vert_lon) * uEast,
                     s(vert_lon) * uRho + c(vert_lon) * uEast,
                     c(vert_lat) * uNorth + s(vert_lat) * s(arm_alt)),
                     dtype=np.float32)  # needed for XLALComputeDetAMResponse

def detector_L(det):
    """
    Return the unit x, y, and z vectors necessary to plot the arms of the
    given interferometer.
    """                  
    v1 =  get_cartesian_coordinates(0, det.xArmAzimuthRadians, det.vertexLatitudeRadians, det.vertexLongitudeRadians)
    v2 =  get_cartesian_coordinates(0, det.yArmAzimuthRadians, det.vertexLatitudeRadians, det.vertexLongitudeRadians)
    return np.array((v1[0], 0, v2[0])), \
           np.array((v1[1], 0, v2[1])), \
           np.array((v1[2], 0, v2[2]))  # pass through origin
           
        
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
    parser.add_option("--primary-ifo",\
        default="H1", metavar="IFO", help="ifo for which to keep 7 day history")
    parser.add_option("-d","--drop-buffers",type="int",\
            default=10,metavar="N",help="Drop first N buffers (whitener settling time)")   
        
    # TODO: Set timezone that defines midnight for the hour computation

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

def atomic_print_figure(fig, dest):
    """
    Multiple threads are writing files. Protect them by making the final plot update atomic.
    """
    tf = tempfile.NamedTemporaryFile(prefix="tmp_hdist", suffix=".png",
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
    hidden_dest = "/.".join(os.path.split(dest))  # same volume as dest, but hidden
    shutil.move(tf.name, hidden_dest)  # leaves target in bad state during transfer
    os.rename(hidden_dest, dest)  # atomic operation on same volume

def signal_handler(signal,frame):
    print >>sys.stderr, "*** SIG %d attempting graceful shutdown... ***" \
     % (signal,)
    bus = pipeline.get_bus()
    bus.post(gst.message_new_eos(pipeline))
    print "Exiting with Signal %s" % signal
    sys.exit(1)

def update_plot(lock, ifos, figure_path, times, h_dist, theta, phi):
    # os.umask() gets and sets at the same time, so we have to set it
    # back after # we know what it is
    umsk = os.umask(0777)
    os.umask(umsk)
   
    #Set up some stuff needed for network response plot
    ind=figure_path.rfind("/")
    network_plot_dest= "%s/network_response.png" % figure_path[0:ind]
    print "Saving network response plot to %s" % network_plot_dest    
    networks = []
    det_list = []
    
    now_line = lines["now"]    
    while True:
        time.sleep(10)

        with lock:
            t_max = 0
            for ifo in ifos:
                if len(times[ifo]):
                    t_max = max(t_max, times[ifo][-1])
            t = timestamp2hour(t_max)

        print("Updating plots!")
                
        now_line[0].set_data([t, t], [0, 10000])  # hopefully 10 Gpc is safe

        for ifo in ifos:
            if ifo == "H1" or ifo == "H2":
                network = cached_detector["LHO_4k"]
            elif ifo == "L1":
                network = cached_detector["LLO_4k"]
            else:
                network = cached_detector["VIRGO"]            
            
            with lock:
                tmp_ifo_times = np.array(times[ifo])  # FIXME: can't slice deques! Do something smarter.
                tmp_h_dist = np.array(h_dist[ifo])  # FIXME: can't slice deques! Do something smarter.
    
            if len(times[ifo]):  
                ts = tmp_ifo_times[-1]
                networks.append(network)
                det_list.append(ifo)
            else:
                ts = 0
                    
            for i, line in enumerate(lines[ifo]):  # each line is a day
                low_ind = tmp_ifo_times.searchsorted(ts - (i + 1) * 86400 * 1e9)
                high_ind = tmp_ifo_times.searchsorted(ts - i * 86400 * 1e9)
    
                #build ends using None to avoid wrap around lines
                updated_times = np.array(map(timestamp2hour, tmp_ifo_times[low_ind:high_ind])) 
                if len(updated_times):
                    mid_ind_high = (np.array(updated_times) - 0).argmax()
                    xlist=[]
                    ylist=[]
                    xlist.extend(updated_times[low_ind:high_ind])
                    xlist.insert(mid_ind_high+1,None)
                    ylist.extend(tmp_h_dist.tolist()[low_ind:high_ind])
                    ylist.insert(mid_ind_high+1,None)
                    line.set_data(xlist,ylist)


            # save data files
            data_file = "history_%s.npz" % ifo
            # mkstemp() ignores umask, creates all files accessible
            # only by owner;  we should respect umask.  note that
            tmpf_1 = "/tmp/%s_horizon_distance.npy" %ifo
            tmpf_2 = "/tmp/%s_times.npy" %ifo
            if os.path.exists(data_file):
                os.chmod(data_file, 0666 & ~umsk)
            if os.path.exists(tmpf_1):
                os.chmod(tmpf_1, 0666 & ~umsk)
            if os.path.exists(tmpf_2):
                os.chmod(tmpf_2, 0666 & ~umsk)
            np.savez(data_file, **{"%s_times" % ifo: tmp_ifo_times, "%s_horizon_distance" % ifo: tmp_h_dist})
            shutil.move(data_file, "/home/bdaudert/public_html/%s_h_dist_data.npz" % ifo) 
            
        #write and save plot
        fig.canvas.draw()
        atomic_print_figure(fig, figure_path)

        #3D Network response plot
        horizons = np.fromiter((h_dist[ifo][-1] for ifo in det_list),np.float)
        if np.isnan(np.sum(horizons)) or np.isinf(sum(horizons)):
            continue
        else:    
            response = network_response(theta, phi, networks, horizons)
            if response is None:
                continue
            else:    
                x, y, z = polaz2xyz(theta, phi, mag=response)
                ax2.plot_surface(x, y, z,cmap=plt.get_cmap('winter'))
                fig2.canvas.draw()
                atomic_print_figure(fig2, network_plot_dest)
        
def make_pipline(opts, fig, lines, times, h_dist, lock):
    def generate_update_callback(lock, opts, ifo, deltaF):
        def update_data(elem): 
            # grab PSD
            buffer = elem.get_property("last-buffer")
            psd = pipeio.array_from_audio_buffer(buffer).squeeze()

            print("updating arrays with {0} timestamp {1:d}".format(ifo, buffer.timestamp))

            # compute horizon distance
            hd = compute_hdist(opts.candle_mass_1, opts.candle_mass_2,
                opts.snr, opts.rate, deltaF, opts.f_min, opts.f_max, psd)

            # update history
            with lock:
                times[ifo].append(buffer.timestamp)  # nanoseconds
                h_dist[ifo].append(hd)
        return update_data

    pipe = gst.Pipeline("NDSTest") 
  
    d_name = {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
    channels = {"H1": "FAKE-STRAIN", "H2": "FAKE-STRAIN", "L1": "FAKE-STRAIN", "V1": "FAKE_h_16384Hz_4R"}
    for ifo in opts.ifos:
        src = mkelem("gds_lvshmsrc", {'shm-name': d_name[ifo]})
        dmx = mkelem("framecpp_channeldemux", {'do-file-checksum':True, 'skip_bad_files': True})
        cnv = mkelem("audioconvert")
        resamp = mkelem("audioresample")
        caps_filt = mkelem("capsfilter", {'caps': gst.Caps("audio/x-raw-float, rate=2048")})
        art = mkelem("audiorate", {'skip-to-first': True, 'silent': True})
        qu1 = mkelem("queue", {'max-size-buffers':0, 'max-size-bytes':0, 'max-size-time':gst.SECOND * 60 })
        qu2 = mkelem("queue", {'max-size-buffers':0, 'max-size-bytes':0, 'max-size-time':gst.SECOND * 60 })
        ts_check=mkelem("lal_checktimestamps")
        art2 = mkelem("audiorate", {'skip-to-first': True, 'silent': True})
        if ifo == "V1":
            stv = mkelem("lal_statevector", {'required-on': 12, 'required-off': ~12 & 0xffffffff})
            pipeparts.src_deferred_link(dmx, "%s:%s" % (ifo, "FAKE_Hrec_Flag_Quality"), qu2.get_pad("sink"))
        else:
            stv = mkelem("lal_statevector", {'required-on': 45})
            pipeparts.src_deferred_link(dmx, "%s:%s" % (ifo, "FAKE-STATE_VECTOR"), qu2.get_pad("sink"))
        gate = mkelem("lal_gate", {'threshold': 1, 'emit-signals':True})        
        appsink = mkelem("appsink", {'caps': gst.Caps("audio/x-raw-float"), 'sync': False,
        'async': False, 'emit-signals': True, 'max-buffers': 1, 'drop': True})
        whiten = mkelem("lal_whiten", {'psd-mode': 0, 'zero-pad': 0, 'fft-length': 8,
        'median-samples': 7, 'average-samples': 128})
        fakesink = mkelem("fakesink", {'sync': False, 'async': False})
        
        pipe.add(src, dmx, cnv, qu1, resamp, caps_filt, art, gate, fakesink, whiten, appsink,qu2, art2, stv)
        gst.element_link_many(src, dmx)
        pipeparts.src_deferred_link(dmx, "%s:%s" % (ifo, channels[ifo]), cnv.get_pad("sink"))
        gst.element_link_many(cnv,qu1, resamp, caps_filt, art, gate, whiten, fakesink)
        gst.element_link_many(qu2, art2, stv)    
        stv.link_pads("src", gate, "control")
        whiten.link_pads("mean-psd", appsink,"sink")

        # hook updater to appsink
        deltaF = whiten.get_property("delta-f")
        appsink.connect_after("new-buffer", generate_update_callback(lock, opts, ifo, deltaF))
    
    return pipe

def run_pipeline(pipeline):
    print "Setting state to PAUSED:", pipeline.set_state(gst.STATE_PAUSED)
    print pipeline.get_state()
    mainloop = gobject.MainLoop()
    handler = simplehandler.Handler(mainloop, pipeline)
    print "Setting state to PLAYING:", pipeline.set_state(gst.STATE_PLAYING)
    mainloop.run()
    return

#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


opts, args = parse_args()

color_dict = {"H1": "red", "H2": "blue", "L1": "green", "V1": "magenta"}

# Set up storage for the history of (time, horizon distance) plus the plot's line collections
times = {}
h_dist = {}
lines = {}  # each IFO can have multiple traces, so this dict maps to a list of lines

#Horizon Distance Plot
fig = plt.figure()
ax1 = fig.add_axes((0.1, 0.1, 0.75, 0.88))
for ifo in opts.ifos:
    if ifo == opts.primary_ifo: 
        history_len = 7
    else:
        history_len = 1
    times[ifo] = deque(maxlen = history_len * 86400/4)  # FIXME: unhardcode 4
    h_dist[ifo] = deque(maxlen = history_len * 86400/4)  # FIXME: unhardcode 4

    # load history
    for fname in opts.history_files:
        if os.path.basename(fname).startswith(ifo) and os.path.exists(fname):
            npz = np.load(fname)
            times[ifo].extend(npz["%s_times" % ifo])
            h_dist[ifo].extend(npz["%s_horizon_distance" % ifo])

    ifo_lines = lines.setdefault(ifo, [])
    line, = ax1.plot([], [], '*-', color=color_dict[ifo], markersize=1.5, lw=1, mec=color_dict[ifo], label=ifo)
    ifo_lines.append(line)
    for j in range(1, history_len):  # add marker-less lines for additional history
        line, = ax1.plot([], [], '-', color=color_dict[ifo], markersize=1, lw=0.5, mec=color_dict[ifo],
            alpha=1. - j / history_len, label="_nolegend_"  )
        ifo_lines.append(line)

vline, = ax1.plot([], [], '-', c='black', lw=0.5)
lines["now"] = [vline]

ax1.grid(True)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_xlim((0, 24))
ax1.set_ylim((330, 360))
ax1.set_xticks(range(25))
ax1.set_xlabel('UTC hour')  # FIXME: Make relative to local time of main IFO
ax1.set_ylabel('Horizon distance (Mpc)')


#3D Network Response Plot
fig2 = plt.figure()
ax2 = p3.Axes3D(fig2)
#line_net_resp, = ax2.plot_surface([], [], [], cmap=col_map)
theta = np.linspace(0, LAL_PI, 401)
phi = np.linspace(0, LAL_TWOPI, 401, endpoint=False)
theta, phi = np.meshgrid(theta, phi)



# Start interactive plotting mode
plt.ion()

# Initialize lock for threading
lock = thrd.Lock()

#create pipeline
pipeline = make_pipline(opts, fig, lines, times, h_dist, lock)

#Add signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

#run threads to save data and plots
t1 = thrd.Thread(target=run_pipeline, args=(pipeline,))
t2 = thrd.Thread(target=update_plot, args=(lock, opts.ifos, opts.figure_path, times, h_dist, theta, phi))
t1.daemon = t2.daemon = True
t1.start()
t2.start()


while 1:
    time.sleep(100)


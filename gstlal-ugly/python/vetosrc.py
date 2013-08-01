#!/usr/bin/python
#
# Copyright (C) 2013 Branson Stephens and Chris Pankow
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import pygtk
pygtk.require("2.0")
import pygst
pygst.require("0.10")
import gobject
import gst

import glob
import gzip
import os
import time
from collections import deque
import numpy as np


#
# =============================================================================
#
#                                Utility Functions
#
# =============================================================================
#


# Older versions of numpy.load choke on .gz files.
# Not sure what version fixes this, but the numpy package 
# for debian squeeze (1.4.1) doesn't. See
# https://github.com/numpy/numpy/issues/1593

def wrapNpLoad(filePath):
    if filePath.endswith('gz'):
        fileObj = gzip.open(filePath)
        return np.load(fileObj)
    else:
        return np.load(filePath)


#
# =============================================================================
#
#                            Pipeline Utility Functions
#
# =============================================================================
#

# Pad probe handler.  Useful for debugging.
#def probeBufferHandler(pad,gst_buffer):
#    print 'gpsstart  = %s' % gst_buffer.timestamp
#    print 'offset    = %s' % gst_buffer.offset
#    return True

# A class for handling the veto timeseries files.

class vetoSource:
    def __init__(self, inputPath, inputPre, inputExt, waitTime):
        # get the file list
        pathPrefix = os.path.join(inputPath,inputPre)
        self.pathPrefix = pathPrefix
        self.inputPre   = inputPre
        self.inputExt   = inputExt
        self.waitTime   = waitTime

        # Initialize the list of files.
        self.check_for_new_files()
        # This is the offset of a given buffer with respect to the global stream in 
        # units of *samples*.
        self.current_offset = 0

        # Determine the rate by looking at the first file in the list.
        # XXX If the files with a given prefix have different rates, this will break.
        # Assume file names are of the form:
        # <input_prefix><training_start>_<training_end>-<gpsstart>-<duration><input_ext>
        # Ex:
        # L-KW_TRIGGERS_ovl_1058313600_1058400000-1058411456-64.npy.gz
        # get the first one.
        filePath = self.fileQueue.popleft()
        # put it back, since this is just for informational purposes.
        self.fileQueue.extendleft([filePath])
        firstVals = wrapNpLoad(filePath)
        training_period, gps_start, rest = filePath[len(pathPrefix):].split('-')
        self.duration = int(rest[:rest.find(inputExt)])
        self.next_output_timestamp = int(gps_start) * gst.SECOND

        # The rate determination assumes the file contains N values with
        # t_i = gpsstart + (i-1)*Delta_t ,
        # such that the last data point is at time t = (gpsstart+duration)-Delta_t.
        # NOTE gstreamer expects an integer rate in Hz.
        # XXX This cast makes me uncomfortable.  That's why I added 0.1.
        self.rate = int(float(firstVals.size)/float(self.duration) + 0.1)
        # Now that we have the rate, we can set the caps for the appsrc
        self.caps = "audio/x-raw-float,width=64,depth=64,channels=1,rate=%d" % self.rate

    def check_for_new_files(self, timestamp=0):
        pattern = self.pathPrefix + '*' + self.inputExt
        def is_current_file(path):
            filePath = os.path.basename(path)
            rest = filePath[len(self.inputPre):]
            if len(rest)>0:
                return int(rest.split('-')[1])*gst.SECOND>=timestamp
            else:
                return None
        filePathList = filter(is_current_file, glob.glob(pattern))
        filePathList.sort()
        self.fileQueue = deque(filePathList)
        return

    def need_data(self, src, need_bytes=None):
        src.info("----------------------------------------------------")
        src.info("Received need-data signal, %s." % time.asctime())
        # Keep circling until you find a file.
        noFile = True
        while noFile:
            # Check if new data has arrived on disk.
            self.check_for_new_files(self.next_output_timestamp)
            try:
                filePath = self.fileQueue.popleft()
                noFile = False
            except IndexError:
                time.sleep(self.waitTime)

        # Get the gpsstart time from the filename.
        rest = filePath[len(self.pathPrefix):]
        gpsstart = int(rest.split('-')[1])
        # Let's re-derive the duration.  maybe it changed?
        rest = rest.split('-')[2]
        duration = int(rest[:rest.find(self.inputExt)])

        # Is this the file we were expecting?  If not, we'll wait 
        # to see if the file we want shows up.
        if gpsstart * gst.SECOND > self.next_output_timestamp:
            time.sleep(self.waitTime)
            self.check_for_new_files(self.next_output_timestamp)
            filePath = self.fileQueue.popleft()
            rest = filePath[len(self.pathPrefix):]
            gpsstart = int(rest.split('-')[1])
            rest = rest.split('-')[2]
            duration = int(rest[:rest.find(self.inputExt)])

            if gpsstart * gst.SECOND > self.next_output_timestamp:
                # Push a gap before continuing to process the file we have.
                buf = gst.buffer_new_and_alloc(0)
                buf.flag_set(gst.BUFFER_FLAG_GAP)
                buf.timestamp = self.next_output_timestamp
                gap_duration = gpsstart * gst.SECOND - buf.timestamp
                gap_samples = int ((gap_duration / gst.SECOND) * self.rate)
                buf.duration = gap_duration
                buf.offset = self.current_offset
                buf.offset_end = self.current_offset + gap_samples
                src.emit("push-buffer", buf)
                src.info("pushed gap with start=%d, duration=%d" %
                    (buf.timestamp/gst.SECOND,gap_duration/gst.SECOND))
                self.next_output_timestamp += buf.duration
                self.current_offset = buf.offset_end

        # Load the numpy array.
        src.info("processing %s" % filePath)
        veto_vals = wrapNpLoad(filePath)
        veto_vals = veto_vals.astype(np.float64)

        # Build the buffer.
        buffer_len = veto_vals.nbytes
        buf = gst.buffer_new_and_alloc(buffer_len)
        buf[:buffer_len-1] = np.getbuffer(veto_vals)
        buf.timestamp = gpsstart * gst.SECOND
        # gst buffers require:
        # buffer_duration * rate / gst.SECOND = (offset_end - offset)
        # The offset is zero since our data begin at the beginning 
        # of the buffer.
        buf.duration = duration * gst.SECOND
        buf.offset = self.current_offset
        buf.offset_end = self.current_offset + duration * self.rate
        buf.caps = self.caps
        # Push the buffer into the stream (a side effect of 
        # emitting this signal).
        src.emit("push-buffer", buf)
        src.info("pushed buffer with start=%d, duration=%d" % (gpsstart,duration))
        self.next_output_timestamp += buf.duration
        self.current_offset = buf.offset_end
        return True




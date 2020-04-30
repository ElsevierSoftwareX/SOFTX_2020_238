#!/usr/bin/python
#
# Copyright (C) 2012 Karsten Wiesner
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
# ==============================================================================
#
#                                   Preamble
#
# ==============================================================================
#
"""
unittest for the gstlal GstLALAdder class
"""
__author__       = "Karsten Wiesner <karsten.wiesner@ligo.org>"
__copyright__    = "Copyright 2013, Karsten Wiesner"

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst
GObject.threads_init()
Gst.init(None)

from gstlal import simplehandler
from gstlal import pipeparts

import unittest
import random

import scipy
import numpy as np

from pylab import figure, show

    
class TestGstLALAdder(unittest.TestCase):

    def __init__(self, testname, 
                 DUT=                "adder", 
                 quiet=              True,
                 volume=             0.25,
                 bits_per_sample=    32,
                 sample_rate=        1000,
                 samples_per_buffer= 1000,
                 num_of_buffers=     6, 
                 timestamp_offs_B=   0,   # timestamp offset at InputB
                 spb_offs_B=         0,   # num of samples per buffer offset at InputB
                 num_bufs_offset_B=  0 ): # num_of_buffers offset at InputB
        
        super(TestGstLALAdder, self).__init__(testname)        
        
        random.seed()

        # tweak this member vars for individual testcases ----------------
        self.DUT=                DUT
        self.quiet=              quiet
        self.volume=             volume
        self.bits_per_sample=    bits_per_sample
        self.sample_rate=        sample_rate
        self.samples_per_buffer= samples_per_buffer
        self.num_of_buffers=     num_of_buffers    
        self.timestamp_offs_B=   timestamp_offs_B
        self.spb_offs_B=         spb_offs_B
        self.num_bufs_offset_B=  num_bufs_offset_B
        
        # internal member vars --------------------------------------------
        self.timestamp_offs_B_in_smpls= self.timestamp_offs_B/1e9*self.sample_rate
        self.idx_buf_dut_out= 0 # index of buffer send on DUT output
        self.accu_buf_dut_out=0 # accumutated output samples at dut out
        # set the numpy output arrays according bps:
        numpy_float_width_map= { 32:np.float32 , 64:np.float64 }
        self.numpy_float_width= numpy_float_width_map[self.bits_per_sample] 
        

    # Test Fixture --------------------------------------------------------
    def setUp(self):

        if(self.quiet==False):
            print()
            print("setUp with bps={0} np-preci={1}".format(self.bits_per_sample, 
                                                           self.numpy_float_width))

        self.pipeline = Gst.Pipeline(name="test_gstlal_adder")
        self.mainloop = GObject.MainLoop()
        self.handler =  simplehandler.Handler(self.mainloop, self.pipeline)
        
        self.dut_out_buf = np.array([])
        self.src_a_buf =   np.array([])
        self.src_b_buf =   np.array([])

        # InputA
        src_a = pipeparts.mkaudiotestsrc(self.pipeline, wave = 0, freq = 880, 
                                         samplesperbuffer = self.samples_per_buffer, 
                                         volume = self.volume, 
                                         num_buffers = self.num_of_buffers, 
                                         name= "InputA")
	capsfilt_a = pipeparts.mkcapsfilter(self.pipeline, src_a, 
        "audio/x-raw, rate={1}".format(self.sample_rate))
        tee_a = pipeparts.mktee(self.pipeline, capsfilt_a)
        if(self.quiet==False): 
            pipeparts.mknxydumpsink(self.pipeline, pipeparts.mkqueue(self.pipeline, 
                                    tee_a), "gstlal_adder_unittest_InputA.dump")

        # InputB
        src_b = pipeparts.mkaudiotestsrc(self.pipeline, wave = 0, freq = 666, 
                                         samplesperbuffer = self.samples_per_buffer+self.spb_offs_B, 
                                         volume = self.volume,
                                         num_buffers = self.num_of_buffers+self.num_bufs_offset_B, 
                                         name= "InputB", 
                                         timestamp_offset= self.timestamp_offs_B)
	capsfilt_b = pipeparts.mkcapsfilter(self.pipeline, src_b,
        "audio/x-raw, rate={1}".format(self.sample_rate))
        tee_b = pipeparts.mktee(self.pipeline, capsfilt_b)
        if(self.quiet==False):               
            pipeparts.mknxydumpsink(self.pipeline, pipeparts.mkqueue(self.pipeline, 
                                    tee_b), "gstlal_adder_unittest_InputB.dump")

        # DUT (Device Under Test)
        adder = Gst.ElementFactory.make(self.DUT, None)
        adder.set_property("name", "DUT")

        if (self.DUT == "lal_adder"):
            #adder.set_property("caps", ... takes a reference to the supplied GstCaps object.
            adder.set_property("sync", True)
        
        self.pipeline.add(adder)
        pipeparts.mkqueue(self.pipeline, tee_a).link(adder)
        pipeparts.mkqueue(self.pipeline, tee_b).link(adder)

        # Output
        tee_out = pipeparts.mktee(self.pipeline, adder)

        if (not self.quiet) and (self.bits_per_sample==32): # autoaudiosink can negotiate on 32 bps only
            pipeparts.mknxydumpsink(self.pipeline, pipeparts.mkqueue(self.pipeline, 
                                    tee_out), "gstlal_adder_unittest_Output.dump")
            sink = Gst.ElementFactory.make("autoaudiosink", None)
            self.pipeline.add(sink)
            pipeparts.mkqueue(self.pipeline, tee_out).link(sink)

        # Fetching buffers from the pipeline
        dut_out_appsink = pipeparts.mkappsink(self.pipeline, 
                                    pipeparts.mkqueue(self.pipeline, tee_out))
        dut_out_appsink.connect("new-buffer", self.on_dut_out_appsink_new_buffer)
        
        # fetch source a
        src_a_appsink = pipeparts.mkappsink(self.pipeline, 
                                    pipeparts.mkqueue(self.pipeline, tee_a))
        src_a_appsink.connect("new-buffer", self.on_src_a_appsink_new_buffer)
        
        # fetch source b
        src_b_appsink = pipeparts.mkappsink(self.pipeline, 
                                    pipeparts.mkqueue(self.pipeline, tee_b))
        src_b_appsink.connect("new-buffer", self.on_src_b_appsink_new_buffer)
        
       

    # callback on dut_out_appsink has received a buffer
    def on_dut_out_appsink_new_buffer(self, element):
       
        buffer = element.emit('pull-buffer')
        samples= len(buffer)/(self.bits_per_sample/8)
        self.accu_buf_dut_out += samples
        
        if(self.quiet==False):
            print("DUT send buffer no.: {0} of size: {1} bytes ; {2} samples ; accu-samples {3}".format( 
                self.idx_buf_dut_out, 
                buffer.data.__sizeof__(), 
                samples,
                self.accu_buf_dut_out))
        
        gst_app_buf= np.ndarray( shape= samples,
                                 buffer= buffer.data, 
                                 dtype= self.numpy_float_width )
        
        self.dut_out_buf = np.concatenate([self.dut_out_buf, gst_app_buf])
        self.idx_buf_dut_out += 1
        
        return True

    # callback on src_a_appsink has received a buffer
    def on_src_a_appsink_new_buffer(self, element):
        buffer = element.emit('pull-buffer')
        gst_app_buf= np.ndarray( shape= (self.samples_per_buffer),
                                 buffer=buffer.data, 
                                 dtype=self.numpy_float_width )
        self.src_a_buf = np.concatenate([self.src_a_buf, gst_app_buf])
        return True

    # callback on src_b_appsink has received a buffer
    def on_src_b_appsink_new_buffer(self, element):
        buffer = element.emit('pull-buffer')
        gst_app_buf= np.ndarray( shape= (self.samples_per_buffer+self.spb_offs_B),
                                 buffer=buffer.data, 
                                 dtype=self.numpy_float_width )
        self.src_b_buf = np.concatenate([self.src_b_buf, gst_app_buf])
        return True


        
    def tearDown(self):
        pass
        
    # Unit tests -------------------------------

    def test_1_plot_signals(self):
        """
        Test 1
        """
        self.pipeline.set_state(Gst.State.PLAYING)
        pipeparts.write_dump_dot(self.pipeline, "test_1_plot_signals", 
                                verbose = True)        
        self.mainloop.run()
 
        zoom_in = 50  # num of samples to display on each side of self.timestamp_offs_B_in_smpls

        # display left side offset region
        fig_l=figure()

        src_a_plot_l= fig_l.add_subplot(1,1,1)
        src_a_plot_l.set_xlim(self.timestamp_offs_B_in_smpls-zoom_in,
            self.timestamp_offs_B_in_smpls+zoom_in) 
        src_a_plot_l.grid()
        src_a_plot_l.plot(self.src_a_buf, 'g.', label='src a')

        src_b_plot_l= fig_l.add_subplot(1,1,1) 
        src_b_plot_l.set_xlim(self.timestamp_offs_B_in_smpls-zoom_in,
                              self.timestamp_offs_B_in_smpls+zoom_in)
        x = scipy.arange(len(self.src_b_buf)) + self.timestamp_offs_B_in_smpls
        src_b_plot_l.plot(x, self.src_b_buf, 'b.--', label='src b')

        dut_out_plot_l= fig_l.add_subplot(1,1,1) 
        dut_out_plot_l.set_xlim(self.timestamp_offs_B_in_smpls-zoom_in,
                                self.timestamp_offs_B_in_smpls+zoom_in)
        dut_out_plot_l.plot(self.dut_out_buf, 'r', label='DUT out')

        dut_out_plot_l.set_xlabel('samples')
        dut_out_plot_l.set_ylabel('volume')
        dut_out_plot_l.legend(loc='upper right')
        src_b_plot_l.set_title('Left side region adding buffer A and B(w/ offset timestamp)')

        # display right side offset region
        fig_r=figure()

        src_a_plot_r= fig_r.add_subplot(1,1,1)
        src_a_plot_r.set_xlim(len(self.dut_out_buf)-self.timestamp_offs_B_in_smpls-zoom_in,
                              len(self.dut_out_buf)-self.timestamp_offs_B_in_smpls+zoom_in) 
        src_a_plot_r.grid()
        src_a_plot_r.plot(self.src_a_buf, 'g.', label='src a')

        src_b_plot_r= fig_r.add_subplot(1,1,1) 
        src_b_plot_r.set_xlim(len(self.dut_out_buf)-self.timestamp_offs_B_in_smpls-zoom_in,
                              len(self.dut_out_buf)-self.timestamp_offs_B_in_smpls+zoom_in)
        x = scipy.arange(len(self.src_b_buf)) + self.timestamp_offs_B_in_smpls
        src_b_plot_r.plot(x, self.src_b_buf, 'b.--', label='src b')

        dut_out_plot_r= fig_r.add_subplot(1,1,1) 
        dut_out_plot_r.set_xlim(len(self.dut_out_buf)-self.timestamp_offs_B_in_smpls-zoom_in,
                                len(self.dut_out_buf)-self.timestamp_offs_B_in_smpls+zoom_in)
        dut_out_plot_r.plot(self.dut_out_buf, 'r', label='DUT out')

        dut_out_plot_r.set_xlabel('samples')
        dut_out_plot_r.set_ylabel('volume')
        dut_out_plot_r.legend(loc='upper left')
        src_b_plot_r.set_title('Right side region adding buffer A and B(w/ offset timestamp)')

        # display all
        fig_all=figure()
        dut_out_all=fig_all.add_subplot(1,1,1)
        dut_out_all.plot(self.src_a_buf, 'g.', label='src a')
        x = scipy.arange(len(self.src_b_buf)) + self.timestamp_offs_B_in_smpls
        dut_out_all.plot(x, self.src_b_buf, 'b.--', label='src b')
        dut_out_all.plot(self.dut_out_buf, 'r', label='DUT out')

        dut_out_all.set_xlabel('samples')
        dut_out_all.set_ylabel('volume')
        dut_out_all.legend(loc='upper right')
        dut_out_all.set_title('Adding buffer A and B(w/ offset timestamp)')

        show()


    def test_2_32bps(self):
        """
        Test 1
        """
        self.pipeline.set_state(Gst.State.PLAYING)

        if(self.quiet==False):
            pipeparts.write_dump_dot(self.pipeline, "test_2_quiet_at_32bps", 
                                     verbose = True)
        self.mainloop.run()

        offs= np.zeros((self.timestamp_offs_B_in_smpls), dtype=float)
        a= np.append(self.src_a_buf, offs)
        b= np.append(offs, self.src_b_buf)

        np_diff= a + b - self.dut_out_buf
        absmax= np.amax(np.absolute(np_diff))

        if(self.quiet==False):
            print() 
            print("maximum of absolute differences from numpy reference add= {0}".format(absmax))
            self.failIf(absmax >= 1.8e-8, "OOOPS!")

        #display all
        #fig_all=figure()
        #np_out_all=fig_all.add_subplot(1,1,1)
        #np_out_all.plot(a+b, 'r')
        #show()

    def test_3_64bps(self):
        """
        Test 1
        """
        self.pipeline.set_state(Gst.State.PLAYING)

        if(self.quiet==False):
            pipeparts.write_dump_dot(self.pipeline, "test_3_quiet_at_64bps", 
                                     verbose = True)
        self.mainloop.run()

        offs= np.zeros((self.timestamp_offs_B_in_smpls), dtype=float)
        a= np.append(self.src_a_buf, offs)
        b= np.append(offs, self.src_b_buf)

        np_diff= a + b - self.dut_out_buf
        absmax= np.amax(np.absolute(np_diff))

        if(self.quiet==False):
            print() 
            print("maximum of absolute differences from numpy reference add= {0}".format(absmax))
            self.failIf(absmax >= 1.0e-20, "OOOPS!")

        #display all
        #fig_all=figure()
        #np_out_all=fig_all.add_subplot(1,1,1)
        #np_out_all.plot(a+b, 'r')
        #show()


# create, customize and run test suite: ------------------------------------------------------

suite= unittest.TestSuite()

#suite.addTest(TestGstLALAdder("test_1_plot_signals"))   # test with the regular adder

#suite.addTest(TestGstLALAdder("test_1_plot_signals", 
#                              DUT= "lal_adder", 
#                              quiet=              False, 
#                              volume=             0.25,
#                              bits_per_sample=    32,
#                              sample_rate=        1000,
#                              samples_per_buffer= 1000,
#                              num_of_buffers=     6, 
#                              timestamp_offs_B=   0.2e9, ## 200 sp (1/sr * spb)
#                              spb_offs_B=         200,
#                              num_bufs_offset_B=  -1))

#suite.addTest(TestGstLALAdder("test_1_plot_signals", 
#                              DUT= "lal_adder", 
#                              quiet=              False, 
#                              volume=             0.25,
#                              bits_per_sample=    32,
#                              sample_rate=        44100,
#                              samples_per_buffer= 1000,
#                              num_of_buffers=     100, 
#                              timestamp_offs_B=   0.0068e9, ## 300 sp (1/sr * spb)
#                              spb_offs_B=         250,      ## 80 buffers at 1250 samples 
#                              num_bufs_offset_B=  -20))     ## ==> 100000 samples


suite.addTest(TestGstLALAdder("test_2_32bps"))   # test with default settings:
suite.addTest(TestGstLALAdder("test_3_64bps"))   # regular gst adder, no timeshift

suite.addTest(TestGstLALAdder("test_2_32bps",
                              DUT= "lal_adder", 
                              quiet=              True, 
                              volume=             0.25,
                              bits_per_sample=    32,
                              sample_rate=        1000,
                              samples_per_buffer= 1000,
                              num_of_buffers=     6, 
                              timestamp_offs_B=   0.2e9,
                              spb_offs_B=         200,
                              num_bufs_offset_B=  -1))

suite.addTest(TestGstLALAdder("test_3_64bps",
                              DUT= "lal_adder", 
                              quiet=              True, 
                              volume=             0.25,
                              bits_per_sample=    64,
                              sample_rate=        1000,
                              samples_per_buffer= 1000,
                              num_of_buffers=     6, 
                              timestamp_offs_B=   0.2e9,
                              spb_offs_B=         200,
                              num_bufs_offset_B=  -1))


# load all:
# suite = unittest.TestLoader().loadTestsFromTestCase(TestGstLALAdder)

unittest.TextTestRunner(verbosity=5).run(suite)



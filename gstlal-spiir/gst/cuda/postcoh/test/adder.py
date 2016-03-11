#!/bin/python

import sys, getopt
from gi.repository import Gst
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")
import gst

def main(argv):

    # Read Args From The Command Line
    print "Gstreamer Adder"
    afile   = ''
    bfile   = ''
    ofile   = ''
    biname  = sys.argv[0]
    try:
        opts, args = getopt.getopt(argv, "ha:b:o:", ["afile=", "bfile=", "ofile="])
    except getopt.GetoptError:
        print biname, '-a <input_0> -b <input_1> -o <output>'
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print biname, '-a <input_0> -b <input_1> -o <output>'
            sys.exit()
        elif opt in ("-a", "--afile"):
            afile = arg
        elif opt in ("-b", "--bfile"):
            bfile = arg
        elif opt in ("-o", "--ofile"):
            ofile = arg
    print "[Input]  a: ", afile, " b: ", bfile
    print "[Output] o: ", ofile

    # Build The Pipeline
    launch_src = "gst-launch "
    launch_src = launch_src + "filesrc location=" + afile
    launch_src = launch_src + " ! capsfilter caps=audio/x-raw-float,width=32,rate=8192,channels=1"
    launch_src = launch_src + " ! adder name=mix"
    launch_src = launch_src + " ! capsfilter caps=audio/x-raw-float,width=32,rate=8192,channels=1"
    launch_src = launch_src + " ! filesink location=" + ofile
    launch_src = launch_src + " filesrc location=" + bfile
    launch_src = launch_src + " ! capsfilter caps=audio/x-raw-float,width=32,rate=8192,channels=1"
    launch_src = launch_src + " ! mix."
    print launch_src

    # eval(launch_src)
    eval("gst-launch")

if __name__ == "__main__":
    main(sys.argv[1:])

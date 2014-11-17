# Copyright (C) 2014 Chris Pankow
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
"""Module holding all gstlal_excesspower related materials"""

import sys
import os
import math

from optparse import OptionParser, OptionGroup

import ConfigParser
from ConfigParser import SafeConfigParser

from pylal import datatypes as laltypes

from glue.ligolw import ligolw
from glue.ligolw import utils as ligolw_utils

from pylal.series import read_psd_xmldoc

from parts import EPHandler

#
# =============================================================================
#
#                             Options Handling
#
# =============================================================================
#

def append_options(parser=None):
    if parser is None:
        parser = OptionParser()

    parser.add_option("-f", "--initialization-file", dest="infile", help="Options to be pased to the pipeline handler. Required.", default=None)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", help="Be verbose.", default=False)
    parser.add_option("-S", "--stream-tfmap", dest="stream_tfmap", action="store", help="Encode the time frequency map to video as it is analyzed. If the argument to this option is \"video\" then the pipeline will attempt to stream to a video source. If the option is instead a filename, the video will be sent to that name. Prepending \"keyframe=\" to the filename will start a new video every time a keyframe is hit.")
    parser.add_option("-r", "--sample-rate", dest="sample_rate", action="store", type="int", help="Sample rate of the incoming data.")
    parser.add_option("-t", "--disable-triggers", dest="disable_triggers", action="store_true", help="Don't record triggers.", default=False)
    parser.add_option("-F", "--trigger-cache-name", dest="file_cache_name", action="store", help="Name of the trigger file cache to be written to. When specified, the LIGO cache entry of each trigger file is written to this file.")
    parser.add_option("-C", "--clustering", dest="clustering", action="store_true", default=False, help="Employ trigger tile clustering before output stage. Default or if not specificed is off." )
    parser.add_option("-m", "--enable-channel-monitoring", dest="channel_monitoring", action="store_true", default=False, help="Emable monitoring of channel statistics like even rate/signifiance and PSD power" )
    parser.add_option("-p", "--peak-over-sample-fraction", type=float, default=None, dest="peak_fraction", help="Take the peak over samples corresponding to this fraction of the DOF for a given tile. Default is no peak." )
    parser.add_option("-o", "--frequency-overlap", type=float, default=0.0, dest="frequency_overlap", help="Overlap frequency bands by this percentage. Default is 0." )
    parser.add_option("-d", "--drop-start-time", type=float, default=120.0, dest="drop_time", help="Drop this amount of time (in seconds) in the beginning of a run. This is to allow time for the whitener to settle to the mean PSD. Default is 120 s.")

    scan_sec = OptionGroup(parser, "Time-frequency scan", "Use these options to scan over a given segment of time for multiple time-frequency maps. Both must be specified to scan a segment of time. If the segment of time begins in the whitening segment, it will be clipped to be outside of it.")
    scan_sec.add_option("--scan-segment-start", type=float, dest="scan_start", help="Beginning of segment to scan.")
    scan_sec.add_option("--scan-segment-end", type=float, dest="scan_end", help="End of segment to scan.")
    parser.add_option_group(scan_sec)
    return parser


def process_options(options, gw_data_source_opts, pipeline, mainloop):
    # Locate and load the initialization file
    if not options.infile:
        print >>sys.stderr, "Initialization file required."
    elif not os.path.exists( options.infile ):
        print >>sys.stderr, "Initialization file path is invalid."
        sys.exit(-1)

    cfg = SafeConfigParser()
    cfg.read(options.infile)

    #
    # This supplants the ligo_data_find step and is mostly convenience
    #
    # TODO: Move to a utility library

    if gw_data_source_opts.data_source == "frames" and gw_data_source_opts.frame_cache is None:
        if gw_data_source_opts.seg is None:
            sys.exit("No frame cache present, and no GPS times set. Cannot query for data without an interval to query in.")

        # Shamelessly stolen from gw_data_find
        print "Querying LDR server for data location." 
        try:
            server, port = os.environ["LIGO_DATAFIND_SERVER"].split(":")
        except ValueError:
            sys.exit("Invalid LIGO_DATAFIND_SERVER environment variable set")
        print "Server is %s:%s" % (server, port)

        try:
            frame_type = cfg.get("instrument", "frame_type")
        except ConfigParser.NoOptionError:
            sys.exit("Invalid cache location, and no frame type set, so I can't query LDR for the file locations.")
        if frame_type == "":
            sys.exit("No frame type set, aborting.")

        print "Frame type is %s" % frame_type
        connection = datafind.GWDataFindHTTPConnection(host=server, port=port)
        print "Equivalent command line is "
        # FIXME: Multiple instruments?
        inst = gw_data_source_opts.channel_dict.keys()[0]
        print "gw_data_find -o %s -s %d -e %d -u file -t %s" % (inst[0], gw_data_source_opts.seg[0], gw_data_source_opts.seg[1], frame_type)
        cache = connection.find_frame_urls(inst[0], frame_type, gw_data_source_opts.seg[0], gw_data_source_opts.seg[1], urltype="file", on_gaps="error")

        tmpfile, tmpname = tempfile.mkstemp()
        print "Writing cache of %d files to %s" % (len(cache), tmpname)
        with open(tmpname, "w") as tmpfile:
        	cache.tofile(tmpfile)
        connection.close()
        gw_data_source_opts.frame_cache = tmpname

    handler = EPHandler(mainloop, pipeline)

    # Enable the periodic output of trigger statistics
    if options.channel_monitoring:
        handler.channel_monitoring = True

    # If a sample rate other than the native rate is requested, we'll need to 
    # keep track of it
    if options.sample_rate is not None:
        handler.rate = options.sample_rate

    # Does the user want a cache file to track the trigger files we spit out?
    # And if so, if you give us a name, we'll update it every time we output,
    # else only at the end of the run
    if options.file_cache_name is not None:
        handler.output_cache_name = options.file_cache_name

    # Clustering on/off
    handler.clustering = options.clustering
    # Be verbose?
    handler.verbose = options.verbose

    # Instruments and channels
    # FIXME: Multiple instruments
    if len(gw_data_source_opts.channel_dict.keys()) == 1:
        handler.inst = gw_data_source_opts.channel_dict.keys()[0]
    else:
        sys.exit("Unable to determine instrument.")

    # FIXME: Multiple instruments
    if gw_data_source_opts.channel_dict[handler.inst] is not None:
        handler.channel = gw_data_source_opts.channel_dict[handler.inst]
    else:
        # TODO: In the future, we may request multiple channels for the same 
        # instrument -- e.g. from a single raw frame
        sys.exit("Unable to determine channel.")
    print "Channel name(s): " + handler.channel 

    # FFT and time-frequency parameters
    # Low frequency cut off -- filter bank begins here
    handler.flow = cfg.getfloat("tf_parameters", "min-frequency")
    # High frequency cut off -- filter bank ends here
    handler.fhigh = cfg.getfloat("tf_parameters", "max-frequency")
    # Frequency resolution of the finest filters
    handler.base_band = cfg.getfloat("tf_parameters", "min-bandwidth")
    # Tile duration should not exceed this value
    handler.max_duration = cfg.getfloat("tf_parameters", "max-duration")
    # Number of resolutions levels. Can't be less than 1, and can't be greater
    # than log_2((fhigh-flow)/base_band)
    handler.max_bandwidth = cfg.getfloat("tf_parameters", "max-bandwidth")
    handler.max_level = int(math.floor(math.log(handler.max_bandwidth / handler.base_band, 2)))+1
    # Frequency band overlap -- in our case, handler uses 1 - frequency overlap
    if options.frequency_overlap > 1 or options.frequency_overlap < 0:
        sys.exit("Frequency overlap must be between 0 and 1.")
    handler.frequency_overlap = options.frequency_overlap

    # DOF options -- this affects which tile types will be calculated
    if cfg.has_option("tf_parameters", "max-dof"):
        handler.max_dof = cfg.getint("tf_parameters", "max-dof")
    if cfg.has_option("tf_parameters", "fix-dof"):
        handler.fix_dof = cfg.getint("tf_parameters", "fix-dof")

    if cfg.has_option("tf_parameters", "fft-length"):
        handler.fft_length = cfg.getfloat("tf_parameters", "fft-length")

    if cfg.has_option("cache", "cache-psd-every"):
        handler.cache_psd = cfg.getint("cache", "cache-psd-every")
        print "PSD caching enabled. PSD will be recorded every %d seconds" % handler.cache_psd
    else:
        handler.cache_psd = None

    if cfg.has_option("cache", "cache-psd-dir"):
        handler.cache_psd_dir = cfg.get("cache", "cache-psd-dir")
        print "Caching PSD to %s" % handler.cache_psd_dir
        
    # Used to keep track if we need to lock the PSD into the whitener
    psdfile = None
    if cfg.has_option("cache", "reference-psd"):
        psdfile = cfg.get("cache", "reference-psd")
        try:
            #FIXME: This will continue to complain about the ContentHandler, but
            # the parsing fails if provided with one.
            #handler.psd = read_psd_xmldoc(ligolw_utils.load_filename(psdfile, contenthandler = ligolw.LIGOLWContentHandler))[handler.inst]
            handler.psd = read_psd_xmldoc(ligolw_utils.load_filename(psdfile))[handler.inst]
            print "Reference PSD for instrument %s from file %s loaded" % (handler.inst, psdfile)
            # Reference PSD disables caching (since we already have it)
            handler.cache_psd = None
            handler.psd_mode = 1
        except KeyError: # Make sure we have a PSD for this instrument
            sys.exit( "PSD for instrument %s requested, but not found in file %s. Available instruments are %s" % (handler.inst, psdfile, str(handler.psd.keys())) )

    # Triggering options
    if cfg.has_option("triggering", "output-file-stride"):
        handler.dump_frequency = cfg.getint("triggering", "output-file-stride")
    if cfg.has_option("triggering", "output-directory"):
        handler.outdir = cfg.get("triggering", "output-directory")
    if cfg.has_option("triggering", "output-dir-format"):
        handler.outdirfmt = cfg.get("triggering", "output-dir-format")

    handler.output = not options.disable_triggers

    # FAP thresh overrides SNR thresh, because multiple resolutions will have 
    # different SNR thresholds, nominally.
    if cfg.has_option("triggering", "snr-thresh"):
        handler.snr_thresh = cfg.getfloat("triggering", "snr-thresh")
    if cfg.has_option("triggering", "fap-thresh"):
        handler.fap = cfg.getfloat("triggering", "fap-thresh")

    if handler.fap is not None:
        print "False alarm probability threshold (in Gaussian noise) is %g" % handler.fap
    if handler.snr_thresh is not None:
        print "Trigger SNR threshold sqrt(E/ndof-1) is %f" % handler.snr_thresh

    # Maximum number of events (+/- a few in the buffer) before which we drop an
    # output file
    if cfg.has_option("triggering", "events_per_file"):
        handler.max_events = cfg.get_int("triggering", "events_per_file")

    return handler

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import sys

try:
    import nds2
except ImportError:
    sys.exit('ERROR: Deactivate virtualenv and try running again')

from astropy.time import Time
import argparse
import numpy as np
import os
import scipy.io as sio
import scipy.signal as sig
import time


def parse_command_line():
    """
    parse command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--duration", "-d",
                        help    = "data segment duration",
                        default = 2048,
        		        dest    = "duration",
                        type    = int)

    parser.add_argument('--fname', '-f',
                        help    = "channel list file name",
                        default = "ChanList_darm.txt",
                        dest    = "fname",
                        type    = str)

    parser.add_argument("--fsup", "-fs",
                        help    = "sample frequency",
                        default = 256,
        		        dest    = "fsup",
                        type    = int)

    parser.add_argument("--ifo", "-i",
                        help    = "interferometer: L1 or H1",
                        default = "L1",
        		        dest    = "ifo",
                        type    = str)

    parser.add_argument("--output", "-o",
                        help    = "output file name",
                        default = None,
        		        dest    = "output")

    parser.add_argument("--portNumber", "-p",
                        help    = "port to connect to",
                        default = 31200,
        		        dest    = "portNumber",
                        type    = int)

    parser.add_argument("--times", "-t",
                        help    = "start time. Ex. 2017-01-04 11:40:00",
                        default = "2017-01-04 11:40:00",
        		        dest    = "times",
                        type    = str)

    params = parser.parse_args()

    return params


# Unpack the command line args
params     = parse_command_line()
dur        = params.duration
fname      = params.fname
fsup       = params.fsup
ifo        = params.ifo
output     = params.output
portNumber = params.portNumber
times      = params.times

nds_osx = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/'
nds_sandbox = '/usr/lib/python2.7/dist-packages/'

if os.path.exists(nds_osx):
    sys.path.append(nds_osx)
elif os.path.exists(nds_sandbox):
    sys.path.append(nds_sandbox)

# channel names
chan_head = ifo + ':'

with open(fname, 'r') as f:
    chanlines = f.read().split()
channels = [chan_head + line for line in chanlines]

# Get data start time
if ifo == 'L1':
    ndsServer = 'nds.ligo-la.caltech.edu'
elif ifo == 'H1':
    ndsServer = 'nds.ligo-wa.caltech.edu'
else:
    sys.exit("unknown IFO specified")

# Setup connection to the NDS
try:
    conn = nds2.connection(ndsServer, portNumber)
except RuntimeError:
    print('ERROR: Need to run `kinit albert.einstein` before nds2 '
          'can establish a connection')
    sys.exit(1)

# Setup start and stop times
t = Time(times, format='iso', scale='utc')
t_start = int(t.gps)

if __debug__:
    print("Output sample rate: {} Hz".format(fsup))
    print("Channel List:\n-------------")
    print("\n".join(channels))

print("Getting data from " + ndsServer + "...")
tic = time.time()
data = conn.fetch(t_start, t_start + dur, channels)

# get the data and stack it into a single matrix where the data are the columns
vdata = []
for k in range(len(channels)):
    fsdown = data[k].channel.sample_rate
    down_factor = int(fsdown // fsup)

    fir_aa = sig.firwin(20 * down_factor + 1, 0.8 / down_factor,
                        window='blackmanharris')

    # Prevent ringing from DC offset
    DC = np.mean(data[k].data)

    # Using fir_aa[1:-1] cuts off a leading and trailing zero
    downdata = sig.decimate(data[k].data, down_factor,
                            ftype = sig.dlti(fir_aa[1:-1], 1.0),
                            zero_phase = True)
    vdata.append(downdata)

# save to a hdf5 format
if output is None:
    funame = 'Data/' + ifo + '_data_array.mat'
else:
    funame = 'Data/' + output

sio.savemat(funame, mdict={'data': vdata, 'fsample': fsup, 'chans': channels},
            do_compression=True)

print("Data saved as " + funame)

if __debug__:
    print("Channel name is " + data[0].channel.name)
    print("Sample rate is " + str(data[0].channel.sample_rate) + " Hz")
    print("Number of samples is " + str(data[0].length))
    print("GPS Start time is " + str(data[0].gps_seconds))
    print("Data retrieval time = " + str(round(time.time() - tic,3)) + " s")

#!/usr/bin/env python
# coding: utf-8
from __future__ import division

from checkSource import checkSource
checkSource()

import os
import numpy as np
from scipy.io import loadmat, savemat
import scipy.signal as sig
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Agg')

from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras import optimizers

import theano

import NonlinearRegression.tools.analyze_run_data as nlr
from NonlinearRegression.tools import preprocessing as ppr
from NonlinearRegression.tools.bilinearHelper import (cost_filter,
                                                      downsample,
                                                      get_cbc,
                                                      load_data,
                                                      normalize,
                                                      plot_cost_asd,
                                                      plot_results,
                                                      plot_training_progress,
                                                      set_plot_style,
                                                      prepare_inputs,
                                                      theano_fft,
                                                      make_parallel,
                                                      subtract_cal)

# Hush tensorflow warnings about AVX instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_plot_style()

def run_test(
    # Training data
    val_frac = 1/4,    # Amount of data to save for validation
    fs_slow  = 32,     # Resample seismic data to this freq
    Tchunk   = 1/4,    # Seconds of data used to predict each DARM sample
    Tbatch   = 1,      # How many seconds of DARM per gradient update
    Nepoch   = 8,      # Number of times to iterate over training data set

    # Neural Network
    Nlayers    = 9,       # Number of Dense layers
    Nlstm      = 3,       # Number of LSTM layers (spaced in powers of 2)
    cost_func  = 'mse',   # Mean Squared Error, i.e. PSD reduction
    optimizer  = 'adam',  # Seems to work well...
    activation = 'elu',   # "Exponential linear unit"
    dropout    = 0.05,    # maybe this helps in training
    verbose    = 1,       # Print training progress to stderr

    # Detector
    IFO        = 'L1',
    remove_cal = False,
    cal_freqs = [22.7, 23.3, 23.9],

    # Cost function
    zero_freqs = [6, 120],
    zero_order = [10, 2],
    pole_freqs = [12, 70],
    pole_order = [10, 2],

    # Output data and plots
    tfft      = 8,
    doPlots   = False,
    plotDir   = 'params/RealRCNN/Figures',
    save_data = False,

    # Whether to look for previously saved downsampled data
    doLoadDownsampled = False,

    # Do bbh optimization
    doCBC = False,
    mass1 = 38.9,
    mass2 = 32.8,
    f_lower = 15.0,
    f_upper = 64.0,

    # do parallel
    doParallel = False,
    gpu_count = 4,

    # Preprocessing
    doWhiten      = True,
    doCoarseGrain = False):

    '''
    Bilinear Regression Code to be run in Parameter Optimization Routine

    Example
    -------
    >>> from RealRCNN import run_test
    >>> output_data = run_test(**kwargs)
    >>> Validation_loss = output_data['history']['val_loss'][-1]
    '''

    ####################
    # Data preparation #
    ####################

    # Make sure the data loads when running from any directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    if doLoadDownsampled:
        if verbose:
            print('Using previous downsampled data!')
        # Load up downsampled datas
        filename = 'params/RealRCNN/DARM_with_bilinear_downsampled.mat'
        datas    = loadmat(filename)

        if datas['fs_slow'][0][0] != fs_slow:
            raise ValueError("Decimated sampling rate from previously saved "
                             " data (%.0f) different from requested (%.0f)" %
                             (datas['fs_slow'][0][0], fs_slow))

        angular     = datas['angular']
        # Npairs      = angular.shape[0]
        beam_spot   = datas['beam_spot']
        Nasc        = datas['Nasc']
        Nspots      = datas['Nspots']

        tar_raw     = datas['tar_raw']
        bg_raw      = datas['bg_raw']
        tar         = datas['tar'][0]
        bg          = datas['bg'][0]

        scale       = datas['scale'][0][0]
        invBP       = datas['invBP']
        fs          = datas['fs'][0][0]
        fs_slow     = datas['fs_slow'][0][0]
        down_factor = int(fs // fs_slow)

    else:
        # Load up data
        try:
            datafile = '../../../../MockData/Data/L1_data_array.mat'
        except IOError:
            print('Data not found!')

        # Unpack the data
        datas = loadmat(datafile)
        fs    = datas['fsample'][0][0]
        Nfast = 20  # how many fast channels


        # Coarse grain if necessary
        if doCoarseGrain:
            bg_raw, tar_raw, wit = ppr.coarseGrainWrap(datafile,
                                                       deltaFy   = 2 / fs,
                                                       data_type = "real")
            asc_controls = wit[:Nfast + 1]
            beam_spot    = wit[Nfast + 1:]
        else:
            tar_raw      = datas['data'][0]
            asc_controls = datas['data'][1:(Nfast+1)]  # {C,D}HARD_{P/Y} + SRC/MICH/PRC
            beam_spot    = datas['data'][(Nfast+1):]

        bg_raw = np.zeros_like(tar_raw)

        # Subtract calibration lines
        if remove_cal:
            tar = subtract_cal(tar_raw, fs, cal_freqs, t_chunk=100)
        else:
            tar = tar_raw

        BP, invBP = cost_filter(fs, zero_freqs, pole_freqs,
                                zero_order, pole_order)

        print("Filtering and Decimating...")

        # remove mean and normalize to std for nicer NN learning
        notch_freqs = [60]
        Q_notch     = 30

        for f_notch in notch_freqs:
            bnot, anot = sig.iirnotch(f_notch/(fs/2), Q_notch)
            tar        = sig.filtfilt(bnot, anot, tar)

        if doWhiten:
            import gwpy.timeseries
            tartime = gwpy.timeseries.TimeSeries(tar,
                                                 sample_rate = fs,
                                                 epoch       = 0.0,
                                                 dtype       = float)
            whitetar = tartime.whiten(1.0,0.5)

            bgtime = gwpy.timeseries.TimeSeries(bg_raw,
                                                sample_rate = fs,
                                                epoch       = 0.0,
                                                dtype       = float)
            whitebg = bgtime.whiten(1.0,0.5)

            Nasc   = asc_controls.shape[0]
            for jj in xrange(Nasc):
                asctime = gwpy.timeseries.TimeSeries(asc_controls[jj,:],
                                                     sample_rate = fs,
                                                     epoch       = 0.0,
                                                     dtype       = float)
                whiteasc = asctime.whiten(1.0,0.5)

                asc_controls[jj,:] = np.array(whiteasc)

        tar, scale = normalize(tar, filter=BP)
        bg, _      = normalize(bg_raw, filter=BP, scale=scale)

        # Get the witness signals ready for training.
        Nasc   = asc_controls.shape[0]
        print("There are " + str(Nasc) + " fast channels in the training inputs.")
        Nspots = beam_spot.shape[0]

        # Shape the ASC control signal with the same filter as DARM
        beam_spot, _ = normalize(beam_spot)
        angular  , _ = normalize(asc_controls, filter=BP)

        # Since we only care about the slow beam spot motion, we
        # don't need full rate information. Decimating the signal
        # reduces the input vector length and the number of neurons
        # we have to train.
        down_factor = int(fs // fs_slow)
        beam_spot   = downsample(beam_spot, down_factor)

        # save downsampled datas
        downsampled_datafile = 'params/RealRCNN/' + IFO + 'DARM_with_bilinear_downsampled.mat'
        datas = {}
        datas['angular']   = angular
        datas['beam_spot'] = beam_spot
        datas['fs']        = fs
        datas['fs_slow']   = fs_slow
        datas['tar_raw']   = tar_raw
        datas['bg_raw']    = bg_raw
        datas['tar']       = tar
        datas['bg']        = bg
        datas['scale']     = scale
        datas['invBP']     = invBP
        datas['Nasc']      = Nasc
        datas['Nspots']    = Nspots

        if not os.path.isfile(downsampled_datafile):
            os.system('mkdir -p params/RealRCNN')
            os.system('touch {}'.format(downsampled_datafile))

        savemat(downsampled_datafile, datas,
                do_compression=True)

    nfft = tfft * fs

    # How many DARM samples are saved for validation
    Nval = int(tar_raw.size * val_frac)

    # How many witness samples are used to predict each DARM sample
    Nchunk = int(Tchunk * fs)
    Nbatch = int(Tbatch * fs)

    Nang   = Nchunk
    Nspot  = Nchunk // down_factor

    # Select training and validation data segments
    training_target = tar[:-Nval]
    training_spt    = beam_spot[:, : -Nval // down_factor]
    training_ang    = angular[:, : -Nval]

    validation_target = tar[-Nval:]
    validation_bg     = bg[-Nval:]
    validation_spt    = beam_spot[:, -Nval // down_factor:]
    validation_ang    = angular[:, -Nval:]

    # Create stacked, strided input arrays
    training_input   = prepare_inputs(training_spt, training_ang, Nchunk)
    validation_input = prepare_inputs(validation_spt, validation_ang, Nchunk)

    # Rescale validation data back to DARM units
    validation_darm = validation_target * scale
    validation_bg  *= scale

    #############################################
    # Construct the neural network and train it #

    # Minimum loss is achieved when target - prediction = bg
    # Thus, MSE = mean(bg**2), i.e. var(bg)
    minLoss = np.var(bg)
    if verbose:
        print('Best achievable cost: {:5g}'.format(minLoss))

    max_features   = 100
    embedding_dims = 5
    maxlen         = training_input.shape[1]

    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))

    model.add(Dropout(0.1))
    model.add(Conv1D(filters     = 128,
                     kernel_size = 7,
                     strides     = 1,
                     padding     = 'same',
                     activation  = 'relu'))

    model.add(MaxPooling1D(7))

    for _ in range(3):
        model.add(Dropout(0.1))
        model.add(Conv1D(64, 2))

    model.add(MaxPooling1D(5))

    model.add(Flatten())

    layer_sizes = range(1, Nlayers)
    layer_sizes.reverse()
    for k in layer_sizes:
        model.add(Dense(2**k, activation=activation))

    model.add(Dense(1, activation='linear'))

    optimizer = optimizers.Adam(lr      = 0.01,
                                beta_1  = 0.9,
                                beta_2  = 0.999,
                                epsilon = 1e-08,
                                decay   = 0.0)
    #############################################

    # define custom cost function if doCBC is true
    if doCBC:
        if not (Nbatch == Nchunk):
            print "Nbatch must be equal to Nchunk for doCBC..."
            exit(0)

        # Compute waveform
        dt = 1.0 / fs
        tt, hp, hc = get_cbc(f_lower, dt, mass1=mass1, mass2=mass2)

        # Ensure waveform and data have same lengths
        if Nbatch > len(hp):
            hp = np.pad(hp,(0,Nbatch-len(hp)),'constant',constant_values=0.0)
            hc = np.pad(hc,(0,Nbatch-len(hc)),'constant',constant_values=0.0)
            if verbose:
                print "zero padding..."
        elif Nbatch < len(hp):
            print "Nbatch smaller than the length of the waveform..."
            exit(0)

        # Compute Theano fft of waveform
        hp_fft, hc_fft = theano_fft(hp), theano_fft(hc)
        hp_freq = np.fft.rfftfreq(len(hp), d=dt)

        idx = np.where((hp_freq >= f_lower) & (hp_freq <= f_upper))[0]

        hp_squared = theano.tensor.sum(theano.tensor.square(hp_fft[0,idx,:]),axis=1)
        hc_squared = theano.tensor.sum(theano.tensor.square(hc_fft[0,idx,:]),axis=1)
        hp_squared = hp_squared / theano.tensor.sum(hp_squared)
        hc_squared = hc_squared / theano.tensor.sum(hc_squared)

        def custom_objective(y_true, y_pred):
            # Difference of predicted and true
            ydiff = y_pred - y_true

            # FFT of the difference
            ydiff_fft = theano.tensor.fft.rfft(ydiff,norm='ortho')
            ydiff_sliced = ydiff_fft[idx,0,:]

            # PSD of the difference
            ydiff_squared = theano.tensor.sum(theano.tensor.square(ydiff_sliced),axis=1)

            # This is something to maximize
            matched_filter = hp_squared / ydiff_squared
            matched_filter_sum = theano.tensor.sum(matched_filter)

            # This is something to minimize
            cost = matched_filter_sum

            return cost

        cost_func = custom_objective

    if doParallel:
        model = make_parallel(model, gpu_count)

    model.compile(optimizer=optimizer, loss=cost_func)

    if verbose:
        print("Starting Network learning...")
    t_start = timer()

    # this does the actual training
    roomba = model.fit(training_input, training_target,
                       validation_data = (validation_input, validation_target),
                       batch_size      = Nbatch,
                       epochs          = Nepoch,
                       verbose         = verbose)
    if verbose:
        print(str(round(timer() - t_start)) + " seconds for Training.")

    #############################
    # Generate final prediction #
    #############################

    if verbose:
        print("Applying model to input data...")
    validation_out = model.predict(validation_input, batch_size=Nbatch)[:, 0]
    validation_out *= scale  # Scale to DARM units

    if verbose:
        print("Unwhitening target and output data...")
    validation_darm = sig.sosfilt(invBP, validation_darm)
    validation_bg   = sig.sosfilt(invBP, validation_bg)
    validation_out  = sig.sosfilt(invBP, validation_out)

    #########################
    # Save and plot results #
    #########################

    model.save('params/RealRCNN/' + IFO + '_RegressionModel.h5')
    if doPlots:
        if not os.path.isdir(plotDir):
            os.makedirs(plotDir)
        if verbose:
            print("making the plots")

        plot_cost_asd(tar, bg, fs, nfft, plotDir = plotDir)
        plot_training_progress(roomba,
                               plotDir = plotDir, minLoss = minLoss)

        plot_results(validation_darm, validation_out, validation_bg,
                     fs, nfft, plotDir = plotDir,
                     plotStrain = False)

    if doCBC:
        from NonlinearRegression.tools.bilinearHelper import compute_overlap

        ff, ptar = sig.welch(tar, fs=fs, nperseg=nfft)
        _,  pbg  = sig.welch(bg,  fs=fs, nperseg=nfft)
        ff_val,  pval = sig.welch(validation_out, fs=fs, nperseg=nfft)

        m_tar = compute_overlap(tfft, fs, f_lower, ff,     ptar, hp=hp, hc=hc)
        m_bg  = compute_overlap(tfft, fs, f_lower, ff,     pbg,  hp=hp, hc=hc)
        m_val = compute_overlap(tfft, fs, f_lower, ff_val, pval, hp=hp, hc=hc)

    if save_data:
        if verbose:
            print("Saving model and processing params...")
        model.summary()

        output_data = {'history': roomba.history,
                       'invBP': invBP,
                       'scale': scale,
                       'fs'   : fs,
                       'nfft' : nfft,
                       }
        savemat('params/RealRCNN/' + IFO + 'Results_TFregression.mat', output_data,
                    do_compression=True)

    PATH  = os.getcwd()
    fname = os.path.basename(__file__)
    nlr.organize_run_data(summary   = model.summary,
                          optimizer = model.optimizer,
                          function  = run_test,
                          PATH      = PATH,
                          name      = fname,
                          minLoss   = minLoss)

    plot_model(model, plotDir + '/model.png')

    return roomba.history['val_loss'][-1]


if __name__ == "__main__":
    import sys
    import argparse

    class helpfulParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('Error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = helpfulParser()
    parser.add_argument('-e', '--epochs',
                        default = 10,
                        type    = int,
                        help    = 'Number of training epochs. Defaults '
                                  ' to %(default)s')

    parser.add_argument('-N', '--Nlayers',
                        default = 10,
                        type    = int,
                        help    = 'Number of layers. Defaults '
                                  ' to %(default)s')

    doLoadDownsampled = False
    if os.path.exists('params/RealRCNN/DARM_with_bilinear_downsampled.mat'):
        doLoadDownsampled = True

    # Get parameters into global namespace
    args    = parser.parse_args()
    Nepoch  = args.epochs
    Nlayers = args.Nlayers

    run_test(Nepoch        = Nepoch,
             Nlayers       = Nlayers,
             save_data     = True,
             remove_cal    = False,
             doPlots       = True,
             doWhiten      = False,
             doCoarseGrain = False,
             doLoadDownsampled = doLoadDownsampled)

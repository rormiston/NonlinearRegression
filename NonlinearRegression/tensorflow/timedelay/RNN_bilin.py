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
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import plot_model
from keras import optimizers

import theano

from NonlinearRegression.tools import analyze_run_data as nlr
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
                                                      make_parallel)

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
    Nlayers    = 6,       # Number of Dense layers
    cost_func  = 'mse',   # Mean Squared Error, i.e. PSD reduction
    optimizer  = 'adam',  # Seems to work well...
    activation = 'elu',   # "Exponential linear unit"
    dropout    = 0.005,   # maybe this helps in training
    verbose    = 1,       # Print training progress to stderr

    # Cost function
    zero_freqs = [6, 130],
    zero_order = [10, 2],
    pole_freqs = [12, 70],
    pole_order = [9, 3],

    # Output data and plots
    tfft      = 8,
    doPlots   = False,
    plotDir   = 'params/RNN_bilin/Figures',
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
    doCoarseGrain = False):

    '''
    Bilinear Regression Code to be run in Parameter Optimization Routine


    Example:
    from RNN_bilin import run_test
    output_data = run_test(**kwargs)
    Validation_loss = output_data['history']['val_loss'][-1]

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
        filename = 'params/RNN_bilin/DARM_with_bilinear_downsampled.mat'
        datas    = loadmat(filename)

        if datas['fs_slow'][0][0] != fs_slow:
            raise ValueError("Decimated sampling rate from previously saved "
                             " data (%.0f) different from requested (%.0f)" %
                             (datas['fs_slow'][0][0], fs_slow))

        angular   = datas['angular']
        beam_spot = datas['beam_spot']
        Npairs    = angular.shape[0]

        tar_raw = datas['tar_raw']
        bg_raw  = datas['bg_raw']
        tar     = datas['tar'][0]
        bg      = datas['bg'][0]

        scale       = datas['scale'][0][0]
        invBP       = datas['invBP']
        fs          = datas['fs'][0][0]
        fs_slow     = datas['fs_slow'][0][0]
        down_factor = int(fs // fs_slow)

    else:
        # Load up data
        try:
            datafile = '../../../../MockData/DARM_with_bilinear.mat'
        except IOError:
            print('Data not found!')

        bg_raw, tar_raw, wit, fs = load_data(datafile)

        if doCoarseGrain:
            bg_raw, tar_raw, wit = ppr.coarseGrainWrap(datafile,
                                                       deltaFy   = 2 / fs,
                                                       data_type = "mock")

        BP, invBP = cost_filter(fs, zero_freqs, pole_freqs, zero_order,
                                pole_order)

        print("Filtering and Decimating...")
        # remove mean and normalize to std for nicer NN learning
        tar, scale = normalize(tar_raw, filter=BP)
        bg, _      = normalize(bg_raw,  filter=BP, scale=scale)

        # Get the witness signals ready for training.
        Npairs = wit.shape[0] // 2  # How many ASC + beam spot pairs

        # Shape the ASC control signal with the same filter as DARM
        beam_spot, _ = normalize(wit[:Npairs])
        angular, _   = normalize(wit[Npairs:], filter=BP)

        # Since we only care about the slow beam spot motion, we
        # don't need full rate information. Decimating the signal
        # reduces the input vector length and the number of neurons
        # we have to train.

        down_factor = int(fs // fs_slow)
        beam_spot = downsample(beam_spot, down_factor)

        # save downsampled datas
        downsampled_datafile = 'DARM_with_bilinear_downsampled.mat'
        datas = {}
        datas['angular']   = angular
        datas['beam_spot'] = beam_spot
        datas['bg']        = bg
        datas['bg_raw']    = bg_raw
        datas['fs']        = fs
        datas['fs_slow']   = fs_slow
        datas['invBP']     = invBP
        datas['scale']     = scale
        datas['tar_raw']   = tar_raw
        datas['tar']       = tar

        savemat(downsampled_datafile, datas, do_compression=True)

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
    training_spt    = beam_spot[:, :-Nval // down_factor]
    training_ang    = angular[:, :-Nval]

    validation_target = tar[-Nval:]
    validation_bg     =  bg[-Nval:]
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
    #############################################

    # Minimum loss is achieved when target - prediction = bg
    # Thus, MSE = mean(bg**2), i.e. var(bg)
    minLoss = np.var(bg)
    if verbose:
        print('Best achievable cost: {:5g}'.format(minLoss))

    # Define the network topology
    input_shape = (Npairs * (Nang + Nspot),)
    model = Sequential()

    model.add(LSTM(32,
                   batch_input_shape = (None, 1, training_input.shape[1]),
                   dropout           = 0.0,
                   recurrent_dropout = 0.1,
                   return_sequences  = True))

    model.add(Dense(32))

    model.add(LSTM(32,
                   dropout           = 0.0,
                   recurrent_dropout = 0.1,
                   return_sequences  = False))


    layers = range(1, Nlayers)
    layers.reverse()
    for l in layers:
        model.add(Dense(2**l, activation='linear'))

    model.add(Dense(1, activation='linear'))

    optimizer = 'adagrad'

    ################################################
    # Define custom cost function if doCBC is true #
    ################################################

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

    # Reshape data
    training_input = training_input.reshape((training_input.shape[0], 1,
                                             training_input.shape[1]))

    training_target  = training_target.reshape((training_target.shape[0],))

    validation_input = validation_input.reshape((validation_input.shape[0], 1,
                                                 validation_input.shape[1]))

    roomba = model.fit(training_input, training_target,
                       validation_data = (validation_input, validation_target),
                       batch_size      = Nbatch,
                       epochs          = Nepoch,
                       verbose         = verbose,
                       shuffle         = True)

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

    if doPlots:
        if not os.path.isdir(plotDir):
            os.makedirs(plotDir)

        plot_cost_asd(tar, bg, fs, nfft, plotDir = plotDir)
        plot_training_progress(roomba, plotDir = plotDir, minLoss = minLoss)

        title_str = 'Bilinear Noise of ' + str(Npairs) + ' pairs of channels'
        plot_results(validation_darm, validation_out, validation_bg,
                     fs, nfft, plotDir = plotDir, title_str=title_str)

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
        model.save('FF_RegressionModel.h5')

        output_data = {'history': roomba.history,
                       'invBP'  : invBP,
                       'scale'  : scale,
                       'fs'     : fs,
                       'nfft'   : nfft,
                       }

        savemat('Results_TFregression.mat', output_data, do_compression=True)

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
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of training epochs. Defaults '
                             ' to %(default)s')

    doLoadDownsampled = False
    if os.path.exists('params/RNN_bilin/DARM_with_bilinear_downsampled.mat'):
        doLoadDownsampled = True

    # Get parameters into global namespace
    args   = parser.parse_args()
    Nepoch = args.epochs

    run_test(Nepoch = Nepoch,
             doPlots = True,
             save_data = True,
             doCoarseGrain = True,
             doLoadDownsampled = doLoadDownsampled)

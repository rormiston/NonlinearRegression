#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import os
import numpy as np
from scipy.io import loadmat, savemat
import scipy.signal as sig
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Agg')

from keras import optimizers

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout

from NonlinearRegression.tools import analyze_run_data as nlr
from NonlinearRegression.tools.bilinearHelper import (cost_filter,
                                                      downsample,
                                                      load_data,
                                                      normalize,
                                                      plot_cost_asd,
                                                      plot_results,
                                                      plot_training_progress,
                                                      set_plot_style,
                                                      prepare_inputs)


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
    Nlayers    = 4,       # Number of Dense layers
    cost_func  = 'mse',   # Mean Squared Error, i.e. PSD reduction
    optimizer  = optimizers.Nadam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0),
    activation = 'elu',   # "Exponential linear unit"
    dropout    = 0.05,    # maybe this helps in training
    verbose    = 1,       # Print training progress to stderr
    # Cost function
    zero_freqs = [5],
    zero_order = [8],
    pole_freqs = [15],
    pole_order = [8],
    # Output data and plots
    tfft      = 8,
    doPlots   = False,
    plotDir   = 'params/scatterRegression/Figures',
    save_data = False,
    # Whether to look for previously saved downsampled data
    doLoadDownsampled = False):

    '''
    Scatter Regression Code to be run in Parameter Optimization Routine


    Example:
    from scatterRegression import run_test
    output_data = run_test(**kwargs)
    Validation_loss = output_data['history']['val_loss'][-1]

    '''

    ####################
    # Data preparation #
    ####################

    if doLoadDownsampled:
        if verbose:
            print('Using previous downsampled data!')
        # Load up downsampled datas
        filename = 'params/scatterRegression/DARM_with_scatter_downsampled.mat'
        datas    = loadmat(filename)

        if datas['fs_slow'][0][0] != fs_slow:
            raise ValueError("Decimated sampling rate from previously saved "
                             " data (%.0f) different from requested (%.0f)" %
                             (datas['fs_slow'][0][0], fs_slow))

        acoustic   = datas['acoustic']
        seismic = datas['seismic']
        Npairs    = acoustic.shape[0]

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
        # Load up datas
        try:
            datafile = '../../../../MockData/DARM_with_scatter.mat'
        except IOError:
            print('Data not found!')

        bg_raw, tar_raw, wit, fs = load_data(datafile)

        BP, invBP = cost_filter(fs, zero_freqs, pole_freqs, zero_order,
                                pole_order)

        print("Filtering and Decimating...")
        # remove mean and normalize to std for nicer NN learning
        tar, scale = normalize(tar_raw, filter=BP)
        bg, _      = normalize(bg_raw,  filter=BP, scale=scale)

        # Get the witness signals ready for training.
        Npairs = wit.shape[0] // 2  # How many ASC + beam spot pairs

        # Shape the ASC control signal with the same filter as DARM
        seismic, _ = normalize(wit[:Npairs])
        acoustic, _   = normalize(wit[Npairs:], filter=BP)

        # Since we only care about the slow beam spot motion, we
        # don't need full rate information. Decimating the signal
        # reduces the input vector length and the number of neurons
        # we have to train.

        down_factor = int(fs // fs_slow)
        seismic = downsample(seismic, down_factor)

        # save downsampled datas
        downsampled_datafile = 'DARM_with_scatter_downsampled.mat'
        datas = {}
        datas['acoustic'] = acoustic
        datas['seismic'] = seismic
        datas['fs'] = fs
        datas['fs_slow'] = fs_slow
        datas['tar_raw'] = tar_raw
        datas['bg_raw'] = bg_raw
        datas['tar'] = tar
        datas['bg'] = bg
        datas['scale'] = scale
        datas['invBP'] = invBP

        savemat(downsampled_datafile, datas, do_compression=True)

    nfft = tfft * fs

    # How many DARM samples are saved for validation
    Nval = int(tar_raw.size * val_frac)

    # How many witness samples are used to predict each DARM sample
    Nchunk = int(Tchunk * fs)
    Nbatch = int(Tbatch * fs)

    Nacous   = Nchunk
    Nseis  = Nchunk // down_factor

    # Select training and validation data segments
    training_target = tar[:-Nval]
    training_sei    = seismic[:, :-Nval // down_factor]
    training_acous    = acoustic[:, :-Nval]

    validation_target = tar[-Nval:]
    validation_bg     =  bg[-Nval:]
    validation_sei    = seismic[:, -Nval // down_factor:]
    validation_acous    = acoustic[:, -Nval:]

    # Create stacked, strided input arrays
    training_input = prepare_inputs(training_sei, training_acous, Nchunk)
    validation_input = prepare_inputs(validation_sei, validation_acous, Nchunk)

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

    # define the network topology  -- -- -- - - -  -  -   -   -    -
    input_shape = (Npairs * (Nacous + Nseis),)
    model = Sequential()
    model.add(Dense(input_shape[0], input_shape=input_shape,
                    activation='linear'))

    # this layer increases training time but not increase performance
    model.add(Dense(input_shape[0], activation=activation))
    model.add(Dropout(dropout))

    # add layers; decrease size of each by half
    layer_sizes = range(1, Nlayers)
    layer_sizes.reverse()
    for k in layer_sizes:
        model.add(Dense(2**k, activation=activation))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='linear'))
    #############################################

    model.compile(optimizer=optimizer, loss=cost_func)

    if verbose:
        print("Starting Network learning...")
    t_start = timer()

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

    if doPlots:
        if not os.path.isdir(plotDir):
            os.makedirs(plotDir)

        plot_cost_asd(tar, bg, fs, nfft, plotDir = plotDir)
        plot_training_progress(roomba, plotDir = plotDir, minLoss = minLoss)

        title_str = 'Scatter Noise of ' + str(Npairs) + ' pairs of channels'
        plot_results(validation_darm, validation_out, validation_bg,
                     fs, nfft, plotDir = plotDir, title_str=title_str)

    if save_data:
        if verbose:
            print("Saving model and processing params...")
        model.summary()
        model.save('FF_ScatterRegressionModel.h5')

        output_data = {'history': roomba.history,
                       'invBP': invBP,
                       'scale': scale,
                       'fs'   : fs,
                       'nfft' : nfft,
                       }
        savemat('Results_ScatterRegression.mat', output_data,
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
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of training epochs. Defaults '
                             ' to %(default)s')

    doLoadDownsampled = False
    if os.path.exists('params/scatterRegression/DARM_with_scatter_downsampled.mat'):
        doLoadDownsampled = True

    # Get parameters into global namespace
    args   = parser.parse_args()
    Nepoch = args.epochs

    run_test(Nepoch    = Nepoch,
             doPlots   = True,
             save_data = True,
             doLoadDownsampled = doLoadDownsampled)

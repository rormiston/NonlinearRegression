#!/usr/bin/env python
# -*- coding: utf-8 -*-
from checkSource import checkSource
checkSource()
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import re
import scipy.io as sio
import sys

import os
# Hush AVX and processor warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

from NonlinearRegression.tools import bilinearHelper as blh
from NonlinearRegression.tools import analyze_run_data as ard
from NonlinearRegression.tools import models as mod
from NonlinearRegression.tools import preprocessing as ppr
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error


# get plot formatting
blh.set_plot_style()


def run_network(
    # BBH Optimization
    doCBC   = False,
    mass1   = 38.9,
    mass2   = 32.8,
    f_lower = 15.0,
    f_upper = 64.0,

    # Data
    data_type    = 'real',
    datafile     = None,
    doFilter     = False,
    doDownsample = False,
    doLoops      = False,
    fs_slow      = 64,
    lookback     = 0.0,
    train_frac   = 0.75,

    # Detector
    ifo = 'L1',

    # Neural network params
    activation = 'elu',
    dropout    = 0.0,
    loss       = 'mse',
    model_type = 'LSTM',
    optimizer  = 'adam',
    recurrent_dropout = 0.00,

    # Optimizer
    beta_1   = 0.9,
    beta_2   = 0.999,
    decay    = None,
    epsilon  = 1e-8,
    lr       = None,
    momentum = 0.0,
    nesterov = False,
    rho      = None,

    # Plotting
    doPlots    = True,
    plotDir    = 'Plots/',
    plotStrain = False,
    save_data  = True,
    tfft       = 8,

    # Preprocessing
    doLines     = False,
    chans       = 'darm',
    width       = 1,
    notch_freqs = [60, 120],

    # Training
    batch_size = 4096,
    epochs     = 10,
    Nlayers    = 8,
    shuffle    = False,
    verbose    = 1):

    ############################
    # Load and preprocess data #
    ############################
    # Make sure the data loads when running from any directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    # Get the model basename and version number
    regex_model = re.compile(r'[a-zA-Z]+')
    basename    = regex_model.findall(model_type)[0]
    regex_num   = re.compile(r'\d+')
    try:
        version = regex_num.findall(model_type)[0]
    except IndexError:
        version = '0'

    # Get the datafile and load the dataset
    datafile = ppr.get_datafile(datafile, data_type, ifo=ifo)
    dataset, sample_rate = ppr.get_dataset(datafile, data_type=data_type)

    if doLoops:
        dataset = ppr.use_cleaned_data(dataset, basename)
        matfile = 'Results-TFregression-{0}.mat'.format(version)
        print('Datafile: params/{0}/{1}'.format(basename, matfile))
    else:
        print('Datafile: {}'.format(datafile))

    # Use an sos filter on witness channels and DARM
    if doFilter:
        dataset = ppr.filter_channels(dataset, fs=sample_rate)

    # Filter and decimate from `sample_rate` to `fs_slow`
    if doDownsample:
        dataset = ppr.downsample_dataset(dataset, sample_rate, fs_slow)
        sample_rate = fs_slow

    # Make sure the values are floats to prevent division errors
    values = dataset.astype('float32')

    # Notch supplied array of lines
    if doLines:
        values = ppr.remove_lines(values, sample_rate,
                                  chans       = chans,
                                  width       = width,
                                  notch_freqs = notch_freqs)
    # Normalize
    scaler    = MinMaxScaler(feature_range=(0, 1))
    stdscaler = StandardScaler()
    scaled    = stdscaler.fit_transform(values)
    scaled    = scaler.fit_transform(scaled)

    # Split into train and test sets
    train_len = int(train_frac * len(scaled))
    train     = scaled[:train_len, :]
    test      = scaled[train_len:, :]

    # Get training and testing feature matrices
    train_X = train[:, 1:]
    test_X  = test[:, 1:]

    # Reshape input to be 3D [samples, timesteps, features]
    train_X  = ppr.do_lookback(train_X, lookback, sample_rate)
    test_X   = ppr.do_lookback(test_X, lookback, sample_rate)

    # Targets have to start `lookback * fs_slow` timesteps in
    train_y = train[int(lookback * sample_rate):, 0]
    test_y  = test[int(lookback * sample_rate):, 0]

    ######################
    # Get network design #
    ######################
    input_shape = (None, train_X.shape[1], train_X.shape[2])
    model = mod.get_model(model_type  = model_type,
                          input_shape = input_shape,
                          dropout     = dropout,
                          Rdropout    = recurrent_dropout,
                          activation  = activation,
                          Nlayers     = Nlayers)

    optimizer = mod.get_optimizer(opt      = optimizer,
                                  decay    = decay,
                                  lr       = lr,
                                  momentum = momentum,
                                  nesterov = nesterov,
                                  beta_1   = beta_1,
                                  beta_2   = beta_2,
                                  epsilon  = epsilon,
                                  rho      = rho)

    if doCBC:
        print('Using custom doCBC loss function.')
        loss = ppr.get_cbc_loss(sample_rate = sample_rate,
                                batch_size  = batch_size,
                                notch_freqs = notch_freqs,
                                width       = width)

    model.compile(loss=loss, optimizer=optimizer)

    ###############
    # Fit network #
    ###############
    if verbose:
        print("Network: {}".format(basename))

    history = model.fit(train_X, train_y,
                        epochs          = epochs,
                        batch_size      = batch_size,
                        validation_data = (test_X, test_y),
                        verbose         = verbose,
                        shuffle         = shuffle)

    ##############################
    # Make Predictions and Plots #
    ##############################
    # make a prediction
    yhat   = model.predict(test_X)
    test_X = ppr.undo_lookback(test_X, lookback, sample_rate)

    # Zero pad untrained lookback portion
    for i in range(int(lookback * sample_rate)):
        yhat = np.insert(yhat, 0, 0)
        test_y = np.insert(test_y, 0, 0)
    yhat = yhat.reshape(len(yhat), 1)
    test_y = test_y.reshape(len(test_y), 1)

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = stdscaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y  = np.concatenate((test_y, test_X), axis=1)
    inv_y  = scaler.inverse_transform(inv_y)
    inv_y  = stdscaler.inverse_transform(inv_y)
    inv_y  = inv_y[:, 0]

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: {}'.format(rmse))

    # Make sure the output directories exist
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    if save_data:
        target = dataset[-len(inv_yhat):, 0]
        inv_yhat = ppr.filter_timeseries(target, inv_yhat)
        subtracted = target - inv_yhat

        matfile = 'params/{0}/subtraction_results-{1}.mat'.format(basename, version)
        ard.track_subtraction_progress(matfile, subtracted, target)

        output_data = {'history': history.history,
                       'fsample': sample_rate,
                       'subtracted': subtracted}

        mat_str  = 'params/{0}/Results_TFregression-{1}.mat'
        mat_name = mat_str.format(basename, version)

        if not os.path.isfile(mat_name):
            os.system('touch {0}'.format(mat_name))

        sio.savemat(mat_name, output_data, do_compression=True)

    if doPlots:
        # Plot history
        minLoss = np.var(scaled[:, 1])
        blh.plot_training_progress(history,
                                   plotDir = plotDir,
                                   minLoss = minLoss,
                                   version = version)

        # Plot cost
        blh.plot_cost_asd(scaled[:, 0], scaled[:, 1], sample_rate,
                          sample_rate * tfft,
                          plotDir = plotDir,
                          version = version)

        # Plot model
        plot_model(model, plotDir + '/model-{0}.png'.format(version))

        # Put everything in the right folders for the webpages
        PATH         = os.getcwd()
        data_path    = PATH + '/' + datafile
        model_params = parse_command_line()
        opt_name     = model_params['optimizer']
        ard.organize_run_data(summary      = model.summary,
                              optimizer    = optimizer,
                              opt_name     = opt_name,
                              model_params = model_params,
                              PATH         = PATH,
                              name         = model_type,
                              minLoss      = minLoss,
                              datafile     = data_path)

        # Plot PSDs
        target = dataset[-len(inv_yhat):, 0]
        bkgd   = scaled[-len(inv_yhat):, 1]
        ard.plot_psd(target, inv_yhat, bkgd, sample_rate, sample_rate * tfft,
                     plotDir  = plotDir,
                     basename = basename,
                     version  = version)


if __name__ == '__main__':


    def parse_command_line():
        """
        parse command line flags. use sensible defaults
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--model_type", "-m",
                            help    = "pick model type to use",
                            default = "LSTM",
                            dest    = "model_type",
                            type    = str)

        parser.add_argument("--train_frac",
                            help    = "ratio of dataset used for training",
                            default = 0.75,
                            dest    = "train_frac",
                            type    = float)

        parser.add_argument("--datafile",
                            help    = "data file to read from",
                            default = None,
                            dest    = "datafile",
                            type    = str)

        parser.add_argument("--data_type", "-d",
                            help    = "real or mock data",
                            default = "real",
                            dest    = "data_type",
                            type    = str)

        parser.add_argument("--dropout", "-D",
                            help    = "dropout regularization",
                            default = 0.0,
                            dest    = "dropout",
                            type    = float)

        parser.add_argument("--recurrent_dropout", "-RD",
                            help    = "recurrent dropout used in RNN memory blocks",
                            default = 0.0,
                            dest    = "recurrent_dropout",
                            type    = float)

        parser.add_argument("--loss",
                            help    = "loss function for neural network",
                            default = 'mse',
                            dest    = "loss",
                            type    = str)

        parser.add_argument("--activation", "-a",
            				help    = "activation function for neural network",
                            default = "elu",
            				dest    = "activation",
                            type    = str)

        parser.add_argument("--optimizer", "-opt",
                            help    = "optimizing function for neural network",
                            default = 'adam',
                            dest    = "optimizer",
                            type    = str)

        parser.add_argument("--epochs", "-e",
                            help    = "Number of iterations of NN training",
                            default = 10,
                            dest    = "epochs",
                            type    = int)

        parser.add_argument("--Nlayers", "-l",
            				help    = "Number of layers for the Dense network",
                            default = 8,
            				dest    = "Nlayers",
                            type    = int)

        parser.add_argument("--batch_size", "-b",
                            help    = "number of samples to be trained at once",
                            default = 4096,
                            dest    = "batch_size",
                            type    = int)

        parser.add_argument("--shuffle", "-s",
                            help    = "shuffle training data",
                            default = False,
                            dest    = "shuffle",
                            action  = 'store_true')

        parser.add_argument("--verbose", "-v",
                            help    = "output verbosity",
                            default = 1,
                            dest    = "verbose",
                            type    = int)

        parser.add_argument("--tfft",
                            help    = "Use to set overlapping segments for PSD",
                            default = 16,
                            dest    = "tfft",
                            type    = int)

        parser.add_argument("--plotDir",
                            help    = "directory to store plots",
                            default = 'Plots/',
                            dest    = "plotDir",
                            type    = str)

        parser.add_argument("--plotStrain",
                            help    = "plot Strain data",
                            default = False,
                            dest    = "plotStrain",
                            action  = 'store_true')

        parser.add_argument("--doPlots",
                            help    = "enable plotting",
                            default = True,
                            dest    = "doPlots",
                            action  = 'store_true')

        parser.add_argument("--lookback", "-lb",
                            help    = "number of SECONDS to look back",
                            default = 0.0,
                            dest    = "lookback",
                            type    = float)

        parser.add_argument("--learning_rate", "-lr",
                            help    = "optimizer learning rate",
                            default = None,
                            dest    = "lr",
                            type    = float)

        parser.add_argument("--decay",
                            help    = "optimizer learning rate decay",
                            default = None,
                            dest    = "decay",
                            type    = float)

        parser.add_argument("--momentum",
                            help    = "optimizer momentum",
                            default = 0.0,
                            dest    = "momentum",
                            type    = float)

        parser.add_argument("--nesterov",
                            help    = "use nesterov momentum",
                            default = False,
                            dest    = "nesterov",
                            action  = 'store_true')

        parser.add_argument("--beta_1",
                            help    = "beta_1 params for optimizer",
                            default = 0.9,
                            dest    = "beta_1",
                            type    = float)

        parser.add_argument("--beta_2",
                            help    = "beta_2 params for optimizer",
                            default = 0.999,
                            dest    = "beta_2",
                            type    = float)

        parser.add_argument("--epsilon",
                            help    = "optimizer param",
                            default = 1e-8,
                            dest    = "epsilon",
                            type    = float)

        parser.add_argument("--rho",
                            help    = "adadelta & rmsprop optimizer params",
                            default = None,
                            dest    = "rho",
                            type    = float)

        parser.add_argument("--interferometer", "-ifo",
                            help    = "L1 or H1",
                            default = "L1",
                            dest    = "ifo",
                            type    = str)

        parser.add_argument("--save_data",
                            help    = "save data to mat file",
                            default = True,
                            dest    = "save_data",
                            action  = 'store_true')

        parser.add_argument("--doLines",
                            help    = "remove lines from raw DARM",
                            default = False,
                            action  = 'store_true',
                            dest    = "doLines")

        parser.add_argument("--chans",
                            help    = "channel(s) to notch. Either 'darm' or 'all'",
                            default = 'darm',
                            dest    = "chans",
                            type    = str)

        parser.add_argument("--width",
                            help    = "notching bin width",
                            default = 1,
                            dest    = "width",
                            type    = int)

        parser.add_argument("--notch_freqs",
                            help    = "frequencies to notch",
                            default = [60, 120],
                            dest    = "notch_freqs",
                            nargs   = '+',
                            type    = float)

        parser.add_argument("--mass1",
                            help    = "mass of mirror 1",
                            default = 38.9,
                            dest    = "mass1",
                            type    = float)

        parser.add_argument("--mass2",
                            help    = "mass of mirror 2",
                            default = 32.8,
                            dest    = "mass2",
                            type    = float)

        parser.add_argument("--f_lower",
                            help    = "frequency band minimum for doCBC",
                            default = 12.0,
                            dest    = "f_lower",
                            type    = float)

        parser.add_argument("--f_upper",
                            help    = "frequency band maximum for doCBC",
                            default = 256.0,
                            dest    = "f_upper",
                            type    = float)

        parser.add_argument("--fs_slow",
                            help    = "downsample rate",
                            default = 64,
                            dest    = "fs_slow",
                            type    = int)

        parser.add_argument("--doCBC",
                            help    = "custom loss for BBH optimization",
                            default = False,
                            action  = 'store_true',
                            dest    = 'doCBC')

        parser.add_argument("--doFilter",
                            help    = "custom loss for BBH optimization",
                            default = False,
                            action  = 'store_true',
                            dest    = 'doFilter')

        parser.add_argument("--doDownsample",
                            help    = "custom loss for BBH optimization",
                            default = False,
                            action  = 'store_true',
                            dest    = 'doDownsample')

        parser.add_argument("--doLoops",
                            help    = "run multi-pass subtraction",
                            default = False,
                            action  = 'store_true',
                            dest    = 'doLoops')

        params = parser.parse_args()

        # Convert params to a dict to feed into run_network as **kwargs
        model_params = {}
        for arg in vars(params):
            model_params[arg] = getattr(params, arg)

        return model_params


    # Get command line flags
    model_params = parse_command_line()

    # Set plotDir to use the current model
    model_type  = model_params['model_type']
    regex_model = re.compile(r'[a-zA-Z]+')
    basename    = regex_model.findall(model_type)[0]
    model_params['plotDir'] = 'params/{}/Figures/'.format(basename)

    # Run it!
    run_network(**model_params)

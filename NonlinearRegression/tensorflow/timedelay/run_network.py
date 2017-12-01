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
matplotlib.rcParams.update({'font.size': 18})
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.optimize import leastsq

from keras.models import Sequential
from keras.layers import (Dense, Dropout, LSTM, Activation, Conv1D,
                          Flatten, Embedding, MaxPooling1D, GRU)
from keras.utils import plot_model
from keras import optimizers

import theano

from NonlinearRegression.tools import analyze_run_data as nlr
from NonlinearRegression.tools import preprocessing as ppr
from NonlinearRegression.tools import nlr_exceptions as ex
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

def lorentzian(x,p):
    numerator =  (p[0]**2 )
    denominator = ( x - (p[1]) )**2 + p[0]**2
    y = p[2]*(numerator/denominator)
    return y

def residuals(p,y,x):
    err = y - lorentzian(x,p)
    return err

def run_test(
    # Training data
    val_frac = 1/4,   # Amount of data to save for validation
    fs_slow  = 32,    # Resample seismic data to this freq
    Tchunk   = 1/4,   # Seconds of data used to predict each DARM sample
    Tbatch   = 1,     # How many seconds of DARM per gradient update
    Nepoch   = 8,     # Number of times to iterate over training data set

    # Neural Network
    network    = 'LSTM',
    Nlayers    = 6,       # Number of Dense layers
    cost_func  = 'mse',   # Mean Squared Error, i.e. PSD reduction
    optimizer  = 'adam',  # Seems to work well...
    activation = 'elu',   # "Exponential linear unit"
    dropout    = 0.005,   # maybe this helps in training
    verbose    = 1,       # Print training progress to stderr
    DenseNet   = False,
    shuffle    = True,

    # Cost function
    zero_freqs = [6, 130],
    zero_order = [10, 2],
    pole_freqs = [12, 70],
    pole_order = [9, 3],

    # Output data and plots
    tfft      = 8,
    doPlots   = False,
    plotDir   = 'params/LSTM/Figures',
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

    # Detector
    ifo        = "L1",
    remove_cal = False,
    cal_freqs  = [22.7, 23.3, 23.9],

    # Preprocessing
    data_type     = "mock",
    doWhiten      = False,
    doLines       = False,
    doCoarseGrain = False):

    """
    Bilinear Regression Code to be run in Parameter Optimization Routine

    Example
    -------
    >>> from run_network import run_test
    >>> output_data = run_test(**kwargs)
    >>> Validation_loss = output_data['history']['val_loss'][-1]
    """

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
        if data_type == "mock":
            datafile = 'Data/DARM_with_bilinear.mat'

        elif data_type == "real":
            datafile = 'Data/{}_data_array.mat'.format(ifo)

        if not os.path.isfile(datafile):
            raise ex.DataNotFound(datafile)

        tar_raw, wit, fs, bg_raw = ppr.load_data(datafile, data_type=data_type)

        if doCoarseGrain:
            bg_raw, tar_raw, wit = ppr.coarseGrainWrap(datafile,
                                                       deltaFy   = 2 / fs,
                                                       data_type = data_type)
        if remove_cal:
            tar = subtract_cal(tar_raw, fs, cal_freqs, t_chunk=100)
        else:
            tar = tar_raw

        BP, invBP = cost_filter(fs, zero_freqs, pole_freqs, zero_order,
                                pole_order)

        print("Filtering and Decimating...")

        notch_freqs = [60,120,180]
        Q_notch     = 30
        #for f_notch in notch_freqs:
        #    bnot, anot = sig.iirnotch(f_notch/(fs/2), Q_notch)
        #    tar        = sig.filtfilt(bnot, anot, tar)

        if doLines:
            notch_freqs = [60,120]
            widths = [2,2] # Hz

            sp = np.fft.rfft(tar)
            freq = np.fft.rfftfreq(tar.shape[-1],d=1/fs)
            for f_notch,width in zip(notch_freqs,widths):
                ind_low = (freq > f_notch-2*width) & (freq < f_notch-width)
                ind_high = (freq < f_notch+2*width) & (freq > f_notch+width)
                ind_mid = (freq < f_notch+width) & (freq > f_notch-width)

                x_bg = np.concatenate((freq[ind_low],freq[ind_high]))
                y_bg_amp = np.concatenate((np.abs(sp[ind_low]),np.abs(sp[ind_high])))
                m_amp, c_amp = np.polyfit(x_bg, y_bg_amp, 1)
                background_amp = m_amp*freq + c_amp
                background_phase = np.random.rand(len(freq),)*2*np.pi - np.pi
                sp[ind_mid] = background_amp[ind_mid]*np.exp(background_phase[ind_mid]*1j)
            tar = np.fft.irfft(sp)

        if doWhiten:
            import gwpy.timeseries
            tartime = gwpy.timeseries.TimeSeries(tar_raw,
                                                 sample_rate = fs,
                                                 epoch       = 0.0,
                                                 dtype       = float)
            tar_raw = tartime.whiten(1.0,0.5)

            bgtime = gwpy.timeseries.TimeSeries(bg_raw,
                                                sample_rate = fs,
                                                epoch       = 0.0,
                                                dtype       = float)
            bg_raw = bgtime.whiten(1.0,0.5)

            Nasc = wit.shape[0]
            for jj in xrange(Nasc):
                wittime = gwpy.timeseries.TimeSeries(wit[jj,:],
                                                     sample_rate = fs,
                                                     epoch       = 0.0,
                                                     dtype       = float)
                whitewit  = wittime.whiten(1.0,0.5)
                wit[jj,:] = np.array(whitewit)

        # remove mean and normalize to std for nicer NN learning
        tar, scale = normalize(tar_raw, filter=BP)
        bg, _      = normalize(bg_raw,  filter=BP, scale=scale)

        # Get the witness signals ready for training.
        Npairs = wit.shape[0] // 2  # How many ASC + beam spot pairs
        Npairs = 0

        # Shape the ASC control signal with the same filter as DARM
        beam_spot, _ = normalize(wit[:Npairs])
        angular, _   = normalize(wit[Npairs:], filter=BP)

        # Since we only care about the slow beam spot motion, we
        # don't need full rate information. Decimating the signal
        # reduces the input vector length and the number of neurons
        # we have to train.

        down_factor = int(fs // fs_slow)
        beam_spot   = downsample(beam_spot, down_factor)

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

    # get nfft
    nfft = tfft * fs

    # How many DARM samples are saved for validation
    Nval = int(tar_raw.size * val_frac)

    # How many witness samples are used to predict each DARM sample
    Nchunk = int(Tchunk * fs)
    Nbatch = int(Tbatch * fs)
    Nang   = Nchunk
    Nspot  = Nchunk // down_factor
    Nspots = beam_spot.shape[0]
    Nasc   = angular.shape[0]

    # Select training and validation data segments
    training_target = tar[:-Nval]
    training_spt    = beam_spot[:, :-Nval // down_factor]
    training_ang    = angular[:, :-Nval]

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
    #############################################

    # Minimum loss is achieved when target - prediction = bg
    # Thus, MSE = mean(bg**2), i.e. var(bg)
    minLoss = np.var(bg)
    if verbose:
        print('Best achievable cost: {:.5f}'.format(minLoss))

    # Define the network topology
    model = Sequential()
    print('Using {} network'.format(network))

    if network == 'LSTM':
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

    elif network == 'MLP':
        input_shape = (((Nasc * Nang) + (Nspots * Nspot)),)

        model.add(Dense(input_shape[0],
                        input_shape = input_shape,
                        activation  = 'linear'))


    elif network == 'GRU':
        model.add(LSTM(32,
                       batch_input_shape = (None, 1, training_input.shape[1]),
                       dropout           = 0.0,
                       recurrent_dropout = 0.0,
                       return_sequences  = True))

        model.add(Dense(32))

        model.add(GRU(32,
                       dropout           = 0.00,
                       recurrent_dropout = 0.01,
                       return_sequences  = True))

        model.add(Dense(32))

        model.add(LSTM(32,
                       dropout           = 0.0,
                       recurrent_dropout = 0.0,
                       return_sequences  = True))

        model.add(GRU(32,
                       dropout           = 0.00,
                       recurrent_dropout = 0.01,
                       return_sequences  = False))

    elif network == 'CNN':
        input_shape = (Npairs * (Nang + Nspot),)

        max_features   = 100
        embedding_dims = 5
        maxlen         = training_input.shape[1]

        model = Sequential()
        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen))

        model.add(Dropout(0.1))
        model.add(Conv1D(filters     = 8,
                         kernel_size = 2,
                         strides     = 1,
                         padding     = 'same',
                         activation  = 'relu'))

        model.add(MaxPooling1D(7))

        for _ in range(3):
            model.add(Dropout(0.1))
            model.add(Conv1D(4, 2))

        model.add(MaxPooling1D(2))

        model.add(Flatten())

    else:
        raise nlr_exceptions.ModelSelectionError()

    if network in ['LSTM', 'GRU']:
        # Reshape data
        training_input   = training_input.reshape((training_input.shape[0], 1,
                                                   training_input.shape[1]))

        training_target  = training_target.reshape((training_target.shape[0],))

        validation_input = validation_input.reshape((validation_input.shape[0], 1,
                                                     validation_input.shape[1]))

    layer_sizes = range(1, Nlayers)
    layer_sizes.reverse()
    for k in layer_sizes:
        model.add(Dense(2**k, activation=activation))

    model.add(Dense(1, activation='linear'))
    optimizer = optimizer
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

    roomba = model.fit(training_input, training_target,
                       validation_data = (validation_input, validation_target),
                       batch_size      = Nbatch,
                       epochs          = Nepoch,
                       verbose         = verbose,
                       shuffle         = shuffle)

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
    data_path = PATH + '/' + datafile
    model_params = parse_command_line()
    nlr.organize_run_data(summary      = model.summary,
                          optimizer    = model.optimizer,
                          opt_name     = model_params['optimizer'],
                          model_params = model_params,
                          PATH         = PATH,
                          name         = network,
                          minLoss      = minLoss,
                          move         = True,
                          datafile     = data_path)

    plot_model(model, plotDir + '/model.png')

    return roomba.history['val_loss'][-1]


if __name__ == "__main__":
    import argparse


    def parse_command_line():
        parser = argparse.ArgumentParser()

        parser.add_argument("--network", "-n",
            				help    = "pick a neural network model",
                            default = 'LSTM',
            				dest    = "network",
                            type    = str)

        parser.add_argument("--data_type", "-d",
                            help    = "Use real or mock data",
                            default = "mock",
                            dest    = "data_type",
                            type    = str)

        parser.add_argument("--val_frac",
                            default = 0.25,
                            dest    = "val_frac",
                            type    = float)

        parser.add_argument("--fs_slow",
                            default = 32,
                            dest    = "fs_slow",
                            type    = int)

        parser.add_argument("--Tchunk",
                            default = 0.25,
                            dest    = "Tchunk",
                            type    = float)

        parser.add_argument("--Tbatch",
                            default = 1,
                            dest    = "Tbatch",
                            type    = int)

        parser.add_argument("--Nepoch", "-e",
                            default = 8,
                            dest    = "Nepoch",
                            type    = int)

        parser.add_argument("--Nlayers",
                            default = 6,
                            dest    = "Nlayers",
                            type    = int)

        parser.add_argument("--cost_func",
                            default = "mse",
                            dest    = "cost_func",
                            type    = str)

        parser.add_argument("--optimizer",
                            default = "adam",
                            dest    = "optimizer",
                            type    = str)

        parser.add_argument("--activation",
                            default = "elu",
                            dest    = "activation",
                            type    = str)

        parser.add_argument("--dropout",
                            default = 0.005,
                            dest    = "dropout",
                            type    = float)

        parser.add_argument("--verbose",
                            default = 1,
                            dest    = "verbose",
                            type    = int)

        parser.add_argument("--DenseNet",
                            default = False,
                            dest    = "DenseNet",
                            type    = bool)

        parser.add_argument("--zero_freqs",
                            default = [6, 130],
                            dest    = "zero_freqs",
                            nargs   = '+',
                            type    = int)

        parser.add_argument("--zero_order",
                            default = [10, 2],
                            dest    = "zero_order",
                            nargs   = '+',
                            type    = int)

        parser.add_argument("--pole_freqs",
                            default = [12, 70],
                            dest    = "pole_freqs",
                            nargs   = '+',
                            type    = int)

        parser.add_argument("--pole_order",
                            default = [9, 3],
                            dest    = "pole_order",
                            nargs   = '+',
                            type    = int)

        parser.add_argument("--tfft",
                            default = 8,
                            dest    = "tfft",
                            type    = int)

        parser.add_argument("--doPlots",
                            default = True,
                            dest    = "doPlots",
                            type    = bool)

        parser.add_argument("--plotDir",
                            default = "params/LSTM/Figures",
                            dest    = "plotDir",
                            type    = str)

        parser.add_argument("--save_data",
                            default = True,
                            dest    = "save_data",
                            type    = bool)

        parser.add_argument("--doCBC",
                            default = False,
                            dest    = "doCBC",
                            type    = bool)

        parser.add_argument("--mass1",
                            default = 38.9,
                            dest    = "mass1",
                            type    = float)

        parser.add_argument("--mass2",
                            default = 32.8,
                            dest    = "mass2",
                            type    = float)

        parser.add_argument("--f_lower",
                            default = 15.0,
                            dest    = "f_lower",
                            type    = float)

        parser.add_argument("--f_upper",
                            default = 64.0,
                            dest    = "f_upper",
                            type    = float)

        parser.add_argument("--doParallel",
                            default = False,
                            dest    = "doParallel",
                            type    = bool)

        parser.add_argument("--gpu_count",
                            default = 4,
                            dest    = "gpu_count",
                            type    = int)

        parser.add_argument("--doLoadDownsampled",
                            default = False,
                            dest    = "doLoadDownsampled",
                            type    = bool)

        parser.add_argument("--doCoarseGrain",
                            default = False,
                            dest    = "doCoarseGrain",
                            type    = bool)

        parser.add_argument("--doWhiten",
                            default = False,
                            dest    = "doWhiten",
                            type    = bool)

        parser.add_argument("--doLines",
                            default = False,
                            dest    = "doLines",
                            type    = bool)

        parser.add_argument("--interferometer", "-ifo",
                            default = "L1",
            				dest    = "ifo",
                            type    = str)

        parser.add_argument("--remove_cal",
                            default = False,
            				dest    = "remove_cal",
                            type    = bool)

        parser.add_argument("--cal_freqs",
                            default = [22.7, 23.3, 23.9],
            				dest    = "cal_freqs",
                            nargs   = '+',
                            type    = float)

        parser.add_argument("--shuffle",
                            default = True,
            				dest    = "shuffle",
                            type    = bool)

        params = parser.parse_args()

        # Convert params to a dict to feed in as **kwargs
        model_params = {}
        for arg in vars(params):
            model_params[arg] = getattr(params, arg)

        # Get the correct plot directory for the given network
        network = model_params['network']
        model_params['plotDir'] = 'params/{}/Figures'.format(network)
        return model_params


    model_params = parse_command_line()

    # Need to change these defaults for cost_func if using real data
    if model_params['data_type'] == 'real':
        model_params['zero_order'] = [10, 2]
        model_params['zero_freqs'] = [6, 120]
        model_params['pole_freqs'] = [12, 70]
        model_params['pole_order'] = [10, 2]

    run_test(**model_params)

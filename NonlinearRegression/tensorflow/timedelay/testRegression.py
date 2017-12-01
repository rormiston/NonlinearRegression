#!/usr/bin/env python

from __future__ import division
import os

from keras import optimizers

from bilinearRegressionReal import run_test

IFO = 'L1'

doLoadDownsampled = False
#if os.path.exists('Data/' + IFO + 'DARM_with_bilinear_downsampled.mat'):
#    doLoadDownsampled = True


run_test(
    # Training data
    val_frac   = 1/4,    # Amount of data to save for validation
    fs_slow    = 64,     # Resample seismic data to this freq
    Tchunk     = 1/4,    # Seconds of data used to predict each DARM sample
    Tbatch     = 1,      # How many seconds of DARM per gradient update
    Nepoch     = 137,    # Number of times to iterate over training data set
    # Neural Network
    DenseNet   = False,   # False = LSTM, True = Dense only
    Nlayers    = 2,       # Number of fully connected layers
    Nlstm      = 6,       # Number of LSTM layers (spaced in powers of 2)
    
    cost_func  = 'mse',   # Mean Squared Error, i.e. PSD reduction
    IFO        = IFO,      # which interferometer

    remove_cal = True,
    cal_freqs  = [22.7, 23.3, 23.9],
    # doing less good than Nadam, even with low LR
    #optimizer  = optimizers.Adamax(lr=3e-5, decay=0.0),

    optimizer  = optimizers.Nadam(lr=1e-5, schedule_decay=0),
    #optimizer = 'adam',
    activation = 'elu',   # "Exponential linear unit"
    dropout    = 0.003,    # maybe this helps in training
    verbose    = 1,       # Print training progress to stderr
    # Cost function
    zero_freqs = [ 1, 70],
    zero_order = [ 6,  3],
    pole_freqs = [11, 30],
    pole_order = [ 6,  3],
    # Output data and plots
    tfft      = 16,
    doPlots   = True,
    plotDir   = 'Figures',
    save_data = True,
    # Whether to look for previously saved downsampled data
    doLoadDownsampled = doLoadDownsampled,
    # Parallize
    doParallel = False, 
    gpu_count = 4)

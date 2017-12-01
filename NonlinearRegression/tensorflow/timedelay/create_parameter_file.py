#!/usr/bin/env python
"""
Creates parameter file useful for bilinearRegression.py
Example :$ python create_parameter_file.py  'sample_parameters.ini'
Issues : help functionality not working
"""
from __future__ import division
import argparse
import configparser
import sys


class helpfulParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        self.print_help()
        sys.exit(2)


parser = helpfulParser()

# Filename : Parameter .ini file
parser.add_argument('--filename', '-f',
                    type    = str,
                    default = 'sample_parameters.ini',
                    nargs   = '?',
                    help    = ("Filename of parameter file. Defaults to "
                               "'%(default)s'"))

########################
# Training_data_params #
########################

parser.add_argument('-fs_slow', '--fs_slow',
                    type    = int,
                    default = 32,
                    nargs   = '?',
                    dest    = 'fs_slow',
                    help    = ("Frequency to which seismic data should be "
                               "resampled. Defaults to %(default)Hz"))

parser.add_argument('-tchunk', '--tchunk',
                    default = 1/4,
                    type    = float,
                    nargs   = '?',
                    help    = ("Duration of witness data window in sec."
                               "Defaults to %(default)s"))

parser.add_argument('-test_frac', '--val_frac',
                    default = 1/4,
                    type    = float,
                    nargs   = '?',
                    help    = ("Fraction of data to be used for testing the "
                               "trained network on unseen data. Defaults to "
                               "%(default)"))


parser.add_argument('-tfft', '--tfft',
                    default = 8,
                    type    = int,
                    nargs   = '?',
                    help    = ("FFT window length for PSDs. Defaults "
                               "to %(default)s"))

###################################
# Invertible_band_pass parameters #
###################################

parser.add_argument('-fz1', '--fz1',
                    default = 4,
                    type    = float,
                    nargs   = '?',
                    help    =("fz1 of [fz1, fz2] used in Invertable band pass "
                              "zeros. Defaults  to %(default) Hz"))

parser.add_argument('-fz2', '--fz2',
                    default = 400,
                    type    = float,
                    nargs   = '?',
                    help    =("fz2 of [fz1, fz2] used in Invertable band pass "
                              "zeros. Defaults  to %(default) Hz"))

parser.add_argument('-fp1', '--fp1',
                    default = 15,
                    type    = float,
                    nargs   = '?',
                    help    = ("fp1 of [fp1, fp2] used in Invertable band pass "
                               "poles: Defaults  to %(default) Hz"))

parser.add_argument('-fp2', '--fp2',
                    default = 150,
                    type    = float,
                    nargs   = '?',
                    help    = ("fp1 of [fp1, fp2] used in Invertable band pass "
                               "poles. Defaults  to %(default) Hz"))

#####################
# Neural_net_params #
#####################

parser.add_argument('-dlayers', '--dense_layers',
                    default = 9,
                    type    = int,
                    nargs   = '?',
                    help    =("Number of dense layers. Defaults to %(default) "
                              "layers"))

parser.add_argument('-dense_activation', '--dense_activation',
                    default = 'elu',
                    type    = str,
                    nargs   = '?',
                    help    = ("Activation function used in dense layers."
                               "Defaults to %(default)"))

parser.add_argument('-dropout_ratio', '--dropout_ratio',
                    default = 0.2,
                    type    = float,
                    nargs   = '?',
                    help    = ("Dropout ratio to minimize overfitting related "
                               "issues. Defaults to %(default)"))

parser.add_argument('-optimizer', '--optimizer',
                    default = 'adam',
                    type    = str,
                    nargs   = '?',
                    help    = ("Optimzer used in minimzation of cost function. "
                               "Defaults to %(default)"))

parser.add_argument('-mycost_func', '--mycost_func',
                    default = 'mse',
                    type    = str,
                    nargs   = '?',
                    help    = ("Metric used for cost function minimization."
                               "Defaults to %(default)"))

######################
# NN_training_params #
######################

parser.add_argument('-batch_size', '--batch_size',
                    default = 1024,
                    type    = int,
                    nargs   = '?',
                    help    = ("Number of samples to be propagated through "
                               "the network. Defaults to %(default) samples"))

parser.add_argument('-epochs', '--epochs',
                    default = 10,
                    type    = int,
                    nargs   = '?',
                    help    = ("Number of epochs to train the network during "
                               "training stage. Defaults to %(default) epochs"))

parser.add_argument('-vsplit', '--validation_split',
                    default = 0.33,
                    type    = float,
                    nargs   = '?',
                    help    = ("Percenatge of data to be used for validation "
                               "during training. Defaults  to %(default)"))

parser.add_argument('-verbose', '--verbose',
                    default = 1,
                    type    = int,
                    nargs   = '?',
                    help    = "verbose. Defaults  to %(default)")

parser.add_argument('-k', '--keywords',
                    default = [],
                    nargs   = argparse.REMAINDER,
                    help    = ("Additional info given to models. Type [model] "
                               "-k -h for model-specific parser arguments."))

# Get parameters into global namespace
args = parser.parse_args()

fs_slow          = args.fs_slow
tchunk           = args.tchunk
val_frac         = args.val_frac
tfft             = args.tfft
fz1              = args.fz1
fz2              = args.fz2
fp1              = args.fp1
fp2              = args.fp2
dense_layers     = args.dense_layers
dense_activation = args.dense_activation
dropout_ratio    = args.dropout_ratio
optimizer        = args.optimizer
mycost_func      = args.mycost_func
batch_size       = args.batch_size
epochs           = args.epochs
validation_split = args.validation_split
verbose          = args.verbose

config = configparser.ConfigParser()

config['training_data_params'] = {'fs_slow' : fs_slow,   # Resample seismic data to this freq
                                  'tchunk'  : tchunk,     # How long windows of witness data are in sec
                                  'val_frac': val_frac,   # Amount of data to save for validation
                                  'tfft'    : tfft}       # FFT window length for PSDs

config['invertible_band_pass'] = {'fz': [fz1, fz2],  # Zeros
                                  'fp': [fp1, fp2]}  # Poles

config['neurel_net_params'] = {'dense_layers'    : dense_layers,      # Number of dense layers
                               'dense_activation': dense_activation,  # Other options: 'relu', tanh, sigmoid
                               'dropout_ratio'   : dropout_ratio,     # to prevent overfitting
                               'optimizer'       : optimizer ,        # cost function optimizer
                               'mycost_func'     : mycost_func }      # cost_function_metric

config['nn_training_params'] = {'batch_size': batch_size,
                                'epochs'    : epochs,
                                'verbose'   : verbose,
                                'validation_split' :validation_split} # partition into training & validation set

with open(args.filename, 'w') as configfile:
     config.write(configfile)

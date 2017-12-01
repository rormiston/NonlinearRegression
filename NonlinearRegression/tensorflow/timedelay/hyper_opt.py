#!/usr/bin/env python
# -*- coding: utf-8 -*-
from checkSource import checkSource
checkSource()
import matplotlib
matplotlib.use('agg')
import argparse
import NonlinearRegression.tools.preprocessing as ppr
import NonlinearRegression.tools.models as mod
import numpy as np
import os
import sys
import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier


# Set the environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

# Make sure we can run this from anywhere
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Set the seed for reproducability
seed = 7
np.random.seed(seed)

# Time how long it takes to run
start_time = time.time()


#################################
# Set up the command line flags #
#################################
def parse_command_line():
    """
    parse command line flags. use sensible defaults
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--activation", "-a",
            		help    = "activation function for neural network",
                        default = "elu",
            		dest    = "activation",
                        nargs   = '+',
                        type    = str)

    parser.add_argument("--batch_size", "-b",
                        help    = "number of samples to be trained at once",
                        default = 1024,
                        nargs   = '+',
                        dest    = "batch_size",
                        type    = int)

    parser.add_argument("--beta_1",
            		help    = "beta_1 params for optimizer",
                        default = 0.9,
                        nargs   = '+',
            		dest    = "beta_1",
                        type    = float)

    parser.add_argument("--beta_2",
            		help    = "beta_2 params for optimizer",
                        default = 0.999,
                        nargs   = '+',
            		dest    = "beta_2",
                        type    = float)

    parser.add_argument("--cut_data", "-cut",
            		help    = "fraction by which to cut down data size",
                        default = 0,
            		dest    = "cut_data",
                        type    = float)

    parser.add_argument("--data_type", "-d",
                        help    = "real or mock data",
                        default = "real",
                        dest    = "data_type",
                        type    = str)

    parser.add_argument("--datafile",
                        help    = "data file to read from",
                        default = 'Data/L1_data_array.mat',
                        dest    = "datafile",
                        type    = str)

    parser.add_argument("--decay",
            		help    = "optimizer learning rate decay",
                        default = None,
            		dest    = "decay",
                        nargs   = '+',
                        type    = float)

    parser.add_argument("--doDownsample",
                        help    = "downsample data",
                        default = False,
                        dest    = "doDownsample",
                        action  = 'store_true')

    parser.add_argument("--doLines",
                        help    = "notch lines",
                        default = False,
                        dest    = "doLines",
                        action  = 'store_true')

    parser.add_argument("--dropout", "-D",
                        help    = "dropout regularization",
                        default = 0.0,
                        nargs   = '+',
                        dest    = "dropout",
                        type    = float)

    parser.add_argument("--epochs", "-e",
                        help    = "Number of iterations of NN training",
                        default = 10,
                        dest    = "epochs",
                        nargs   = '+',
                        type    = int)

    parser.add_argument("--epsilon",
            		help    = "optimizer param",
                        default = 1e-8,
            		dest    = "epsilon",
                        type    = float)

    parser.add_argument("--fs_slow",
                        help    = "downsample rate",
                        default = 64,
                        nargs   = "+",
                        dest    = "fs_slow",
                        type    = int)

    parser.add_argument("--ifo", "-i",
                        help    = "IFO: L1 or H1",
                        default = "L1",
                        dest    = "ifo",
                        type    = str)

    parser.add_argument("--kinit",
            		help    = "initialize weights",
                        default = 'glorot_uniform',
            		dest    = "kinit",
                        type    = str)

    parser.add_argument("--learning_rate", "-lr",
            		help    = "optimizer learning rate",
                        nargs   = '+',
                        default = None,
            		dest    = "lr",
                        type    = float)

    parser.add_argument("--lookback", "-lb",
            		help    = "timesteps to look back",
                        default = 0.0,
                        nargs   = '+',
            		dest    = "lookback",
                        type    = float)

    parser.add_argument("--loss",
                        help    = "loss function for neural network",
                        default = 'mse',
                        nargs   = '+',
                        dest    = "loss",
                        type    = str)

    parser.add_argument("--model_type", "-m",
                        help    = "pick model type to use",
                        default = "LSTM",
                        dest    = "model_type",
                        type    = str)

    parser.add_argument("--momentum",
            		help    = "optimizer momentum",
                        nargs   = '+',
                        default = 0.0,
            		dest    = "momentum",
                        type    = float)

    parser.add_argument("--nesterov",
            		help    = "use nesterov momentum",
                        default = False,
            		dest    = "nesterov",
                        nargs   = '+')

    parser.add_argument("--neurons", "-N",
        		help    = "number of neurons in input layer",
                        default = 8,
                        nargs   = '+',
        		dest    = "neurons",
                        type    = int)

    parser.add_argument("--Nlayers", "-l",
            	        help    = "Number of layers for the Dense network",
                        default = 8,
                        nargs   = '+',
            	        dest    = "Nlayers",
                        type    = int)

    parser.add_argument("--optimizer", "-opt",
                        help    = "optimizing function for neural network",
                        default = 'adam',
                        nargs   = '+',
                        dest    = "optimizer",
                        type    = str)

    parser.add_argument("--recurrent_dropout", "-RD",
                        help    = "recurrent dropout used in RNN memory blocks",
                        default = 0.0,
                        nargs   = '+',
                        dest    = "recurrent_dropout",
                        type    = float)

    parser.add_argument("--rho",
            		help    = "adadelta & rmsprop optimizer params",
                        default = None,
                        nargs   = '+',
            		dest    = "rho",
                        type    = float)

    parser.add_argument("--shuffle", "-s",
                        help    = "shuffle training data",
                        default = False,
                        dest    = "shuffle",
                        action  = 'store_true')

    parser.add_argument("--train_frac",
                        help    = "ratio of dataset used for training",
                        default = 0.90,
                        nargs   = "+",
                        dest    = "train_frac",
                        type    = float)

    parser.add_argument("--verbose", "-v",
                        help    = "output verbosity",
                        default = 0,
                        dest    = "verbose",
                        type    = int)

    parser.add_argument("--doFilter",
                        help    = "lowpass filter data",
                        default = False,
                        dest    = "doFilter",
                        action  = 'store_true')

    params = parser.parse_args()

    # Convert params to a dict
    model_params = {}
    for arg in vars(params):
        model_params[arg] = getattr(params, arg)

    return model_params


# Get command line flags
model_params = parse_command_line()

# Make sure that single args aren't part of a list
# and build the param grid
param_grid = {}
for key, val in model_params.items():
    if isinstance(val, list):
        if len(val) == 1:
            model_params[key] = val[0]
        if len(val) > 1:
            param_grid[key] = val

###########################
# Unpack the command line #
###########################
activation   = model_params['activation']
batch_size   = model_params['batch_size']
beta_1       = model_params['beta_1']
beta_2       = model_params['beta_2']
cut_data     = model_params['cut_data']
datafile     = model_params['datafile']
data_type    = model_params['data_type']
decay        = model_params['decay']
doFilter     = model_params['doFilter']
doLines      = model_params['doLines']
doDownsample = model_params['doDownsample']
dropout      = model_params['dropout']
epochs       = model_params['epochs']
epsilon      = model_params['epsilon']
fs_slow      = model_params['fs_slow']
ifo          = model_params['ifo']
kinit        = model_params['kinit']
lookback     = model_params['lookback']
loss         = model_params['loss']
lr           = model_params['lr']
model_type   = model_params['model_type']
momentum     = model_params['momentum']
nesterov     = model_params['nesterov']
neurons      = model_params['neurons']
Nlayers      = model_params['Nlayers']
optimizer    = model_params['optimizer']
rho          = model_params['rho']
Rdropout     = model_params['recurrent_dropout']
train_frac   = model_params['train_frac']
verbose      = model_params['verbose']

##################################
# Get the training and test data #
##################################
datafile = ppr.get_datafile(datafile, data_type, ifo=ifo)
dataset, sample_rate = ppr.get_dataset(datafile, data_type=data_type)

if cut_data:
    size = int(dataset.shape[0] * (1 - cut_data))
    temp = np.zeros(shape=(size, dataset.shape[1]))
    for i in range(dataset.shape[1]):
        temp[:, i] = dataset[:size, i]
    dataset = temp

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

# Data input shape
input_shape = (None, train_X.shape[1], train_X.shape[2])

###################################
# Build the model to parameterize #
###################################
def create_model(# Model
                 activation  = activation,
                 batch_size  = batch_size,
                 dropout     = dropout,
                 epochs      = epochs,
                 input_shape = input_shape,
                 kinit       = kinit,
                 loss        = loss,
                 model_type  = model_type,
                 neurons     = neurons,
                 Nlayers     = Nlayers,
                 Rdropout    = Rdropout,
                 verbose     = verbose,

                 # Optimizer
                 beta_1    = beta_1,
                 beta_2    = beta_2,
                 decay     = decay,
                 epsilon   = epsilon,
                 lr        = lr,
                 momentum  = momentum,
                 nesterov  = nesterov,
                 optimizer = optimizer,
                 rho       = rho):

    model = mod.get_model(activation  = activation,
                          batch_size  = batch_size,
                          dropout     = dropout,
                          input_shape = input_shape,
                          kinit       = kinit,
                          model_type  = model_type,
                          neurons     = neurons,
                          Nlayers     = Nlayers)

    optimizer = mod.get_optimizer(beta_1   = beta_1,
                                  beta_2   = beta_2,
                                  decay    = decay,
                                  epsilon  = epsilon,
                                  lr       = lr,
                                  momentum = momentum,
                                  nesterov = nesterov,
                                  opt      = optimizer,
                                  rho      = rho)

    model.compile(loss=loss, optimizer=optimizer, metrics=['mae', 'accuracy'])

    return model


######################################################
# Do the hyperparameter search and print the results #
######################################################
model = KerasClassifier(batch_size = batch_size,
                        build_fn   = create_model,
                        epochs     = epochs,
                        verbose    = verbose)

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(train_X, train_y)

###################
# Get the results #
###################
best_score  = grid_result.best_score_
best_params = grid_result.best_params_
best_output = 'Best: {0} using {1}'.format(best_score, best_params)

means  = grid_result.cv_results_['mean_test_score']
stds   = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

############################
# Print to stdout and save #
############################
PATH  = os.getcwd() + '/params/{0}'.format(model_type)
fname = PATH + '/hyper_opt_results.txt'.format(model_type)

if not os.path.isdir(PATH):
    os.makedirs(PATH)
if not os.path.isfile(fname):
    os.system('touch {0}'.format(fname))

with open(fname, 'a+') as f:
    title = 'Results from {}\n'.format(time.strftime("%c"))
    dash  = '{}\n'.format('=' * len(title))
    f.write(dash + title + dash)
    f.write('{}\n\n'.format(best_output))
    print(dash + title + dash.strip('\n'))
    print(best_output + '\n')
    for mean, stdev, param in zip(means, stds, params):
        results = '{0} ({1}) with: {2}'.format(mean, stdev, param)
        print(results)
        f.write(results + '\n')

    f.write('\nCommand\n-------\n')
    cmd = 'python ' + ' '.join(sys.argv) + '\n\n\n'
    f.write(cmd)

end_time = time.time()
print("Done. Optimization took {:.2f} seconds".format(end_time - start_time))

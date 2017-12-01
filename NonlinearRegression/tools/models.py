import re
import sys

import os
# Hush TF AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

from NonlinearRegression.tools import nlr_exceptions as ex
from keras.models import Sequential
from keras import optimizers
from keras.layers import (Dense, Dropout, LSTM, GRU, Flatten, Conv1D, MaxPooling1D)
from keras.constraints import maxnorm
from keras import regularizers


def get_model(model_type  = None,
              input_shape = None,
              dropout     = 0.0,
              batch_size  = None,
              kinit       = 'glorot_uniform',
              neurons     = 8,
              Rdropout    = 0.0,
              activation  = None,
              Nlayers     = 8):


    # Strip the model number away
    regex_model = re.compile(r'[a-zA-Z]+')
    model_type  = regex_model.findall(model_type)[0]

    model = Sequential()

    if model_type == 'LSTM':
        if len(input_shape) == 3:
            model.add(LSTM(32,
                           batch_input_shape  = input_shape,
                           dropout            = dropout,
                           recurrent_dropout  = Rdropout,
                           kernel_initializer = kinit,
                           return_sequences   = True))
        else:
            model.add(LSTM(32,
                           input_shape        = input_shape,
                           dropout            = dropout,
                           recurrent_dropout  = Rdropout,
                           kernel_initializer = kinit,
                           return_sequences   = True))

        model.add(Dense(32))

        for _ in range(3):
            model.add(LSTM(32,
                           dropout              = dropout,
                           recurrent_activation = 'sigmoid',
                           recurrent_dropout    = Rdropout,
                           kernel_initializer   = kinit,
                           return_sequences     = True))

            model.add(Dense(32))

        model.add(LSTM(32,
                       dropout              = dropout,
                       recurrent_activation = 'sigmoid',
                       recurrent_dropout    = Rdropout,
                       kernel_initializer   = kinit,
                       return_sequences     = False))

        model.add(Dense(32))

        model.add(Dense(1, activation='linear'))

    elif model_type == 'MLP':
        if len(input_shape) == 3:
            model.add(Dense(256,
                            batch_input_shape = input_shape,
                            kernel_initializer = kinit,
                            activation         = activation))
        else:
            model.add(Dense(256,
                            input_shape = input_shape,
                            kernel_initializer = kinit,
                            activation  = activation))

        for _ in range(5):
            model.add(Dense(256))
            model.add(Dropout(dropout))

        layer_sizes = range(1, Nlayers)
        layer_sizes.reverse()
        for k in layer_sizes:
            model.add(Dense(2**k,
                            activation = activation,
                            kernel_initializer = kinit))

        model.add(Flatten())

        model.add(Dense(1, activation='linear'))

    elif model_type == 'GRU':
        if len(input_shape) == 3:
            model.add(GRU(16,
                          batch_input_shape  = input_shape,
                          dropout            = dropout,
                          recurrent_dropout  = Rdropout,
                          kernel_initializer = kinit,
                          return_sequences   = True))
        else:
            model.add(GRU(16,
                          input_shape        = input_shape,
                          dropout            = dropout,
                          recurrent_dropout  = Rdropout,
                          kernel_initializer = kinit,
                          return_sequences   = True))

        model.add(GRU(16,
                      dropout            = dropout,
                      recurrent_dropout  = Rdropout,
                      kernel_initializer = kinit,
                      return_sequences   = False))

        model.add(Dense(1, activation='linear'))


    elif model_type == 'CNN':
        if len(input_shape) == 3:
            model.add(Conv1D(64, 3, batch_input_shape=input_shape))
        else:
            model.add(Conv1D(filters     = 16,
                             kernel_size = 2,
                             input_shape = input_shape,
                             strides     = 1,
                             padding     = 'valid',
                             activation  = 'relu'))

        for _ in range(15):
            model.add(Conv1D(64, 1))
            model.add(MaxPooling1D(1))
            model.add(Dropout(dropout))

            model.add(Conv1D(32, 1))
            model.add(MaxPooling1D(1))
            model.add(Dropout(dropout))

            model.add(Conv1D(16, 1))
            model.add(MaxPooling1D(1))
            model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(16))
        model.add(Dense(8))
        model.add(Dense(1, activation='linear'))

    elif model_type == 'random':
        if len(input_shape) == 3:
            model.add(Dropout(dropout, batch_input_shape=input_shape))
        else:
            model.add(Dropout(dropout, input_shape=input_shape))

        for _ in range(Nlayers):
            model.add(Dense(neurons,
                            kernel_constraint  = maxnorm(3),
                            kernel_initializer = kinit,
                            activation         = activation))

            model.add(Dropout(dropout))

        model.add(Dense(1, kernel_constraint=maxnorm(3), activation='linear'))
        model.add(Flatten())

    else:
        raise ex.ModelSelectionError()

    return model


def get_optimizer(opt,
                  decay    = None,
                  lr       = None,
                  momentum = 0.0,
                  nesterov = False,
                  beta_1   = 0.9,
                  beta_2   = 0.999,
                  epsilon  = 1e-8,
                  rho      = None):

    ###############################
    # Stochastic Gradient Descent #
    ###############################
    if opt == 'sgd':

        if lr is None:
            lr = 0.01

        if decay is None:
            decay = 0.0

        optimizer = optimizers.SGD(lr       = lr,
                                   momentum = momentum,
                                   decay    = decay,
                                   nesterov = nesterov)
    ########
    # Adam #
    ########
    elif opt == 'adam':

        if lr is None:
            lr = 0.001

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)
    ##########
    # Adamax #
    ##########
    elif opt == 'adamax':

        if lr is None:
            lr = 0.002

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)
    #########
    # Nadam #
    #########
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'nadam':

        if lr is None:
            lr = 0.002

        if decay is None:
            decay = 0.004

        optimizer = optimizers.Adam(lr      = lr,
                                    beta_1  = beta_1,
                                    beta_2  = beta_2,
                                    epsilon = epsilon,
                                    decay   = decay)

    ###########
    # RMSprop #
    ###########
    # It is recommended to leave the parameters of this
    # optimizer at their default values (except the learning
    # rate, which can be freely tuned).
    elif opt == 'rmsprop':

        if lr is None:
            lr = 0.001

        if decay is None:
            decay = 0.0

        if rho is None:
            rho = 0.9

        optimizer = optimizers.RMSprop(lr      = lr,
                                       rho     = rho,
                                       epsilon = epsilon,
                                       decay   = decay)
    ###########
    # Adagrad #
    ###########
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'adagrad':

        if lr is None:
            lr = 0.01

        if decay is None:
            decay = 0.0

        optimizer = optimizers.Adagrad(lr      = lr,
                                       decay   = decay,
                                       epsilon = epsilon)

    ############
    # Adadelta #
    ############
    # It is recommended to leave the parameters of this
    # optimizer at their default values.
    elif opt == 'adadelta':

        if lr is None:
            lr = 1.0

        if decay is None:
            decay = 0.0

        if rho is None:
            rho = 0.95

        optimizer = optimizers.Adadelta(lr      = lr,
                                        rho     = rho,
                                        epsilon = epsilon,
                                        decay   = decay)

    else:
        print('ERROR: Unknown optimizer')
        sys.exit(1)

    return optimizer

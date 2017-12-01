#!/usr/bin/env python
# coding: utf-8

# Use Keras + TensorFlow to train a Neural Network to do Nonlinear Regression:
# Remove bilinear (ASC) noise from DARM

from __future__ import division

from checkSource import checkSource
checkSource()

import numpy as np
import os

import subprocess, sys
#from threading import Thread

from scipy.io import loadmat, savemat
import scipy.signal as sig
from timeit import default_timer as timer

import matplotlib.pyplot as plt

#  from keras.utils import plot_model
#  from IPython.display import SVG
#  from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import losses
# should we set the keras backend explicitly??

from bilinearHelper import true_coupling, stride_wit, set_plot_style
set_plot_style()

#print("python:{}, keras:{}, tensorflow: {}".format(sys.version, keras.__version__, tf.__version__))

############################
# Training Data Parameters #
############################
fs_slow  = 32   # Resample seismic data to this freq
tchunk   = 1/4  # How long windows of witness data are in sec
val_frac = 1/4  # Amount of data to save for validation
tfft     = 8    # FFT window length for PSDs

####################
# Data preperation #
####################

# Make sure the data loads when running from any directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Load up datas
datas   = loadmat('../../../../MockData/' + 'DARM_with_bilinear.mat')
bg_raw  = datas['background'][0]
tar_raw = datas['darm'][0]
wit     = datas['wit']
fs      = datas['fs'][0][0]
nfft    = tfft * fs

# Check how good we can really do
if __debug__:
    best_outcome  = true_coupling(wit[0], wit[1], fs)
    best_residual = tar_raw - best_outcome
    error         = bg_raw  - best_residual
    ff, pp        = sig.welch([tar_raw, best_residual, bg_raw, error],
                              fs=fs, nperseg=nfft, axis=-1)

    plt.figure(33)
    plt.loglog(ff, np.sqrt(pp).T)
    plt.legend(['Background + ASC',
                'Best case residual',
                'True Background',
                'Mismatch'],
               loc='best')
    plt.xlim([8, 210])
    plt.ylim([9e-21, .31e-16])
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    plt.savefig('MockData-Bilinear.pdf')

# DARM cost weighting highpass
#f1    = 15
#f2    = 100
#sosHP = sig.butter(8, [f1/(fs/2), f2/(fs/2)], btype='bandpass', output='sos')

# invertable band pass
fz = [4, 400]
fp = [15, 20, 150]

_,z0,_ = sig.butter(10, fz[0]/(fs/2), btype='lowpass', output='zpk')
_,p0,_ = sig.butter( 8, fp[0]/(fs/2), btype='lowpass', output='zpk')
_,p1,_ = sig.butter( 1, fp[1]/(fs/2), btype='lowpass', output='zpk')
_,p2,_ = sig.butter( 3, fp[1]/(fs/2), btype='lowpass', output='zpk')
_,z1,_ = sig.butter( 2, fz[1]/(fs/2), btype='lowpass', output='zpk')

zs    = np.concatenate((z0, z1))
ps    = np.concatenate((p0, p1))
BP    = sig.zpk2sos(zs, ps, 1, pairing='keep_odd')
invBP = sig.zpk2sos(ps, zs, 1, pairing='keep_odd')
#sosBP = np.concatenate([sosHP, HP], axis = 0)

# Shape and scale the background and DARM signals
bg    = sig.sosfilt(BP,  bg_raw)
tar   = sig.sosfilt(BP, tar_raw)

scale = np.std(tar)
bg   /= scale
tar  /= scale

# Visualize cost function
if __debug__:
    from scipy.integrate import cumtrapz
    ff, ptar = sig.welch(tar, fs=fs, nperseg=nfft)
    ff, pbg  = sig.welch(bg,  fs=fs, nperseg=nfft)
    tar_rms  = np.flipud(np.sqrt(-cumtrapz(ptar[::-1], ff[::-1], initial=0)))
    bg_rms   = np.flipud(np.sqrt(-cumtrapz( pbg[::-1], ff[::-1], initial=0)))
    print('Best achievable cost: {:5g}'.format(bg_rms[0]**2))

    plt.figure(911)
    ht, = plt.loglog(ff, np.sqrt(ptar), label='Target')
    hb, = plt.loglog(ff, np.sqrt(pbg), label='Background')
    plt.semilogx(ff, tar_rms, linestyle='dashed', color=ht.get_color())
    plt.semilogx(ff, bg_rms, linestyle='dashed', color=hb.get_color())
    plt.legend()
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    plt.title(r'Shaped DARM cost for NN training')
    plt.xlabel(r'Frequency [Hz]')
    plt.ylabel(r'ASD [1/$\sqrt{\rm Hz}$]')
    plt.savefig('cost_function_spectrum.pdf')

# Get the witness signals ready for training.

print("Filtering, Chunking, and Decimating...")

# remove mean and normalize to std for nicer NN learning
beam_spot  = (wit[0] - np.mean(wit[0]))
beam_spot /= np.std(beam_spot)

angular    = (wit[1] - np.mean(wit[1]))
# Shape the ASC control signal with the same filter as DARM
angular    = sig.sosfilt(BP, angular)
angular   /= np.std(angular)

# Break witnesses up into array that provides the window of past data that
# leads up to each training sample.
angular     = stride_wit(angular,   int(tchunk*fs))
beam_spot   = stride_wit(beam_spot, int(tchunk*fs))

# Since we only care about the slow beam spot motion, we don't need full rate
# information. Decimating the signal reduces the input vector length and the
# number of neurons we have to train.

# Let's decimate! ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~~  ~  ~ ~ ~ ~ ~ ~~ ~ ~ ~~~~  ~
t_start     = timer()
down_factor = int(fs // fs_slow)
fir_aa      = sig.firwin(20 * down_factor + 1, 0.8 / down_factor,
                    window='blackmanharris')
# Using fir_aa[1:-1] cuts off a leading and trailing zero
beam_spot   = sig.decimate(beam_spot, down_factor,
                         ftype = sig.dlti(fir_aa[1:-1], 1.0),
                         zero_phase = True)

if __debug__:
    print(str(round(timer() - t_start)) + " seconds for Decimating.")
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~~  ~  ~ ~ ~ ~ ~ ~~ ~ ~ ~~~~  ~

# How many DARM samples are saved for validation
Nval = int(tar.shape[0] * val_frac)

# Throw away the DARM samples that don't have enough witness history available
# to generate predictions.
tar_raw = tar_raw[-angular.shape[0]:]
bg_raw  =  bg_raw[-angular.shape[0]:]
tar     =     tar[-angular.shape[0]:]
bg      =      bg[-angular.shape[0]:]

Nang  = angular.shape[-1]    # how many samples of angular noise to use
Nspot = beam_spot.shape[-1]  # how many samples of spot motion noise to use

# Define training and validation data
training_input = np.concatenate([angular[:-Nval,:], beam_spot[:-Nval,:]],
                                axis=1)
training_target = tar[:-Nval]

validation_input = np.concatenate([  angular[-Nval:,:],
                                   beam_spot[-Nval:,:]],
                                   axis=1)

# Rescale validation data back to DARM units
validation_darm = tar[-Nval:]*scale
validation_bg   =  bg[-Nval:]*scale

if __debug__:
    print(angular.shape)
    print(beam_spot.shape)

#############################################
# Construct the neural network and train it #
#############################################


# define the network topology  -- -- -- - - -  -  -   -   -    -
model = Sequential()
#model.add(Dense(Nang+Nspot, input_shape=(Nang+Nspot,), activation = 'linear'))

# this layer increases trainging time but seems to not increase performance
#model.add(Dense(Nang+Nspot, activation = 'elu'))
#model.add(Dense(Nang+Nspot, activation = 'elu'))  # try one more fully connected
model.add(LSTM(128, return_sequences=False, stateful=False,
               input_shape=(None, Nang+Nspot),
               ))
#model.add(LSTM(128, stateful=False, activation='tanh'))

g = range(9) # add layers; decrease size of each by half
g.reverse()
for k in g[0:-2]:
    model.add(Dense(2**k, activation = 'elu'))

model.add(Dense(1,    activation = 'linear'))
# -- -- -- - - -  -  -   -   -    - -- -- -- - - -  -  -   -   -    -

# define the cost function
mycost_func = losses.mean_squared_error

# some common network optimizers
# RMSprop unstable; adam is slightly better than Adadelta
from keras import optimizers
#sgd = optimizers.RMSprop(lr  = 0.01, rho = 0.90, epsilon = 1e-08, decay = 0.0)
#sgd = optimizers.Adadelta(lr = 1.00, rho = 0.95, epsilon = 1e-08, decay = 0.0)
#sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
sgd = 'adam'

model.compile(optimizer = sgd, loss = mycost_func)

print("Starting Network learning...")
t_start = timer()


roomba = model.fit(training_input,
          training_target,
          batch_size = int(fs),  # How many samples per gradient update
          epochs     = 7,        # No. of iterations on the whole training set
          validation_split=0.1,
          verbose    = 1)

if __debug__:
    print(str(round(timer() - t_start)) + " seconds for Training.")

################################
# Evaluate the trained network #
print("Applying model to input data...")
validation_out      = model.predict(validation_input)[:,0]
validation_out     *= scale  # Scale to DARM units

print("Unwhitening target and output data...")
validation_darm     = sig.sosfilt(invBP, validation_darm)
validation_bg       = sig.sosfilt(invBP, validation_bg)
validation_out      = sig.sosfilt(invBP, validation_out)
validation_residual = validation_darm - validation_out

if __debug__:
    print "Diagnostics:"
    model.summary()
#    print('{:3g}'.format(np.std(scale*angular[-Nval:,:])))
#    print('{:3g}'.format(np.std(validation_out)))
#    print('{:3g}'.format(np.std(validation_darm)))


print("Saving model and processing params...")
model.save('LSTM_RegressionModel.h5')
output_data = {'history': roomba.history,
               'invBP' : invBP,
               'scale' : scale,
               'fs'    : fs,
               'nfft'  : nfft,
              }
savemat('Results_LSTM.mat', output_data, do_compression=True)

################################

ff1,pp = sig.welch([validation_darm, validation_out,
                   validation_residual, validation_bg],
                  fs=fs, nperseg=nfft, axis=-1)

ff2,co = sig.coherence(validation_darm, validation_out,
                      fs=fs, nperseg=nfft)

if __debug__:
    plot_training_progress(roomba)

# make plots to evaluate success / failure of the regression
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6,8))

ax1.loglog(ff1, np.sqrt(pp).T,
           alpha=0.7)
ax1.legend(['DARM', 'Prediction', 'Subtracted','Background'])
ax1.set_xlim([3, 400])
ax1.set_ylim([1e-20, 2e-13])
ax1.grid(True, which='minor')
ax1.set_ylabel(r'DARM (m/$\sqrt{\rm Hz}$)')
ax1.set_title(r'NN/LSTM Bilinear Noise Regression')

ax2.semilogx(ff2, co,
             label='DARM/Prediction')
ax2.grid(True, which='minor')
ax2.set_ylim([0, 1])
ax2.set_xlabel(r'Frequency (Hz)')
ax2.set_ylabel(r'Coherence')
ax2.legend()

plt.subplots_adjust(hspace=0.05)

# save figure, backing up previous versions for easy comparison
figname = 'ValidationLSTM'
try:
    cpStr = 'cp -p ' + figname + '_1' + '.pdf ' + figname + '_2' + '.pdf'
    subprocess.call(cpStr, shell=True)
except:
    print('Error: ' + cpStr)

try:
    cpStr = 'cp -p ' + figname +        '.pdf ' + figname + '_1' + '.pdf'
    subprocess.call(cpStr, shell=True)
except:
    print('Error: ' + cpStr)

plt.savefig(figname + '.pdf')

#  plot_model(model, to_file='model.png', show_shapes=True)
#  SVG(model_to_dot(model).create(prog='dot', format='svg'))

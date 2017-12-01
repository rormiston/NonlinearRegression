from __future__ import division
import numpy as np
import scipy.signal as sig

from keras.models import Sequential
from keras.layers import Dense


def stride_wit(x, nperseg, step=1):

    nperseg = int(nperseg)
    step = int(step)
    noverlap = nperseg - step

    shape = x.shape[:-1] + ((x.size-noverlap)//step, nperseg)

    strides = x.strides[:-1] + (step*x.strides[-1], x.strides[-1])

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def stride_ts_wit(x, timesteps, nperseg, step=1):

    timesteps = int(timesteps)
    nperseg = int(nperseg)
    step = int(step)
    noverlap = nperseg - step

    shape = ((x.size-noverlap)//step-timesteps+1, timesteps, nperseg)

    strides = (step*x.strides[-1], step*x.strides[-1], x.strides[-1])

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def get_signals(order, sec=64, fs=2048, fpole=None, q=10):

    if fpole is None:
        fpole = fs*.05  # 20 samples per cycle, i.e. ~100Hz for fs=2048
    if order < 2:
        raise ValueError('Need at least 2 poles')

    N = int(sec*fs)
    Npad = 2*fs

    w = -2*np.pi*fpole

    zz = []
    pp = []
    kk = 1

    # Q = Im(w)/2Re(w) = tan(theta)/2
    theta = np.arctan(2*q)

    p = w * np.exp(1j*theta)
    p = [p, np.conj(p)]

    z = []

    k = np.prod(p)/np.prod(z)  # Unity gain at 0 Hz
    #  k = 1  # Unity gain at inf

    zd, pd, kd, _ = sig.cont2discrete((z, p, k), 1/fs, method='bilinear')
    sos = sig.zpk2sos(zd, pd, kd)

    zz += list(zd)
    pp += list(pd)
    kk *= kd

    order -= 2


    if order > 0:  # Use some cheby1 nonsense
        zd, pd, kd = sig.cheby1(order, 6, fpole/(fs/2), output='zpk')
        section = sig.zpk2sos(zd, pd, kd)
        np.concatenate((sos, section))

        zz += list(zd)
        pp += list(pd)
        kk *= kd


    x = np.random.randn(N+Npad)
    y = sig.sosfilt(sos, x)

    kk /= np.std(y)
    y /= np.std(y)

    sys = sig.dlti(zz, pp, kk, dt=1/fs)

    x = x[Npad//2:-Npad//2]
    y = y[Npad//2:-Npad//2]

    return x, y, sys

def make_short_NN(input_dim, **kwargs):

    if 'optimizer' not in kwargs:
        kwargs['optimizer'] = 'adam'
    if 'loss' not in kwargs:
        kwargs['loss'] = 'mse'

    activation = kwargs.pop('activation', 'tanh')

    model = Sequential()
    model.add(Dense(input_dim//2, input_dim=input_dim, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1, activation='linear'))

    model.compile(**kwargs)

    return model

def make_NN(input_dim, nlayer=2, **kwargs):

    if nlayer < 2 or not isinstance(nlayer, int):
        raise ValueError('nlayer needs to be integer greater than 1')
    if 'optimizer' not in kwargs:
        kwargs['optimizer'] = 'adam'
    if 'loss' not in kwargs:
        kwargs['loss'] = 'mse'

    activation = kwargs.pop('activation', 'tanh')
    N = [int(np.round(n)) for n in
         np.logspace(np.log10(input_dim), 0, nlayer+1)[1:-1]]

    model = Sequential()
    model.add(Dense(N[0], input_dim=input_dim, activation = activation))

    for n in N[1:]:
        model.add(Dense(n, activation = activation))

    model.add(Dense(1, activation = 'linear'))

    model.compile(**kwargs)

    return model

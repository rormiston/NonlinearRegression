from __future__ import division
import matplotlib
matplotlib.use('agg')
from coarseGrain import coarseGrainReal
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

import os
# Hush TF AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

import NonlinearRegression.tools.bilinearHelper as blh
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
import sys
import tensorflow as tf


def coarseGrainWrap(datafile,
                    deltaFy   = None,
                    flowy     = None,
                    relative  = False,
                    data_type = "mock"):
    """
    coarseGrainWrap is a wrapper for the coarse grain function
    shared output library (compiled with SWIG).

    Parameters
    ----------
    datafile : `string`
        Path to the mat file containing the time series arrays

    deltaFy : `float`
        New coarse graining bin width. If deltaFy is None, the
        returned arrays are identical to the input arrays

    flowy : `float`
        Starting frequency bin

    relative : `bool`
       When set to False, the input value for deltaFy is interpreted
       as typed. When set to True however, deltaFy is how many times
       more coarse the output will be in relation to the default setting.
       e.g., if the default binning is 1/4 Hz, then entering deltaFy = 4
       when relative is True will result in deltaFy = 4 * 1/4 = 1Hz.

    data_type : `str`
        determines whether to use real or mock data

    Returns
    -------
        tar_raw: `numpy.ndarray`
            darm (target) array

        bg_raw: `numpy.ndarray`
            array of lists containing the background time series data

        wit: `numpy.ndarray`
            array of lists containing the witness channels time series data.
    """

    mat_file = sio.loadmat(datafile)

    if data_type == "mock":
        bg_raw   = mat_file['background'][0]
        tar_raw  = mat_file['darm'][0]
        wit      = mat_file['wit']
        fs       = mat_file['fs'][0][0]

    elif data_type == "real":
        tar_raw = mat_file['data'][0]
        wit     = mat_file['data'][1:]
        fs      = mat_file['fsample'][0][0]
        bg_raw  = np.zeros_like(tar_raw)

    ########
    # DARM #
    ########
    fft      = np.fft.fft(tar_raw)
    real_fft = np.array([np.real(x) for x in fft])
    imag_fft = np.array([np.imag(x) for x in fft])
    real_fs  = FrequencySeries(real_fft)

    # Extract metadata
    Nx      = len(real_fft)
    flowx   = 0
    deltaFx = 1.0 / float(fs)

    # Calculate default values
    if deltaFy is None:
        deltaFy = deltaFx

    elif deltaFy is not None:
        if relative:
            deltaFy = deltaFy * deltaFx
        else:
            pass

    if flowy is None:
        flowy = flowx + (deltaFy - deltaFx) * 0.5

    # Coarse grain the data
    size  = 2 + len(wit)
    count = 1
    sys.stdout.write('\rCoarse graining: {0}/{1}'.format(count, size))
    sys.stdout.flush()
    real_cg = np.array(coarseGrainReal(real_fft, Nx, flowx,
                                       deltaFx, flowy, deltaFy))
    count += 1
    sys.stdout.write('\rCoarse graining: {0}/{1}'.format(count, size))
    sys.stdout.flush()
    imag_cg = np.array(coarseGrainReal(imag_fft, Nx, flowx,
                                       deltaFx, flowy, deltaFy))

    # Combine the real and imaginary parts
    fft_cg = np.array(real_cg) + 1j * np.array(imag_cg)

    # Take the real part of the IFFT to get back to the time series data
    darm_cg = np.real(np.fft.ifft(fft_cg))

    ##############
    # Background #
    ##############
    fft      = np.fft.fft(bg_raw)
    real_fft = np.array([np.real(x) for x in fft])
    imag_fft = np.array([np.imag(x) for x in fft])
    real_fs  = FrequencySeries(real_fft)

    # Coarse grain the data
    real_cg = np.array(coarseGrainReal(real_fft, Nx, flowx,
                                       deltaFx, flowy, deltaFy))

    imag_cg = np.array(coarseGrainReal(imag_fft, Nx, flowx,
                                       deltaFx, flowy, deltaFy))

    # Combine the real and imaginary parts
    fft_cg = np.array(real_cg) + 1j * np.array(imag_cg)

    # Take the real part of the IFFT to get back to the time series data
    bkg_cg = np.real(np.fft.ifft(fft_cg))

    #############
    # Witnesses #
    #############
    new_wit = []
    for witness in wit:

        count += 1
        sys.stdout.write('\rCoarse graining: {0}/{1}'.format(count, size))
        sys.stdout.flush()

        if count == size:
            print('\n')

        fft      = np.fft.fft(witness)
        real_fft = np.array([np.real(x) for x in fft])
        imag_fft = np.array([np.imag(x) for x in fft])
        real_fs  = FrequencySeries(real_fft)

        # Coarse grain the data
        real_cg = np.array(coarseGrainReal(real_fft, Nx, flowx,
                                           deltaFx, flowy, deltaFy))

        imag_cg = np.array(coarseGrainReal(imag_fft, Nx, flowx,
                                           deltaFx, flowy, deltaFy))

        # Combine the real and imaginary parts
        fft_cg = np.array(real_cg) + 1j * np.array(imag_cg)

        # Take the real part of the IFFT to get back to the time series data
        ts_cg = np.real(np.fft.ifft(fft_cg))
        new_wit.append(ts_cg)

    new_wit = np.array(new_wit)

    return bkg_cg, darm_cg, new_wit


def load_data(datafile, data_type='mock'):

    mat_file = sio.loadmat(datafile)

    if data_type == 'mock':
        darm = mat_file['darm'][0]
        wit  = mat_file['wit']
        fs   = mat_file['fs'][0][0]
        bkgd = mat_file['background'][0]

    elif data_type == 'real':
        darm  = mat_file['data'][0]
        wit   = mat_file['data'][1:]
        fs    = mat_file['fsample'][0][0]
        bkgd  = np.zeros_like(darm)

    return darm, wit, fs, bkgd


def get_dataset(datafile, data_type='real'):
    """
    get_test_and_train_data reads in the data and returns the proper
    test and training data given a normalization type (including `None`)
    and the test fraction

    Parameters
    ----------
    datafile : `string`
        full path to mat file

    Returns
    -------
        dataset : `numpy.ndarray`
            test data. includes all channels except darm
    """
    mat_file = sio.loadmat(datafile)

    if data_type == 'mock' or data_type == 'scatter':
        bg_raw   = mat_file['background']
        tar_raw  = mat_file['darm']
        wit      = mat_file['wit']
        fs       = mat_file['fs'][0][0]

    elif data_type == 'real':
        tar_raw = mat_file['data'][0]
        tar_raw = tar_raw.reshape(1, tar_raw.shape[0])
        wit     = mat_file['data'][1:]
        fs      = mat_file['fsample'][0][0]
        bg_raw  = np.zeros_like(tar_raw)

    # Pack the data together
    features = np.sum(tar_raw.shape[0] + bg_raw.shape[0] + wit.shape[0])
    length   = tar_raw.shape[1]
    dataset  = np.zeros(shape=(features, length))

    dataset[0, :] = tar_raw
    dataset[1, :] = bg_raw
    for i, w in enumerate(wit):
        i += 2
        dataset[i, :] = w

    # Set to have shape = (samples, features)
    dataset = dataset.T

    return dataset, fs


def get_datafile(datafile, data_type, ifo='L1'):
    if datafile is None:
        if data_type == 'real':
            if ifo == 'L1':
                datafile = 'Data/L1_data_array.mat'
            elif ifo == 'H1':
                datafile = 'Data/H1_data_array.mat'
            else:
                print('ERROR: No valid data file given')
                sys.exit(1)

        elif data_type == 'mock':
            datafile = 'Data/DARM_with_bilinear.mat'

        elif data_type == 'scatter':
            datafile = 'Data/DARM_with_scatter.mat'

        else:
            print('ERROR: Not a valid data type')
            sys.exit(1)

    return datafile


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    convert series to supervised learning
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df     = pd.DataFrame(data)
    cols   = []
    names  = []

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def remove_lines(dataset, fs,
                 chans       = 'darm',
                 width       = 1,
                 notch_freqs = [60, 120]):

    if chans == 'darm':
        notched_chans = 1
    elif chans == 'all':
        notched_chans = dataset.shape[1]
    else:
        print('WARNING: Unrecognized keyword "{}"'.format(chans))
        print('Notching frequencies from DARM only.')
        notched_chans = 1

    widths = [width, width] # Hz

    for i in range(notched_chans):
        tar  = dataset[:, i]
        sp   = np.fft.rfft(tar)
        freq = np.fft.rfftfreq(tar.shape[-1], d=1.0 / fs)

        for f_notch, width in zip(notch_freqs, widths):

            ind_low  = (freq > f_notch - 2 * width) & (freq < f_notch-width)
            ind_high = (freq < f_notch + 2 * width) & (freq > f_notch+width)
            ind_mid  = (freq < f_notch + width) & (freq > f_notch-width)

            x_bg     = np.concatenate((freq[ind_low], freq[ind_high]))
            y_bg_amp = np.concatenate((np.abs(sp[ind_low]), np.abs(sp[ind_high])))

            m_amp, c_amp     = np.polyfit(x_bg, y_bg_amp, 1)
            background_amp   = m_amp * freq + c_amp
            background_phase = np.random.rand(len(freq),) * 2 * np.pi - np.pi

            sp[ind_mid] = background_amp[ind_mid] *\
                          np.exp(background_phase[ind_mid] * 1j)

        tar = np.fft.irfft(sp)
        dataset[:, i] = tar

    return dataset


def get_cbc_loss(batch_size  = 512,
                 sample_rate = 256,
                 Tbatch      = None,
                 Tchunk      = None,
                 f_lower     = 15.0,
                 f_upper     = 64.0,
                 mass1       = 38.9,
                 mass2       = 32.8,
                 notch_freqs = [60.0, 120.0],
                 width       = 1.0):

    if Tbatch is not None:
        Nbatch     = int(Tbatch * sample_rate)
        Nchunk     = int(Tchunk * sample_rate)
        batch_size = Nbatch
        if Nbatch != Nchunk:
            print('ERROR: Nbatch must be equal to Nchunk')
            sys.exit(1)

    # Compute waveform
    dt = 1.0 / sample_rate
    tt, hp, hc = blh.get_cbc(f_lower, dt, mass1=mass1, mass2=mass2)
    # why is the waveform calculation being repeated for each call to the cost function?

    # Ensure waveform and data have same lengths
    if batch_size > len(hp):
        hp = np.pad(hp, (0, batch_size - len(hp)), 'constant', constant_values=0.0)
        hc = np.pad(hc, (0, batch_size - len(hc)), 'constant', constant_values=0.0)

    elif batch_size < len(hp):
        print("batch_size smaller than the length of the waveform.")
        print("Cutting down arrays...")
        hp = hp[:batch_size]
        hc = hc[:batch_size]

    # make FFTs of the inspiral templates
    hp_fft, hc_fft = blh.tensorflow_fft(hp), blh.tensorflow_fft(hc)
    freq           = np.fft.fftfreq(len(hp), d=dt)
    freq_tensor    = tf.cast(tf.Variable(freq), tf.float64)

    widths = np.full_like(notch_freqs, width)
    for f_notch, bin_width in zip(notch_freqs, widths):
        min_val = tf.cast(tf.Variable(f_notch - bin_width), tf.float64)
        max_val = tf.cast(tf.Variable(f_notch + bin_width), tf.float64)
        hp_fft  = tf.where(tf.logical_and(tf.less(freq_tensor, max_val),
                                          tf.greater(freq_tensor, min_val)),
                                          tf.zeros_like(hp_fft), hp_fft)

    idx = np.where((freq >= f_lower) & (freq <= f_upper))[0]
    idx = idx.astype('int32')

    hp_sliced  = tf.slice(hp_fft, [idx[0]], [len(idx)])
    hp_squared = tf.cast(tf.square(hp_sliced), tf.float64)
    hp_squared = hp_squared / tf.reduce_sum(hp_squared)

    # this is the inverse whitening filter for OAF-CAL_DARM
    darm_weight = 1
    afterApril2017 = False   # for Jan data from O2

    if afterApril2017:
        f1 = 0.3
        f2 = 30
        npoles = 6
    else:
        f1 = 1
        f2 = 100
        npoles = 5

    ff = freq
    for i in range(npoles):
        darm_weight *= np.abs((f1/f2) * (f2 + 1j*ff)/(f1 + 1j*ff))

    def custom_objective(y_true, y_pred, hp_squared=hp_squared, idx=idx):
        # Difference of predicted and true
        ydiff = y_pred - y_true
        ydiff = tf.cast(ydiff, tf.complex64)

        # FFT of the difference
        ydiff_fft = tf.fft(ydiff)
        # apply whitening filter here:
        ydiff_fft *= darm_weight

        ydiff_sliced = tf.slice(ydiff_fft, [idx[0], 0], [len(idx), 1])
        ydiff_sliced = tf.squeeze(ydiff_sliced, axis=1)

        # PSD of the difference
        ydiff_squared = tf.square(ydiff_sliced)
        ydiff_squared = tf.cast(ydiff_squared, tf.float64)

        # This is something to maximize
        matched_filter = tf.divide(hp_squared, ydiff_squared)
        matched_filter = tf.cast(matched_filter, tf.float32)

        matched_filter_sum = tf.reduce_sum(matched_filter, axis=0)

        # This is something to minimize
        cost = matched_filter_sum

        return cost

    loss = custom_objective

    return loss


def downsample_dataset(dataset, fs, fs_slow):
    down_factor = int(fs // fs_slow)
    data_len    = int(dataset.shape[0]/down_factor)
    channels    = dataset.shape[1]
    resampled   = np.zeros(shape=(data_len, channels))

    fir_aa = sig.firwin(20 * down_factor + 1, 0.8 / down_factor,
                        window='blackmanharris')[1:-1]

    for i in range(dataset.shape[1]):
        resampled[:, i] = sig.decimate(dataset[:, i], down_factor,
                                       ftype      = sig.dlti(fir_aa, 1.0),
                                       zero_phase = True,
                                       axis       = -1)
    return resampled


def do_lookback(dataset, tseg, fs, fs_slow=None, min_samples=64):
    # Downsample if necessary
    if fs_slow is not None:
        dataset = downsample_dataset(dataset, fs, fs_slow)
        fs = fs_slow

    # Time segments must fit evenly into data
    lookback = int(fs * tseg)
    Nsegs = int(dataset.shape[0] - lookback)

    # Exit if there isn't enough data to return
    if Nsegs < min_samples:
        print('ERROR: Not enough samples with given lookback and sample rate')
        print('Try decreasing the lookback or downsampling less')
        sys.exit(1)

    chans = dataset.shape[1]
    data = np.zeros(shape=(Nsegs, lookback + 1, chans))

    # Fill the new array with the lookback data
    for chan in range(chans):
        for seg in range(Nsegs):
            info = dataset[seg: (seg + lookback + 1), chan]
            info = np.flip(info.reshape(len(info)), axis=0)
            data[seg, :, chan] = info

    return data


def undo_lookback(dataset, tseg, fs):
    Nsegs = int(dataset.shape[0] + dataset.shape[1] - 1)
    chans = dataset.shape[2]
    data  = np.zeros(shape=(Nsegs, chans))

    for chan in range(chans):
        data[:len(dataset[0, :, chan]), chan] = np.flip(dataset[0, :, chan], axis=0)

    for chan in range(chans):
        for j in range(dataset.shape[1] + 1, Nsegs + 1):
            data[j - 1, chan] = dataset[j - len(dataset[0, :, chan]), 0, chan]

    return data


def filter_channels(dataset, N=8, fknee=3, fs=256, btype='highpass'):
    f = fknee / ( fs / 2.0)
    z, p, k = sig.butter(N, f, btype=btype, output='zpk')
    sos = sig.zpk2sos(z, p, k)

    if 'low' in btype:
        start_freq = 20
        step0 = int(start_freq * fs)
        for ii in range(dataset.shape[1]):
            dataset[step0:, ii] = sig.sosfilt(sos, dataset[step0:, ii])

    else:
        for ii in range(dataset.shape[1]):
            dataset[:, ii] = sig.sosfilt(sos, dataset[:, ii])

    return dataset


def use_cleaned_data(dataset, model_basename):
    if isinstance(dataset, str):
        dataset, _ = ppr.get_dataset(dataset)

    # The mat file just created is the one we want to use
    PATH = 'params/{0}/'.format(model_basename)
    all_files  = os.listdir(PATH)
    mat_files  = [PATH + af for af in all_files if af.endswith('.mat')]
    recent_mat = max(mat_files, key=os.path.getctime)

    # Load it, get the new target data and reshape it
    chan_data  = sio.loadmat(recent_mat)
    subtracted = chan_data['subtracted'].T
    subtracted = subtracted.reshape(len(subtracted))

    # Cut dataset down to the size of the cleaned target array
    temp_dataset = np.zeros(shape=(len(subtracted), dataset.shape[1]))
    for chan in range(dataset.shape[1] - 1):
        temp_dataset[:, chan + 1] = dataset[-len(subtracted):, chan + 1]

    # Set darm (channel 0) to be the cleaned version
    temp_dataset[:, 0] = subtracted

    return temp_dataset


def nonlinear_channels(textfile, datafile):
    # Read the channel combinations
    combinations = []
    nums = re.compile(r'\d+')
    with open(textfile) as f:
        lines = f.readlines()
        datafile = lines[0].strip('\n').split(' ')[1].strip()
        for line in lines[2:]:
            channels = line.split('\t')[0]
            channels = nums.findall(channels)
            if len(channels) > 0:
                channels = map(int, channels)
                combinations.append(channels)

    dataset, fs = ppr.get_dataset(datafile)
    previous    = sio.loadmat(datafile)
    prev_chans  = previous['chans']

    chan_set = []
    for comb in combinations:
        for i in comb:
            chan_set.append(i)
    chan_set = list(set(chan_set))

    new_chans = [prev_chans[i] for i in chan_set]

    num_chans = len(combinations) + len(chan_set) + 1
    temp = np.ones(shape=(dataset.shape[0], num_chans))
    temp[:, 0] = dataset[:, 0]
    count = 0
    for ix, chan in enumerate(chan_set):
        temp[:, ix + 1] = dataset[:, chan]
        count = ix

    for combos in combinations:
        count += 1
        for i in combos:
            temp[:, count] *= dataset[:, i]

    output = {'data':temp, 'fsample':fs, 'chans':new_chans}
    sio.savemat('nonlinear_channels.mat', output, do_compression=True)


def filter_timeseries(target, prediction):
    target_fft = np.fft.fft(target)
    pred_fft   = np.fft.fft(prediction)
    new_pred   = []

    for i in range(len(target_fft)):
        if np.absolute(target_fft[i]) >= np.absolute(pred_fft[i]):
            new_pred.append(prediction[i])
        else:
            new_pred.append(0)

    return new_pred

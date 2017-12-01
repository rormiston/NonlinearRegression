from __future__ import division
import numpy as np
import scipy
import scipy.signal as sig
import matplotlib as mpl
import matplotlib.pyplot as plt
import subprocess
import theano

import os
# Hush TF AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

from cycler import cycler
from datetime import datetime
from numpy.lib.stride_tricks import as_strided
from scipy.io import loadmat

from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:
                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape,
                                         arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            #merged.append(merge(outputs, mode='concat', concat_axis=0))
            merged.append(concatenate(outputs,axis=0))

        return Model(inputs=model.inputs, outputs=merged)

def theano_fft(waveform):

    from theano.tensor.fft import rfft
    x = theano.tensor.matrix('x', dtype='float64')
    rfft = rfft(x, norm='ortho')
    f_rfft = theano.function([x], rfft)

    N = len(np.array(waveform))
    waveform_array = np.zeros((1, N), dtype='float64')
    waveform_array[:] = waveform
    waveform_fft = f_rfft(waveform_array)
    return waveform_fft

def tensorflow_fft(waveform):

    waveform_fft = tf.fft(waveform)
    return waveform_fft

def store_old_plots(figname):
    cmdStr = 'cp -p '
    try:
        cpStrPNG = cmdStr + figname + '_1' + '.png ' + figname + '_2' + '.png'
        cpStrPDF = cmdStr + figname + '_1' + '.pdf ' + figname + '_2' + '.pdf'
        subprocess.call(cpStrPNG, shell=True)
        subprocess.call(cpStrPDF, shell=True)
    except:
        print('Error: ' + cpStr)

    try:
        cpStrPNG = cmdStr + figname +        '.png ' + figname + '_1' + '.png'
        cpStrPDF = cmdStr + figname +        '.pdf ' + figname + '_1' + '.pdf'
        subprocess.call(cpStrPNG, shell=True)
        subprocess.call(cpStrPDF, shell=True)
    except:
        print('Error: ' + cpStr)

def get_cbc(f_lower, delta_t, mass1=38.9, mass2=32.8):

    filename = 'waveforms/cbc_%.1f_%.1f.dat' % (mass1, mass2)
    if not os.path.isfile(filename):
        try:
            from pycbc.waveform import get_td_waveform
        except:
            print("Waveform not found and no pycbc installed...")
            exit(0)
        hp, hc = get_td_waveform(
            approximant="SEOBNRv2",
            mass1=mass1,
            mass2=mass2,
            f_lower=f_lower,
            delta_t=1.0/16384.0,
            distance=1)
        tt = np.array(hp.sample_times)
        t0 = tt[0]

        q = int(16384.0*delta_t)
        hp = scipy.signal.decimate(np.array(hp), q)
        hc = scipy.signal.decimate(np.array(hc), q)
        tt = tt[::q]

        f = open(filename, "w+")
        for i in xrange(len(tt)):
            f.write("%.15e %.15e %.15e\n" % (tt[i] - t0, hp[i], hc[i]))
        f.close()

    data_out = np.loadtxt(filename)
    tt, hp, hc = data_out[:, 0], data_out[:, 1], data_out[:, 2]
    dt = tt[1] - tt[0]
    if dt != delta_t:
        raise ValueError("Waveform delta t (%.0f) different from "
                         "requested(%.0f)" % (dt, delta_t))

    return tt, hp, hc


def compute_overlap(tfft, fs, f_lower, ff, psd_array, hp=[], hc=[]):

    nfft = int(tfft * fs)
    flen = int(nfft / 2)
    df = 1.0 / tfft
    dt = 1.0 / fs

    if len(hp) == 0 or len(hc) == 0:
        tt, hp, hc = get_cbc(f_lower, dt, mass1=38.9, mass2=32.8)
    freq = np.fft.rfftfreq(len(hp), d=dt)
    rfft = np.fft.rfft(hp)

    flog = np.log(ff)
    slog = np.log(psd_array)

    psd_interp = scipy.interpolate.interp1d(flog, slog)
    psd = np.exp(psd_interp(np.log(freq)))

    m = np.sum(np.abs(rfft)**2 / psd)

    return m


def normalize(x, scale=None, filter=None):
    '''
    Create zero mean, unit std version of signal; optionally filtered
    '''
    # Transpose gymnastics neccesary for row-wise subtraction
    x = (x.T - np.mean(x, axis=-1)).T

    if filter is not None:
        x = sig.sosfilt(filter, x, axis=-1)

    if scale is None:
        scale = np.std(x, axis=-1)
    x = (x.T / scale).T

    return x, scale


def stride_wit(x, nperseg, step=1):
    '''
    Memory efficient way of breaking a 1D array up into overlapping segments
    '''
    step     = int(step)
    noverlap = nperseg - step

    shape   = ((x.shape[-1] - noverlap) // step, ) + x.shape[:-1] + (nperseg, )
    strides = (step * x.strides[-1],) + x.strides[:-1] + (x.strides[-1],)

    # Using `writeable = False` avoids destructive manipulations
    return as_strided(x, shape=shape, strides=strides, writeable=False)


def cost_filter(fs, zeros, poles, zero_order, pole_order):
    s_zeros = []
    s_poles = []

    if sum(zero_order) != sum(pole_order):
        raise ValueError('Bandpass not invertible!')

    for ii, fz in enumerate(zeros):
        _, sz, _ = sig.butter(zero_order[ii], fz / (fs / 2), output='zpk')
        s_zeros.append(sz)

    for ii, fp in enumerate(poles):
        _, sp, _ = sig.butter(pole_order[ii], fp / (fs / 2), output='zpk')
        s_poles.append(sp)

    s_zeros = np.concatenate(s_zeros)
    s_poles = np.concatenate(s_poles)
    BP      = sig.zpk2sos(s_zeros, s_poles, 1, pairing='keep_odd')
    invBP   = sig.zpk2sos(s_poles, s_zeros, 1, pairing='keep_odd')

    return BP, invBP


def cal_darm(f, x, NewEpoch=True):
    '''
    This applies a frequency domain calibration to DARM spectra.
    OAF-CAL_DARM => Strain
    '''
    pss = np.array([1, 1, 1, 1, 1])
    if NewEpoch:
        ps = 0.3 * pss
        zs = 30  * pss
    else:
        ps = 1   * pss
        zs = 100 * pss

    calTF = np.ones_like(x)
    for k in range(len(ps)):
        calTF *= np.abs( (ps[k]/zs[k]) * (zs[k] + 1j*f)/(ps[k] + 1j*f))

    strain = x * calTF

    return strain


def downsample(x, down_factor):
    # Using fir_aa[1:-1] cuts off a leading and trailing zero
    fir_aa = sig.firwin(20 * down_factor + 1, 0.8 / down_factor,
                        window='blackmanharris')[1:-1]
    out = sig.decimate(x, down_factor, ftype=sig.dlti(fir_aa, 1.0),
                       zero_phase=True, axis=-1)

    return out


def set_plot_style():
    # Now alter my matplotlib parameters
    plt.style.use('bmh')
    mpl.rcParams.update({
        'axes.grid': True,
        'axes.titlesize': 'medium',
        'font.family': 'serif',
        'font.size': 12,
        'grid.color': 'w',
        'grid.linestyle': '-',
        'grid.alpha': 0.5,
        'grid.linewidth': 1,
        'legend.borderpad': 0.2,
        'legend.fancybox': True,
        'legend.fontsize': 13,
        'legend.framealpha': 0.7,
        'legend.handletextpad': 0.1,
        'legend.labelspacing': 0.2,
        'legend.loc': 'best',
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{txfonts}'
    })

    mpl.rc("savefig", dpi=100)
    mpl.rc("figure", figsize=(7, 4))


def plot_results(target, prediction, bg, fs, nfft,
                 plotDir='.', title_str=None, plotStrain=False):

    residual = target - prediction

    ff1, pp = sig.welch([target, prediction, residual, bg],
                        fs=fs, nperseg=nfft, axis=-1)
    ff2, co = sig.coherence(target, prediction,
                            fs=fs, nperseg=nfft)

    title_str = 'IFO = ' + 'L1' + ' Nlayers = ' + str(7)
    datestr = '{:%y%m%d_%H%M}'.format(datetime.now())
    info_str = 'Dense NN ' + '3 ' + 'N_layers = ' + '7'
    # make plots to evaluate success / failure of the regression
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                       figsize=(6, 8),
                                       gridspec_kw={'height_ratios':[3, 1]})


    if plotStrain:
        strain = cal_darm(ff1, np.sqrt(pp),
                          NewEpoch=False)
        srd = loadmat('Data/SRD_25W.mat')
        strain = strain.T
    else:
        strain = np.sqrt(pp).T

    ax1.set_prop_cycle(cycler('color',
                                      ['xkcd:jade',
                                       'xkcd:blood orange',
                                       'xkcd:burple',
                                       'xkcd:cement']))
    labels = ['DARM', 'Prediction', 'Subtracted']

    if plotStrain:
        ax1.loglog(srd['f'].T, srd['x'].T,
                   c = 'xkcd:cement',
                   alpha = 0.5,
                   label = 'Quantum/Thermo',
                   rasterized = True)
        ax1.set_ylim([3e-21, 3e-15])

    for i in range(3):
        ax1.loglog(ff1.T, strain[:,i],
                    alpha = 0.8,
                    rasterized=True,
                       label=labels[i])

    ax1.legend(fontsize='small')
    ax1.set_xlim([7, 150])
    ax1.grid(True, which='minor')
    ax1.set_ylabel(r'DARM (m/$\sqrt{\rm Hz}$)')
    ax1.set_title(title_str)

    ax2.semilogx(ff2.T, co,
                 label = 'DARM/Prediction',
                 c = 'xkcd:poop brown',
                 rasterized = True)
    ax2.grid(True, which='minor')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel(r'Frequency (Hz)')
    ax2.set_ylabel(r'Coherence')
    foo = ax2.text(51, 0.7, info_str,
                       color = 'xkcd:burple',
                           fontsize='xx-small',
                           horizontalalignment='right',
                           verticalalignment='center')
    ax2.legend(fontsize='x-small')

    plt.subplots_adjust(hspace=0.05)

    # save figure,
    # backing up previous versions for easy comparison
    figName = '{}/TF-bilinear-validation'.format(plotDir)

    try:
        get_ipython
        plt.show()
    except NameError:
        store_old_plots(figName)
        plt.savefig('{}.png'.format(figName))
        plt.savefig('{}.pdf'.format(figName))


def plot_training_progress(model,
                           plotDir = './',
                           minLoss = 0,
                           version = 0):
    plt.figure(123456)
    plt.semilogy(model.history['loss'],
                 marker='o', markeredgewidth=0, alpha=0.7,
                 mfc='cyan', label='Train')
    plt.semilogy(model.history['val_loss'],
                 marker='o', markeredgewidth=0, alpha=0.7,
                 mfc='purple', label='Test')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.axhline(y = minLoss, color='coral', label='Minimum')
    # plt.ylim((0.01, 1.1))
    plt.legend()


    try:
        get_ipython
        plt.show()
    except NameError:
        figName = '{0}/loss_history_comparison-{1}.png'.format(plotDir, version)
        figNamePDF = '{0}/loss_history_comparison-{1}.pdf'.format(plotDir, version)
        plt.savefig(figName)
        plt.savefig(figNamePDF)
        plt.close()


def load_data(filename):
    datas   = loadmat(filename)
    bg_raw  = datas['background'][0]
    tar_raw = datas['darm'][0]
    wit     = datas['wit']
    fs      = datas['fs'][0][0]
    return bg_raw, tar_raw, wit, fs


def plot_cost_asd(tar, bg, fs, nfft, plotDir='./', version=0):
    from scipy.integrate import cumtrapz

    ff, ptar = sig.welch(tar, fs=fs, nperseg=nfft)
    #ff, pbg  = sig.welch(bg,  fs=fs, nperseg=nfft)
    ff = np.squeeze(ff)

    tar_rms  = np.flipud(np.sqrt(-cumtrapz(ptar[::-1], ff[::-1],
                                           initial=0)))
    #bg_rms   = np.flipud(np.sqrt(-cumtrapz( pbg[::-1], ff[::-1],
    #                                        initial=0)))

    plt.figure(911)
    ht, = plt.loglog(ff, np.sqrt(ptar),
                     label='Target',rasterized=True)
    #hb, = plt.loglog(ff, np.sqrt(pbg), label='Background')
    plt.loglog(ff, tar_rms,
               linestyle='dashed', color=ht.get_color(),rasterized=True)
    #plt.semilogx(ff, bg_rms,  linestyle='dashed', color=hb.get_color())
    plt.legend()
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    plt.title(r'Shaped DARM cost for NN training')
    plt.xlabel(r'Frequency [Hz]')
    plt.ylabel(r'ASD [1/$\sqrt{\rm Hz}$]')

    try:
        get_ipython
        plt.show()
    except NameError:
        fig = '{0}/cost_function_spectrum-{1}.png'
        figPDF = '{0}/cost_function_spectrum-{1}.pdf'
        plt.savefig(fig.format(plotDir, version))
        plt.savefig(figPDF.format(plotDir, version))
        plt.close()


def prepare_inputs(beam_spot, angular, chunk_size):

    Nasc        = angular.shape[0]
    Nspots      = beam_spot.shape[0]
    down_factor = angular.shape[-1] // beam_spot.shape[-1]

    # Zero pad each signal, so that there is a window for every target sample
    spot_chunk_size = chunk_size // down_factor
    padded_spot     = np.pad(beam_spot, ((0, 0), (spot_chunk_size - 1, 0)),
                         mode='constant', constant_values=0)
    strided_spot    = stride_wit(padded_spot, chunk_size // down_factor)

    # Stack the beam spot signals
    stacked_spot = strided_spot.reshape((strided_spot.shape[0],
                                         Nspots * spot_chunk_size))

    # Spot signal is downsampled, so just repeat each slow window for
    # successive fast target samples
    stacked_spot = np.repeat(stacked_spot, down_factor, axis=0)

    padded_ang = np.pad(angular, ((0, 0), (chunk_size - 1, 0)),
                        mode='constant', constant_values=0)
    strided_ang = stride_wit(padded_ang, chunk_size)

    # Stack the angular signals
    stacked_ang = strided_ang.reshape((strided_ang.shape[0],
                                       Nasc * chunk_size))

    # Stack them all together!
    out = np.concatenate([stacked_spot, stacked_ang], axis=-1)
    return out


def subtract_cal(target, fs, fsub, t_chunk=100):
    if target.ndim > 1:
        raise ValueError('1D signals only')

    fsub = np.asarray(fsub)
    tt = np.arange(target.size) / float(fs)
    linefit = np.zeros(tt.shape)

    for f in fsub:
        if f < fs / 2:
            yy = fft_list(target, f, t_chunk, fs)
            mag = np.mean(np.abs(yy))
            ph = np.mean(np.unwrap(np.angle(yy)))

            linefit += mag * np.sin(2 * np.pi * f * tt + ph)

    result = target - linefit
    return result


def fft_list(x, fint, t_chunk, fs):

    window = 'flattop'
    detrend = 'constant'

    x = np.asarray(x)
    x = sig.detrend(x, type=detrend)

    if x.size == 0:
        return np.empty(x.shape)

    nperseg = int(np.round(t_chunk * fs))
    nperseg = min(x.shape[-1], nperseg)
    noverlap = nperseg // 2

    win = sig.get_window(window, nperseg)

    result = _segdemod(x, fint, fs, win, nperseg, noverlap)
    scale = 2 * np.sqrt(1 / (win.sum()**2))
    result *= scale

    return result


def _segdemod(x, f, fs, win, nperseg, noverlap):
    # Make sin in phase
    d = 1j * np.exp(-1j * 2 * np.pi * f * (np.arange(len(x)) / fs))

    step = nperseg - noverlap
    shape = ((x.shape[-1] - noverlap) // step, nperseg)
    xstrides = (step * x.strides[-1], x.strides[-1])
    dstrides = (step * d.strides[-1], d.strides[-1])

    xs = np.lib.stride_tricks.as_strided(x, shape=shape, strides=xstrides)
    ds = np.lib.stride_tricks.as_strided(d, shape=shape, strides=dstrides)

    # Apply window by multiplication
    result = ds * xs * win
    result = result.sum(axis=-1)

    return result

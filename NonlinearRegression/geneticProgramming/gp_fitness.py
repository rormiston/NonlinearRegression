import sys
import numpy as np
import scipy.signal as sig
import noisesub
from mockdata import SEC, FS, NOISE_LEVEL


def basic_fitness(times, target, y1, y2, toolbox, individual):
    try:
        f_guess = np.vectorize(toolbox.compile(expr=individual))
        diff = target - coupling_filter(f_guess(y1, y2))
    except ValueError:
        return sys.float_info.max,
    except OverflowError:
        return sys.float_info.max,

    fitness =  (NOISE_LEVEL**(-2))*diff.dot(diff)/len(target),# must be tuple
    return fitness

def inner_product(x, y, nbands):
    xfft = np.fft.rfft(x)[:-1]
    yfft = np.fft.rfft(y)[:-1]
    nperband = xfft.size//nbands
    xfft = np.reshape(xfft, (nbands,nperband))
    yfft = np.conj(np.reshape(yfft, (nbands,nperband)))
    # detrend
    xfft = xfft-np.mean(xfft, axis=1, keepdims=True)
    yfft = yfft-np.mean(yfft, axis=1, keepdims=True)
    return np.einsum('ij,ij->i', xfft, yfft)

def coherence_sub(tar, wit, td_seg=True, signif=True):
    '''
    calculate frequency domain coherence between guessed function and target
    '''
    N = tar.size

    bins = 128 # bins per frequency band
    bands = N//(2*bins) # number of frequency bands

    if(td_seg):
        ff, coherence, coefs = noisesub.mcoherence(tar, wit, FS, nperseg=2*bands)
            # noverlap=0, window="boxcar")
        coefs = coefs[0]
    else:
        csd = inner_product(tar, wit, bands)
        tpsd = inner_product(tar, tar, bands)
        wpsd = inner_product(wit, wit, bands)
        coefs = csd/wpsd
        coherence = np.real((csd*np.conj(csd)/(tpsd*wpsd)))

    # if signif=True, don't perform subtraction at freqs with low coherence
    if(signif):
        set_0 = np.less(abs(coherence), .05)
        coefs[set_0] = 0.0
        low = 10*SEC//bins
        coefs[0:low] = np.zeros(low)

    if(td_seg):
        # maybe there's some way to apply windowing to the coefs to match the csd?
        coefs = np.repeat(coefs, bins)
        coefs = coefs[bins//2:-1*bins//2]
        #coefs = np.append(coefs, 0.0)
    else:
        coefs = np.repeat(coefs, bins)
        #coefs = np.append(coefs, 0.0)

    sub = np.fft.rfft(wit)[:-1]
    # discard nyquist bin so that the bands fit evenly
    sub = sub*coefs
    tarfft = np.fft.rfft(tar)[:-1]

    diff = tarfft - sub
    y_guess = np.fft.irfft(diff, len(tar))

    return diff, coherence, y_guess

def fitness_in_band(y_guess, lowf=50.0, highf=400.0, nfft=2048):
    ff, y_psd = sig.welch(y_guess, FS, nperseg=nfft)
    # measure fitness in specified band by average psd compared to noise level
    bins = len(y_guess)//nfft
    fit = y_psd[lowf*SEC//bins:highf*SEC//bins]
    fit = np.sum(fit)/(NOISE_LEVEL**2*len(fit))
    return fit

def coh_fitness(times, target, y1, y2, toolbox, individual, verbose=False):
    '''
    determine the 'fitness' of a candidate function by:
        1. evaluating the function with the test data (y1, y2) as input
        2. calculating the coherence of the result with the target signal
        3. subtracting the product of the evaluated data and the optimal
           transfer function for each frequency band
        4. estimating fitness by average psd in relevant frequency band
    '''
    try:
        f_guess = np.vectorize(toolbox.compile(expr=individual))
        test_wit = NOISE_LEVEL*f_guess(y1,y2)
        yl = len(y1)
        diff, coh, y_guess = coherence_sub(target, test_wit, td_seg=False)
        fit = fitness_in_band(y_guess)
        if(np.isnan(fit)==False):
            return fit,
        else:
            return sys.float_info.max,
    except ValueError:
        return sys.float_info.max,
    except OverflowError:
        return sys.float_info.max,
    except ZeroDivisionError:
        return sys.float_info.max,
    except np.linalg.linalg.LinAlgError:
        return sys.float_info.max,

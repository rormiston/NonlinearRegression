import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as sig
import noisesub
from regplots import plot_results


# run paramters

# how many variables get stored and interpreted
latent_dim = 200
# how many time slices of the spectrogram to use each step
batch_size = 8
lrate = 1e-3
# True for simple data, False to read from make_STFT_data files
use_simple_data = False

# simple data to try: including filters on each witness and overall
def make_simple_data():
    ts_len = 2048*2048
    seglen = 512
    y1_ts = np.random.randn(ts_len)
    y2_ts = np.random.randn(ts_len)
    bg_ts = np.random.randn(ts_len)

    b1, a1 = sig.butter(1, [.1, .4], 'bandpass')
    b2, a2 = sig.butter(1, [.3], 'lowpass')
    b3, a3 = sig.butter(3, [.36, .37], 'bandpass')

    # add nonlinear noise with filters to background
    noise = np.sin(.3*sig.lfilter(b2, a2, y1_ts) + .7*sig.lfilter(b3, a3, y2_ts))
    darm_ts = 0.2*bg_ts + sig.filtfilt(b1, a1, noise)

    freqs, ts, darm = noisesub.stft(darm_ts, fs=2048, nperseg=seglen, return_onesided=True)
    freqs, ts, y1 = noisesub.stft(y1_ts, fs=2048, nperseg=seglen, return_onesided=True)
    freqs, ts, y2 = noisesub.stft(y2_ts, fs=2048, nperseg=seglen, return_onesided=True)
    freqs, ts, data = noisesub.stft(0.2*bg_ts, fs=2048, nperseg=seglen, return_onesided=True)
    print "ts length", bg_ts.shape
    return darm, data, y1, y2

# Choose where to get data
if(use_simple_data):
    darm, data, y1, y2 = make_simple_data()
    asd_norm=1
else:
    darm = np.load("STFT_darm.npy")
    asd_norm=np.real(np.max(darm))
    darm /= asd_norm
    data = np.load("STFT_data.npy")
    data /= asd_norm
    y1 = np.load("STFT_wit1.npy")
    y2 = np.load("STFT_wit2.npy")

lenFFT = darm.shape[0]
lenT = darm.shape[1]
seglen = 2*(lenFFT-1)

print lenFFT
loss_weights = np.zeros(lenFFT)
loss_weights[np.floor(50*(lenFFT/1024)):np.floor(65*(lenFFT/1024))]=1.0

# used to convert spectrogram to psd
win = sig.get_window('hann', 512)
rescale = 2*win.sum()/(256*(win*win).sum())

# variable initialization: each complex weight is combination of 2 real vars
def init_vars(shape, std=0.05, mean=0.0, imean=0.0):
    real = mean + tf.random_normal(shape, stddev=std, dtype=tf.float64)
    imag = imean + tf.random_normal(shape, stddev=std, dtype=tf.float64)
    return tf.complex(tf.Variable(real),tf.Variable(imag))

# set up weights and biases according to network architecture
weights = {
    # first stage of map is encoding: info about both witness spectra is
    # stored in "latent variables" which have some probability distribution
    # stored as a mean and variance. See encode()
    'enc_1': init_vars([2*lenFFT, 256], std=0.05),
    'enc_2': init_vars([256, 256], std=0.05),
    'enc_mean': init_vars([256, latent_dim], std=0.05),
    'enc_std': init_vars([256, latent_dim], std=0.5),

    # second stage is decoding: latent variables are interpreted to transform
    # the witnesses into a guess of the noise. The nonlinearity comes from
    # how the transformation depends on the input spectra (through the latent
    # variables). See decode()
    'dec_1': init_vars([latent_dim, 256], std=0.2),
    'dec_2': init_vars([256, 256], std=0.2),
    'dec_out': init_vars([256, 2*lenFFT], std = 0.01)
}

biases = {
    'enc_1': init_vars([1,256], std=0.05),
    'enc_2': init_vars([1,256], std=0.05),
    'enc_mean': init_vars([1, latent_dim], std=0.05),
    'enc_std': init_vars([1, latent_dim], std=0.5, mean=-5.0),

    'dec_1': init_vars([1, 256], std=0.2),
    'dec_2': init_vars([1, 256], std=0.2),
    'dec_out': init_vars([1,2*lenFFT], std=0.05, mean=0.0)
}

# Overall gain parameter keeps weights from all getting immediately pushed to
# zero (rather than finding real coherence)
# gets softplused to keep from crossing zero and training weights in
# opposite directions
overall_gain = tf.Variable(-6*tf.ones([], dtype=tf.float64))


def to_csd(spec1, spec2):
    '''
    Turn complex spectrograms into CSDs by multiplying, rescaling and averaging
    '''
    csd_grid = tf.multiply(spec1, tf.conj(spec2))
    csd_real = rescale*tf.real(csd_grid)
    csd_imag = rescale*tf.imag(csd_grid)
    csd_mean = tf.complex(
        tf.reduce_mean(csd_real, 0),
        tf.reduce_mean(csd_imag, 0))
    return csd_mean

def to_psd(spec1):
    '''
    Turn a complex spectrogram into a PSDs by squaring, rescaling and averaging
    '''
    psd = tf.reduce_mean(
        rescale*tf.complex_abs(spec1*tf.conj(spec1)), 0)
    return psd # should be real anyway

def get_coherence(x, y):
    '''
    calculate coherence between two complex spectrograms
    '''
    mag_csd = tf.complex_abs(to_csd(x, y))
    return tf.square(mag_csd)/(to_psd(x)*to_psd(y))

def encode(x):
    '''
    Generate latent variable probablility distribtions for each time slice
    of the witness spectrograms.
    '''
    # x: shape (batch_size, 2, lenFFT), the 2 is for 2 channels
    x = tf.reshape(x, (batch_size, -1))
    hid1 = tf.tanh(tf.matmul(x, weights['enc_1']) + biases['enc_1'])
    hid2 = tf.tanh(tf.matmul(hid1, weights['enc_2']) + biases['enc_2'])

    mean_z = tf.matmul(hid2, weights['enc_mean']) + biases['enc_mean']
    std_z = tf.matmul(hid2, weights['enc_std']) + biases['enc_std']
    std_z = tf.nn.softplus(tf.real(std_z))
    # output distribution in latent space
    return mean_z, std_z # each has shape (batch_size, latent_dim)


def decode(x, z):
    # z: shape (batch_size, latent_dim)
    # x: hape (batch_size, 2, lenFFT), the 2 is for 2 channels
    hid1 = tf.tanh(tf.matmul(z, weights['dec_1']) + biases['dec_1'])
    hid2 = tf.tanh(tf.matmul(hid1, weights['dec_2'])+ biases['dec_2'])

    final = tf.matmul(hid2, weights['dec_out']) + biases['dec_out']

    # transform_x is noise estimate before normalization
    transform_x = tf.reduce_sum(x*tf.reshape(final, [batch_size, 2, -1]), 1)
    norm = tf.sqrt(tf.reduce_mean(tf.square(tf.complex_abs(transform_x))))
    return transform_x*comp(tf.nn.softplus(overall_gain)/norm)

# tensorflow doesn't let you multiply complex and real types, so you
# have to explicity convert to complex first
def comp(x):
    return tf.complex(tf.to_double(x), tf.to_double(0.0*x))

# Specify expected inputs to the model
spec_x = tf.placeholder(tf.complex128, [batch_size, 2, lenFFT])
spec_y = tf.placeholder(tf.complex128, [batch_size, lenFFT])
epsilon = tf.placeholder(tf.complex128, [1, latent_dim])
eval_weights = tf.placeholder(tf.float64, [lenFFT])

# Actual steps of model compuation:

# get latent distribution representation of input channel behaviors
mean_z, std_z = encode(spec_x)

# probabalistically sample latent distribution
z = mean_z + epsilon*comp(std_z)

# get darm subtraction term
noise_guess = decode(spec_x, z)

# calculate losses: cross entropy and KL divergence according to
# standard variable auto-encoder; coherence can be interprested as probability
# add an extra term for rms; use a constant to balance it with other terms

# maybe slice to choose relevant freqs, or multiply by weight vector
coherence = get_coherence(noise_guess, spec_y)
loss_xent = -1*tf.reduce_sum(tf.log(coherence)*eval_weights)

D_j = 1 + tf.log(tf.square(std_z)) - tf.square(tf.complex_abs(mean_z)) - tf.square(std_z)
loss_div = tf.reduce_mean(-0.5*tf.reduce_sum(D_j, 1))

loss_rms = 100*tf.sqrt(tf.reduce_sum(to_psd(spec_y - noise_guess)*eval_weights))\
            /tf.sqrt(tf.reduce_sum(to_psd(spec_y)*eval_weights))

loss = loss_xent + loss_div + loss_rms
optimizer = tf.train.AdamOptimizer(learning_rate = lrate)
train_step = optimizer.minimize(loss)

# evaluate how much of the residual is the original data: NOT to be
# used in loss calculations!
dat = tf.placeholder(tf.complex128, [batch_size,lenFFT])
data_coherence = tf.reduce_mean(get_coherence(dat, spec_y-noise_guess))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
epochs = 10


def reconstruct_ts(spec):
    t, ts = noisesub.istft(spec, fs=2048, nperseg=seglen, input_onesided=True)
    return np.real(ts)

#Main loop
for i in range(0, epochs):

    start = 0
    end = start + batch_size
    my_dict = {}
    total_loss = 0
    while (end < (lenT-batch_size-1)): # save last batch for test
        x_in = np.rollaxis(np.array([
            y1[:,start:end],
            y2[:,start:end],
        ]), 2)
        y_in = np.rollaxis(np.array(darm[:,start:end]), 1)
        ep = np.random.normal(size=(1,latent_dim))+ 1j*np.random.normal(size=(1,latent_dim))
        my_dict = {spec_x:x_in, spec_y:y_in, epsilon:ep, eval_weights:loss_weights}
        sess.run(train_step, feed_dict = my_dict)
        start = end
        end = start + batch_size
        total_loss += sess.run(loss, feed_dict = my_dict)
    print "Epoch", i
    print "Total Loss", total_loss
    print "Coherence", np.mean(sess.run(coherence, feed_dict = my_dict)*loss_weights)/np.mean(loss_weights)
    print "RMS", sess.run(loss_rms/100, feed_dict = my_dict)
    dat_in = np.rollaxis(
        np.array(data[:,start-batch_size:start]), 1)
    my_dict[dat] = dat_in
    print "Data Coherence", sess.run(data_coherence, feed_dict = my_dict)
    print sess.run(tf.nn.softplus(overall_gain))

    # After several iterations, give an update on test performance
    if(i%10 == 9 or i==epochs-1):
        # get thr correct chunk of data
        start = lenT - batch_size
        end = start + batch_size
        test_x_in = np.rollaxis(np.array([
            y1[:,start:end],
            y2[:,start:end],
        ]), 2)

        test_y_in = np.rollaxis(np.array(darm[:,start:end]), 1)
        ep = np.random.normal(size=(1,latent_dim))+ 1j*np.random.normal(size=(1,latent_dim))
        my_dict = {spec_x:test_x_in, spec_y:test_y_in,
                   epsilon:ep, eval_weights:loss_weights}

        # check performance
        print "TEST RMS", sess.run(loss_rms/100, feed_dict = my_dict)
        test_dat_in = np.rollaxis(
            np.array(data[:,start:end]), 1)
        my_dict[dat] = test_dat_in
        print "TEST Data Coherence", sess.run(data_coherence, feed_dict = my_dict)


# end of computation: trainintg complete
start = lenT - batch_size
end = start + batch_size
y1_valid = reconstruct_ts(y1[:,start:end])
y2_valid = reconstruct_ts(y2[:,start:end])
data_valid = reconstruct_ts(data[:,start:end])
darm_valid = reconstruct_ts(darm[:,start:end])

noise_guess_test = sess.run(noise_guess, feed_dict = my_dict)
noise_guess_valid = reconstruct_ts(np.rollaxis(noise_guess_test, 1))

print "data segment", data[:, start:end].shape
print "data_valid", data_valid.shape
print y1_valid.shape, y2_valid.shape, noise_guess_valid.shape, darm_valid.shape
print "all data", data.shape
print "all reconst", reconstruct_ts(data).shape

# this call makes ASD, coherence, and timeseries plots
plot_results(darm_valid*asd_norm, y1_valid, y2_valid,
    noise_guess_valid*asd_norm, data_valid*asd_norm)

# make timeseries plots
# plt.figure()
# plt.plot((darm_valid-data_valid)[0:200], label="real noise")
# plt.plot(noise_guess_valid[0:200], label="noise guess")
# plt.legend()
# plt.savefig("fit_%d.png" %i)
# plt.close()
#
# plt.figure()
# plt.plot(data_valid[0:200], label="bg")
# # plt.plot(darm_test[0:200], label="darm")
# plt.plot(darm_valid[0:200] - noise_guess_valid[0:200], label="resid")
# plt.legend()
# plt.savefig("res_%d.png" %i)
# plt.close()

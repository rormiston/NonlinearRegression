import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import scipy.signal as sig
from scipy import interpolate
from mockdata import starting_data
from mockdata.mock_noise import scatfunk as f
from regplots import plot_results
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
from mockdata.params import *
from tf_2dhelpers import *

# change to True to print loss on each iteration: slows down computation
# because it's a separate evaluation of the tensorflow model
print_loss = True
# make animation of lattice evolution. Takes a while, maybe code could be
# improved to use less time/memory
animate = False

# Set number of iterations
nIter = 75

# get data: right now it's hardcoded to use the scattering model
times, darm, wit1, wit2, data = starting_data(
    include_filt=True, scat_model=True)

# format data for model input
wit1 = wit1.astype(np.float32)
xscale = np.std(wit1)
wit2 = wit2.astype(np.float32)
yscale = np.std(wit2)
print "Scaling:", xscale, yscale

# highpass to cut out low freq noise we can't subtract
fs = int(1/times[1])
b,a = sig.butter(4, 2*30.0/fs, btype='highpass')
darm_scaled = (sig.filtfilt(b,a,darm)/NOISE_LEVEL).astype(np.float32)
data_scaled = (sig.filtfilt(b,a,data)/NOISE_LEVEL).astype(np.float32)


# bandpass to use for loss calculation: filtering is done *within*
# tensorflow model using built in convolution funcitons. Its purpose
# is to prioritize the relevant freequencies while evaluation the model's
# subtraction success.
fir_bandpass = sig.firwin(N+1, [50, 400], pass_zero=False, nyq = fs/2.0)


# Explanation of the lattice concept:
# We want to estimate a nonlinear function of 2 witness channels. We
# assume that the function is relatively smooth, in the sense that it
# doesn't suddenly jump around with small variations of its input.
# We can then normalize the witnesses to 0 mean and unit variance and
# plot each sample in the time series as an (x,y) grid point
# (wit1(t), wit2(t)). To allow the inclusion of a few outlying points
# without sacrificing resolution, the values are passed through tanh to map
# them all between -1 and 1.
# The function value at each (wit1(t), wit2(t)) point is estimated by
# interpolation from a lattice of regularly spaced points in (wit1, wit2)
# space associated with nonlinear function guesses.
#
# The simplest way would be to find the function values on the lattice
# independently. However, this allows for too much overfitting, as well
# as many useless weights. Instead, we use the assumption that the
# function is smooth to adjust a smaller subset of parameters representing
# the first n terms in the 2d fourier transform of the lattice. The
# gradients for each of these terms is surprisingly correlated to the
# quality of the eventual subtraction result, meaning the model converges
# quickly.


# lattice parameters
lattice_res = 100 # resolution of the lattice
# to avoid infinite results in arctanh, lattice points are calculated
# from [-1+epsilon, 1-epsilon] instead of [-1,1]
epsilon = 1e-5
# how many fourier terms to include for each dimension
n = 10

# generate a grid of values according to fourier coefficients
# (magnitude and phase for a combination of x and y frequencies)
# These coeffieicents are immediately interpreted and mapped to
# the normalized (x,y) space; the computation does NOT take place
# directly in the 2d frequenct space.
def generate_grids(weights, phases):
    '''
    Given 2d Fourier coefficients, generate lattice_res by lattice_res
    candidate grids of estimated function values.
    Number of grids generated is equal to first dimension of arguments.
    '''
    # weights: nHidden1 x n x 2n
    # phases: nHidden1 x n x 2n
    # out: nHidden1 x lattice_res x lattice_res
    # each grid is sum over n x 2n fourier terms, so before contraction the
    # grids tensor has shape [nHidden1, n, 2n, lattice_res, lattice_res]
    coords = tf.linspace(-1 + epsilon, 1-epsilon, lattice_res)
    xs = tf.reshape(coords, [1, 1, 1, lattice_res, 1])
    ys = tf.reshape(coords, [1, 1, 1, 1, lattice_res])
    fx = tf.reshape(tf.linspace(0.0, n-1, n), [1, n, 1, 1, 1])
    fy = tf.reshape(tf.linspace(-n*1.0, n-1, 2*n), [1, 1, 2*n, 1, 1])

    phi = tf.reshape(phases, [-1, n, 2*n, 1, 1])
    amp = tf.reshape(weights, [-1, n, 2*n, 1, 1])
    grids = amp*tf.sin(np.pi*(fx*xs + fy*ys)/2 + phi) # rank 5
    # sum up terms to make grids, reshape for matmul compatibility later
    out = tf.reshape(tf.reduce_sum(grids, [1,2]), [-1, lattice_res*lattice_res])
    return out

# copied from Rory's example.
def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    '''
    Initialize TensorFlow variables with values distributed according
    to arguments. Xavier method is intended to take netowrk population
    into account to share responsibility fairly.
    '''
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(
            tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
        high = 8*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high))

def multilayer_perceptron(weights, biases):
    '''
    Evaluate network inputs to calculate function value estimates on lattice.
    '''
    # rank 3 tensor [nHidden1, lattice_res, lattice_res]
    # instead of matrix multiplication, this step turns the variables
    # into lattics values by interpreting them as Fourier terms
    layer_1 = generate_grids(weights['h1'], biases['b1']) # rank 3 tensor

    # combine candidate lattices according to learned weights.
    layer_2 = tf.add(tf.matmul(weights['h2'], layer_1), biases['b2'])
    # layer_2 = tf.nn.tanh(layer_2)

    # Current settings have only one unit in layer 2 so this step isn't
    # strictly necessary; however it may be useful to combine lattices
    # through multiple layers instead.
    out_layer = tf.add(tf.matmul(weights['out'], layer_2), biases['out'])
    return out_layer

# choose how many neurons in each layer
nHidden1 = 3
nHidden2 = 1

# Creating these dictionaries with init_weights adds their contents to
# the TensorFlow model
weights =  {
    'h1': init_weights( # these weights are really Fourier amplitudes
        [nHidden1, n, 2*n],

        'uniform'),

    'h2': init_weights(
        [nHidden2, nHidden1],
        'xavier',
        xavier_params=(nHidden1, nHidden2)),
     'out': init_weights(
        [1, nHidden2],
        # 'zeros')
        'xavier',
        xavier_params=(nHidden2,1))
}

biases = {
    'b1': init_weights([nHidden1, n, 2*n],'uniform'), # random phases to start
    'b2': init_weights([nHidden2, 1],'zeros'),
    'out': init_weights([1, 1],'zeros')
}

# Telling the model to expect the following as inputs
witness1 = tf.placeholder(tf.float32, [None])
witness2 = tf.placeholder(tf.float32, [None])
target = tf.placeholder(tf.float32, [None])
bandpass = tf.placeholder(tf.float32, [None])

# This tells the model to evaluate the network when its output is needed;
# it doesn't cause the multilayer_perceptron computation to be perfomed
# right away. Instead it gets reevaluated anytime a call to sess.run()
# needs to know the value of latticeEstimate.
latticeEstimate = tf.reshape(
    multilayer_perceptron(weights, biases),
    [lattice_res, lattice_res])

# call function in tf_2dhelpers to interpolate all the (wit1,wit2) points
# and construct a timeseries for the nonlinear witness.
interpolated_wit = extract_wit(latticeEstimate, witness1, witness2, xscale, yscale)

# loss = tf.reduce_mean(tf.abs(target-interpolated_wit), 0)

# The nonlinear witness gets wiener filtered before subtraction
# I copied the brute force version of Q's algorithm into TensorFlow
# operations.

# Calculate wiener filter for subtraction
filt = tf_wiener_fir(target, interpolated_wit)
# apply the filter to the constructed timeseries
filtered_wit = tf_filt(interpolated_wit, filt)
# subtract the filtered wintess from darm
resid = tf.slice(target, [N], [-1]) - filtered_wit

# loss is the mean absolute value of the residual, bandpassed to
# look at relevant freqs (this worked a little better than rms)
loss = tf.reduce_mean(tf.abs(tf_filt(resid, bandpass)), 0)

# anytime you change the loss function, the learning rate needs to change
optimizer = tf.train.AdamOptimizer(learning_rate=1e-1, epsilon=1e-9)
train_step = optimizer.minimize(loss)

# when called with sess.run(), tells Tensorflow to initialize its
# variables with starting values.
init = tf.global_variables_initializer()

def main():
    # Launch the graph. Any computation that uses Tensorflow executes only
    # when called by the session using sess.run()
    #
    # Evaluations that depend on inputs (defined as placeholders above)
    # need to be given a dictionary of input values.

    sess = tf.Session()
    sess.run(init)

    if(animate):
        frames = np.zeros((nIter+1, lattice_res, lattice_res))
    if(print_loss):
        # loss at the beginning, before any optimization has occurred
        best = sess.run(
            loss,
            feed_dict={
                witness1:wit1,
                witness2:wit2,
                target:darm_scaled,
                bandpass:fir_bandpass})
    # perfect subtraction would just leave data, so give it clean data
    # with nothing to subtract to find lowest possible loss value
    goal = sess.run(
        loss,
        feed_dict={
            witness1:np.zeros(wit1.size),
            witness2:np.zeros(wit2.size),
            target:data_scaled,
            bandpass:fir_bandpass})
    print "Loss goal:", goal

    # MAIN LOOP
    for i in xrange(nIter):
        if(animate):
            # save current lattice values for animation
            frames[i,:,:] = latticeEstimate.eval(session=sess)
        # this is where the model gets trained. Each call to train_step
        # updates all the variables once.
        sess.run(
            train_step,
            feed_dict={
                witness1:wit1,
                witness2:wit2,
                target:darm_scaled,
                bandpass:fir_bandpass})
        print "ITERATION:", i
        # It's nice to have loss updates, but this does actually slow it down
        # good to enable if you're testing a changed hyperparameter
        if(print_loss):
            loss_i = sess.run(
                loss,
                feed_dict={
                    witness1:wit1,
                    witness2:wit2,
                    target:darm_scaled,
                    bandpass:fir_bandpass})
            if loss_i < best:
                print loss_i, "**New best**"
                best = loss_i
                # best_lattice = latticeEstimate.eval(session=sess)
            else:
                print loss_i

    # get final model outputs
    end_lattice = sess.run(latticeEstimate) # doesn't need input
    end_loss = sess.run(
        loss,
        feed_dict={
            witness1:wit1,
            witness2:wit2,
            target:darm_scaled,
            bandpass:fir_bandpass})
    end_wit = sess.run(
        interpolated_wit,
        feed_dict={witness1:wit1,
                   witness2:wit2,
                   target:darm_scaled,
                   bandpass:fir_bandpass})

    print "Final loss:", end_loss

    # make 'perfect solution' lattice for comparison
    lattice_real = np.zeros((lattice_res, lattice_res), dtype = np.float32)

    coords = np.linspace(-1+epsilon, 1-epsilon, lattice_res)
    # map lattice coordinates to witness channel values
    xs_grid = np.arctanh(coords)*xscale
    ys_grid = np.arctanh(coords)*yscale
    # calculate the actual nonlinear function value at lattice points
    for i in range (0, lattice_res):
        lattice_real[:,i] = f(xs_grid, ys_grid[i])

    # plot lattice values as images
    # the units are arbitrary
    plt.figure()
    plt.subplot(1,2,1)
    plt.pcolor(np.real(end_lattice))
    plt.subplot(1,2,2)
    plt.pcolor(lattice_real)
    # plt.show()
    plt.savefig("result_tf.png")

    # save data so it can be processed further
    np.save("tfend_darm", darm)
    np.save("tfend_wit1", wit1)
    np.save("tfend_wit2", wit2)
    np.save("tfend_data", data)
    np.save("tfend_wit_result", end_wit)
    np.save("tfend_lattice_result", end_lattice)

    # make animation
    if(animate):
        frames[-1, :, :] = end_lattice
        print 'Animating...'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        fig = plt.figure()
        def plot_frame(frame):
            plt.pcolor(frame)
        anim = animation.FuncAnimation(
            fig, plot_frame, frames, interval=50)
        anim.save('lattice_evolution.mp4', writer=writer)
        print 'Done.'


    # plot ASDs and coherence. This also makes a timeseries plot but it's
    # broken right now
    plot_results(
        darm, wit1, wit2, end_wit, data,
        file_end="_2dft", plot_all=True, filt=True)
if __name__ == "__main__":
    main()

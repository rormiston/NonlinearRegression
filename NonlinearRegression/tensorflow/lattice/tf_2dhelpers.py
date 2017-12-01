import tensorflow as tf
import numpy as np
import scipy.signal as sig
from scipy import interpolate
from mockdata import starting_data
from mockdata.mock_noise import scatfunk as f
from mockdata.mock_noise import lambduh


N = 100 # wiener fir filter order
ind = np.zeros((N+1, N+1), dtype=np.int32)
for i in range(0, N+1):
    ind[i,:] = np.arange(N-i, 2*N+1-i)
indices = tf.reshape(tf.constant(ind), [(N+1),(N+1), 1])

# helper functions for extract_wit: switch between lattice coordinates
# [-1,1] to indices [0, lattice_res)
def coord2ind(x, coords):
    return (x - coords[0])/(coords[1]-coords[0])

def ind2coord(x, coords):
    return tf.cast(x, tf.float32)*(coords[1]-coords[0]) + coords[0]

# get list of lattice values from list of coordinates
def get_values_at_coordinates(input, coords):
    input_as_vector = tf.reshape(input, [-1])
    coords_as_indices = (coords[:, 0] * tf.shape(input)[1]) + coords[:, 1]
    return tf.gather(input_as_vector, coords_as_indices)

def extract_wit(lattice, witness1, witness2, xscale, yscale):
    '''
    Given a lattice and witness timeseries as TensorFlow Tensors, return
    a Tensor of the timeseries found by linearly interpolating the lattice
    at the coordinates given by the witnesses.

    The backpropagation from the final loss to the Fourier coefficients
    passes through this function.
    '''
    coords = tf.linspace(-1.0, 1.0, tf.shape(lattice)[0])

    # convert witnesses to contiuous lattice indices between 0 and lattice_res
    xpoints = coord2ind(tf.tanh(witness1/xscale), coords)
    ypoints = coord2ind(tf.tanh(witness2/yscale), coords)
    C = tf.transpose(tf.reshape(tf.concat(0,[xpoints, ypoints]) , [2, -1]))

    # reference: http://stackoverflow.com/questions/34902782

    # first make arrays of lattice indexes that surround each point
    # so for a point (8.2, 33.5) these would be (8, 33), (8, 34), (9,33)
    # and (9, 34): the indices of the 4 lattice points surrounding each
    # point to interpolate
    UL = tf.cast(tf.floor(C), tf.int32)
    # print sess.run(UL)
    UR = tf.cast(
        tf.concat(1, [tf.floor(C[:, 0:1]), tf.ceil(C[:, 1:2])]), tf.int32)
    LL = tf.cast(
        tf.concat(1, [tf.ceil(C[:, 0:1]), tf.floor(C[:, 1:2])]), tf.int32)
    LR = tf.cast(tf.ceil(C), tf.int32)

    # get lattice values that surround each input point
    UL_vals = get_values_at_coordinates(lattice, UL)
    UR_vals = get_values_at_coordinates(lattice, UR)
    LL_vals = get_values_at_coordinates(lattice, LL)
    LR_vals = get_values_at_coordinates(lattice, LR)

    # find offsets from block corners: Varies between 0.0 and 1.0.
    horizontal_offset = C[:, 0] - tf.cast(UL[:, 0], tf.float32)
    horizontal_interpolated_top = (
        ((1.0 - horizontal_offset) * UL_vals)
        + (horizontal_offset * UR_vals))

    horizontal_interpolated_bottom = (
        ((1.0 - horizontal_offset) * LL_vals)
        + (horizontal_offset * LR_vals))

    vertical_offset = C[:, 1] - tf.cast(UL[:, 1], tf.float32)

    # linearly interpolate value at each point from surrounding values
    interpolated_result = (
        ((1.0 - vertical_offset) * horizontal_interpolated_top)
        + (vertical_offset * horizontal_interpolated_bottom))

    return interpolated_result

# can be used to validate results from extract_wit. Not compatible with
# TensorFlow backpropagation.
def extract_wit_numpy(lattice, wit1, wit2):
    epsilon = 1e-5
    x_ins = np.tanh(wit1/xscale)
    print x_ins[0:10]
    y_ins = np.tanh(wit2/yscale)
    print y_ins[0:10]
    # x_ins = wit1
    # y_ins = wit2
    print "done."
    coords = np.linspace(-1+epsilon, 1-epsilon, lattice.shape[0])

    f_guess = interpolate.RectBivariateSpline(coords, coords, lattice)
    print "interp done"

    wit_guess = f_guess(x_ins, y_ins, grid=False)
    print wit_guess[0:10]
    return wit_guess

def tf_wiener_fir(tar, wit):
    '''
    Single witness, backprop compatible verison of noisesub.wiener_fir
    Returns fir wiener subtraction filter to apply to the witness.
    The backpropagation from the final loss to the Fourier coefficients
    passes through this function.
    '''
    # there's only one target and one witness aka M=1

    # get cross-correlation between target and witness
    # P is a 1-d tensor, each element is the correlation with a different
    # time delay
    # we only care about the half where the witness precedes the target,
    # hence the slice operation
    P = tf.slice(tf_xcorr(tar, wit, N), [N,0], [N+1,1])

    # m = ii = 0
    # get the witness's autocorrelation
    r = tf_xcorr(wit, wit, N)

    # This call to gather generates a warning about converting sparse to dense.
    # I rewrote this part to use slicing and reshaping instead, but it took
    # just as long and was hard to understand, so this is the best I can do for
    # now.
    # This step makes a Toeplitz matrix out of the autocorrelation values,
    # by repeatedly pulling the same element of r out, specified by the
    # indices matrix defined above. There should be a more efficient way
    # to do this...
    R = tf.gather(r, indices)

    # Solve the matrix equation to find the filter coefficients.
    # As far as I know this is the speed limiting step- maybe try to replicate
    # block_levinson instead
    W = tf.matrix_solve(tf.reshape(R, [N+1,N+1]), P)
    return W

def tf_xcorr(x, y, n):
    '''
    Use TensorFlow's built in convolution to find cross-correlation between
    inputs.
    '''
    paddings = tf.constant([[n, n]])
    x_in = tf.reshape(tf.pad(x, paddings), [ 1, -1, 1])
    y_in = tf.reshape(y, [-1, 1, 1])

    return tf.reshape(tf.nn.conv1d(x_in, y_in, 1, padding='VALID'), [-1,1])

def tf_filt(wit, filt):
    '''
    Apply a 1-dimensional fir filter. Since valid padding is chosen, the
    output will be shorter than the input. 
    '''
    wit_in = tf.reshape(wit, [1, -1, 1])
    filt_in = tf.image.flip_up_down(tf.reshape(filt, [-1, 1, 1]))

    return tf.reshape(tf.nn.conv1d(wit_in, filt_in, 1, padding='VALID'), [-1])

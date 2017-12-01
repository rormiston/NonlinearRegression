import tensorflow as tf
import numpy as np

DARM = np.load("STFT_darm.npy").T
DARM /= np.min(DARM)
y1 = np.load("STFT_wit1.npy").T
#y2 = np.load("y2.npy")
lenFFT = DARM.shape[1]
lenT = DARM.shape[0]

def init_weights(shape, init_method='xavier', xavier_params = (None, None)):

    if init_method == 'zeros':
        return tf.complex(tf.Variable(tf.zeros(shape, dtype=tf.float32)),
                          tf.Variable(tf.zeros(shape, dtype=tf.float32)))

    elif init_method == 'uniform':
        return tf.complex(tf.Variable(tf.random_normal(shape, stddev=0.01,
                                                       dtype=tf.float32)),
                          tf.Variable(tf.random_normal(shape, stddev=0.01,
                                                       dtype=tf.float32)))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {tanh:4, sigmoid:1}
        high = 8*np.sqrt(6.0/(fan_in + fan_out))
        return tf.complex(tf.Variable(tf.random_uniform(shape, minval=low,
                                                        maxval=high)),
                          tf.Variable(tf.random_uniform(shape, minval=low,
                                                        maxval=high)))

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with tanh activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Hidden layer with tanh activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    # Hidden layer with tanh activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)
    # Hidden layer with tanh activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.tanh(layer_4)
    # Hidden layer with tanh activation
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.tanh(layer_5)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])
    return out_layer

nHidden1 = 128
nHidden2 = 64
nHidden3 = 32
nHidden4 = 16
nHidden5 = 8
# Store layers weight & bias
weights = {
    'h1': init_weights(
        [lenFFT, nHidden1],
        'xavier',
        xavier_params=(lenFFT, nHidden1)),
    'h2': init_weights(
        [nHidden1, nHidden2],
        'xavier',
        xavier_params=(nHidden1, nHidden2)),
    'h3': init_weights(
        [nHidden2, nHidden3],
        'xavier',
        xavier_params=(nHidden2, nHidden3)),
     'h4': init_weights(
        [nHidden3, nHidden4],
        'xavier',
        xavier_params=(nHidden3, nHidden4)),
     'h5': init_weights(
        [nHidden4, nHidden5],
        'xavier',
        xavier_params=(nHidden4, nHidden5)),
    'out': init_weights(
        [nHidden4, lenFFT],
        'xavier',
        xavier_params=(nHidden5, lenFFT))
}


biases = {
    'b1': init_weights([1,nHidden1],'zeros'),
    'b2': init_weights([1,nHidden2],'zeros'),
    'b3': init_weights([1,nHidden3],'zeros'),
    'b4': init_weights([1,nHidden4],'zeros'),
    'b5': init_weights([1,nHidden5], 'zeros'),
    'out': init_weights([1,lenFFT],'zeros')
}

witness = tf.placeholder(tf.complex64, [None, lenFFT])

nonlinearWitness = multilayer_perceptron(witness, weights, biases)

y_ = tf.placeholder(tf.complex64, [None, lenFFT])

################################################################################################################################################

DARM_minus_NLN = tf.sub( y_, nonlinearWitness )

# produces a frequency vector which is averaged over time
#  DARM_minus_NLN_time_average = tf.reduce_mean( tf.abs(DARM_minus_NLN), 0 ) # produces a frequency vector which is averaged over time
DARM_minus_NLN_time_average = tf.sqrt( tf.reduce_mean( tf.square(
                                tf.abs(DARM_minus_NLN)), 0))

# Minimize

NLW_NLW = tf.abs(tf.reduce_sum(tf.mul(tf.conj(nonlinearWitness),
                                      nonlinearWitness), reduction_indices=0))

DARM_DARM = tf.abs(tf.reduce_sum(tf.mul(tf.conj(y_), y_), reduction_indices=0))

NLW_DARM = tf.square(tf.abs(tf.reduce_sum(tf.mul(tf.conj(nonlinearWitness),
                                                 y_), reduction_indices=0)))

C = tf.div( NLW_DARM, tf.mul(NLW_NLW, DARM_DARM ) )

loss = 1.0 - (1./lenFFT)*tf.reduce_sum(C)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
accuracy = 1.0 - (1./lenFFT)*tf.reduce_sum(C)
nIter = 50

sess = tf.Session()
sess.run(init)
for i in xrange(nIter):
        sess.run(train_step, feed_dict={witness: y1, y_: DARM})
        print"%e\n"%(sess.run(accuracy, feed_dict={witness: y1, y_: DARM}))

import pylab as plt
a = nonlinearWitness.eval(session=sess, feed_dict={witness:y1})
plt.loglog(np.sqrt(np.mean(np.abs(DARM)**2, 0)), label='DARM')
plt.loglog(np.sqrt(np.mean(np.abs(a)**2, 0)), label='NLW')
plt.legend()
plt.savefig("cleanedDARM.png")
plt.clf()
np.save("NLW_estimate", a)

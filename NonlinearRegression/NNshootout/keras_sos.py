from __future__ import division
import numpy as np
from keras import backend as K
from keras import activations, initializations
from keras.models import Sequential
from keras.engine import InputSpec
from keras.layers.recurrent import Recurrent


class SOS_RNN(Recurrent):
    def __init__(self, init='normal', activation='linear', sos=None, **kwargs):

        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        #  if sos is None:
        #      self.sos=np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
        #  else:
        #      self.sos = np.asarray(sos)

        kwargs['stateful'] = True
        kwargs['unroll'] = True
        super(SOS_RNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]
        self.output_dim = self.input_dim

        self.reset_states()

        self.W_gain = self.add_weight(
            shape=(1, self.output_dim),
            initializer=self.init,
            name='{}_gain'.format(self.name))

        self.W_a1 = self.add_weight(
            shape=(1, self.output_dim),
            initializer=self.init,
            name='{}_a1'.format(self.name))

        self.W_a2 = self.add_weight(
            shape=(1, self.output_dim),
            initializer=self.init,
            name='{}_a2'.format(self.name))

        self.W_b0 = self.add_weight(
            shape=(1, self.input_dim),
            initializer=self.init,
            name='{}_b0'.format(self.name))

        self.W_b1 = self.add_weight(
            shape=(1, self.input_dim),
            initializer=self.init,
            name='{}_b1'.format(self.name))

        self.W_b2 = self.add_weight(
            shape=(1, self.input_dim),
            initializer=self.init,
            name='{}_b2'.format(self.name))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def reset_states(self):
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('Input shape needs to be completely specified')
        input_dim = input_shape[2]

        self.states = [K.zeros((1, input_dim)) for _ in range(4)]

    def step(self, x, states):

        x_m0 = x
        x_m1 = states[2]
        x_m2 = states[3]

        y_m1 = states[0]
        y_m2 = states[1]

        #  output = x_m0 * self.sos[0]
        #  output += x_m1 * self.sos[1]
        #  output += x_m2 * self.sos[2]
        #  output -= y_m1 * self.sos[-2]
        #  output -= y_m2 * self.sos[-1]

        output = x_m0 * self.W_b0
        output += x_m1 * self.W_b1
        output += x_m2 * self.W_b2
        output -= y_m1 * self.W_a1
        output -= y_m2 * self.W_a2
        output *= self.W_gain  # Gain = 1/a0

        return output, [output, y_m1, x_m0, x_m1]


if __name__ == '__main__':
    model = Sequential()
    model.add(SOS_RNN(batch_input_shape=(1, 1, 1)))
    model.compile(optimizer='rmsprop', loss='mse')

    import scipy.signal as sig
    x = np.random.randn(1000, 1)
    xt = x.reshape(-1, 1, 1)
    sos = sig.butter(2, 0.1, output='sos')
    y = sig.sosfilt(sos, x[:, 0])

    # Why doesn't this converge?
    nb_epoch = 5
    for _ in range(nb_epoch):
        model.fit(xt, y, nb_epoch=1, batch_size=1, shuffle=False)
        model.reset_states()

    pred = model.predict(xt, batch_size=1)[:, 0]

    import matplotlib.pyplot as plt
    plt.plot(y, label='target')
    plt.plot(pred, label='prediction')
    plt.title('SOS Keras layer test')
    plt.legend()
    plt.show()

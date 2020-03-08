################################
# KERAS CUSTOM LAYER COMPONENT #
################################

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda, Concatenate, Activation, Dot, RepeatVector
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Model


class BaselineASR(Model):
    def __init__(self, filters, kernel_size, strides, padding, n_lstm_units, n_dense_units):
        super(BaselineASR, self).__init__()
        self.conv1d        = Conv1D(filters,
                                    kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    activation='relu')
        self.lstm_forward  = LSTM(n_lstm_units,
                                  return_sequences=True,
                                  activation='tanh')
        self.lstm_backward = LSTM(n_lstm_units,
                                  return_sequences=True,
                                  go_backwards=True,
                                  activation='tanh')
        self.bilstm        = Bidirectional(self.lstm_forward, backward_layer=self.lstm_backward)
        self.dense         = Dense(n_dense_units)

    def call(self, x):
        x = self.conv1d(x)
        x = self.bilstm(x)
        x = self.dense(x)
        return x


class BasicASREncoder(Model):
    def __init__(self, vocab_len, n_lstm, batch_size):
        super(BasicASREncoder, self).__init__()
        self.vocab_len = vocab_len
        self.n_lstm = n_lstm
        self.batch_size = batch_size

        self.bilstm = Bidirectional(LSTM(self.n_lstm,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='orthogonal'))  # or 'glorot_uniform'

    def call(self, X, hidden):
        """
        Parameters
        ----------
        X : shaoe=(m, Tx, n_mfcc)
            audio sequence input

        Returns
        -------
        output : shape=(m, Tx, n_lstm)
            output from lSTM
        state : shape=(m, n_lstm)
            final LSTM state
        """
        output, *state = self.bilstm(X, initial_state=hidden)
        return output, state  # shape=(batch_size, 4*n_lstm)


class BahdanauAttention(tf.keras.Model):
    def __init__(self, n_dense):
        super(BahdanauAttention, self).__init__()
        self.n_dense = n_dense
        self.W1 = Dense(n_dense)  # no activation g(x) = x
        self.W2 = Dense(n_dense)  # no activation g(x) = x
        self.V = Dense(1)  # no activation g(x) = x

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class PStacker(Layer):
    def __init__(self):
        super(PStacker, self).__init__()
        self.even = Lambda(lambda x: x[:, 0::2, :])
        self.odd = Lambda(lambda x: x[:, 1::2, :])
        self.concat = Concatenate(axis=-1)

    def call(self, inputs):
        even_sequence = self.even(inputs)
        odd_sequence = self.odd(inputs)
        outputs = self.concat([even_sequence, odd_sequence])
        return outputs


class MLPAttention(Layer):
    def __init__(self, n_dense):
        super(MLPAttention, self).__init__()
        self.densor_1 = Dense(n_dense, )
        self.densor_2 = Dense(n_dense//2)
        self.densor_3 = Dense(n_dense//4)
        self.densor_4 = Dense(1, activation='relu')

    def call(self, X):
        return self.densor_4(self.densor_3(self.densor_2(self.densor_1(X))))


class MLPOutput(Layer):
    def __init__(self, n_dense):
        super(MLPOutput, self).__init__()
        self.densor_1 = Dense(n_dense, activation='relu')
        self.densor_2 = Dense(n_dense//2, activation='relu')

    def call(self, X):
        return self.densor_2(self.densor_1(X))


class LuongAttention(Model):
    def __init__(self, n_dense):
        super(LuongAttention, self).__init__()
        self.concatenator = Concatenate(axis=-1)
        self.densor = MLPAttention(n_dense)
        self.activator = Activation('softmax', name='attention_weights')
        self.dotor = Dot(axes=1)

    def call(self, inputs):
        encoder_outputs, *decoder_prev_states = inputs
        Tx = K.int_shape(encoder_outputs)[1]

        decoder_prev_states = self.concatenator(decoder_prev_states)
        decoder_prev_states = RepeatVector(Tx)(decoder_prev_states)
        concat = self.concatenator([encoder_outputs, decoder_prev_states])

        e = self.densor(concat)
        alphas = self.activator(e)
        context_vector = self.dotor([alphas, encoder_outputs])

        return context_vector, alphas


class DecoderLSTM(Model):
    def __init__(self, n_lstm, n_dense, vocab_len):
        super(DecoderLSTM, self).__init__()
        self.lstm_1 = LSTM(n_lstm, return_sequences=True, return_state=True)
        self.lstm_2 = LSTM(n_lstm, return_sequences=True, return_state=True)
        self.mlp = MLPOutput(n_dense)
        self.dense = Dense(vocab_len, activation='softmax')

    def call(self, inputs):
        context_vector, *initial_states = inputs

        lstm_1_output, *lstm_1_states = self.lstm_1(context_vector, initial_state=initial_states[0:2])
        lstm_2_output, *lstm_2_states = self.lstm_2(lstm_1_output, initial_state=initial_states[2:4])
        outputs = self.mlp(lstm_2_output)
        outputs = self.dense(outputs)

        return outputs, [*lstm_1_states, *lstm_2_states]


class EncoderLSTM(Model):
    def __init__(self, n_lstm):
        super(EncoderLSTM, self).__init__()
        self.pstack = PStacker()
        self.encoder_1 = Bidirectional(LSTM(n_lstm//4, return_sequences=True))
        self.encoder_2 = Bidirectional(LSTM(n_lstm//2, return_sequences=True))
        self.encoder_3 = Bidirectional(LSTM(n_lstm, return_sequences=True, return_state=True))

    def call(self, inputs):
        stack_1 = self.pstack(self.encoder_1(inputs))
        stack_2 = self.pstack(self.encoder_2(stack_1))
        encoder_outputs, *encoder_states = self.encoder_3(stack_2)

        return encoder_outputs, encoder_states

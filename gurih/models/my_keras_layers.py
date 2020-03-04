################################
# KERAS CUSTOM LAYER COMPONENT #
################################

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dense


class BaselineASR(tf.keras.Model):
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


class ASREncoder(tf.keras.Model):
    def __init__(self, vocab_len, n_lstm, batch_size):
        super(ASREncoder, self).__init__()
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

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_lstm)

    def initialize_hidden_state(self):
        return [tf.zeros((None, self.n_lstm)) for i in range(4)]


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


class ASRDecoder(tf.keras.Model):
    def __init__(self, vocab_len, n_lstm, n_units, batch_size):
        super(ASRDecoder, self).__init__()
        self.vocab_len = vocab_len
        self.n_lstm = n_lstm
        self.n_units = n_units
        self.batch_size = batch_size

        self.lstm = LSTM(self.n_lstm,
                         return_sequences=True,
                         return_state=True,
                         recurrent_initializer='orthogonal')  # or 'glorot_uniform'
        self.fc = Dense(self.vocab_len)
        self.attention = BahdanauAttention(self.n_units)

    def call(self, x, hidden, enc_output):
        """
        Parameters
        ----------

        """
        hidden = tf.concat(hidden, axis=1)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = tf.expand_dims(x, axis=-1)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, *state = self.lstm(x)

        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, tf.concat(state, axis=1), attention_weights

    # def call(self, X, hidden, enc_output):
    #     dec_hidden = tf.concat(hidden, axis=1)
    #     outputs = []

    #     for t in range(0, X.shape[1]):
    #         x = tf.expand_dims(X[:, t], axis=-1)
    #         x = tf.expand_dims(x, axis=-1)

    #         context_vector, attention_weights = self.attention(dec_hidden, enc_output)
    #         x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    #         output, *state = self.lstm(x)

    #         output = tf.reshape(output, (-1, output.shape[2]))
    #         x = self.fc(output)
    #         x = tf.expand_dims(x, axis=1)
    #         outputs.append(x)
    #         dec_hidden = tf.concat(state, axis=1)

    #     return tf.concat(outputs, axis=1), dec_hidden, attention_weights

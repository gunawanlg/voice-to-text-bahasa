################################
# KERAS CUSTOM LAYER COMPONENT #
################################

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, Dense


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

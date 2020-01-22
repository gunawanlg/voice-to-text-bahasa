"""
Defines abstract class for model.

All script in this file is expected to be run from /project/notebooks/modelling directory.

See notebook for example usage.
"""
from math import ceil

from tensorflow.keras import Model
import tensorflow.keras.models
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Lambda, Dense, Dropout, LSTM, Activation, Masking, Conv1D, Bidirectional, TimeDistributed
# from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt

class _BaseModel:
    """
    Abstract class wrapper for tf.keras.Model. Defines how to:
        - save() model.json and weights.h5
        - load() model.json and weights.h5
        - save and plot history during training
    
    It also defines default dir_path and doc_path if this class was to be used in directory
    /notebooks/modelling/
    """
    def __init__(self, X_train, y_train, X_val=None, y_val=None, dir_path="../../models/", doc_path="../../docs/"):
        self._X_train = X_train
        self._X_val = X_val
        self._y_train = y_train
        self._y_val = y_val
        self._dir_path = dir_path
        self._doc_path = doc_path
        self.model = None
        self.history = None
        self.fig = None

        self._show_summary()

    def compile(self):
        """Model.compile"""
        raise NotImplementedError

    def fit(self):
        """Model.fit"""
        raise NotImplementedError

    def evaluate(self):
        """Model.evaluate"""
        raise NotImplementedError

    def predict(self):
        """Model.predict"""
        raise NotImplementedError

    def save(self):
        """Save model in .json and weights in .h5 format"""
        if self.model is None: 
            raise Exception("Model not created and trained")

        model_json = self.model.to_json()
        with open(self._dir_path + self.model.name + ".json", "w") as f:
            f.write(model_json)
        print("Model " + self.model.name + " saved at: " + self._dir_path + " " + self.model.name + ".json")
        self.model.save_weights(self._dir_path + self.model.name + ".h5")
        print("Weights serialized at: " + self._dir_path + self.model.name + ".h5")
        
    def load(self):
        """Load model from .json and weights from .h5"""
        json_file = open(self._dir_path + self.model.name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self._dir_path + self.model.name + ".h5")
        print("Loaded model " + self.model.name + " from disk")

    def plot_history(self):
        """Plot train/val loss figure created from _save_history()"""
        if self.fig is None:
            raise AttributeError("Model is not fitted. No history figure found. Call fit() method first")
        plt.show(self.fig)

    def _show_summary(self):
        """Show summary of the model after calling __init__"""
        print("Model directory is set to " + self._dir_path)
        print("Documentation directory is set to " + self._doc_path)

    def _save_history_figure(self):
        """Create train/val loss figure from history"""
        if self.history is None: 
            raise AttributeError("Model is not fitted. No history found. Call fit() method first.")

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(self.history.history['loss'])
        ax.plot(self.history.history['val_loss'])
        ax.set_title('model loss')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(['train', 'val'], loc='upper left')
        plt.savefig(self._doc_path+self.model.name+'.png', dpi=300)

        self.fig = fig

    @property
    def dir_path(self):
        return self._dir_path

    @dir_path.setter
    def dir_path(self, new_val):
        if (new_val == self._dir_path):
            print("Path already set to " + self._dir_path)
        else:
            self._dir_path = new_val
            print("Model directory is set to " + self._dir_path)

    @property
    def doc_path(self):
        return self._doc_path

    @doc_path.setter
    def doc_path(self, new_val):
        if (new_val == self._doc_path):
            print("Path already set to " + self._doc_path)
        else:
            self._doc_path = new_val
            print("Model directory is set to " + self._doc_path)

class BaselineASRModel(_BaseModel):
    """
    BaselineASRModel(_BaseModel)

    Baseline Automatic Speech Recognition (ASR) model using CTC Loss.
    Architecture:
        Conv1D --> Bidrectional(LSTM) --> Dense(vocab_len)

    Example
    -------
    >>> import string
    >>> import numpy
    >>> from sklearn.model_selection import train_test_split
    >>> vocab = set(string.ascii_lowercase)
    >>> vocab |= {' ', '>', '%'} # space_token, end_token, blank_token
    >>> vocab_index = list(range(len(vocab)))
    >>> X = np.array([[5 for x in range(1000)] for y in range(300)])
    >>> X = np.expand_dims(X, axis=0)
    >>> X.shape
    (1, 300, 1000)
    >>> y = np.random.choice(vocab_index, 100)
    >>> y = np.expand_dims(y, axis=0)
    >>> y.shape
    (1, 100)
    >>> X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.4, random_state=42, stratify=Y)
    >>> BaselineASR = BaselineASRModel(X_train, X_val, y_train, y_val)
    >>> BaselineASR.create()
    >>> BaselineASR.compile()
    >>> BaselineASR.fit(epochs=1, batch_size=32) # history is now callable
    >>> BaselineASR.plot_history()
    """
    def __init__(self, X_train, y_train, X_val=None, y_val=None, sample_rate=16000, lang='ind'):
        super().__init__(X_train, y_train, X_val, y_val)
        self._input_length = X_train.shape[1]
        self._sample_rate = sample_rate
        self._lang = lang
        self._filters      = None
        self._kernel_size  = None
        self._strides      = None
        self._padding      = None
        self._n_lstm_units = None
        self.vocab_len     = None

    def create(self, filters=200, kernel_size=11, strides=2, padding='valid', n_lstm_units=200, vocab_len=29):
        self._filters      = filters
        self._kernel_size  = kernel_size
        self._strides      = strides
        self._padding      = padding
        self._n_lstm_units = n_lstm_units
        self.vocab_len     = vocab_len

        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args

            return ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = Input(shape=[200],dtype='float32',name="CTC_Label")
        input_length = Input(shape=[1],dtype='int32',name="input_length")
        label_length = Input(shape=[1],dtype='int32',name="label_length")

        # self.model = Sequential()
        # self.model.add(Input(shape=self._X_train.shape[1:]))
        # self.model.add(Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='relu'))
        # self.model.add(Bidirectional(LSTM(n_lstm_units, return_sequences=True, activation='tanh')))
        # self.model.add(Dense(vocab_len))
        
        # self.model.summary()

        input_in = Input(shape=self._X_train.shape[1:])
        conv1D = Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='relu')(input_in)
        biLSTM = Bidirectional(LSTM(n_lstm_units, return_sequences=True, activation='tanh'))(conv1D)
        y_pred = Dense(vocab_len)(biLSTM)
        
        # Print model summary
        tensorflow.keras.models.Model(inputs=input_in, outputs=y_pred).summary()

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        self.model = tensorflow.keras.models.Model(inputs=[input_in, labels, input_length, label_length], outputs=loss_out)
        self.model._name = '_'.join(["BaselineASR", self._lang+'-'+str(self._sample_rate),
                             'f'+str(filters), 
                             'k'+str(kernel_size), 
                             's'+str(strides), 
                             'p'+padding, 
                             'nlstm'+str(n_lstm_units), 
                             'ndense'+str(vocab_len)])
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    def compile(self):
        # self.model.compile(loss=_ctc_loss_func, optimizer="adam")
        pass

    def fit(self, epochs=100, batch_size=32):
        checkpoint_path = self._dir_path+self.model.name+'.h5'

        if (self._X_val == None) and (self._y_val == None):
             self.history = self.model.fit(self._X_train,
                                      self._y_train, 
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      shuffle=True)
        else:
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                    monitor='val_loss',
                                    save_best_only=True,
                                    mode='min',
                                    save_weights_only=False)

            es_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=1e-4,
                                        patience=2,
                                        mode='min')

            self.history = self.model.fit(self._X_train,
                                        self._y_train, 
                                        validation_data=(self._X_val, self._y_val),
                                        callbacks=[cp_callback, es_callback],
                                        epochs=epochs, 
                                        batch_size=batch_size, 
                                        shuffle=True)
        
        self._save_history_figure()

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred


####################################
### KERAS CUSTOM LAYER COMPONENT ###
####################################

class BaselineASR(Model):
    def __init__(self, filters, kernel_size, strides, padding, n_lstm_units, n_dense_units):
        super(BaselineASR, self).__init__()
        self.conv1d        = Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='relu')
        self.lstm_forward  = LSTM(n_lstm_units, return_sequences=True, activation='tanh')
        self.lstm_backward = LSTM(n_lstm_units, return_sequences=True, go_backwards=True, activation='tanh')
        self.bilstm        = Bidirectional(self.lstm_forward, backward_layer=self.lstm_backward)
        self.dense         = Dense(n_dense_units)

    def call(self, x):
        x = self.conv1d(x)
        x = self.bilstm(x)
        x = self.dense(x)

        return x

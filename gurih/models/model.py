"""
Defines abstract class for model.

All script in this file is expected to be run from /project/notebooks/modelling directory.

See notebook for example usage.
"""
from math import ceil

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Lambda, Dense, Dropout, LSTM, Activation 
from tensorflow.keras.layers import Masking, Conv1D, Bidirectional, TimeDistributed
# from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt

# from .my_keras_layers import BaselineASR


class _BaseModel:
    """
    Abstract class wrapper for tf.keras.Model. Defines how to:
        - save() model.json and weights.h5
        - load() model.json and weights.h5
        - save and plot history during training
    
    It also defines default dir_path and doc_path if this class was to be used in directory /notebooks/modelling/

    Available convenience function for subclassing:
        _create_callbacks : create basic ModelCheckpoint and EarlyStoppping callback
            the callbacks created will be saved in object.
        _fit : keras model fit method
            defines basic fit method, with ModelCheckpoint and EarlyStopping callbacks if validation data is provided 
            and _create_callbacks is called.

    Parameters
    ----------
        dir_path : str
            directory to store model checkpoint
        doc_path : str
            directory to store model training documentation

    Attributes
    ----------
        model : tensorflow.keras.models.Model
            instance of model, defined in _create()
        history : keras History object
            history data during model training using fit() or fit_generator()
        fig : matplotlib.figure.Figure
            plot of loss and other metrics during training
        callbacks : list of keras.callbacks.callbacks.Callback
            callbacks defined in _callbacks(), if not implemented, will not store either checkpoint or training metrics
    """
    def __init__(self, dir_path="../../models/", doc_path="../../docs/"):
        self._dir_path = dir_path
        self._doc_path = doc_path
        self.model = None
        self.history = None
        self.fig = None
        self.callbacks = None
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
        
        print(f"Model {self.model.name} saved at: {self._dir_path}{self.model.name}.json")
        self.model.save_weights(self._dir_path + self.model.name + ".h5")
        print(f"Weights serialized at: {self._dir_path}{self.model.name}.h5")
        
    def load(self):
        """Load model from .json and weights from .h5"""
        json_file = open(self._dir_path + self.model.name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self._dir_path + self.model.name + ".h5")
        print(f"Loaded model {self.model.name} from disk.")

    def plot_history(self):
        """Plot train/val loss figure created from _save_history()"""
        if self.fig is None:
            raise AttributeError("Model is not fitted. No history figure found. Call fit() method first")
        return self.fig

    def _fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, min_delta=1e-4, patience=2):
        """
        Base fit model for easier implementation of child instance fit() method.

        Parameters
        ----------
        X_train : np.ndarray[shape=(num_training_examples, *dims)]
            input data
        y_train : np.ndarray[shape=(num_training_examples, num_classes)]
            label data
        X_val : np.ndarray[shape=(num_training_examples, *dims)]
            input validation data, will do early stopping if provided
        y_val : np.ndarray[shape=(num_training_examples, num_classes)]
            label validation data, will do early stopping if provided
        epoch : int,
            number of epoch to run
        batch_size : int,
            batch size for training, usually in the power of two
        """
        if (X_val == None) and (y_val == None):
            self.history = self.model.fit(X_train,
                                          y_train, 
                                          epochs=epochs, 
                                          batch_size=batch_size, 
                                          shuffle=True)
        else:
            self.history = self.model.fit(X_train,
                                          y_train, 
                                          validation_data=(X_val, y_val),
                                          callbacks=self.callbacks,
                                          epochs=epochs, 
                                          batch_size=batch_size, 
                                          shuffle=True)
        
        self._save_history_figure()

    def _callbacks(self, min_delta=1e-4, patience=2):
        """
        Basic keras fit callbacks.

        Save best model and its weights according to best validation loss, while also performs
        early stopping when it starts to overfit.
        """
        checkpoint_path = self._dir_path+self.model.name+'.h5'
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='min',
                                          save_weights_only=False)

        es_callback = EarlyStopping(monitor='val_loss',
                                    min_delta=1e-4,
                                    patience=2,
                                    mode='min')

        self.callbacks = [cp_callback, es_callback]
        
    def _show_summary(self):
        """Show summary of the model after calling __init__"""
        print(f"Model directory is set to {self._dir_path}")
        print(f"Documentation directory is set to {self._doc_path}")
        print()

    def _save_history_figure(self):
        """Create train/val loss figure from history"""
        if self.history is None: 
            raise AttributeError("Model is not fitted. No history found. Call fit() method first.")
        
        history_dict = self.history.history

        # No validation training
        if 'val_loss' in history_dict:
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(self.history.history['loss'])
            ax.plot(self.history.history['val_loss'])
            ax.set_title('model loss')
            ax.set_ylabel('loss')
            ax.set_xlabel('epoch')
            ax.legend(['train', 'val'], loc='upper left')
            plt.savefig(self._doc_path+self.model.name+'.png', dpi=300)
        # With validation training
        else:
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(self.history.history['loss'])
            ax.set_title('model loss')
            ax.set_ylabel('loss')
            ax.set_xlabel('epoch')
            ax.legend(['train'], loc='upper left')
            plt.savefig(self._doc_path+self.model.name+'.png', dpi=300)

        self.fig = fig
        plt.close(fig)

    @property
    def dir_path(self):
        return self._dir_path

    @dir_path.setter
    def dir_path(self, new_val):
        if (new_val == self._dir_path):
            print("Model directory already set to " + self._dir_path)
        else:
            self._dir_path = new_val
            print("Model directory is set to " + self._dir_path)

    @property
    def doc_path(self):
        return self._doc_path

    @doc_path.setter
    def doc_path(self, new_val):
        if (new_val == self._doc_path):
            print("Documentation directory already set to " + self._doc_path)
        else:
            self._doc_path = new_val
            print("Documentation directory is set to " + self._doc_path)

class BaselineASRModel(_BaseModel):
    """
    BaselineASRModel(_BaseModel)

    Baseline Automatic Speech Recognition (ASR) model using CTC Loss.
    Architecture:
        Conv1D --> Bidrectional(LSTM) --> Dense(vocab_len+1) 
        +1 in Dense layer is for blank token needed in CTC.

    Example
    -------
        See 3.0-glg-baseline-model.ipynb for detailed example
    """
    def __init__(self, input_shape, vocab_len, filters=200, kernel_size=11, strides=2, padding='valid', 
                 n_lstm_units=200, **kwargs):
        super().__init__(**kwargs)
        self.vocab_len     = vocab_len
        self.input_shape   = input_shape
        self._filters      = filters
        self._kernel_size  = kernel_size
        self._strides      = strides
        self._padding      = padding
        self._n_lstm_units = n_lstm_units

        self._create()
        
    def _create(self):
        """Create the baseline ASR with CTC Model"""
        def _ctc_lambda_func(args):
            """Lambda function to calculate CTC loss in keras"""
            y_pred, labels, input_length, label_length = args
            # y_pred = y_pred[:, 2:, :]
            return ctc_batch_cost(labels, y_pred, input_length, label_length)

        # Calculate output shape as len of vocab +1 for CTC blank token
        output_shape = self.vocab_len + 1

        input_in = Input(shape=self.input_shape, name="the_input")
        mask     = Masking(mask_value=0, name="masking")(input_in)
        conv1D   = Conv1D(self._filters, 
                          self._kernel_size, 
                          strides=self._strides, 
                          padding=self._padding, 
                          activation='relu', 
                          name="conv1")(mask)
        biLSTM   = Bidirectional(LSTM(self._n_lstm_units, 
                                      return_sequences=True, 
                                      activation='tanh'), 
                                 name="bidirectional")(conv1D)
        y_pred   = TimeDistributed(Dense(output_shape, activation='softmax'), name="the_output")(biLSTM)

        labels       = Input(shape=[None],dtype='float32',name="the_labels")
        input_length = Input(shape=[1],dtype='int32',name="input_length")
        label_length = Input(shape=[1],dtype='int32',name="label_length")
        loss_out     = Lambda(_ctc_lambda_func, 
                              output_shape=(1,), 
                              name='ctc')([y_pred, labels, input_length, label_length])

        self.model = Model(inputs=[input_in, labels, input_length, label_length], outputs=[loss_out])
        self.model._name = '_'.join(["BaselineASR",
                                     'f'+str(self._filters), 
                                     'k'+str(self._kernel_size), 
                                     's'+str(self._strides), 
                                     'p'+self._padding, 
                                     'nlstm'+str(self._n_lstm_units), 
                                     'ndense'+str(self.vocab_len)])

        # See the model summary before calculating custom CTC loss
        # for clarity of the architecture of the model
        tmp_model = Model(inputs=input_in, outputs=y_pred)
        tmp_model._name = '_'.join(["BaselineASR",
                                    'f'+str(self._filters), 
                                    'k'+str(self._kernel_size), 
                                    's'+str(self._strides), 
                                    'p'+self._padding, 
                                    'nlstm'+str(self._n_lstm_units), 
                                    'ndense'+str(self.vocab_len)])
        tmp_model.summary()

    def compile(self, optimizer='adam', **kwargs):
        """
        Compile the model, a.k.a building the graph.

        Parameters
        ----------
        optimizer : str, optional
            optimizer string or class from keras.optimizers, defaulting to:
                Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        """
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Fit the model.
        
        TODO: currently not implemented. use fit_generator instead.
        """
        # self._fit(*args, **kwargs)
        raise AttributeError("Method fit() is not implemented. Please use fit_generator() instead.")

    def fit_generator(self, train_generator, validation_generator=None, epochs=1, **kwargs):
        """
        Fit the model using generator. 
        
        This fit method prefer explicit input on training and validation data, rather than using validation_split. It is 
        best to first create your data, for example, by using train_test_split method from scikit-learn.
        
        Parameters
        ----------
        train_generator : keras.utils.Sequence
            yield (inputs, outputs) where:
            inputs = {
                'the_input':     np.ndarray[shape=(batch_size, max_seq_length, mfcc_features)]: input audio data
                'the_labels':    np.ndarray[shape=(batch_size, max_transcript_length)]: transcription data
                'input_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of frames) in output layer
                'label_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of letters) in y
            },
            outputs = {
                'ctc':           np.ndarray[shape=(batch_size, 1)]: dummy data for dummy loss function
            }
        validation_generator : keras.utils.Sequence, optional
            validation data generator, yield same inputs, outputs shape as train_generator
        epochs : int
            number of iteration throughout the whole datasetd
        """
        self.history = self.model.fit_generator(generator=train_generator, 
                                 validation_data=validation_generator,
                                 epochs=epochs,
                                 callbacks=self.callbacks,
                                 **kwargs)
        self._save_history_figure()

    def evaluate(self, X_test, y_test):
        raise NotImplementedError("evaluate() method not implemented for this model.")

    def predict(self, X_test):
        """
        Compute CTC Matrix of input.

        Parameters
        ----------
        X_test : np.array[shape=(m, max_seq_length, features)]
            m examples mfcc input of audio data 
        
        Return
        ------
        ctc_matrix : np.array[shape=(m, ctc_input_length, features)]
            ctc_matrix output of X_test
        """
        input_data = self.model.get_layer('the_input').input
        y_pred = self.model.get_layer('ctc').input[0]
        pred_func = K.function([input_data], [y_pred])

        ctc_matrix = pred_func(X_test)

        return ctc_matrix[0] # get only the matrix
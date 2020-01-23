from random import shuffle as shuf
import string

import numpy as np
import pandas as pd
from tensorflow.keras import keras

class AudioDataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras.
    
    Parameters
    ----------
        df : pd.DataFrame
            columns = ['sentence_string', ']
    
    
    Vocab used = alphabet(26) + space_token(' ') + end_token('>')
    """
    def __init__(self, df, max_seq_output_length, batch_size=32, n_classes=28, num_batch=0, shuffle=True):
        self.df = df
        self.max_seq_output_length = max_seq_output_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.num_batch = num_batch
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.df))

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        cal_num_batch = int(np.floor(self.df.shape[0] / self.batch_size))
        if (self.num_batch == 0) | (self.num_batch > cal_num_batch):
            self.num_batch = cal_num_batch

        return self.num_batch
    
    def __getitem__(self, batch_index):
        """
        Generate one batch of data
                
        Return
        ------
        inputs : dict
            'the_input':     np.ndarray[shape=(batch_size, max_seq_length, mfcc_features)]: input audio data
            'the_labels':    np.ndarray[shape=(batch_size, max_transcript_length)]: transcription data
            'input_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of frames) in output layer
            'label_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of letters) in y
        outputs : dict
            'ctc':           np.ndarray[shape=(batch_size, 1)]: dummy data for dummy loss function
        """

        # Generate indexes of current batch
        indexes_in_batch = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        # Shuffle indexes within current batch if shuffle=true
        if self.shuffle:
            shuf(indexes_in_batch)

        X, y, input_length, label_length = self.__data_generation()

        inputs = {'the_input': X,
                  'the_labels': y,
                  'input_length': np.array([input_length]),
                  'label_length': np.array([label_length])}

        outputs = {'ctc': np.zeros([self.batch_size])} # dummy data for dummy loss function
        # print(inputs)
        # print(outputs)
        # print(X.shape)
        # print(y.shape)
        return inputs, outputs

    def on_epoch_end(self):
        pass
        
    def __data_generation(self):
        """
        Generates data containing batch_size samples
        
        TODO: This method currently implement dummy data. Please change to approriate methods.
        """
        X = self.__input_from_audio()
        y = self.__labels_from_string()

        input_length = self.max_seq_output_length # equals to output shape of Model, e.g. Model(X), 
                                                  # this is the input length to the CTC
        label_length = self.n_classes

        return X, y, input_length, label_length

    def __input_from_audio(self):
        """
        TODO: change implementation by using self.df instead of dummy data
        """
        X = np.array([[5 for x in range(39)] for y in range(300)]) # shape = (300, 39)
        X = np.expand_dims(X, axis=0) # shape = (1, 300, 39)

        return X

    def __labels_from_string(self):
        """
        TODO: change implementation by using self.df instead of dummy data
        """
        vocab = set(string.ascii_lowercase)
        vocab |= {' ', '>'}

        char_to_index = {}
        for i, v in enumerate(vocab):
            char_to_index[v] = i

        # Dummy data
        vocab_index = list(range(len(vocab)))
        y = np.random.choice(vocab_index, 100)
        y = np.expand_dims(y, axis=0)

        return y
        

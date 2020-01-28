from random import shuffle as shuf
import string
import librosa

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from gurih.features.extractor import MFCCFeatureExtractor
from gurih.data.normalizer import AudioNormalizer
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.utils.data_utils import Sequence
from sklearn.pipeline import Pipeline

class DataGenerator(Sequence):
    """
    Generates data for Keras.
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing 
    batch_size : int
        subset size of the training sample
    num_batch : int
        number of batch
    shuffle : bool
        allow shuffling of indexes or not
    """
    def __init__(self, df, batch_size=32, num_batch=0, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batch = num_batch
        self.shuffle = shuffle

        # Initialize indexes
        self.indexes = np.arange(len(self.df))

        # Initialize character map
        self.char_map = {chr(i) : i - 96 for i in range(97, 123)}
        self.char_map[" "] = 0
        self.char_map[">"] = 27


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

        # Load audio and transcripts
        X_data_raw, y_data_raw = self._load_data(self.df, indexes_in_batch)

        # Preprocess and pad data
        X_data, input_length = self._extract_features_and_pad(X_data_raw)
        y_data, label_length = self._convert_transcript_and_pad(y_data_raw)

        # X, y, input_length, label_length = self.__data_generation()

        inputs = {'the_input': X_data,
                  'the_labels': y_data,
                  'input_length': np.array([input_length]),
                  'label_length': np.array([label_length])}

        outputs = {'ctc': np.zeros([self.batch_size])}

        return inputs, outputs
    
    def _load_data(self, df, indexes_in_batch):
        """
        Loads the the corresponding frames (audio time series) from 
        dataframe containing filename, filesize, transcript.

        Parameters
        ----------
        df : pd.DataFrame
            dataframe containing filename, transcript
        
        indexes_in_batch: list 
            list containing the indexes of the audio filenames in the 
            dataframe that is to be loaded.

        Returns
        -------
        X_data_raw: list
            list containing loaded audio time series
        y_data_raw: list 
            list containing transcripts corresponding to 
            loaded audio
        """

        X_data_raw = []
        y_data_raw = []

        for i in indexes_in_batch:
            # Read the path of the audio
            path = df.iloc[i]['filename']
            X_data_raw.append(path)

            # Read transcript data
            y_txt = df.iloc[i]['transcript']
            y_data_raw.append(y_txt)

        return X_data_raw, y_data_raw

    def _extract_features_and_pad(self, X_data_raw):
        """
        Converts list of audio time series to MFCC 
        Zero-pads each sequence to be equal length to the longest
        sequence. Stores the length of each feature-sequence before
        padding for the CTC.

        Parameters
        ----------
        X_data_raw : list
            List of the data containing path of the audio

        Returns
        -------
        X_data : np.array
            Array of the newly appended data (n, max_X_length, 
            default_coefficients)
        input_length : int
            Length of the input
        """

        norm_feature_extractor = Pipeline(
            steps = [
                ("normalizer", AudioNormalizer()),
                ("mfcc_feature_extractor", MFCCFeatureExtractor(append_delta=True))
            ]
        )

        # Fit transform the audio files (normalized and have
        # their feature extracted)
        X_transformeds = norm_feature_extractor.fit_transform(X_data_raw)

        # Get the longest frame
        max_X_length = len(max(X_transformeds, key=lambda x: x.shape[0]))
        
        # Initialize empty data for padding
        X_data = np.empty([0, max_X_length, 39])
        X_seq_lengths = []

        for i in range(0, len(X_transformeds)):
            X_transformeds[i] = X_transformeds[i].T
            X_transformed_shape = X_transformeds[i].shape[0]

            # Add zero to the end of the X.shape[1] (after transposed)
            X_transformed_padded = pad_sequences(X_transformeds[i], maxlen=max_X_length, dtype='float', padding='post', truncating='post')

            X_transformed_padded = X_transformed_padded.T
            X_data = np.insert(X_data, i, X_transformed_padded, axis=0)
            
            # Append the length of the X and discards the first
            # two outputs
            X_seq_lengths.append(X_transformed_shape - 2)

        input_length = np.array(X_seq_lengths)

        return X_data, input_length

    def _convert_text_to_int_sequence(self, text):
        """
        Converts text to the corresponding int sequence.

        Parameters
        ----------
        text : str
            the transcripts that are going to be
            transcribed to int.

        Returns
        -------
        int_sequence : list
            list of the corresponding int sequence
            of the given text.
        """
        int_sequence = []

        for c in text:
          int_sequence.append(char_map[c])  

        return int_sequence

    def _convert_transcript_and_pad(self, y_data_raw):
        """
        Converts text to the corresponding int sequence.

        Parameters
        ----------
        y_data_raw : np.array
            List of the data containing path of the audio

        Returns
        -------
        y_data : np.array
            Array of the newly int-encoded data 
        input_length : int
            Length of the input
        """

        # Find longest sequence in y for padding
        max_y_length = len(max(y_data_raw, key=len))

        y_data = np.empty([0, max_y_length])

        y_seq_length = []

        # Converts to int and pads to be equal max_y_length
        for i in range(0, len(y_data_raw)):
            y_int = self._convert_text_to_int_sequence(y_data_raw[i])

            y_seq_length.append(len(y_int))

            for j in range(len(y_int), max_y_length):
                y_int.append(0)
            
            y_data = np.insert(y_data, i, y_int, axis=0)

        label_length = np.array(y_seq_length)

        return y_data, label_length


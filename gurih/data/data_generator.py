import math
from random import shuffle as shuf
import string
import glob

import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from gurih.features.extractor import MFCCFeatureExtractor
from gurih.data.normalizer import AudioNormalizer


class DataGenerator(Sequence):
    """
    Generates data for ASRModel.
    
    Parameters
    ----------
    input_dir : str
        directory of .npy and .txt transcription
    max_seq_length : int
        maximum sequence length of ASR model
    char_to_idx_map : dict
        dictionary mapping character to index
    batch_size : int
        subset size of the training sample
    num_batch : int
        number of batch
    shuffle : bool
        allow shuffling of indexes or not

    Example
    -------
    >>> import string
    >>> char_to_idx_map = {chr(i) : i - 96 for i in range(97, 123)}
    >>> char_to_idx_map[" "] = 0
    >>> char_to_idx_map["."] = 27
    >>> char_to_idx_map[","] = 28
    >>> char_to_idx_map["%"] = 29
    >>> generator = DataGenerator("input_dir/", 300, char_to_idx_map)
    >>> generator[0][0]
    {'the_input': array([[[5, 5, 5, ..., 5, 5, 5],
         [5, 5, 5, ..., 5, 5, 5],
         [5, 5, 5, ..., 5, 5, 5],
         ...,
         [5, 5, 5, ..., 5, 5, 5],
         [5, 5, 5, ..., 5, 5, 5],
         [5, 5, 5, ..., 5, 5, 5]]]),
    'the_labels': array([[11, 23,  4,  0,  1, 13, 27, 21, 13, 13, 12,  4,  3, 14, 21,  0,
            17,  8,  7, 12,  5, 24, 14,  7,  5, 18, 15,  6, 14, 16, 16,  2,
            14, 25, 18,  1, 11, 21, 14, 25, 11, 10, 13, 16,  9,  7, 25, 27,
            16, 15,  0, 14,  2, 25,  4,  7, 26, 27, 15, 23, 12,  6, 22,  6,
            25, 12, 24,  4,  5, 12,  1,  4, 18, 13, 21, 14,  6, 13, 22, 15,
            14, 19, 11, 23, 21, 23,  6,  6, 23,  0,  1,  5, 23, 25, 25, 24,
            6, 19,  4,  6]]),
    'input_length': array([145]),
    'label_length': array([28])}
    """
    def __init__(self, input_dir, max_seq_length, char_to_idx_map, batch_size=32, num_batch=0, shuffle=True):
        self.input_dir = input_dir
        self.max_seq_length = max_seq_length
        self.char_to_idx_map = char_to_idx_map
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.shuffle = shuffle

        features_filename = sorted(glob.glob(input_dir+"*.npy"))
        transcription_filename = sorted(glob.glob(input_dir+"*.txt"))

        n_features = len(features_filename)
        n_transcription = len(transcription_filename)
        msg = f"Incosistent input length {n_features} != {n_transcription}"
        assert len(features_filename) == len(transcription_filename), msg
        self._m = len(features_filename)

        # Initialize indexes
        self.indexes = np.arange(self._m)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """

        cal_num_batch = int(np.floor(self._m / self.batch_size))
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

        # Load audio and transcript
        X = []
        y = []
        input_length = [self.max_seq_length]*len(indexes_in_batch)
        # input_length = math.ceil(float(input_length - 11 + 1) / float(2))
        label_length = []

        for idx in indexes_in_batch:
            x_tmp = np.load(f"{self.input_dir}{idx}.npy")
            if x_tmp.shape[0] > self.max_seq_length:
                raise ValueError(f"Found input sequence {x_tmp.shape[0]} more than {self.max_seq_length}")
            elif x_tmp.shape[0] < self.max_seq_length:
                tmp = np.empty([0, self.max_seq_length, x_tmp.shape[1]])
                tmp[:x_tmp.shape[0]] = x_tmp
                X.append(x_tmp)
            else:
                X.append(x_tmp)

            with open(f"{self.input_dir}{idx}.txt", 'r') as f:
                y_str = f.readlines()[0]
            y_str = [self.char_to_idx_map[c] for c in y_str]
            y.append(y_str)
            label_length.append(len(y_str))

        # Cast to np.array
        X = np.array(X)
        y = np.array(y)
        input_length = np.array(input_length)
        label_length = np.array(label_length)
        
        inputs = {
            'the_input': X,
            'the_labels': y,
            'input_length': input_length,
            'label_length': label_length
        }

        outputs = {
            'ctc': np.zeros([self.batch_size])
        }

        return inputs, outputs
    
    # def _load_data(self, df, indexes_in_batch):
    #     """
    #     Loads the the corresponding frames (audio time series) from 
    #     dataframe containing filename, filesize, transcript.

    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         dataframe containing filename, transcript
        
    #     indexes_in_batch: list 
    #         list containing the indexes of the audio filenames in the 
    #         dataframe that is to be loaded.

    #     Returns
    #     -------
    #     X_data_raw: list
    #         list containing loaded audio time series
    #     y_data_raw: list 
    #         list containing transcripts corresponding to 
    #         loaded audio
    #     """

    #     X_data_raw = []
    #     y_data_raw = []

    #     for i in indexes_in_batch:
    #         # Read the path of the audio
    #         path = df.iloc[i]['filename']
    #         X_data_raw.append(path)

    #         # Read transcript data
    #         y_txt = df.iloc[i]['transcript']
    #         y_data_raw.append(y_txt)

    #     return X_data_raw, y_data_raw

    # @staticmethod
    # def _extract_features_and_pad(X_data_raw):
    #     """
    #     Converts list of audio time series to MFCC 
    #     Zero-pads each sequence to be equal length to the longest
    #     sequence. Stores the length of each feature-sequence before
    #     padding for the CTC.

    #     Parameters
    #     ----------
    #     X_data_raw : list
    #         List of the data containing path of the audio

    #     Returns
    #     -------
    #     X_data : np.array
    #         Array of the newly appended data (n, max_X_length, 
    #         default_coefficients)
    #     input_length : int
    #         Length of the input
    #     """

    #     norm_feature_extractor = Pipeline(
    #         steps = [
    #             ("normalizer", AudioNormalizer()),
    #             ("mfcc_feature_extractor", MFCCFeatureExtractor(append_delta=True))
    #         ]
    #     )

    #     # Fit transform the audio files (normalized and have
    #     # their feature extracted)
    #     X_transformeds = norm_feature_extractor.fit_transform(X_data_raw)

    #     # Get the longest frame
    #     max_X_length = len(max(X_transformeds, key=lambda x: x.shape[0]))
        
    #     # Initialize empty data for padding
    #     X_data = np.empty([0, max_X_length, 39])
    #     X_seq_lengths = []

    #     for i in range(0, len(X_transformeds)):
    #         X_transformed_shape = X_transformeds[i].shape[0]
    #         X_transformeds[i] = X_transformeds[i].T

    #         # Add zero to the end of the X.shape[1] (after transposed)
    #         X_transformed_padded = pad_sequences(X_transformeds[i], maxlen=max_X_length, dtype='float', padding='post', truncating='post')

    #         X_transformed_padded = X_transformed_padded.T
    #         X_data = np.insert(X_data, i, X_transformed_padded, axis=0)
            
    #         # Append the length of the X and discards the first
    #         # two outputs
    #         # X_seq_lengths.append(X_transformed_shape - 2)
    #         X_seq_lengths.append(X_transformed_shape)

    #     input_length = np.array(X_seq_lengths)

    #     return X_data, input_length

    # def _convert_text_to_int_sequence(self, text):
    #     """
    #     Converts text to the corresponding int sequence.

    #     Parameters
    #     ----------
    #     text : str
    #         the transcripts that are going to be
    #         transcribed to int.

    #     Returns
    #     -------
    #     int_sequence : list
    #         list of the corresponding int sequence
    #         of the given text.
    #     """
    #     int_sequence = []

    #     for c in text:
    #       int_sequence.append(self.char_map[c])  

    #     return int_sequence

    # def _convert_transcript_and_pad(self, y_data_raw):
    #     """
    #     Converts text to the corresponding int sequence.

    #     Parameters
    #     ----------
    #     y_data_raw : np.array
    #         List of the data containing path of the audio

    #     Returns
    #     -------
    #     y_data : np.array
    #         Array of the newly int-encoded data 
    #     input_length : int
    #         Length of the input
    #     """

    #     # Find longest sequence in y for padding
    #     max_y_length = len(max(y_data_raw, key=len))

    #     y_data = np.empty([0, max_y_length])

    #     y_seq_length = []

    #     # Converts to int and pads to be equal max_y_length
    #     for i in range(0, len(y_data_raw)):
    #         y_int = self._convert_text_to_int_sequence(y_data_raw[i])

    #         y_seq_length.append(len(y_int))

    #         for j in range(len(y_int), max_y_length):
    #             y_int.append(0)
            
    #         y_data = np.insert(y_data, i, y_int, axis=0)

    #     label_length = np.array(y_seq_length)

    #     return y_data, label_length


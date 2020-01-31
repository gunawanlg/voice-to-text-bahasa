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
            x_tmp_padded = self._pad_sequence(x_tmp, self.max_seq_length)
            X.append(x_tmp_padded)

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
    
    @staticmethod
    def _pad_sequence(x, max_seq_length):
        """Zero pad input features sequence"""
        out = None
        if x.shape[0] > max_seq_length:
                raise ValueError(f"Found input sequence {x.shape[0]} more than {max_seq_length}")
        elif x.shape[0] < max_seq_length:
            out = np.empty([0, max_seq_length, x.shape[1]])
            out[:x.shape[0]] = x
        else:
            out = x
        
        return out
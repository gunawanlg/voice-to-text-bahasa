import glob
from random import shuffle as shuf

import numpy as np
from kapre.time_frequency import Melspectrogram
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Generates data for ASRModel.

    Parameters
    ----------
    input_dir : str
        directory of .npz and .txt transcription
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
    def __init__(self, input_dir, max_seq_length, max_label_length, ctc_input_length,
                 char_to_idx_map, batch_size=32, num_batch=0, shuffle=True, sr=16000, n_mfcc=39):
        self.input_dir = input_dir
        self.max_seq_length = max_seq_length
        self.max_label_length = max_label_length
        self.ctc_input_length = ctc_input_length
        self.char_to_idx_map = char_to_idx_map
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.shuffle = shuffle
        self.sr = sr
        self.n_mfcc = n_mfcc

        features_filename = sorted(glob.glob(input_dir + "*.wav"))
        transcription_filename = sorted(glob.glob(input_dir + "*.txt"))

        n_features = len(features_filename)
        n_transcription = len(transcription_filename)
        msg = f"Incosistent input length {n_features} != {n_transcription}"
        assert len(features_filename) == len(transcription_filename), msg
        self._m = len(features_filename)

        # Kapre
        self.spectrogrator = Melspectrogram(n_dft=512, n_hop=256, input_shape=(None, None),
                                            padding='same', sr=self.sr, n_mels=self.n_mfcc,
                                            fmin=0.0, fmax=self.sr/2, power_melgram=2.0,
                                            return_decibel_melgram=True, trainable_fb=False,
                                            trainable_kernel=False,
                                            name='melspectrogram')

        # Initialize indexes
        self.indexes = np.arange(self._m)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        cal_num_batch = int(np.ceil(self._m / self.batch_size))
        if (self.num_batch == 0) | (self.num_batch > cal_num_batch):
            self.num_batch = cal_num_batch

        return self.num_batch

    def __getitem__(self, batch_index):
        """
        Generate one batch of data

        Return
        ------
        inputs : dict
            'the_input':     np.ndarray[shape=(batch_size, max_seq_length, mfcc_features)]
            'the_labels':    np.ndarray[shape=(batch_size, max_transcript_length)]
            'input_length':  np.ndarray[shape=(batch_size, 1)]: ctc input length
            'label_length':  np.ndarray[shape=(batch_size, 1)]: ctc input label length
        outputs : dict
            'ctc':           np.ndarray[shape=(batch_size, 1)]: dummy data for dummy loss function
        """

        # Generate indexes of current batch
        indexes_in_batch = self.indexes[
            batch_index * self.batch_size:(batch_index + 1) * self.batch_size
        ]

        # Shuffle indexes within current batch if shuffle=true
        if self.shuffle:
            shuf(indexes_in_batch)

        # Load audio and transcript
        X = []
        y = []
        input_length = [self.ctc_input_length] * len(indexes_in_batch)
        # input_length = math.ceil(float(input_length - 11 + 1) / float(2))
        label_length = [self.max_label_length] * len(indexes_in_batch)

        for idx in indexes_in_batch:
            # On the fly spectrogram
            input_data = tf.io.read_file(f"{self.input_dir}{idx}.wav")
            audio, sr = tf.audio.decode_wav(input_data)
            audio = tf.expand_dims(tf.transpose(audio, [1, 0]), axis=0)

            log_mel_spectrum = tf.squeeze(self.spectrogrator(audio), axis=-1)
            log_mel_spectrum = tf.transpose(log_mel_spectrum, [0, 2, 1])
            padded_spectrum = pad_sequences(log_mel_spectrum,
                                            maxlen=self.max_seq_length,
                                            dtype='float32',
                                            padding='post',
                                            truncating='post',
                                            value=0.0)
            X.append(padded_spectrum[0])

            # x_tmp = np.load(f"{self.input_dir}{idx}.npz")
            # x_tmp = x_tmp['arr_0']
            # x_tmp_padded = self._pad_sequence(x_tmp, self.max_seq_length)
            # X.append(x_tmp_padded)

            with open(f"{self.input_dir}{idx}.txt", 'r') as f:
                y_str = f.readlines()[0]
            y_tmp = [self.char_to_idx_map[c] for c in y_str]
            y_tmp_padded = self._pad_transcript(y_tmp, self.max_label_length)
            y.append(y_tmp_padded)

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
            'ctc': np.zeros([len(indexes_in_batch)])
        }

        return inputs, outputs

    @staticmethod
    def _pad_sequence(x, max_seq_length):
        """Zero pad input features sequence"""
        out = None
        if x.shape[0] > max_seq_length:
            raise ValueError(f"Found input sequence {x.shape[0]} more than {max_seq_length}")
        elif x.shape[0] < max_seq_length:
            out = np.zeros([max_seq_length, x.shape[1]])
            out[:x.shape[0]] = x
        else:
            out = x

        return out

    def _pad_transcript(self, y, max_label_length):
        """Zero pad input label transcription"""
        out = None
        if len(y) > max_label_length:
            raise ValueError(f"Found label transcript {len(y)} more than {max_label_length}")
        elif len(y) < max_label_length:
            # out = np.full([max_label_length], len(self.char_to_idx_map)-1, dtype=int)
            out = np.full([max_label_length], 0, dtype=int)
            out[:len(y)] = y
        else:
            out = y

        return out


def iterate_data_generator(data_generator):
    """
    Create generator function for DataGenerator class yielding inputs.

    Parameters
    ----------
        data_generator : gurih.data.data_generator.DataGenerator
            DataGenerator class

    Returns
    -------
        input_gen : generator
            input sequence generator yielding tuples of input sequence and
            the corresponding label
    """
    i = 0
    while True:
        item = data_generator.__getitem__(i)
        inputs = item[0]
        the_input = inputs['the_input']
        the_label = inputs['the_labels']
        out = [the_input, the_label]
        if the_input.size == 0:
            break
        else:
            i += 1
            yield out


def iterate_y_data_generator(data_generator):
    """
    Create generator function for DataGenerator class yielding inputs.

    Parameters
    ----------
        data_generator : gurih.data.data_generator.DataGenerator
            DataGenerator class

    Returns
    -------
        input_gen : generator
            input sequence generator yielding tuples of input sequence and
            the corresponding label
    """
    i = 0
    while True:
        item = data_generator.__getitem__(i)
        inputs = item[0]
        the_label = inputs['the_labels']
        if the_label.size == 0:
            break
        else:
            i += 1
            for y in the_label:
                yield y


def get_y_true_data_generator(idx_to_char_map, data_generator):
    return [
        ''.join([idx_to_char_map[c] for c in lbl])
        for lbl in iterate_y_data_generator(data_generator)
    ]


def validate_dataset_dir(dataset_dir):
    npzs = sorted(glob.glob(dataset_dir + "*.npz"))
    txts = sorted(glob.glob(dataset_dir + "*.txt"))

    # assert not empty
    assert len(npzs) != 0, f"No npz found in {dataset_dir}."
    assert len(txts) != 0, f"No txt found in {dataset_dir}."

    # assert same length
    assert len(npzs) == len(txts), f"Inconsistent input length {len(npzs)} != {len(txts)}"

    # assert same name conventions
    for npz, txt in zip(npzs, txts):
        assert npz[:-4] == txt[:-4], f"Found inconsistent naming {npz[:-4]} and {txt[:-4]}"

    print(f"{dataset_dir} checks passed.")

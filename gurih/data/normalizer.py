import os
import json
from datetime import datetime

import numpy as np
import librosa
from sklearn.base import TransformerMixin

__all__ = ['AudioNormalizer']


class AudioNormalizer(TransformerMixin):
    """
    Normalize all of the audio files so that they will have
    the same sample_rate, downsized to 1 channel and
    bit_depth of {-1, 1}

    Parameters
    ----------
    sample_rate : int, default=16000
        Size of the sample rate.

    mono : bool, default=True
        Number of channels of the audio, `True` means 1 channel,
        `False` means stereo.

    write_audio_output : bool, default=False
        Store the normalized audio data.

    output_dir : string, default='../data/processed/normalized'
        Output directory of where the normalized audio data
        will be stored.

    encode : bool. default=True
        Encode the filename as ids (True) or just leave it as the filename,
        the idea is to compress the number of strings since appending the full
        filename will eventually increase the file size.

    Examples
    --------
    Given an array of audio filename, we let the audio normalizer
    transform the raw audio data to a normalized audio signal data for the
    next preprocessing pipeline.

    >>> normalizer = AudioNormalizer(is_training=True, encode=True)
    >>> X = ["OSR_us_000_0010_8k.wav", "OSR_us_000_0011_8k.wav"]
    >>> normalizer.transform(X)
    {0: array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), 1:
    array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)}
    """

    def __init__(self, sample_rate=16000, mono=True, write_audio_output=False, output_dir=".",
                 encode=False, is_training=False):
        self.sample_rate = sample_rate
        self.mono = mono
        self.output_dir = output_dir
        self.write_audio_output = write_audio_output
        self.encode = encode
        self.is_training = is_training

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform X to normalized audio files

        Parameters
        ----------
        X : 1-d array
            List of audio filenames

        Returns
        -------
        signal_dict: dict
            Dictionary of signal and corresponding ids (if encode=True) or
            filenames
        """
        # Create a dictionary to store key-value pairs of
        if self.is_training:
            signal_dict = {}
        else:
            signals = []

        # Create a dictionary to encode the name of the sample of ids
        if self.encode:
            id_dict = {}

        processed_data_directory  = self.output_dir
        date = datetime.today().strftime("%Y%m%d")

        for i, filename in enumerate(X):
            signal, sample_rate = librosa.load(filename, sr=self.sample_rate, mono=self.mono)
            if signal.ndim == 1:
                signal = np.expand_dims(signal, axis=0)

            if self.write_audio_output:
                if not os.path.exists(processed_data_directory):
                    os.mkdir(processed_data_directory)

                filename = filename.split("/")[-1]

                # Generate filename consisting of {output_dir}_{original_file_name}_{date}_
                # normalized and convert thedm to wav
                filename = f"{processed_data_directory}/{filename[:-4]}_{date}_normalized.wav"
                librosa.output.write_wav(filename, signal, sample_rate)

            # Save the filenames and signal and ecode them into int
            if self.encode:
                id_dict[i] = filename
                signal_dict[i] = signal
            else:
                filename = filename[:-4].split("/")[-1]

                if self.is_training:
                    signal_dict[filename] = signal
                else:
                    signals.append(signal)

        # Write the .json file to store the corresponding ids and filenames
        if self.encode:
            with open(f"{self.output_dir}/{date}_audio_encoding.json", "w") as f:
                json.dump(id_dict, f)

        # return signal_dict or array of signal
        if self.is_training:
            return signal_dict
        else:
            return signals

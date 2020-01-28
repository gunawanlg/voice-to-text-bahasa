from sklearn.base import TransformerMixin
import librosa
import os

from datetime import datetime

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

    output_dir: string, default='../data/processed/normalized'
        Output directory of where the normalized audio data
        will be stored.

    Examples
    --------
    Given an array of audio filename, we let the audio normalizer
    transform the raw audio data to a normalized audio data for the
    next ML pipeline.

    >>> normalizer = AudioNormalizer()
    >>> X = ["OSR_us_000_0010_8k.wav", "OSR_us_000_0011_8k.wav"]
    >>> normalizer.transform(X)
    >>> mfcc.transform(X)
    ["OSR_us_000_0010_8k_22012020_normalized.wav",
    "OSR_us_000_0011_8k_22012020_normalized.wav"]
    """

    def __init__(self, sample_rate=16000, mono=True, output_dir="."):
        self.sample_rate = sample_rate
        self.mono = mono
        self.output_dir = output_dir

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
        X_out: 1-d array
            List of normalized .wav audio filenames
        """

        X_out = []

        processed_data_directory  = self.output_dir
        date = datetime.today().strftime("%Y%m%d")

        if not os.path.exists(processed_data_directory):
            os.mkdir(processed_data_directory)

        for filename in X:
            signal, sample_rate = librosa.load(filename, sr=self.sample_rate, mono=self.mono)

            filename = filename.split("/")[-1]

            # Generate new_filename consisting of {output_dir}_{original_file_name}_{date}_
            # normalized and convert them to wav
            new_filename = f"{processed_data_directory}/{filename[:-4]}_{date}_normalized.wav"
            librosa.output.write_wav(new_filename, signal, sample_rate)
            X_out.append(new_filename)

        return X_out
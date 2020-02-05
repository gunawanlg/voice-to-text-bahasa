import numpy as np
from scipy.fftpack import dct
from sklearn.base import TransformerMixin
import librosa
import os
from datetime import datetime
import pickle

__all__ = [
    'MFCCFeatureExtractor'
]

class _BaseFeatureExtractor(TransformerMixin):
    """
    Base class for feature extractor that includes the code to
    precalculate the values for advanced feature extraction like
    MFCC.
    """

    def _apply_pre_emphasis(self, signal, coefficient=0.97):
        """
        Apply pre-emphasis to amplify high frequencies

        Parameters
        ----------
        signal : 1-D np.Array
            Array of signal from the audio.

        coefficient : float, default=0.97
            Coefficient for pre-emphasis.


        Returns
        -------
        emphasized_signal : 1-D np.Array:
            Pre-emphasized signal
        """

        emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])

        return emphasized_signal

    def _frame_signal(self, signal, sample_rate, frame_size=0.025, frame_stride=0.01, winfunc=lambda x: np.ones((x,))):
        """
        Split signal into short-time frames

        Parameters
        ----------
        signal : 1-D np.Array
            Array of signal from the audio.

        sample_rate : int
            Size of the sample rate.

        frame_size : float, default=0.025
            Size of the audio frame.

        frame_stride : float, default=0.01
            Length of the frame stride.


        Returns
        -------
        emphasized_signal : 1-D np.Array:
            Pre-emphasized signal
        """

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate

        signal_length = len(signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))

        num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = int((num_frames - 1) * frame_step + frame_length)

        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.concatenate((signal, z))

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.tile(winfunc(frame_length), (num_frames, 1))

        return frames

    def _convert_hz_to_mel(self, hz):
        """
        Convert frequency to Mel frequency.

        Parameters
        ----------
        hz : int
            Frequency.

        Returns
        -------
        mel : float
            Mel Frequency.
        """

        return 2595 * np.log10(1 + hz / 700.0)


    def _convert_mel_to_hz(self, mel):
        """
        Convert Mel frequency to frequency.

        Parameters
        ----------
        mel : float
            Mel frequency.

        Returns
        -------
        hz : float
            frequency.
        """
        return 700 * (10**(mel / 2595.0) - 1)

    def _get_filter_banks(self, filter_num=20, NFFT=512, sample_rate=16000, low_freq=0, high_freq=None):

        """
        Get the Mel filter banks.

        Parameters
        ----------
        filter_num : int, default=20
            Number of Mel filters.
        
        NFFT : int, default=512
            Size of the Fast Fourier Transform
        
        sample_rate : int, default=16000
            Sampling rate

        low_freq : int, default=0
            Lowest frequency

        high_freq : int, default=None
            Highest frequency

        Returns
        -------
        filter_bank : 1-D np.array
            Filter bank
        """
        
        low_mel = self._convert_hz_to_mel(low_freq)

        high_freq = high_freq or sample_rate / 2
        high_mel = self._convert_hz_to_mel(high_freq)

        mel_points = np.linspace(low_mel, high_mel, filter_num + 2)
        
        hz_points = self._convert_mel_to_hz(mel_points)
        
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)
        
        filter_bank = np.zeros([filter_num, int(NFFT / 2 + 1)])
        for j in range(0, filter_num):
            for i in range(int(bin[j]), int(bin[j+1])):
                filter_bank[j,i] = (i-bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                filter_bank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        return filter_bank

    def _apply_fourier_transform(self, frames, NFFT=512):
        """
        Apply Fast Fourier Transform and calculate magnitude of the
        spectrum

        Parameters
        ----------
        frames : 1-D np.Array
            Array of signal from the audio.

        NFFT : int, default=512
            Size of the Fast Fourier Transform

        Returns
        -------
        pow_frames : 1-D np.Array:
            Power pectrum.
        """

        complex_spectrum = np.absolute(np.fft.rfft(frames, NFFT))
        power_spectrum = 1.0/NFFT * np.square(complex_spectrum)
        return power_spectrum

    def _compute_delta(self, features, N=2):
        """
        Compute delta features from a feature vector sequence.

        Parameters
        ----------
        features : 1-D np.Array
            Array of features.

        N : int, default=2
            Preceding and following N frames

        Returns
        -------
        delta_features : 1-D np.Array:
            Array of delat features
        """
        NUMFRAMES = len(features)
        features = np.concatenate(([features[0] for i in range(N)], features, [features[-1] for i in range(N)]))
        denom = sum([2*i*i for i in range(1, N+1)])
        delta_features = []
        for j in range(NUMFRAMES):
            delta_features.append(np.sum([n*features[N+j+n] for n in range(-1*N, N+1)], axis=0)/denom)
        return delta_features

    def _apply_lifter(self, cepstra, L=22):
        """
        Apply lifter function

        Parameters
        ----------
        cepstra : 1-D np.Array
            Coefficients of the MFCC.

        L : int, default=22
            Number of lifter.

        Returns
        -------
        lift * cepstra : int
            A multiplication of lifter and MFCC coeff
        """
        if L > 0:
            (nframes, ncoeff) = np.shape(cepstra)
            n = np.arange(ncoeff)
            lift = 1 + (L/2) * np.sin(np.pi * n/L)
            return lift * cepstra
        else:
            return cepstra

class MFCCFeatureExtractor(_BaseFeatureExtractor):
    """
    Extract Mel features from audio files

    The input for this transformer should be an array-like of
    strings. The features from the audio are extracted through an 
    MFCC way. This results in an array of integers of Mel features
    for each audio.

    Parameters
    ----------
    sample_rate : int, default=16000
        Size of the sample rate.

    frame_size : float, default=0.025
            Size of the audio frame.

    frame_stride : float, default=0.01
        Length of the frame stride.

    filter_num : int, default=26
        Number of Mel filters.

    NFFT : int, default=512
        Size of the Fast Fourier Transform.

    low_freq : int, default=0
        Lowest frequency.

    high_freq : int, default=None
        Highest frequency.

    pre_emphasis_coeff : float, default=0.97
        Coefficient for pre-emphasis.

    cep_lifter : int, default=22
        Number of lifter.

    cep_num: int, default=13
        Number of cepstral coefficients

    dct_type : int, default=2
        Type of numpy discrete consine transform.
    
    dct_norm : str, default="ortho"
        Normalization mode of the dct.

    append_energy:
        Replace the zeroth cepstral coeff with the energy 

    append_delta : bool, default=False
        Append the delta features to features.
    
    write_output : bool, default=False
        Store the Mel features to pickle.

    output_dir : string, path, default='../data/processed/extracted'
        Output directory of where the normalized audio data
        will be stored.

    Examples
    --------
    Given an array of audio filename, we let the feature extractor
    transform the audio signal data to a Mel's feature.

    >>> mfcc = MFCCFeatureExtractor()
    >>> X = ["OSR_us_000_0010_8k.wav"]
    >>> mfcc.fit(X)
    >>> mfcc.filter_bank_
    [[0.   0.5  1.   ... 0.   0.   0.  ]
    [0.   0.   0.   ... 0.   0.   0.  ]
    ...]]
    >>> mfcc.transform(X)
    [array([[-7.12404039e+01,  4.03084620e+00, -2.14802821e+01, ...,
        1.74105473e-01, -2.86706693e-01,  1.62723292e-01],
        ...]
    """

    def __init__(self, sample_rate=16000, frame_size=0.025, frame_stride=0.01, filter_num=26, cep_num=13, NFFT=512, low_freq=0, high_freq=None,pre_emphasis_coeff=0.97, cep_lifter=22, dct_type=2, dct_norm="ortho", append_energy=True, append_delta=False, write_output=True, output_dir="."):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.filter_num = filter_num
        self.NFFT = NFFT
        self.low_freq = low_freq
        self.high_freq = self.sample_rate / 2 or self.high_freq
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.cep_lifter = cep_lifter
        self.cep_num = cep_num
        self.dct_type = dct_type
        self.dct_norm = dct_norm
        
        self.append_delta = append_delta
        self.append_energy = True
        self.write_output = write_output
        self.output_dir = output_dir
    
    def fit(self, X, y=None):
        """
        Fit MFCCFeatureExtractor to X.

        Parameters
        ----------
        X : 1-d np.array
            The data to determine the filter bank

        Returns
        -------
        self
        """
        self.filter_bank_ = super()._get_filter_banks(self.filter_num, self.NFFT, self.sample_rate, self.low_freq, self.high_freq)

        return self

    def transform(self, X):
        """
        Transform X to Mel features.

        Parameters
        ----------
        X : dict
            The data to transform

        Returns
        -------
        mfcc_features_dict : 2-d array
            Dict of array of Mel features
        """

        mfcc_features_dict = {}

        if self.write_output:
            processed_data_directory  = self.output_dir
            date = datetime.today().strftime("%Y%m%d")
            if not os.path.exists(processed_data_directory):
                os.mkdir(processed_data_directory)
        
        for filename in X.keys():
            signal = X[filename]

            signal = super()._apply_pre_emphasis(signal, self.pre_emphasis_coeff)

            frames = super()._frame_signal(signal, self.sample_rate, self.frame_size, self.frame_stride)

            spec_power = super()._apply_fourier_transform(frames, self.NFFT)

            energy = np.sum(spec_power, 1)
            energy = np.where(energy==0, np.finfo(float).eps, energy)

            filter_bank_ = self.filter_bank_

            features = np.dot(spec_power, filter_bank_.T)
            features = np.where(features==0, np.finfo(float).eps, features)

            features = np.log(features)
            features = dct(features, type=self.dct_type, axis=1, norm=self.dct_norm)[:, :self.cep_num]

            features = super()._apply_lifter(features, self.cep_lifter)

            if self.append_energy:
                features[:, 0] = np.log(energy)

            if self.append_delta:
                delta_features = super()._compute_delta(features)
                delta_features_delta = super()._compute_delta(delta_features)
                features = np.concatenate((features, delta_features, delta_features_delta), axis=1)

            mfcc_features_dict[filename] = features

            if self.write_output:
                npz_filename = f"{filename}.npz"
                np.save(f"{processed_data_directory}/{npz_filename}", features)

        return mfcc_features_dict

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


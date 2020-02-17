import math
import json
import warnings

import numpy as np
from sklearn.base import TransformerMixin
from pydub import AudioSegment
# from pydub.utils import mediainfo
from tqdm.auto import tqdm


class Splitter(TransformerMixin):
    """
    Split audio files into chunks.

    Parameters
    ----------
    max_frame_length : int, [default=80000]
        maximum frame length of splited audio, e.g. if sampling rate is
        16000 Hz, then 80000 values equal to 5 secs of splitted audio.
    strides : int, [default=80000]
        if strides equals frame_length, then there will be no overlap in
        splitted audio.
    padding : str, [default='same']
        if 'same', zero pad chunks to size of max_frame_length
        if 'valid', does not do padding
    low_memory : bool, [default=False]
        will return a generator object if True

    Raises
    ------
    ValueError
        if strides > frame_length, causing cropped audio.

    Returns
    -------
    out : list of lists or numpy.ndarray
        return list of lists if padding='same' as output chunks will not have
        equal shape. Return numpy array otherwise.

    """
    def __init__(self, max_frame_length=80000, strides=80000, padding='same',
                 low_memory=False):
        if strides > max_frame_length:
            raise ValueError(f"Strides value of {strides} exceed frame_length \
                             of {max_frame_length}.")

        self.max_frame_length = max_frame_length
        self.strides = strides
        self.padding = padding
        self.low_memory = low_memory

    def fit(self, X, y=None):
        """Do nothing."""
        return self

    def transform(self, X):
        """X is numpy array of shape (m, sample_rate*duration)"""
        if (self.low_memory is True):
            return self._tranform_generator(X)
        else:
            out = []
            for x in X:
                seq_length = x.shape[0]
                if seq_length <= self.max_frame_length:
                    if (seq_length - self.strides) > 0:  # can get at least two chunks
                        out.append(self.split(x))
                    else:
                        warnings.warn(f"Found input shape {x.shape[0]} <= \
                                        {self.max_frame_length}, skipping \
                                        split.")
                        out.append(x)
                else:
                    out.append(self.split(x))

            if self.padding == 'same':
                out = np.array(out)

            return out

    def _tranform_generator(self, X):
        for x in X:
            seq_length = x.shape[0]
            if seq_length <= self.max_frame_length:
                if (seq_length - self.strides) > 0:  # can get at least two chunks
                    yield self.split(x)
                else:
                    warnings.warn(f"Found input shape {x.shape[0]} <= \
                                    {self.max_frame_length}, skipping \
                                    split.")
                    yield x
            else:
                yield self.split(x)

    def split(self, x):
        chunks = []
        for i in range(0, x.shape[0], self.strides):
            chunk = list(x[i:i + self.max_frame_length])  # python handles out of bound indexing
            if self.padding == 'same':
                if len(chunk) < self.max_frame_length:
                    padding = [0.0] * (self.max_frame_length - len(chunk))
                    chunk += padding
            chunks.append(chunk)

        msg = f"Output shape mismatch {len(chunks)} and \
                {math.ceil(x.shape[0] / self.max_frame_length)}"
        assert len(chunks) == (math.ceil(x.shape[0] / self.strides)), msg

        return chunks


class AeneasSplitter:
    """
    Split aligned aeneas .json output files into fragments of mp3s and its
    correponding .txt transcription

    Parameters
    ----------
    input_dir : str, optional
        directory where to find both audio and txt file, default to current
        directory
    output_dir : str, optional
        directory where to find output audio and transcription fragments

    Example
    -------
    >>> splitter = AeneasSplitter("input_dir/", "output_dir/")
    >>> fragments = splitter.load("aeneas_aligned.json")
    >>> splitter.split_and_write(fragments)
    """
    def __init__(self, input_dir=None, output_dir=None):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def load(self, json_filename):
        """
        Load aeenas aligned .json files.

        Parameters
        ----------
        json_filename : str
            input string of aeneas .json filename

        Return
        ------
        fragments : list of dict
            list of aeneas fragment dictionary
        """
        with open(self.input_dir + json_filename, 'r') as f:
            data = ''.join(f.readlines())

        # convert to dict, all data is stored in 'fragments' keys
        fragments = json.loads(data)['fragments']
        self._filename = json_filename[:-5]  # omit the .json

        return fragments

    def split_and_write(self, fragments):
        """
        Split and write output fragments audio and transcription.

        Parameters
        ----------
        fragments : list of dict
            list of aeneas fragment dictionary
        """
        input_filename = self.input_dir + self._filename + ".mp3"
        audio = AudioSegment.from_mp3(input_filename)

        for fragment in tqdm(fragments):
            trimmed_audio = self._trim(audio, fragment)
            self._write_audio(trimmed_audio, fragment['id'])
            self._write_text(fragment['lines'], fragment['id'])

    def _trim(self, audio, fragment):
        """
        Trim audio file.

        Parameters
        ----------
        audio : pydub.audio_segment.AudioSegment
            input audio
        fragment : dict
            element from fragments list

        Return
        ------
        trimmed_audio : pydub.audio_segment.AudioSegment
            trimmed audio segment
        """
        t_begin = float(fragment['begin']) * 1000  # miliseconds
        t_end = float(fragment['end']) * 1000  # miliseconds
        trimmed_audio = audio[t_begin:t_end]

        return trimmed_audio

    def _write_audio(self, trimmed_audio, id, audio_format="mp3", **kwargs):
        """
        write trimmed audio.

        Parameters
        ----------
        trimmed_audio : pydub.audio_segment.AudioSegment
            trimmed audio segment
        id : str,
            unique id identifier of the fragment
        audio_format : str, optional
            output format keyword arguments from AudioSegment.export
        **kwrags
            pydub.audio_segment.AudioSegment keyword arguments
        """
        out_filename = self.output_dir + self._filename + '_' + id + ".mp3"
        trimmed_audio.export(out_filename, format="mp3", **kwargs)

    def _write_text(self, lines, id):
        text_filename = self.output_dir + self._filename + '_' + id + ".txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.writelines(lines)

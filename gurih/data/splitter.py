import json

from pydub import AudioSegment
from pydub.utils import mediainfo
from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm as tqdm


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
        with open(self.input_dir+json_filename, 'r') as f:
            data = ''.join(f.readlines())
        
        # convert to dict, all data is stored in 'fragments' keys
        fragments = json.loads(data)['fragments']
        self._filename = json_filename[:-5] # omit the .json

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
        t_begin = float(fragment['begin']) * 1000 # miliseconds
        t_end = float(fragment['end']) * 1000 # miliseconds
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

        

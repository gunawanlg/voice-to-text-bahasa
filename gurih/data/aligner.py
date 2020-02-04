import os
import json
import glob

from tqdm.auto import tqdm
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from aeneas.logger import Logger
from aeneas.exacttiming import TimeValue
from aeneas.dtw import DTWAlgorithm
from aeneas.runtimeconfiguration import RuntimeConfiguration
from sklearn.base import TransformerMixin

class Aligner(TransformerMixin):
    """
    Aligner(TransformerMixin)

    Parameters
    ----------
    language : str
        three character string indicated by ISO 639-3 language code.
    aligner_type : str
        by default use 'aeneas'
    
    Example
    -------
    >>> X = [ (r"path/to/audio.mp3", r"path/to/audio_transcription.txt) ]
    >>> aligner = Aligner('ind', 'aeneas')
    >>> alignment_dict = aligner.transform(X)
    >>> print(aligned_dict[0]['fragments'])
    """
    def __init__(self, language, aligner_type='aeneas', text_type='plain', output_type='json', write_output=False, max_audio_length=500, print_log=False):
        self.language = language
        self.aligner_type = aligner_type
        self.text_type = text_type
        self.output_type = output_type
        self.write_output = write_output
        self.max_audio_length = unicode(str(max_audio_length))
        self.print_log = print_log
        self.res = []

    def _validate_availability(self, X):
        """
        Validate if .json file exists or not

        Parameters
        ----------
        X: 1-d array
            list of data to be validated

        Return
        ------
        json_availability_dict : dict
            dictionary of json file availability
        """

        # Get a list of existing json files
        finished_jsons = glob.glob("*.json")
        self.res += finished_jsons

        # Get a list of all possible generated jsons
        possible_json_file_path_absolutes =  [x[0][:-4] + ".json" for x in X]

        # Create a dictionary for faster querying
        json_availability_dict = {possible_json_file_path_absolute: True for possible_json_file_path_absolute in possible_json_file_path_absolutes}

        for finished_json in finished_jsons:
            json_availability_dict[finished_json] = False

        return json_availability_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform tuples of path of (audio, transcript) into 
        aligned audio and sentence in form of 
        AENEAS json output file.

        Parameters
        ----------
        X : 1-d np.array
            list of tuples of 
            (r"path/to/audio.mp3", r"path/to/audio_transcription.txt)

        Returns
        -------
        self.res : 1-d array
            list of json files of aligned audio and 
            sentence
        """

        config_string = u"task_language="+self.language+"|is_text_type="+self.text_type+"|os_task_file_format="+self.output_type

        json_availability_dict = self._validate_availability(X)

        # Create Task
        for x in tqdm(X):
            json_file_path_absolute = x[0][:-4] + ".json"

            if json_availability_dict[json_file_path_absolute]:
                try:
                    task = Task(config_string=config_string)
                    task.audio_file_path_absolute = x[0]
                    task.text_file_path_absolute = x[1]
                
                    # Process Task
                    if self.print_log:
                        logger = Logger(tee=True)
                    else:
                        logger = None

                    rconf = None
                    rconf = RuntimeConfiguration()
                    rconf.set_granularity(3)
                    rconf[RuntimeConfiguration.MFCC_MASK_NONSPEECH] = False
                    rconf[RuntimeConfiguration.TASK_MAX_AUDIO_LENGTH] = TimeValue(self.max_audio_length)
                    rconf[RuntimeConfiguration.DTW_ALGORITHM] = DTWAlgorithm.STRIPE
                    rconf[RuntimeConfiguration.VAD_MIN_NONSPEECH_LENGTH] = TimeValue(u"0.050")
                    rconf.set_tts(3)
                    ExecuteTask(task, rconf=rconf, logger=logger).execute()

                    self.res.append(json_file_path_absolute)

                    if self.write_output: 
                        # Output sync map to file
                        task.sync_map_file_path_absolute = json_file_path_absolute
                        task.output_sync_map_file()
                except:
                    self.res.append(None)

        return self.res
#!/usr/bin/env python
# coding=utf-8

import json
import glob

from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from sklearn.base import TransformerMixin

from gurih.utils import validate_nonavailability

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
    >>> cligner = Aligner('ind', 'aeneas')
    >>> alignment_dict = aligner.transform(X)
    >>> print(aligned_dict[0]['fragments'])
    """
    def __init__(self, language, aligner_type='aeneas', audio_type='mp3', text_type='plain', output_type='json', write_output=False):
        self.language = language
        self.aligner_type = aligner_type
        self.audio_type = audio_type
        self.text_type = text_type
        self.output_type = output_type
        self.write_output = write_output
        self.res = []

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

        json_availability_dict = validate_nonavailability(X, "json")

        # Create Task
        for x in X:
            json_file_path_absolute = x[0][:-4] + ".json"

            if json_availability_dict[json_file_path_absolute]:
                try:
                    task = Task(config_string=config_string)
                    task.audio_file_path_absolute = x[0]
                    task.text_file_path_absolute = x[1]
                
                    # Process Task
                    ExecuteTask(task).execute()

                    self.res.append(json_file_path_absolute)

                    if self.write_output: 
                        # Output sync map to file
                        task.sync_map_file_path_absolute = json_file_path_absolute
                        task.output_sync_map_file()
                except:
                    self.res.append(None)

        return self.res
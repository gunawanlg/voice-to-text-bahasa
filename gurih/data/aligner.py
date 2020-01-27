#!/usr/bin/env python
# coding=utf-8

import json
import glob

from aeneas.executetask import ExecuteTask
from aeneas.task import Task

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
        config_string = u"task_language="+self.language+"|is_text_type="+self.text_type+"|os_task_file_format="+self.output_type

        json_availability_dict = self._validate_availability(X)

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
                
                    json_output = json.loads(task.sync_map.json_string)
                    self.res.append(json_output)

                    if self.write_output: 
                        # Output sync map to file
                        task.sync_map_file_path_absolute = json_file_path_absolute
                        task.output_sync_map_file()
                except:
                    self.res.append(None)

        return self.res
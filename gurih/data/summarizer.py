import json

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class Summarizer(TransformerMixin):
    """
    Given a list of .json files from the aligner ouput
    Summarizer will return a pd.DataFrame of the statistical summary of 
    each split.

    Parameters
    ----------
    write_output : bool, default=False
        Store the pd.DataFrame to csv.

    output_dir : str, default="."
        Output path for the results of summarizer.

    output_filename : str, default="statistical_summary.csv"
        Output filename for the results of summarizer.
    """

    def __init__(self, write_output=True, output_dir=".", output_filename="statistical_summary.csv"):
        self.write_output = write_output
        self.output_dir = output_dir
        self.output_filename = output_filename

    def fit(self, X, y=None):
        return X

    def transform(self, X):
        # Generate empty list for storing row outputs 
        ids = []
        audio_durations = []
        transcripts = []
        transcript_lengths = []
        total_commas = []
        split_by_comma_clengths = []
        max_split_by_comma_clengths = []
        min_split_by_comma_clengths = []
        split_by_comma_wlengths = []
        max_split_by_comma_wlengths = []
        min_split_by_comma_wlengths = []

        for filename in X:
            with open(filename) as f:
                text = json.load(f)

                # Strip the filename so it can be used to create subfilenames
                filename_stripped = filename[:-5]
                fragments = text["fragments"]

                for i, fragment in enumerate(fragments):
                    id_name = f"{filename_stripped}_{str(i).zfill(3)}"
                    ids.append(id_name)

                    begin_time = float(fragment["begin"])
                    end_time = float(fragment["end"])

                    # Calculate the duration of the audio
                    audio_duration = end_time - begin_time
                    audio_duration = round(audio_duration, 3)
                    audio_durations.append(audio_duration)

                    transcript = fragment["lines"][0]
                    transcripts.append(transcript)
                    
                    transcript_length = len(transcript)
                    transcript_lengths.append(transcript_length)

                    total_comma = transcript.count(",")
                    total_commas.append(total_comma)

                    # Split the transcript by comma and find the length
                    # of each split, the idea of splitting it by comma is to
                    # find out if a sentence contains more than one sentences
                    # by inferring it from the length of the splits by comma
                    split_by_comma = transcript.split(",")
                    split_by_comma_clength = [len(sentence) for sentence in split_by_comma]
                    split_by_comma_wlength = [len(sentence.split(" ")) for sentence in split_by_comma]

                    split_by_comma_clengths.append(split_by_comma_clength)
                    max_split_by_comma_clengths.append(max(split_by_comma_clength))
                    min_split_by_comma_clengths.append(min(split_by_comma_clength))

                    split_by_comma_wlengths.append(split_by_comma_wlength)
                    max_split_by_comma_wlengths.append(max(split_by_comma_wlength))
                    min_split_by_comma_wlengths.append(min(split_by_comma_wlength))
        
        # Generate DataFrame for the ouput
        df = pd.DataFrame({
                "id_name": ids,
                "audio_duration": audio_durations,
                "transcript": transcripts,
                "transcript_length": transcript_lengths,
                "total_comma" : total_commas,
                "split_by_comma_length": split_by_comma_clengths,
                "max_split_by_comma_length" : max_split_by_comma_clengths,
                "min_split_by_comma_length" : min_split_by_comma_clengths,
                "split_by_comma_wlength": split_by_comma_wlengths,
                "max_split_by_comma_wlength" : max_split_by_comma_wlengths,
                "min_split_by_comma_wlength" : min_split_by_comma_wlengths
            })

        # Write the output to .csv if True
        if self.write_output:
            output_filename = f"{self.output_dir}/{self.output_filename}"
            df.to_csv(output_filename)

        return df

    def fit_transform(self, X, y=None):
        return fit(X).transform(X)
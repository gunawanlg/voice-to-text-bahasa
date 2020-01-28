import os
import glob
import unittest

from pydub import AudioSegment
from pydub.utils import mediainfo
from gurih.data.splitter import AeneasSplitter

class AeneasSplitterTest(unittest.TestCase):
    """
    Test case for class in splitter.py.

    Current config to be run from test/ directory
    """
    def test_equal_num_output_files(self):
        """
        Total files should equal to total fragments list
        """
        splitter = AeneasSplitter("./test_data/", "./test_data/")
        fragments = splitter.load("INDASV_1CH_1.json")
        splitter.split_and_write(fragments)

        num_fragment = len(fragments)
        num_output_files = len(glob.glob("./test_data/INDASV_1CH_1_f*.mp3"))
        
        msg = "Total output files must equal to total fragments length"
        self.assertEqual(num_fragment, num_output_files, msg=msg)

    def test_equal_total_duration(self):
        """
        Total duration of splitted files should equal to original unsplitted
        audio file. Each file should have the correct length according to the
        aligned json file.
        """
        input_filename = "./test_data/INDASV_1CH_1.mp3"
        splitter = AeneasSplitter("./test_data/", "./test_data/")
        fragments = splitter.load("INDASV_1CH_1.json")
        splitter.split_and_write(fragments)

        total_output_duration = 0
        out_filenames = sorted(glob.glob("./test_data/INDASV_1CH_1_f*.mp3"))
        for out_filename, fragment in zip(out_filenames, fragments):
            test_audio = AudioSegment.from_mp3(out_filename)

            t_begin = float(fragment['begin'])
            t_end = float(fragment['end'])
            in_duration = (t_end - t_begin) / 1000
            out_duration = test_audio.duration_seconds
            
            msg = "Output input duration doesn't match"
            self.assertEqual(out_duration, in_duration, msg=msg)

            total_output_duration += out_duration

        input_info = mediainfo(input_filename)
        msg = "Total duration of output should add up to input"
        self.assertEqual(float(input_info['duration']), 
                         total_output_duration,
                         msg=msg)

if __name__ == "__main__":
    unittest.main()

    # Clean the output files
    out_filenames = sorted(glob.glob("./test_data/INDASV_1CH_1_f*.mp3"))
    for out_filename in out_filenames:
        os.remove(out_filename)

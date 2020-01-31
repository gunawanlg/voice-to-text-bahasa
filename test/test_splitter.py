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

    BUGS: using mediainfo instead of duration_seconds yield different results
    """
    @classmethod
    def setUpClass(cls):
        splitter = AeneasSplitter("./test_data/", "./test_data/")
        fragments = splitter.load("INDASV_1CH_1.json")
        splitter.split_and_write(fragments)
        cls._fragments = fragments

    @classmethod
    def tearDownClass(cls):
        # Clean the output files
        out_filenames = sorted(glob.glob("./test_data/INDASV_1CH_1_f*.mp3"))
        for out_filename in out_filenames:
            os.remove(out_filename)

        txt_filenames = sorted(glob.glob("./test_data/INDASV_1CH_1_f*.txt"))
        for txt_filename in txt_filenames:
            os.remove(txt_filename)
        
    def test_equal_num_output_files(self):
        """
        Total files should equal to total fragments list
        """
        num_fragment = len(self._fragments)
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

        total_output_duration = 0
        out_filenames = sorted(glob.glob("./test_data/INDASV_1CH_1_f*.mp3"))
        for out_filename, fragment in zip(out_filenames, self._fragments):
            test_audio = AudioSegment.from_mp3(out_filename)

            # t_begin = float(Decimal(fragment['begin']) * 1000) # miliseconds
            # t_end = float(Decimal(fragment['end']) * 1000) # miliseconds
            t_begin = float(fragment['begin'])
            t_end = float(fragment['end'])
            in_duration = round(t_end - t_begin, 3) # seconds

            # out_duration = float(test_info['duration'])
            out_duration = test_audio.duration_seconds
            
            msg = f"Output input duration doesn't match {out_filename}"
            self.assertEqual(out_duration, in_duration, msg=msg)
            total_output_duration += (t_end - t_begin)

        input_audio = AudioSegment.from_mp3(input_filename)
        msg = "Total duration of output should add up to input"
        self.assertAlmostEqual(input_audio.duration_seconds, 
                         total_output_duration,
                         delta=0.05,
                         msg=msg)

if __name__ == "__main__":
    unittest.main()

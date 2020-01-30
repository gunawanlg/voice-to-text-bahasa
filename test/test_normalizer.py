import unittest
from datetime import datetime
import glob

import numpy as np
from tinytag import TinyTag

from gurih.data.normalizer import AudioNormalizer

class AudioNormalizerTest(unittest.TestCase):

    def setUp(self):
        self.output_dir = "test_data"
        self.X = ["INDASV_GEN_2.mp3"]
        self.X = [f"{self.output_dir}/{x}" for x in self.X]
        self.audio_normalizer = AudioNormalzer()

    def test_normalizer_output(self):
        """
        Test if the normalizer returns the right outputs.
        """

        normalized_signals = self.audio_normalizer.fit_transform(self.X)

        self.assertEqual(type(normalized_signals), dict)
        self.assertEqual(len(normalized_signals[0].shape), 1)
        self.assertGreater(normalized_signals[0].shape[0], 0)

    def test_normalizer_output_details(self):
        """
        Test if the normalizer returns the audio files with the right
        specifications (sample rate of 16000, bit depth of [-1, 1] and 
        1 channeled audio).
        """

        audio_normalizer = AudioNormalizer(write_audio_output=True, output_dir="test_data")
        normalized_signals = audio_normalizer.fit_transform(self.X)
        date = datetime.today().strftime("%Y%m%d")
        new_filename = f"{self.X[0][:-4]}_{date}_normalized.wav"

        tag = TinyTag.get(new_filename)

        self.assertEqual(tag.samplerate, 16000)
        self.assertEqual(tag.channels, 1)

if __name__ == "__main__":
    unittest.main()
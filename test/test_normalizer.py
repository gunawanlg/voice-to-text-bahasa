from datetime import datetime
import os
import unittest

import librosa
from tinytag import TinyTag

from gurih.data.normalizer import AudioNormalizer


class NormalizerTest(unittest.TestCase):

    def setUp(self):
        self.normalizer_default = AudioNormalizer()
        self.x = "test_data/test_20kb.wav"
        self.signal, _ = librosa.load(f"{self.x}", sr=16000)
        self.frame_size = (1, 1, self.signal.shape[0])
        self.output_dir = "test_data"
        self.date = datetime.today().strftime("%Y%m%d")
        self.output_name = f"{self.x.split('/')[-1][:-4]}_{self.date}_normalized.wav"

    def tearDown(self):
        try:
            output_file_full_path = f"{self.output_dir}/{self.output_name}"
            os.remove(output_file_full_path)
        except OSError:
            print("File does not exist")

    def test_normalizer_output_shape(self):
        """
        Test the shape of default output of AudioNormalizer and make sure that
        AudioNormalizer doesn't change the shape
        """
        signal = self.normalizer_default.fit_transform([self.x])

        self.assertEqual(signal.shape, self.frame_size)

    def test_normalizer_output_file(self):
        """
        Test the shape of default output of AudioNormalizer and make sure that
        AudioNormalizer doesn't change the shape
        """
        normalizer = AudioNormalizer(
            write_audio_output=True,
            output_dir="test_data")

        normalizer.fit_transform([self.x])

        output_file_full_path = f"{self.output_dir}/{self.output_name}"

        tag = TinyTag.get(output_file_full_path)
        self.assertEqual(tag.samplerate, 16000)
        self.assertTrue(os.path.exists(output_file_full_path))

    def test_normalizer_encoded_output(self):
        """
        Test the shape of default output of AudioNormalizer and make sure that
        AudioNormalizer doesn't change the shape
        """
        normalizer = AudioNormalizer(
            write_audio_output=True,
            output_dir="test_data",
            is_training=True,
            encode=True)

        output = normalizer.fit_transform([self.x])

        self.assertEqual(len(output.keys()), 1)
        self.assertEqual(list(output.keys()), [0])


if __name__ == "__main__":
    unittest.main()

import unittest

import librosa

from gurih.features.extractor import MFCCFeatureExtractor


class IntegrationTest(unittest.TestCase):

    def setUp(self):
        self.mfcc_default = MFCCFeatureExtractor()
        self.mfcc_39_coefficients = MFCCFeatureExtractor(append_delta=True)
        self.signal, _ = librosa.load("test_data/test_20kb.wav", sr=16000)

    def test_preemphasis(self):
        """
        Test if the preemphasis still returns similar shape as the signal
        """
        preemphasized_signal = self.mfcc_default._apply_pre_emphasis(self.signal)

        self.assertEqual(self.signal.shape, preemphasized_signal.shape)

    def test_frame_signal(self):
        """
        Test if the `_frame_signal` function complies the working `num_frames`
        1 + ⌈|signal_length - (frame_size * sample_rate)| / frame_step⌉ formula
        """
        framed_signal = self.mfcc_default._frame_signal(self.signal, sample_rate=16000)

        self.assertEqual(framed_signal.shape, (34, 400))

    def test_frame_signal_8000k_sr(self):
        """
        Test if even in the change of sample rate, the function still returns
        the right shape.
        """
        framed_signal = self.mfcc_default._frame_signal(self.signal, sample_rate=8000)

        self.assertEqual(framed_signal.shape, (70, 200))

    def test_mffc_single(self):
        """
        Test if the extracted features' shape complies the shape
        """
        extracted_features = self.mfcc_default.fit_transform([self.signal])

        self.assertEqual(extracted_features.shape, (1, 34, 13))


if __name__ == "__main__":
    unittest.main()
import unittest
from gurih.features.extractor import MFCCFeatureExtractor
from gurih.data.normalizer import AudioNormalizer
from sklearn.pipeline import Pipeline

from tinytag import TinyTag
from datetime import datetime

class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.X = ["test_data/INDASV_GEN_1.mp3"]
        self.output_dir = "test_data"
        self.audio_normalizer = AudioNormalizer(output_dir=self.output_dir)

    def test_normalizer_output_details(self):
        """
        Test if the normalizer returns the right outputs.
        """

        X_out = audio_normalizer.fit_transform(self.X)
        date = datetime.today().strftime("%Y%m%d")
        new_filename = f"{X[0][:-4]}_{date}_normalized.wav"

        tag = TinyTag.get(X_out[0])

        self.assertEqual(X_out[0], new_filename)
        self.assertEqual(tag.samplerate, 16000)
        self.assertEqual(tag.channels, 1)


    def test_pipeline(self):
        """
        Test if the pipeline returns the correct shape.
        """

        norm_feature_extractor = Pipeline(
            steps = [
                ("normalizer", self.audio_normalizer),
                ("mfcc_feature_extractor", MFCCFeatureExtractor())
            ]
        )

        X_out = norm_feature_extractor.fit_transform(self.X)

        self.assertEqual(X_out[0].shape[1], 13)

    def test_delta_delta_pipeline(self):
        """
        Test if the delta delta pipeline returns the 
        correct shape.
        """

        norm_feature_extractor = Pipeline(
            steps = [
                ("normalizer", self.audio_normalizer),
                ("mfcc_feature_extractor", MFCCFeatureExtractor(append_delta=True))
            ]
        )

        X_out = norm_feature_extractor.fit_transform(self.X)

        self.assertEqual(X_out[0].shape[1], 39)

if __name__ == "__main__":

    unittest.main()
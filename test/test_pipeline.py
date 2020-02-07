import unittest
from datetime import datetime

from tinytag import TinyTag
from sklearn.pipeline import Pipeline

from gurih.features.extractor import MFCCFeatureExtractor
from gurih.data.normalizer import AudioNormalizer

class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.output_dir = "test_data"
        self.X = ["INDASV_GEN_1.mp3"]
        self.X = [f"{self.output_dir}/{x}" for x in self.X]
        self.audio_normalizer = AudioNormalizer(output_dir=self.output_dir)

    def test_pipeline(self):
        """
        Test if the pipeline returns the correct shape.
        """

        norm_feature_extractor = Pipeline(
            steps = [
                ("normalizer", self.audio_normalizer),
                ("mfcc_feature_extractor", MFCCFeatureExtractor(write_output=True))
            ]
        )

        pipeline_outputs = norm_feature_extractor.fit_transform(self.X)

        self.assertEqual(pipeline_outputs[0].shape[1], 13)

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

        pipeline_outputs = norm_feature_extractor.fit_transform(self.X)

        self.assertEqual(pipeline_outputs[0].shape[1], 39)

if __name__ == "__main__":
    unittest.main()
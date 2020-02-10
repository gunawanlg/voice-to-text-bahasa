import os
import glob
import string
import unittest

import numpy as np

from sklearn.pipeline import Pipeline
from gurih.data.normalizer import AudioNormalizer
from gurih.data.data_generator import DataGenerator
from gurih.features.extractor import MFCCFeatureExtractor
from gurih.models.utils import CharMap


class DataGeneratorTest(unittest.TestCase):
    """Test suite for DataGenerator class"""
    @classmethod
    def setUpClass(cls):
        """
        Make dummy .npz and .txt
        """
        input_dir = "test_data/data_generator/"
        
        X = glob.glob(input_dir+"*.mp3")

        pipeline = Pipeline(
            steps = [
                ("normalizer", AudioNormalizer(output_dir=input_dir)),
                ("mfcc_feature_extractor", MFCCFeatureExtractor(write_output=True,
                                                                output_dir=input_dir,
                                                                append_delta=True))
            ]
        )
        _ = pipeline.fit_transform(X)
        cls.input_dir = input_dir

    @classmethod
    def tearDownClass(cls):
        """
        Delete dummy .npz and .txt
        """
        npz_files = glob.glob(cls.input_dir+"*.npz")
        json_files = glob.glob(cls.input_dir+"*.json")
        for npz_file in npz_files:
            os.remove(npz_file)
        for json_file in json_files:
            os.remove(json_file)

    def test_get_item(self):
        CHAR_TO_IDX_MAP = CharMap.CHAR_TO_IDX_MAP

        MAX_SEQ_LENGTH = 2500
        MAX_LABEL_LENGTH = 100
        BATCH_SIZE = 1

        generator = DataGenerator(input_dir=self.input_dir,
                                max_seq_length=MAX_SEQ_LENGTH,
                                max_label_length=MAX_LABEL_LENGTH,
                                ctc_input_length=1245,
                                char_to_idx_map=CHAR_TO_IDX_MAP,
                                batch_size=BATCH_SIZE)

        batch0, _ = generator.__getitem__(0)
        batch1, _ = generator.__getitem__(1)

        x0 = batch0.get("the_input")
        x1 = batch1.get("the_input")
        y0 = batch0.get("the_labels")
        y1 = batch1.get("the_labels")
        input_length = batch0.get("input_length")
        label_length = batch0.get("label_length")

        self.assertTupleEqual(x0.shape, (1, MAX_SEQ_LENGTH, 39))
        self.assertTupleEqual(x1.shape, (1, MAX_SEQ_LENGTH, 39))
        self.assertTupleEqual(y0.shape, (1, MAX_LABEL_LENGTH))
        self.assertTupleEqual(y1.shape, (1, MAX_LABEL_LENGTH))
        self.assertEqual(input_length.shape[0], 1)
        self.assertEqual(label_length.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
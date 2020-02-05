import os
import glob
import unittest

from gurih.data.normalizer import AudioNormalizer
from gurih.data.data_generator import DataGenerator
from gurih.features.extractor import MFCCFeatureExtractor
from gurih.models.model import BaselineASRModel
from gurih.models.model_utils import CharMap
from sklearn.pipeline import Pipeline

class BaselineASRModelTest(unittest.TestCase):
    """Test suite for BaselineASRModel class"""
    @classmethod
    def setUpClass(cls):
        input_dir = 'test_data/data_generator/'
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

        cls._input_dir = input_dir
        cls._CHAR_TO_IDX_MAP = CharMap.CHAR_TO_IDX_MAP
        cls._IDX_TO_CHAR_MAP = CharMap.IDX_TO_CHAR_MAP

        cls._MAX_SEQ_LENGTH = 2500
        cls._MAX_LABEL_LENGTH = 100
        cls._BATCH_SIZE = 1
        
    @classmethod
    def tearDownClass(cls):
        npzs = glob.glob(cls._input_dir+"*.npz")
        jsons = glob.glob(cls._input_dir+"*.json")
        for npz in npzs:
            os.remove(npz)
        for json in jsons:
            os.remove(json)
        
        pngs = glob.glob(cls._input_dir+"*.png")
        for png in pngs:
            os.remove(png)

    def test_do_compile(self):
        try:
            model = BaselineASRModel(input_shape=(self._MAX_SEQ_LENGTH, 39),
                                     vocab_len=len(CharMap()))
            model.compile()
        except ():
            model = None
        
        self._model = model
        self.assertIsNotNone(model)
        
    def test_do_fit_generator(self):
        model = BaselineASRModel(input_shape=(self._MAX_SEQ_LENGTH, 39),
                                 vocab_len=len(CharMap()))
        model.doc_path = self._input_dir
        model.dir_path = self._input_dir

        model.compile()
        CTC_INPUT_LENGTH = model.model.get_layer('the_output').output.shape[1]

        train_generator = DataGenerator(input_dir=self._input_dir,
                                        max_seq_length=self._MAX_SEQ_LENGTH,
                                        max_label_length=self._MAX_LABEL_LENGTH,
                                        ctc_input_length=CTC_INPUT_LENGTH,
                                        char_to_idx_map=self._CHAR_TO_IDX_MAP,
                                        batch_size=self._BATCH_SIZE)
        model.fit_generator(train_generator=train_generator,
                            epochs=1)


if __name__ == "__main__":
    unittest.main()
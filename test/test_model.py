import os
import glob
import unittest

from gurih.data.data_generator import DataGenerator
from gurih.models.model import BaselineASRModel
from gurih.models.utils import CharMap
from gurih.utils import generator_to_numpy


class BaselineASRModelTest(unittest.TestCase):
    """Test suite for BaselineASRModel class"""
    @classmethod
    def setUpClass(cls):
        input_dir = 'test_data/data_generator/'

        cls._input_dir = input_dir
        cls._CHAR_TO_IDX_MAP = CharMap.CHAR_TO_IDX_MAP
        cls._IDX_TO_CHAR_MAP = CharMap.IDX_TO_CHAR_MAP

        cls._MAX_SEQ_LENGTH = 3000
        cls._MAX_LABEL_LENGTH = 300
        cls._BATCH_SIZE = 1

    @classmethod
    def tearDownClass(cls):
        pngs = glob.glob(cls._input_dir+"*.png")
        for png in pngs:
            os.remove(png)

    def test_do_compile(self):
        try:
            model = BaselineASRModel(input_shape=(self._MAX_SEQ_LENGTH, 39),
                                     vocab_len=len(CharMap()),
                                     training=False)
            model.compile()
        except ():
            model = None

        self._model = model
        self.assertIsNotNone(model)

    def test_do_fit_generator(self):
        model = BaselineASRModel(input_shape=(self._MAX_SEQ_LENGTH, 39),
                                 vocab_len=len(CharMap()),
                                 training=False)
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

    def test_do_fit(self):
        with self.assertRaises(NotImplementedError):
            model = BaselineASRModel(input_shape=(self._MAX_SEQ_LENGTH, 39),
                                     vocab_len=len(CharMap()),
                                     training=False)
            model.compile()
            model.fit()

    def test_predict_low_memory_data_gen(self):
        model = BaselineASRModel(input_shape=(self._MAX_SEQ_LENGTH, 39),
                                 vocab_len=len(CharMap()),
                                 training=False)
        model.compile()
        CTC_INPUT_LENGTH = model.model.get_layer('the_output').output.shape[1]

        test_generator = DataGenerator(input_dir=self._input_dir,
                                       max_seq_length=self._MAX_SEQ_LENGTH,
                                       max_label_length=self._MAX_LABEL_LENGTH,
                                       ctc_input_length=CTC_INPUT_LENGTH,
                                       char_to_idx_map=self._CHAR_TO_IDX_MAP,
                                       batch_size=self._BATCH_SIZE)
        ctc_matrix = model.predict(test_generator, low_memory=True)
        ctc_matrix = generator_to_numpy(ctc_matrix)

        self.assertEqual(ctc_matrix.shape[0], 2)
        self.assertEqual(ctc_matrix.shape[1], CTC_INPUT_LENGTH)
        self.assertEqual(ctc_matrix.shape[2], len(CharMap()) + 1)

    def test_predict_not_low_memory_data_gen(self):
        model = BaselineASRModel(input_shape=(self._MAX_SEQ_LENGTH, 39),
                                 vocab_len=len(CharMap()),
                                 training=False)
        model.compile()
        CTC_INPUT_LENGTH = model.model.get_layer('the_output').output.shape[1]

        test_generator = DataGenerator(input_dir=self._input_dir,
                                       max_seq_length=self._MAX_SEQ_LENGTH,
                                       max_label_length=self._MAX_LABEL_LENGTH,
                                       ctc_input_length=CTC_INPUT_LENGTH,
                                       char_to_idx_map=self._CHAR_TO_IDX_MAP,
                                       batch_size=self._BATCH_SIZE)
        ctc_matrix = model.predict(test_generator, low_memory=False)

        self.assertEqual(ctc_matrix.shape[0], 2)
        self.assertEqual(ctc_matrix.shape[1], CTC_INPUT_LENGTH)
        self.assertEqual(ctc_matrix.shape[2], len(CharMap()) + 1)

    def test_evaluate(self):
        model = BaselineASRModel(input_shape=(self._MAX_SEQ_LENGTH, 39),
                                 vocab_len=len(CharMap()),
                                 training=False)
        model.compile()
        CTC_INPUT_LENGTH = model.model.get_layer('the_output').output.shape[1]

        train_generator = DataGenerator(input_dir=self._input_dir,
                                        max_seq_length=self._MAX_SEQ_LENGTH,
                                        max_label_length=self._MAX_LABEL_LENGTH,
                                        ctc_input_length=CTC_INPUT_LENGTH,
                                        char_to_idx_map=self._CHAR_TO_IDX_MAP,
                                        batch_size=self._BATCH_SIZE)

        loss, ctc_matrix = model.evaluate(train_generator, low_memory=True)

        self.assertEqual(loss.shape[0], 2)

        train_generator = DataGenerator(input_dir=self._input_dir,
                                        max_seq_length=self._MAX_SEQ_LENGTH,
                                        max_label_length=self._MAX_LABEL_LENGTH,
                                        ctc_input_length=CTC_INPUT_LENGTH,
                                        char_to_idx_map=self._CHAR_TO_IDX_MAP,
                                        batch_size=10000)
        X = train_generator[0][0]['the_input']
        y = train_generator[0][0]['the_labels']

        loss, ctc_matrix = model.evaluate(X, y, low_memory=True)

        self.assertEqual(loss.shape[0], 2)


if __name__ == "__main__":
    unittest.main()

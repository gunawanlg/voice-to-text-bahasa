import inspect
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from gurih.models.decoder import CTCDecoder
from gurih.models.model import BaselineASRModel
from gurih.models.utils import CharMap


class TranscriptorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # 39 MFCC featrues, 1000 sequence length, 1 input example
        cls._X = np.random.rand(5, 1000, 39)

        cls._y = [
            "adam, set, enos, kenan, mahalaleel, yared, henokh, metusalah, lamekh, nuh, sem, ham \
            dan yafet.",
            "keturunan yafet ialah gomer, magog, madai, yawan, tubal, mesekh dan tiras.",
            "keturunan gomer ialah askenas, difat dan togarma.",
            "keturunan yawan ialah elisa, tarsis, orang kitim dan orang rodanim.",
            "keturunan ham ialah kush, misraim, put dan kanaan."
        ]

        cls.model = BaselineASRModel(input_shape=(1000, 39), vocab_len=29, training=False)
        cls.model.compile()

    @classmethod
    def tearDownClass(cls):
        del cls._X
        del cls._y
        del cls.model

    def test_fit_predict_with_y(self):
        transcriptor = CTCDecoder(self.model, CharMap.IDX_TO_CHAR_MAP)
        transcriptor.fit(self._X, self._y)
        y_pred = transcriptor.predict(self._X)

        self.assertEqual(len(y_pred), len(self._y))
        self.assertIsNotNone(transcriptor.wer_)

    def test_fit_predict_without_y(self):
        transcriptor = CTCDecoder(self.model, CharMap.IDX_TO_CHAR_MAP)
        transcriptor.fit(self._X)
        y_pred = transcriptor.predict(self._X)

        self.assertEqual(len(y_pred), len(self._y))

    def test_fit_predict_low_memory_with_y(self):
        transcriptor = CTCDecoder(self.model, CharMap.IDX_TO_CHAR_MAP, True)
        transcriptor.fit(self._X, self._y)
        y_pred = transcriptor.predict(self._X)

        self.assertTrue(inspect.isgenerator(transcriptor.ctc_matrix_))
        self.assertEqual(len(y_pred), len(self._y))
        self.assertIsNotNone(transcriptor.wer_)

    def test_fit_predict_low_memory_without_y(self):
        transcriptor = CTCDecoder(self.model, CharMap.IDX_TO_CHAR_MAP, True)
        transcriptor.fit(self._X)
        y_pred = transcriptor.predict(self._X)

        self.assertTrue(inspect.isgenerator(transcriptor.ctc_matrix_))
        self.assertEqual(len(y_pred), len(self._y))

    def test_predict_without_fit(self):
        with self.assertRaises(NotFittedError):
            transcriptor = CTCDecoder(self.model, CharMap.IDX_TO_CHAR_MAP)
            transcriptor.predict(self._X)


if __name__ == "__main__":
    unittest.main()

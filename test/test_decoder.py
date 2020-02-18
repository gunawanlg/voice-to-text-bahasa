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

        cls._model = BaselineASRModel(input_shape=(1000, 39), vocab_len=29, training=False)
        cls._model.compile()

    @classmethod
    def tearDownClass(cls):
        del cls._X
        del cls._y
        del cls._model

    def test_predict_non_generator(self):
        ctc_matrix = self._model.predict(self._X)
        decoder = CTCDecoder(CharMap.IDX_TO_CHAR_MAP)
        decoder.fit(ctc_matrix)
        y_pred = decoder.predict(ctc_matrix)

        self.assertEqual(len(y_pred), len(self._y))

    def test_predict_generator(self):
        ctc_matrix = self._model.predict(self._X, low_memory=True)
        decoder = CTCDecoder(CharMap.IDX_TO_CHAR_MAP)
        decoder.fit(ctc_matrix)
        y_pred = decoder.predict(ctc_matrix)

        self.assertEqual(len(y_pred), len(self._y))

    def test_predict_without_fit(self):
        with self.assertRaises(NotFittedError):
            decoder = CTCDecoder(CharMap.IDX_TO_CHAR_MAP)
            decoder.predict(self._X)


if __name__ == "__main__":
    unittest.main()

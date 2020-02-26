import os
import unittest

from gurih.models.utils import CharMap, cer, wer, wer_and_cer


class ModelsUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._y_true_list = ['aku dan dia', 'dia dan kamu', 'kamu dan aku']
        cls._y_pred_list = ['aky dan dia', 'diaa dan kamu', 'kamu aku']
        cls._y_true_str = 'aku, kamu, dan dia'
        cls._y_pred_str = 'ak, dia, dan'

    @classmethod
    def tearDownClass(cls):
        os.remove("diff.html")

    def test_len_charmap(self):
        self.assertEqual(len(CharMap()), 29)

    def test_cer(self):
        self.assertAlmostEqual(cer(self._y_true_list, self._y_pred_list).mean(), 16.919191919191917)
        self.assertAlmostEqual(cer(self._y_true_str, self._y_pred_str), 50.0)

    def test_wer(self):
        self.assertAlmostEqual(wer(self._y_true_list, self._y_pred_list).mean(), 33.33333333333333)
        self.assertAlmostEqual(wer(self._y_true_str, self._y_pred_str), 75.0)

    def test_wer_and_cer(self):
        out = wer_and_cer(self._y_true_list, self._y_pred_list)
        self.assertAlmostEqual(out['wer'].mean(), 33.33333333333333)
        self.assertAlmostEqual(out['cer'].mean(), 16.919191919191917)

        out = wer_and_cer(self._y_true_str, self._y_pred_str)
        self.assertAlmostEqual(out['wer'], 75.0)
        self.assertAlmostEqual(out['cer'], 50.0)

    def test_write_html(self):
        with self.assertRaises(FileNotFoundError):
            _ = wer(self._y_true_list, self._y_pred_list, html_filename="diff.html")
            with open("diff.html", 'r') as f:
                _ = f.readlines()

        _ = wer(self._y_true_str, self._y_pred_str, html_filename="diff.html")
        with open("diff.html", 'r') as f:
            out = f.readlines()

        self.assertNotEqual(out, "")

    def test_stats(self):
        wer_, stats = wer(self._y_true_list, self._y_pred_list, return_stats=True)

        self.assertEqual(wer_.shape[0], len(self._y_true_list))
        self.assertEqual(stats.shape[0], len(self._y_true_list))
        self.assertAlmostEqual(stats.mean(), 0.75)
        self.assertListEqual(list(stats[0]), [2, 1, 0, 0])
        self.assertListEqual(list(stats[1]), [2, 1, 0, 0])
        self.assertListEqual(list(stats[2]), [2, 0, 1, 0])

    def test_error(self):
        with self.assertRaises(TypeError):
            _ = wer(self._y_true_list, self._y_pred_str)


if __name__ == "__main__":
    unittest.main()

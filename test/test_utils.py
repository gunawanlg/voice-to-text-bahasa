import glob
import inspect
import unittest

import numpy as np

import gurih.utils as gt


class UtilsTest(unittest.TestCase):
    def test_generator_to_numpy(self):
        data = (x**2 for x in range(10))
        numpy_data = gt.generator_to_numpy(data)

        self.assertEqual(numpy_data.shape[0], 10)

    def test_sample_numpy(self):
        data = np.random.rand(5, 1000, 39)
        out = gt.sample_numpy(data, 2)

        self.assertTupleEqual(out.shape, (2, 1000, 39))

    def test_batch(self):
        data = np.random.rand(10)
        out = gt.batch(data, b=3)
        out_list = list(out)

        self.assertTrue(inspect.isgenerator(out))
        self.assertEqual(len(out_list), 4)
        self.assertEqual(len(out_list[-1]), 1)

        out2 = gt.batch(data, n=3)
        out2_list = list(out2)

        self.assertTrue(inspect.isgenerator(out2))
        self.assertEqual(len(out2_list), 3)
        self.assertEqual(len(out2_list[-1]), 4)

    def test_generate_filename(self):
        r = glob.glob("*.mp3")
        h = gt.generate_filenames(".")

        self.assertEqual(len(r), len(h))

    def test_validate_nonavailability(self):
        X = ["0.mp3", "1.mp3"]

        out = gt.validate_nonavailability(X, "npz")

        self.assertTrue(out["0.npz"])
        self.assertTrue(out["1.npz"])


if __name__ == "__main__":
    unittest.main()

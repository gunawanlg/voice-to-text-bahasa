import os
import glob
import string
import unittest

import numpy as np

from gurih.data.data_generator import DataGenerator

class DataGeneratorTest(unittest.TestCase):
    """Test suite for DataGenerator class"""
    @classmethod
    def setUpClass(cls):
        """
        Make dummy .npy and .txt
        """
        if not os.path.exists("test_generator/"): os.makedirs("test_generator/")
        X_dummy = np.random.uniform(-1, 1, size=(10, 300, 39))
        for i, x_dummy in enumerate(X_dummy):
            np.save(f"test_generator/{i}.npy", x_dummy)
        
        vocab = list(string.ascii_lowercase)
        vocab.insert(0, ' ')
        vocab.extend(['.', ',']) # space_token, end_token
        Y_dummy = np.random.choice(vocab, size=(10, 100))
        for i, y_dummy in enumerate(Y_dummy):
            with open(f"test_generator/{i}.txt", 'w', encoding='utf-8') as f:
                f.writelines(y_dummy)

    @classmethod
    def tearDownClass(cls):
        """
        Delete dummy .npy and .txt
        """
        npy_files = glob.glob("test_generator/*.npy")
        txt_files = glob.glob("test_generator/*.txt")
        for npy_file, txt_file in zip(npy_files, txt_files):
            os.remove(npy_file)
            os.remove(txt_file)
        
        os.removedirs("test_generator/")

    def test_get_item(self):
        char_to_idx_map = {chr(i) : i - 96 for i in range(97, 123)}
        char_to_idx_map[" "] = 0
        char_to_idx_map["."] = 27
        char_to_idx_map[","] = 28
        char_to_idx_map["%"] = 29

        generator = DataGenerator("test_generator/", 300, char_to_idx_map, batch_size=6)
        batch0, _ = generator.__getitem__(0)
        batch1, _ = generator.__getitem__(1)

        x0 = batch0.get("the_input")
        x1 = batch1.get("the_input")
        y0 = batch0.get("the_labels")
        y1 = batch1.get("the_labels")
        input_length = batch0.get("input_length")
        label_length = batch0.get("label_length")

        self.assertTupleEqual(x0.shape, (6, 300, 39))
        self.assertTupleEqual(x1.shape, (4, 300, 39))
        self.assertTupleEqual(y0.shape, (6, 100))
        self.assertTupleEqual(y1.shape, (4, 100))
        self.assertEqual(input_length.shape[0], 6)
        self.assertEqual(label_length.shape[0], 6)


if __name__ == "__main__":
    unittest.main()
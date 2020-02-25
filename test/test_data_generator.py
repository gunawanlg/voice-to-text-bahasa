import glob
import unittest

import gurih.data.data_generator as gd
from gurih.data.data_generator import DataGenerator
from gurih.models.utils import CharMap
from gurih.utils import generator_to_numpy


class DataGeneratorTest(unittest.TestCase):
    """Test suite for DataGenerator class"""
    @classmethod
    def setUpClass(cls):
        input_dir = "test_data/data_generator/"
        cls.input_dir = input_dir

    @classmethod
    def tearDownClass(cls):
        del cls.input_dir

    def test_get_item(self):
        CHAR_TO_IDX_MAP = CharMap.CHAR_TO_IDX_MAP

        MAX_SEQ_LENGTH = 3000
        MAX_LABEL_LENGTH = 300
        BATCH_SIZE = 1

        generator = DataGenerator(input_dir=self.input_dir,
                                  max_seq_length=MAX_SEQ_LENGTH,
                                  max_label_length=MAX_LABEL_LENGTH,
                                  ctc_input_length=1495,
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


class DataGeneratorUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_dir = "test_data/data_generator/"
        CHAR_TO_IDX_MAP = CharMap.CHAR_TO_IDX_MAP
        MAX_SEQ_LENGTH = 3000
        cls.MAX_LABEL_LENGTH = 300
        BATCH_SIZE = 1
        cls.generator = DataGenerator(input_dir=cls.input_dir,
                                      max_seq_length=MAX_SEQ_LENGTH,
                                      max_label_length=cls.MAX_LABEL_LENGTH,
                                      ctc_input_length=1495,
                                      char_to_idx_map=CHAR_TO_IDX_MAP,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False)

    @classmethod
    def tearDownClass(cls):
        del cls.generator

    def test_iterate_generator(self):
        gen = gd.iterate_data_generator(self.generator)
        gen_list = list(gen)

        self.assertEqual(len(gen_list), 2)

        gen = gd.iterate_y_data_generator(self.generator)
        gen_np = generator_to_numpy(gen)

        self.assertTupleEqual(gen_np.shape, (2, self.MAX_LABEL_LENGTH))

        gen = gd.get_y_true_data_generator(CharMap.IDX_TO_CHAR_MAP, self.generator)
        gen_list = list(gen)
        gen_list = [seq.strip() for seq in gen_list]  # remove the padding

        txts = glob.glob(self.input_dir+"*.txt")
        y_true = []
        for txt in txts:
            with open(txt, 'r') as f:
                y_true.append(f.readline())

        self.assertEqual(len(gen_list), 2)
        self.assertListEqual(gen_list, y_true)

    def test_validate_dataset_dir(self):
        gd.validate_dataset_dir(self.input_dir)


if __name__ == "__main__":
    unittest.main()

import unittest

from gurih.data.data_generator import DataGenerator
from gurih.models.utils import CharMap


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


if __name__ == "__main__":
    unittest.main()

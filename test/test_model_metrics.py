import unittest

import numpy as np
import tensorflow as tf

from gurih.models.my_keras_metrics import CER


class MyKerasMetricsTest(unittest.TestCase):
    def test_my_sequence_edit_distance(self):
        np.random.seed(42)
        Y_pred = np.random.rand(100, 250, 32) * 0.15
        Y_true = np.random.randint(0, 32, size=(100, 250))
        Y_pred = tf.nn.softmax(Y_pred)
        Y_true = tf.one_hot(Y_true, 32)

        cer = CER(Y_true, Y_pred)

        self.assertAlmostEqual(cer.numpy(), 88.38, delta=0.01)


if __name__ == "__main__":
    unittest.main()

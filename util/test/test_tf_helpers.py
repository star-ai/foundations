import unittest

from util.tf_helpers import TFHelper
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os

tfe.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class TestTFHelper(unittest.TestCase):
  def setUp(self):
    pass

  def test_argmax_2D(self):
    x = np.zeros((2, 5, 5, 1))
    x[0][0][3][0] = 1.0
    x[1][4][2][0] = 1.0
    x = tf.constant(x, dtype=tf.float32)
    # print(x)

    y = TFHelper.argmax_2D(x)
    # print(y.numpy())
    self.assertEqual(y.numpy()[0][0], 0)
    self.assertEqual(y.numpy()[0][1], 3)
    self.assertEqual(y.numpy()[1][0], 4)
    self.assertEqual(y.numpy()[1][1], 2)


  def test_tf_multinomial(self):
    p = np.array([[0.05, 0.05, 0.05],
                  [0.05, 0.60, 0.05],
                  [0.05, 0.05, 0.05],
                  ])
    p = np.expand_dims(p, axis=2)
    p = np.expand_dims(p, axis=0) # build [1, 3, 3, 1]

    p = tf.constant(p)

    y = TFHelper.choose_2D(p)
    # print(y)


if __name__ == "__main__":
  unittest.main()
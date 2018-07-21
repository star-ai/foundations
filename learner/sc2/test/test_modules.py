import unittest
from learner.modules import ConvModule, DenseModule, softmax2D

import os
import tensorflow.contrib.eager as tfe
import tensorflow as tf

tfe.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class TestConvModule(unittest.TestCase):

  def test_conv_module(self):
    x = tf.random_normal((2, 48, 48, 3))
    mod = ConvModule(24)
    y = mod(x)
    self.assertEqual(24, y.shape[3])


  def test_conv_module_softmax(self):
    x = tf.random_uniform((2, 5, 5, 3))
    mod = ConvModule(1, activation=softmax2D)
    y = mod(x)

    sum_y = tf.reduce_sum(y).numpy()
    self.assertAlmostEqual(2.0, sum_y, 2)

  def test_softmax2D(self):
    x = tf.random_uniform((1, 5, 5, 1))
    y = softmax2D(x)

    sum_y = tf.reduce_sum(y).numpy()
    self.assertAlmostEqual(1.0, sum_y, 2)



  def test__dense_module(self):
    x = tf.random_normal((2, 128))

    mod = DenseModule(16)
    y = mod(x)

    self.assertEqual(16, y.shape[1])

  def test__dense_module_no_activation(self):
    x = tf.random_normal((2, 128))

    mod = DenseModule(16, activation=None)
    y = mod(x)

    self.assertEqual(16, y.shape[1])

  def test__dense_module_softmax_activation(self):
    x = tf.random_normal((2, 128))

    mod = DenseModule(16, activation=tf.nn.softmax)
    y = mod(x)

    self.assertAlmostEqual(2.0, tf.reduce_sum(y).numpy(), 2)

if __name__ == "__main__":
  unittest.main()
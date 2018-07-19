import unittest

from learner.dense import SimpleDenseLearner

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tfe.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class TestSimpleDenseLearner(unittest.TestCase):
  def setUp(self):
    self.model = SimpleDenseLearner()

  def test_call(self):
    x = tf.random_normal((3, 4))
    p = self.model(x)

    # print(x)
    # print(p.shape)
    # print(p)
    self.assertEqual((3, 2), p.shape)

  def test_train(self):
    """
    test the full training by passing in a batch of
     s_0, a, s_1, r, done
    """
    s_0 = np.random.rand(1, 4)
    a = np.array([2])
    s_1 = np.random.rand(1, 4)
    r = np.array([[5]])
    done = np.array([[False]])

    p_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    for i in range(50):
      self.model.train(s_0, a, s_1, r, done, 2, p_optimizer)

    s_0 = tf.constant(s_0, dtype=tf.float32)
    p = self.model(s_0)



if __name__ == "__main__":
  unittest.main()
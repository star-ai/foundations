import unittest

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from learner.gym.reinforce import SimpleDenseLearner

tfe.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class TestSimpleDenseLearner(unittest.TestCase):
  def setUp(self):
    self.model = SimpleDenseLearner(nb_actions=2, learning_rate=0.1)

  def test_call(self):
    x = tf.random_normal((3, 4))
    p = self.model(x)

    # print(x)
    # print(p.shape)
    # print(p)
    self.assertEqual((3, 2), p.shape)

  def test_train(self):
    s_0 = np.random.rand(1, 4)
    a = np.array([1])
    r = np.array([[5]])

    for i in range(50):
      self.model.train(s_0, a, r)

    s_0 = tf.constant(s_0, dtype=tf.float32)
    p = self.model(s_0)
    self.assertAlmostEqual(1, p.numpy()[0][1], 1)



if __name__ == "__main__":
  unittest.main()
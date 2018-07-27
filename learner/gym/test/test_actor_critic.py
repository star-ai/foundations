import unittest

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from learner.gym.actor_critic import SimpleMCAdvantageAC

tfe.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class TestSimpleMCAdvantageAC(unittest.TestCase):
  def setUp(self):
    self.model = SimpleMCAdvantageAC(nb_actions=3)

  def test_call(self):
    x = tf.random_normal((3, 4))
    p = self.model.P(x)
    v = self.model.V(x)

    self.assertEqual((3, 3), p.shape)
    self.assertEqual((3, 1), v.shape)

  def test_train(self):
    s_0 = np.random.rand(1, 4)
    a = np.array([1])
    s_1 = np.random.rand(1, 4)
    r = np.array([[5]])
    done = np.array([False])

    p = self.model.P(tf.constant(s_0, dtype=tf.float32))

    for i in range(100):
      self.model.train(s_0, a, s_1, r, done)

    s_0 = tf.constant(s_0, dtype=tf.float32)
    p = self.model.P(s_0)
    # self.assertAlmostEqual(1, p.numpy()[0][1], 0)
    self.assertTrue(p.numpy()[0][1] > 0.7)



if __name__ == "__main__":
  unittest.main()
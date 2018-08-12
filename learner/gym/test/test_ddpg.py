import unittest

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
from learner.gym.ddpg import SimpleDPGActorCritic, SimplePolicy
from util.tf_helpers import TFHelper

tfe.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class TestSimplePolicy(unittest.TestCase):
  def setUp(self):
    self.model = SimplePolicy()

  def test_upper_lower_bands(self):
    x = tf.random_normal((16, 4))
    p = self.model(x, training=True).numpy()

    greater_than = (p >= -1).all()
    less_than = (p <= 1).all()
    self.assertTrue(greater_than and less_than)

  def test_noise(self):

    x = tf.constant([[1, 2], [1, 2]], dtype=tf.float32)
    p = self.model(x).numpy()

    # print(p)
    self.assertEqual(p[0], p[1])

    p = self.model(x, training=True).numpy()
    # print(p)
    self.assertNotEqual(p[0], p[1])


class TestDDPG(unittest.TestCase):

  def test_call_policy(self):
    self.model = SimpleDPGActorCritic(2, action_high=[2, 1], action_low=[-6, -1])

    x = tf.random_normal((3, 4))
    p = self.model.P(x).numpy()
    # print(p)

    greater_than = (p > -6).all()
    less_than = (p < 2).all()
    self.assertTrue(greater_than and less_than)
    self.assertEqual(p.shape, (3, 2))

  def test_call_Q(self):
    self.model = SimpleDPGActorCritic(2, action_high=[1, 1], action_low=[-1, -1])

    x = tf.random_normal((3, 4))
    a = tf.random_normal((3, 2))


    q = self.model.Q([x, a])

    # print(q)
    self.assertEqual(q.shape, (3, 1))


  # def test_train(self):
  #   self.model = SimpleDPGActorCritic(2, action_high=[1, 1], action_low=[-1, -1])
  #   s_0 = np.array([[1, 1, 1, 1]])
  #   a_0 = np.array([[0.5, -0.5]])
  #   s_1 = np.array([[1, 2, 2, 1]])
  #   r_1 = np.array([[1]])
  #   done = np.array([[False]])
  #
  #   for i in range(1):
  #     self.model.train(s_0, a_0, s_1, r_1, done)
  #
  #     p = self.model.P(tf.constant(s_0, dtype=tf.float32))

      # print("Policy", p)

    # print(len(self.model.variables))

  def test_set_weights_to_clone(self):
    s_0 = tf.constant([[1., 1., 1., 1.]])
    a_0 = tf.constant([[0.5, -0.5]])

    self.model = SimpleDPGActorCritic(2, action_high=[1, 1], action_low=[-1, -1])
    self.model.P(s_0)
    self.model.Q([s_0, a_0])

    self.target = SimpleDPGActorCritic(2, action_high=[1, 1], action_low=[-1, -1])

    if len(self.target.variables) == 0:
      # if variables are not initialized then initialize it and set weights to be
      self.target.P(s_0)
      self.target.Q([s_0, a_0])
      self.target.set_weights(self.model.get_weights())

    self.assertEqual(len(self.model.variables), len(self.target.variables))

  # @unittest.skip
  def test_save(self):
    s_0 = tf.constant([[1., 1., 1., 1.]])
    a_0 = tf.constant([[0.5, -0.5]])

    self.model = SimpleDPGActorCritic(2, action_high=[1, 1], action_low=[-1, -1])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    y_p = self.model.P(s_0)
    y_q = self.model.Q([s_0, a_0])

    # print(y_p, y_q)

    TFHelper.save_eager("unittest_ddpg", self.model)

    self.model = SimpleDPGActorCritic(2, action_high=[1, 1], action_low=[-1, -1])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    TFHelper.load_eager("unittest_ddpg", self.model)

    self.assertEqual(y_p, self.model.P(s_0))
    self.assertEqual(y_q, self.model.Q([s_0, a_0]))

    y_p = self.model.P(s_0)
    y_q = self.model.Q([s_0, a_0])
    #
    print(y_p, y_q)

if __name__ == '__main__':
    unittest.main()
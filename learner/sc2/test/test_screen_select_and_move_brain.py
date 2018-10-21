import unittest

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from learner.sc2.screen_only import ScreenSelectAndMoveLearner

tfe.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class TestScreenSelectAndMoveBrain(unittest.TestCase):
  def setUp(self):
    self.model = ScreenSelectAndMoveLearner()

  def test_call(self):
    x = tf.random_normal((2, 84, 84, 17))
    p, v = self.model(x)

    # print(p.shape, v.shape)
    # print(p)
    self.assertEqual((2, 7056), p.shape)
    self.assertEqual((2, 1), v.shape)


  # def test_train_policy(self):
  #   x = tf.random_normal((1, 10, 10, 17))
  #   y = np.zeros((1, 100))
  #   y[0][3] = 1.0
  #   y = tf.constant(y, dtype=tf.float32)
  #
  #   # print(y[:, :1, :, :])
  #
  #   y_hat, v = self.model(x)
  #   y_hat_loss = tf.losses.mean_squared_error(y, y_hat)
  #   # print(y_hat[:, :1, :, :])
  #   # print("loss before training:", y_hat_loss)
  #   for i in range(50):
  #     self.model.train_policy(x, y, 1.)
  #
  #   y_hat_trained, v = self.model(x)
  #   y_hat_loss = tf.losses.mean_squared_error(y, y_hat_trained)
  #   # print("loss after training:", y_hat_loss)
  #   # print(y_hat_trained[:, :1, :, :])
  #   self.assertAlmostEqual(1.0, y_hat_trained.numpy()[0][3], 1)


  # def test_train_value(self):
  #   x = tf.random_uniform((1, 10, 10, 17))
  #   y = np.ones((1, 1)) * 5 # target = 5.
  #   p, y_hat = self.model(x)
  #   # print(y_hat)
  #
  #   for i in range(50):
  #     self.model.train_value(x, y)
  #
  #   p, y_hat = self.model(x)
  #   self.assertAlmostEqual(5.0, y_hat.numpy()[0][0], 0)

  def test_train(self):
    """
    test the full training by passing in a batch of 
     s_0, a, s_1, r, done
     (1, w, h, 1)
    (1, 1)
    (1, w, h, 1)
    (1, 1)
    (1, 1)
    """
    s_0 = np.random.rand(1, 10, 10, 17)
    a = np.array([3])
    s_1 = s_0 # np.random.rand(1, 10, 10, 17)
    r = np.array([[50]])
    done = np.array([[True]])

    for i in range(100):
      self.model.train(s_0, a, s_1, r, done, 10*10)

    s_0 = tf.constant(s_0, dtype=tf.float32)
    p, v = self.model(s_0)
    # print(p)
    # print(v)

    self.assertEqual(3, p.numpy().argmax())

if __name__ == "__main__":
  unittest.main()
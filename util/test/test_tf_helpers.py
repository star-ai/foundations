import unittest

from util.tf_helpers import TFHelper
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os

tfe.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class SimpleModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dense = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)

  def call(self, inputs, training=False):
    return self.dense(inputs)

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


  def test_choose_2D(self):
    p = np.array([[0.05, 0.05, 0.05],
                  [0.05, 0.60, 0.05],
                  [0.05, 0.05, 0.05],
                  ])
    p = np.expand_dims(p, axis=2)
    p = np.expand_dims(p, axis=0) # build [1, 3, 3, 1]

    p = tf.constant(p)

    y = TFHelper.choose_2D(p)
    # print(y)

  def test_coords_to_2D_onehot(self):
    p = np.array([[0.05, 0.05, 0.05],
                  [0.05, 0.60, 0.05],
                  [0.05, 0.05, 0.05],
                  ])
    p = np.expand_dims(p, axis=2)
    p = np.expand_dims(p, axis=0) # build [1, 3, 3, 1]

    p = tf.constant(p)

    y = TFHelper.choose_2D(p)
    # print(y)

    z = TFHelper.coords_to_2D_onehot(y, p.shape)
    # print(z)

  def test_get_action(self):
    def policy_func(s_0):
      return tf.constant([[1e-5, 1e-5, 1.0 - 2*1e-5]])

    a = TFHelper.get_action(policy_func, np.arange(3), [0])
    self.assertEqual(a, 2)


  def test_copy_weights(self):

    x = tf.random_normal((1, 2))

    model_a = SimpleModel()
    out = model_a(x)

    # print("out", out)

    model_b = SimpleModel()
    model_b(x)
    model_b.set_weights(model_a.get_weights())

    original_weights = model_a.get_weights()
    # print("model a weights:", model_a.get_weights())
    # print("model b weights:", model_b.get_weights())
    for i in range(len(model_a.variables)):
      model_a.variables[i].assign_add(1)

    TFHelper.update_target_model(model_a, model_b, 0.2)

    # print("model a weights:", original_weights)
    # print("model b weights:", model_b.get_weights())

    original_weight = original_weights[0][0]
    self.assertEqual(model_b.get_weights()[0][0], original_weight * 0.8 + (original_weight + 1) * 0.2)


  # @unittest.skip
  def test_save(self):
    x = tf.constant([[1., 1., 1., 1.]])

    model = SimpleModel()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    y = model(x)

    TFHelper.save_eager("unittest_save", model, optimizer)

    model = SimpleModel()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    self.assertNotEqual(y, model(x))

    TFHelper.load_eager("unittest_save", model, optimizer)

    self.assertNotEqual(y, model(x))



if __name__ == "__main__":
  unittest.main()
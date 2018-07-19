import tensorflow as tf
import tensorflow.contrib.eager as tfe

from learner.modules import ConvModule, DenseModule, softmax2D
from learner.train import train_vanilla_pg_policy

"""
Simple Dense Learner
"""
class SimpleDenseLearner(tf.keras.Model):
  def __init__(self, learning_rate=0.01, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.dense1 = DenseModule(2, activation=tf.nn.softmax)
    self.dense1 = tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)  # seems batch norm doesn't work very well here
    # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


  def call(self, inputs, training=False):
    # policy = self.dense1(inputs, training=training)
    policy = self.dense1(inputs)
    return policy

  def train(self, s_0, a_0, s_1, r_1, done, num_actions, p_optimizer):
    s_0 = tf.constant(s_0, dtype=tf.float32)
    a_0 = tf.one_hot(a_0, depth=num_actions, dtype=tf.float32)
    # s_1 = tf.constant(s_1, dtype=tf.float32)
    r_1 = tf.constant(r_1, dtype=tf.float32)
    # done = tf.constant(1-done, dtype=tf.float32)

    train_vanilla_pg_policy(self, self, p_optimizer, s_0, a_0, r_1)
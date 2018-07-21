import tensorflow as tf
import tensorflow.contrib.eager as tfe

from learner.modules import ConvModule, DenseModule, softmax2D
from learner.train import train_vanilla_pg_policy

"""
Simple Dense Learner
"""
class SimpleDenseLearner(tf.keras.Model):
  def __init__(self, nb_actions=2, learning_rate=0.01, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.dense1 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu,
    #                                     kernel_regularizer=tf.keras.regularizers.l2)  # seems batch norm doesn't work very well here
    self.dense2 = tf.keras.layers.Dense(units=nb_actions, activation=tf.nn.softmax,
                                        kernel_regularizer=tf.keras.regularizers.l2)  # seems batch norm doesn't work very well here

    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self.nb_actions = nb_actions


  def call(self, inputs, training=False):
    # policy = self.dense1(inputs)
    policy = self.dense2(inputs)
    return policy

  def train(self, s_0, a_0, r_1):
    s_0 = tf.constant(s_0, dtype=tf.float32)
    a_0 = tf.one_hot(a_0, depth=self.nb_actions, dtype=tf.float32)
    advantage = tf.constant(r_1, dtype=tf.float32)

    train_vanilla_pg_policy(self, self, self.optimizer, s_0, a_0, advantage)
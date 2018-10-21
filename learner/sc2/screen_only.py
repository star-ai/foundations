import tensorflow as tf
import tensorflow.contrib.eager as tfe

from learner.modules import ConvModule, DenseModule, softmax2D
from learner.train import train_vanilla_pg_policy, train_vanilla_pg_value
"""
Assuming 84x84x17 screen input
and only produces a click_move output on screen
"""
class ScreenSelectAndMoveLearner(tf.keras.Model):
  def __init__(self, learning_rate=0.001, gamma=0.99, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.gamma = gamma

    self.core_conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
    self.core_conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
    self.core_conv_final = tf.keras.layers.Conv2D(8, kernel_size=(3,3), padding="same", activation=tf.nn.relu)

    self.policy1 = tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding="same", activation=None)

    self.value0 = tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding="same", activation=tf.nn.relu)
    self.flatten = tf.keras.layers.Flatten()
    self.value1 = tf.keras.layers.Dense(units=128, kernel_regularizer=tf.keras.regularizers.l2, activation=tf.nn.relu)
    self.value2 = tf.keras.layers.Dense(units=128, kernel_regularizer=tf.keras.regularizers.l2, activation=tf.nn.relu)
    self.value3 = tf.keras.layers.Dense(units=32, kernel_regularizer=tf.keras.regularizers.l2, activation=tf.nn.relu)
    self.value4 = tf.keras.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l2, activation=None)

    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  def call(self, inputs):
    """
    :param inputs: state
    :param training: whether we're in training mode
    :return: 
    """
    policy = self.P(inputs)
    value = self.V(inputs)
    return policy, value

  def P(self, inputs, training=False):
    core = self.core_conv1(inputs)
    core = self.core_conv2(core)
    core = self.core_conv_final(core)

    policy = self.policy1(core)
    policy = tf.reshape(policy, (policy.shape[0], -1))
    policy = tf.nn.softmax(policy)
    return policy

  def V(self, inputs, training=False):
    core = self.core_conv1(inputs)
    core = self.core_conv2(core)
    core = self.core_conv_final(core)

    value = self.value0(core)
    value = self.flatten(value)
    value = self.value1(value)
    # value = self.value2(value)
    value = self.value3(value)
    value = self.value4(value)
    return value

  def train(self, s_0, a_0, s_1, r_t, done, num_actions):
    s_0 = tf.constant(s_0, dtype=tf.float32)
    a_0 = tf.one_hot(a_0, depth=num_actions, dtype=tf.float32)
    s_1 = tf.constant(s_1, dtype=tf.float32)
    r_t = tf.constant(r_t, dtype=tf.float32)
    done = tf.constant(1-done, dtype=tf.float32)

    v_0 = self.V(s_0)
    v_1 = self.V(s_1)

    # Q_pi(s,a)
    Q = r_t + self.gamma * done * v_1

    # TD_error
    Adv = Q - v_0

    train_vanilla_pg_value(self.V, self, self.optimizer, s_0, Q)
    train_vanilla_pg_policy(self.P, self, self.optimizer, s_0, a_0, Adv)


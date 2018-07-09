import tensorflow as tf
import tensorflow.contrib.eager as tfe

from brain.modules import ConvModule, DenseModule, softmax2D

"""
Assuming 84x84x17 screen input
and only produces a click_move output on screen
"""
class ScreenSelectAndMoveBrain(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.core_conv1 = ConvModule(32)
    self.core_conv2 = ConvModule(32)
    self.core_conv_final = ConvModule(8)

    self.policy1 = ConvModule(1, activation=softmax2D)

    self.value0 = ConvModule(1) #84*84
    self.flatten = tf.keras.layers.Flatten()
    self.value1 = DenseModule(1024)
    self.value2 = DenseModule(128)
    self.value3 = DenseModule(32)
    self.value4 = DenseModule(1, activation=None)


  def call(self, inputs, training=False):
    core = self.core_conv1(inputs)
    core = self.core_conv2(core)
    core = self.core_conv_final(core)

    policy = self.policy1(core)

    value = self.value0(core)
    value = self.flatten(value)
    value = self.value1(value)
    value = self.value2(value)
    value = self.value3(value)
    value = self.value4(value)

    return policy, value


  def train_policy(self, optimizer, inputs, targets, reward):
    with tfe.GradientTape() as tape:
      p, v = self(inputs, training=True)
      """p shape and target shape must be the same"""
      loss = p * tf.cast(targets, dtype=tf.float32)
      loss = tf.reshape(loss, (p.shape[0], -1))
      loss = tf.reduce_sum(loss, reduction_indices=1)
      loss = tf.log(loss)

      final_loss = -tf.reduce_mean(loss * reward)

    grads = tape.gradient(final_loss, self.variables)
    optimizer.apply_gradients(zip(grads, self.variables), global_step=tf.train.get_or_create_global_step())



  def train_value(self, optimizer, inputs, targets):
    with tfe.GradientTape() as tape:
      p, v = self(inputs, training=True)
      loss = tf.losses.mean_squared_error(targets, v)

    grads = tape.gradient(loss, self.variables)
    optimizer.apply_gradients(zip(grads, self.variables), global_step=tf.train.get_or_create_global_step())


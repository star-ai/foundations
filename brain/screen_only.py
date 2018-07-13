import tensorflow as tf
import tensorflow.contrib.eager as tfe

from brain.modules import ConvModule, DenseModule, softmax2D

"""
Assuming 84x84x17 screen input
and only produces a click_move output on screen
"""
class ScreenSelectAndMoveBrain(tf.keras.Model):
  def __init__(self, gamma=0.99, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.gamma = gamma

    self.core_conv1 = ConvModule(32)
    self.core_conv2 = ConvModule(32)
    self.core_conv_final = ConvModule(8)

    self.policy1 = ConvModule(1, activation=None)

    self.value0 = ConvModule(1) #84*84
    self.flatten = tf.keras.layers.Flatten()
    self.value1 = DenseModule(1024)
    self.value2 = DenseModule(128)
    self.value3 = DenseModule(32)
    self.value4 = DenseModule(1, activation=None)


  def call(self, inputs, training=False):
    """
    Bit of a redundancy when it calls the model here as it's doing both the the policy and value
    
    :param inputs: state
    :param training: whether we're in training mode
    :return: 
    """
    core = self.core_conv1(inputs)
    core = self.core_conv2(core)
    core = self.core_conv_final(core)

    policy = self.policy1(core)
    policy = tf.reshape(policy, (policy.shape[0], -1))
    policy = tf.nn.softmax(policy)

    value = self.value0(core)
    value = self.flatten(value)
    value = self.value1(value)
    value = self.value2(value)
    value = self.value3(value)
    value = self.value4(value)

    return policy, value


  def train_policy(self, optimizer, inputs, targets, advantage):
    with tfe.GradientTape() as tape:
      p, v = self(inputs, training=True)
      """p shape and target shape must be the same"""
      p = tf.clip_by_value(p, 1e-6, 0.999999)
      loss = p * tf.cast(targets, dtype=tf.float32)
      loss = tf.reshape(loss, (p.shape[0], -1))
      loss = tf.reduce_sum(loss, reduction_indices=1)
      loss = tf.log(loss)

      print("P Loss:", loss)
      final_loss = -tf.reduce_mean(loss * advantage)

    grads = tape.gradient(final_loss, self.variables)
    optimizer.apply_gradients(zip(grads, self.variables), global_step=tf.train.get_or_create_global_step())



  def train_value(self, optimizer, inputs, targets):
    with tfe.GradientTape() as tape:
      p, v = self(inputs, training=True)
      loss = tf.losses.mean_squared_error(targets, v)
      print("V Loss:", loss)

    grads = tape.gradient(loss, self.variables)
    optimizer.apply_gradients(zip(grads, self.variables), global_step=tf.train.get_or_create_global_step())

  def train(self, s_0, a_0, s_1, r_1, done, num_actions, p_optimizer, v_optimizer):
    s_0 = tf.constant(s_0, dtype=tf.float32)
    a_0 = tf.one_hot(a_0, depth=num_actions, dtype=tf.float32)
    s_1 = tf.constant(s_1, dtype=tf.float32)
    r_1 = tf.constant(r_1, dtype=tf.float32)
    done = tf.constant(1-done, dtype=tf.float32)

    p, v_0 = self(s_0)
    p, v_1 = self(s_1)

    # Q_pi(s,a)
    Q = r_1 #+ self.gamma * done * v_1

    # TD_error
    A = Q - v_0

    self.train_value(v_optimizer, s_0, Q)
    self.train_policy(p_optimizer, s_0, a_0, A)

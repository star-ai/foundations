import tensorflow as tf
import numpy as np
from learner.train import train_vanilla_ddpg_policy, train_vanilla_pg_value
from learner.modules import DenseModule

import tensorflow.contrib.eager as tfe

from util.tf_helpers import TFHelper

"""
Simple Deterministic Actor Critic
"""

class SimplePolicy(tf.keras.Model):
  def __init__(self, nb_actions=1, action_high=[1], action_low=[-1], noise_stddev=0.1, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.policy1 = tf.keras.layers.Dense(units=32, kernel_regularizer=tf.keras.regularizers.l2)
    # self.policy2 = tf.keras.layers.Dense(units=nb_actions, activation=tf.nn.tanh,
    #                                      kernel_regularizer=tf.keras.regularizers.l2)
    self.initializer = tf.initializers.random_uniform(minval=-0.003, maxval=0.003)
    self.policy1 = DenseModule(400)
    self.policy2 = DenseModule(300)
    self.policy3 = tf.keras.layers.Dense(units=nb_actions, activation=tf.nn.tanh, kernel_initializer=self.initializer,
                                         kernel_regularizer=tf.keras.regularizers.l2)
    self.noise = tf.keras.layers.GaussianNoise(stddev=noise_stddev)

    self.nb_actions = nb_actions
    action_high = np.array(action_high)
    action_low = np.array(action_low)
    self.action_mean = (action_high + action_low) / 2
    self.action_range = action_high - self.action_mean

  def call(self, inputs, training=False):
    """
    :param inputs: state 
    :param training: 
    :return: 
    """
    policy = inputs
    policy = self.policy1(policy)
    policy = self.policy2(policy)
    policy = self.policy3(policy)
    policy = self.noise(policy, training=training)
    policy = tf.clip_by_value(policy, -1., 1.)
    policy = policy * self.action_range + self.action_mean
    return policy

class SimpleQ(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.q_state_1 = tf.keras.layers.Dense(units=32, kernel_regularizer=tf.keras.regularizers.l2)
    self.initializer = tf.initializers.random_uniform(minval=-0.003, maxval=0.003)
    self.q_state_1 = DenseModule(400)
    self.q_state_2 = DenseModule(300)
    self.q_concat = tf.keras.layers.Concatenate()
    self.q1 = tf.keras.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer,
                                    kernel_regularizer=tf.keras.regularizers.l2)

  def call(self, inputs, training=False):
    """
    :param inputs: [state, action] 
    :param training: 
    :return: 
    """
    [state, action] = inputs

    q = state
    q = self.q_state_1(q)
    q = self.q_concat([q, action])
    q = self.q_state_2(q)
    q = self.q1(q)
    return q


class SimpleDDPGActorCritic(tf.keras.Model):
  """
    
  """
  def __init__(self, nb_actions=1, action_high=[1], action_low=[-1], actor_lr=0.01, critic_lr=0.01, gamma=0.99, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.policy = SimplePolicy(nb_actions=nb_actions, action_high=action_high, action_low=action_low)
    self.Q_network = SimpleQ()

    self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr)
    self.q_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr)

    self.gamma = gamma


  def call(self, inputs, training=False):
    """
    :param inputs: [state, action] 
    :param training: 
    :return: 
    """
    [state, action] = inputs
    policy = self.P(state, training)
    value = self.Q(inputs, training)
    return policy, value

  def P(self, inputs, training=False):
    """
    :param inputs: state 
    :param training: 
    :return: 
    """
    return self.policy(inputs, training=training)

  def Q(self, inputs, training=False):
    """
    :param inputs: [state, action] 
    :param training: 
    :return: 
    """
    return self.Q_network(inputs, training=training)



class DoubleSimpleDDPGActorCritic():
  def __init__(self, nb_actions=1, action_high=[1], action_low=[-1], actor_lr=0.01, critic_lr=0.01, gamma=0.99,
               tau=0.01, *args, **kwargs):
    self.tau = tau
    self.learner = SimpleDDPGActorCritic(nb_actions=nb_actions, action_high=action_high, action_low=action_low,
                                         actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma)
    self.learner_target = SimpleDDPGActorCritic(nb_actions=nb_actions, action_high=action_high, action_low=action_low,
                                                actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma)

  def getAction(self, s_0, training=False):
    s_0 = tf.constant([s_0], dtype=tf.float32)
    action = self.learner.P(s_0, training=training).numpy()
    return np.squeeze(action, axis=0)

  def save(self, name):
    TFHelper.save_eager(name, self.learner, self.learner.optimizer)
    # TFHelper.save_eager(name + "/learner", self.learner, self.learner.optimizer)
    # TFHelper.save_eager(name + "/target", self.learner_target)

  def load(self, name):
    TFHelper.load_eager(name, self.learner, self.learner.optimizer)
    # TFHelper.load_eager(name + "/learner", self.learner, self.learner.optimizer)
    # TFHelper.load_eager(name + "/target", self.learner_target)

  def train(self, s_0, a_0, s_1, r_1, done):
    s_0 = tf.constant(s_0, dtype=tf.float32)
    a_0 = tf.constant(a_0, dtype=tf.float32)
    s_1 = tf.constant(s_1, dtype=tf.float32)
    r_1 = tf.constant(r_1, dtype=tf.float32)
    done = tf.constant(1-done, dtype=tf.float32)

    if len(self.learner_target.variables) == 0:
      # if variables are not initialized then initialize it and set weights to be
      print("initializing target network")
      self.learner_target.P(s_0)
      self.learner_target.Q([s_0, a_0])
      self.learner_target.set_weights(self.learner.get_weights())

    a_1 = self.learner_target.P(s_1, training=False)
    q_1 = self.learner_target.Q([s_1, a_1])

    # Q_pi(s,a)
    Q_0_target = r_1 + self.learner.gamma * done * q_1

    train_vanilla_pg_value(self.learner.Q, self.learner.Q_network, self.learner.q_optimizer, [s_0, a_0], Q_0_target)

    train_vanilla_ddpg_policy(self.learner.P, self.learner.Q, self.learner.policy, self.learner.policy_optimizer, s_0)

    TFHelper.update_target_model(self.learner, self.learner_target, self.tau)


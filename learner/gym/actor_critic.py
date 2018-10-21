import tensorflow as tf

from learner.train import train_vanilla_pg_policy, train_vanilla_pg_value

"""
Simple Advantage Actor Critic
"""
class SimpleMCPolicyGradientWithBaseline(tf.keras.Model):
  """
    Monte Carlo
  
    # CartPole-v0 settings: policy_lr=0.01, value_lr=0.01, gamma=0.99 
    # Acrobot-v1 settings: policy_lr=0.001, value_lr=0.001, gamma=0.99  does a pretty good job after 1k episodes
    # MountainCar-v0 settings: can't learn anything, can never accidentally hit the target
    # LunarLander-v2 settings: policy_lr=0.001, value_lr=0.001, gamma=0.99  after 10000 (20k?) episodes 
                               it achieves 200+ consistently. slight stability issues
  """
  def __init__(self, nb_actions=2, policy_lr=0.01, value_lr=0.01, gamma=0.99, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.policy1 = tf.keras.layers.Dense(units=32, kernel_regularizer=tf.keras.regularizers.l2, activation=tf.nn.relu)
    self.policy2 = tf.keras.layers.Dense(units=32, kernel_regularizer=tf.keras.regularizers.l2, activation=tf.nn.relu)
    self.policy_out = tf.keras.layers.Dense(units=nb_actions, activation=tf.nn.softmax,
                                            kernel_regularizer=tf.keras.regularizers.l2)
    self.value1 = tf.keras.layers.Dense(units=32, kernel_regularizer=tf.keras.regularizers.l2, activation=tf.nn.relu)
    self.value2 = tf.keras.layers.Dense(units=32, kernel_regularizer=tf.keras.regularizers.l2, activation=tf.nn.relu)
    self.value_out = tf.keras.layers.Dense(units=1, activation=None,
                                           kernel_regularizer=tf.keras.regularizers.l2)

    self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=policy_lr)
    self.value_optimizer = tf.train.AdamOptimizer(learning_rate=value_lr)
    self.nb_actions = nb_actions
    self.gamma = gamma


  def call(self, inputs, training=False):
    policy = self.P(inputs, training)
    value = self.V(inputs, training)
    return policy, value

  def P(self, inputs, training=False):
    policy = self.policy1(inputs)
    policy = self.policy2(policy)
    policy = self.policy_out(policy)
    return policy

  def V(self, inputs, training=False):
    value = self.value1(inputs)
    value = self.value2(value)
    value = self.value_out(value)
    return value

  def train(self, s_0, a_0, s_1, r_1, done):
    """
    :param s_0: 
    :param a_0: 
    :param s_1: 
    :param r_1: This is the total discounted reward from Q(s_0, a_0) to the end of the episode 
    :param done: 
    :return: 
    """
    s_0 = tf.constant(s_0, dtype=tf.float32)
    a_0 = tf.one_hot(a_0, depth=self.nb_actions, dtype=tf.float32)
    s_1 = tf.constant(s_1, dtype=tf.float32)
    r_1 = tf.constant(r_1, dtype=tf.float32)
    done = tf.constant(1-done, dtype=tf.float32)

    v_0 = self.V(s_0)

    # Q_pi(s,a)
    Q = r_1

    # Advantage
    A = Q - v_0

    train_vanilla_pg_value(self.V, self, self.value_optimizer, s_0, Q)
    train_vanilla_pg_policy(self.P, self, self.policy_optimizer, s_0, a_0, A)


class SimpleTDLossAC(SimpleMCPolicyGradientWithBaseline):
  """
    TD0
    
    # CartPole-v0 settings: policy_lr=0.01, value_lr=0.01, gamma=0.99     300+ episodes
    # Acrobot-v1 settings: policy_lr=0.001, value_lr=0.001, gamma=0.99    10k episodes and it learns well
    # MountainCar-v0 settings: can't learn anything 
    # LunarLander-v2 settings: policy_lr=0.001, value_lr=0.001, gamma=0.99  8000 episodes it lands nicely with 200+ scores
  """
  def train(self, s_0, a_0, s_1, r_1, done):
    s_0 = tf.constant(s_0, dtype=tf.float32)
    a_0 = tf.one_hot(a_0, depth=self.nb_actions, dtype=tf.float32)
    s_1 = tf.constant(s_1, dtype=tf.float32)
    r_1 = tf.constant(r_1, dtype=tf.float32)
    done = tf.constant(1-done, dtype=tf.float32)

    v_0 = self.V(s_0)
    v_1 = self.V(s_1)

    # Q_pi(s,a)
    Q = r_1 + self.gamma * done * v_1

    # TD_error
    A = Q - v_0

    train_vanilla_pg_value(self.V, self, self.value_optimizer, s_0, Q)
    train_vanilla_pg_policy(self.P, self, self.policy_optimizer, s_0, a_0, A)
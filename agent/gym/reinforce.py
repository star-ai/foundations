from agent.gym.base_agent import BaseAgent
from learner.dense import SimpleDenseLearner
from memory.memory import EpisodicMemory, Transition, Gt

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tfe.enable_eager_execution()

class REINFORCE(BaseAgent):
  def __init__(self):
    super().__init__()
    self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    self.s_0 = None
    self.a_0 = None
    self.s_1 = None
    self.r_1 = None
    self.done = None
    self.gt = Gt()

  def setup(self, observation_space, action_space):
    super().setup(observation_space, action_space)
    self.learner = SimpleDenseLearner()
    self.memory = EpisodicMemory()

  def reset(self):
    super().reset()
    self.s_0 = None
    self.a_0 = None
    self.s_1 = None
    self.r_1 = None
    self.done = None
    self.train()


  def step(self, obs):
    super().step(obs)
    self.s_1 = obs[0]
    self.r_1 = obs[1]
    self.done = obs[2]

    if self.s_0 is not None and self.a_0 is not None:
      a = Transition(self.s_0, self.a_0, self.s_1, self.r_1, self.done)
      self.memory.push(a)
      self.s_0 = self.s_1
      self.a_0 = None

    if self.s_0 is None:
      self.s_0 = self.s_1

    self.a_0 = self.getAction(self.s_0)
    return self.a_0


  def getAction(self, s_0):
    s_0 = tf.constant([s_0], dtype=tf.float32)
    action_probability = self.learner(s_0).numpy()
    action = np.random.choice([0, 1], p=action_probability[0])
    return action

  def train(self):
    if len(self.memory) == 0:
      return

    s_0, a_0, s_1, r_1, done = self.memory.sample()

    baseline = self.gt.add_episode_rewards(np.squeeze(r_1))
    baseline = np.expand_dims(baseline, axis=1)

    A = r_1 - baseline
    self.learner.train(s_0=s_0, a_0=a_0, s_1=s_1, r_1=A, done=done, num_actions=self.action_space.n,
                     p_optimizer=self.optimizer)
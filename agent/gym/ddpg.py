import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from agent.gym.base_agent import BaseAgent, RLAgent
from learner.gym.ddpg import DoubleSimpleDDPGActorCritic
from learner.train import train_vanilla_pg_policy, train_vanilla_pg_value
from memory.memory import ReplayMemory, EpisodicMemory
from memory.sarsd import SARSD
from util.tf_helpers import TFHelper

tfe.enable_eager_execution()

class DDPGActorCritic(RLAgent):
  """
    Monte Carlo actor critic
  """
  def setup(self, observation_space, action_space):
    super().setup(observation_space, action_space)
    print(action_space.shape, action_space.high, action_space.low)
    self.num_actions = action_space.shape[0]
    self.learner = DoubleSimpleDDPGActorCritic(nb_actions=self.num_actions, action_high=action_space.high,
                                               action_low=action_space.low,
                                               actor_lr=0.0001, critic_lr=0.001, gamma=0.99, tau=0.01)

    # self.memory = EpisodicMemory(gamma=0.99, accum_reward=False)
    self.memory = ReplayMemory(capacity=1000000)

  def save(self, name):
    self.learner.save(name)

  def load(self, name):
    self.learner.load(name)

  def getAction(self, s_0):
    return self.learner.getAction(s_0, self.training)


  def train(self):
    if len(self.memory) <= 20000:
      return

    for i in range(100):
      s_0, a_0, s_1, r_1, done = self.memory.sample(batch_size=64)
      self.learner.train(s_0=s_0, a_0=a_0, s_1=s_1, r_1=r_1, done=done)


import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from agent.gym.base_agent import BaseAgent, RLAgent
from learner.gym.reinforce import SimpleDenseLearner
from memory.memory import EpisodicMemory, Gt
from memory.sarsd import SARSD
from util.tf_helpers import TFHelper

tfe.enable_eager_execution()

class REINFORCE(RLAgent):
  """
  REINFORCE with a baseline for discrete actions
  """
  def __init__(self):
    super().__init__()
    self.gt = Gt()

  def setup(self, observation_space, action_space):
    super().setup(observation_space, action_space)
    self.learner = SimpleDenseLearner(nb_actions=action_space.n, learning_rate=0.1)
    self.memory = EpisodicMemory()
    self.actions = np.arange(self.action_space.n)

  def getAction(self, s_0):
    s_0 = tf.constant([s_0], dtype=tf.float32)
    action_probability = self.learner(s_0).numpy()
    action = np.random.choice(self.actions, p=action_probability[0])
    return action

  def train(self):
    if len(self.memory) == 0:
      return

    s_0, a_0, s_1, r_1, done = self.memory.sample()

    baseline = self.gt.add_episode_rewards(np.squeeze(r_1))
    baseline = np.expand_dims(baseline, axis=1)

    A = r_1 - baseline
    self.learner.train(s_0=s_0, a_0=a_0, r_1=A)
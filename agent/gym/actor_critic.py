import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from agent.gym.base_agent import BaseAgent, RLAgent
from learner.gym.actor_critic import SimpleMCAdvantageAC, SimpleTDLossAC
from memory.memory import EpisodicMemory
from memory.sarsd import SARSD
from util.tf_helpers import TFHelper

tfe.enable_eager_execution()

class MCActorCritic(RLAgent):
  """
    Monte Carlo actor critic
  """

  def setup(self, observation_space, action_space):
    super().setup(observation_space, action_space)
    self.learner = SimpleMCAdvantageAC(nb_actions=action_space.n, policy_lr=0.01, value_lr=0.01, gamma=0.99)
    self.memory = EpisodicMemory(gamma=0.99)
    self.actions = np.arange(self.action_space.n)

  def getAction(self, s_0):
    return TFHelper.get_action(self.learner.P, self.actions, s_0)

  def train(self):
    if len(self.memory) == 0:
      return

    s_0, a_0, s_1, r_1, done = self.memory.sample()
    self.learner.train(s_0=s_0, a_0=a_0, s_1=s_1, r_1=r_1, done=done)





class TD0ActorCritic(MCActorCritic):
  def setup(self, observation_space, action_space):
    super().setup(observation_space, action_space)
    self.learner = SimpleTDLossAC(nb_actions=action_space.n, policy_lr=0.01, value_lr=0.01, gamma=1/0.99)
    self.memory = EpisodicMemory(accum_reward=False)
    self.actions = np.arange(self.action_space.n)
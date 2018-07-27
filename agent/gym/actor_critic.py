import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from agent.gym.base_agent import BaseAgent
from learner.gym.actor_critic import SimpleMCAdvantageAC, SimpleTDLossAC
from memory.memory import EpisodicMemory, Gt
from memory.sarsd import SARSD
from util.tf_helpers import TFHelper

tfe.enable_eager_execution()

class MCActorCritic(BaseAgent):
  """
    Monte Carlo actor critic
  """
  def __init__(self):
    super().__init__()
    self.gt = Gt()
    self.sarsd = SARSD()

  def setup(self, observation_space, action_space):
    super().setup(observation_space, action_space)
    self.learner = SimpleMCAdvantageAC(nb_actions=action_space.n, policy_lr=0.01, value_lr=0.01, gamma=0.99)
    self.memory = EpisodicMemory(gamma=0.99)
    self.actions = np.arange(self.action_space.n)

  def reset(self):
    super().reset()
    self.sarsd.reset()
    self.train()


  def step(self, obs):
    super().step(obs)
    state = obs[0]
    reward = obs[1]
    done = obs[2]

    transition = self.sarsd.observe(state, reward, done)
    if transition is not None: self.memory.push(transition)

    action = self.getAction(state)
    self.sarsd.act(action)

    return action

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
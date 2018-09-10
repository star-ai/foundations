from abc import ABCMeta, abstractmethod

from memory.sarsd import SARSD
from util.tf_helpers import TFHelper


class BaseAgent(metaclass=ABCMeta):
  """A base agent to write custom scripted agents.
  """

  def __init__(self):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.reward_history = []
    self.training = True

  @abstractmethod
  def setup(self, observation_space, action_space):
    self.action_space = action_space
    self.observation_space = observation_space

  def reset(self):
    if self.steps > 0:
      self.reward_history.append(self.reward)

    self.reward = 0
    self.steps = 0
    self.episodes += 1

  def step(self, obs):
    self.steps += 1
    self.reward += obs[1]
    return self.action_space.sample()




class RLAgent(BaseAgent):
  def __init__(self):
    super().__init__()
    self.sarsd = SARSD()

  def reset(self):
    super().reset()
    self.sarsd.reset()
    if self.training:
      self.train()


  def step(self, obs):
    super().step(obs)
    state = obs[0]
    reward = obs[1]
    done = obs[2]

    if hasattr(self, "memory"):
      transition = self.sarsd.observe(state, reward, done)
      if transition is not None: self.memory.push(transition)

    action = self.getAction(state)
    self.sarsd.act(action)

    return action

  @abstractmethod
  def getAction(self, s_0):
    pass

  # don't like this but oh well, could use a decorator instead
  def save(self, name):
    if hasattr(self, "learner"):
      TFHelper.save_eager(name, self.learner)

  def load(self, name):
    if hasattr(self, "learner"):
      TFHelper.load_eager(name, self.learner)

  @abstractmethod
  def train(self):
    pass


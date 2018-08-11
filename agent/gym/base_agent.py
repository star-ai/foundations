class BaseAgent(object):
  """A base agent to write custom scripted agents.
  """

  def __init__(self):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.reward_history = []
    self.training = True

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

  def save(self, filename):
    print("save not implemented")

  def load(self, filename):
    print("save not implemented")

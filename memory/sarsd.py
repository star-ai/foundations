from memory.memory import Transition

class SARSD():
  def __init__(self):
    self.reset()

  def observe(self, state, reward=0, done=False):
    transition = None

    if self.s_0 is not None and self.a_0 is not None:
      transition = Transition(self.s_0, self.a_0, state, reward, done)

    self.s_0 = state
    self.a_0 = None

    return transition

  def act(self, action):
    self.a_0 = action

  def reset(self):
    self.s_0 = None
    self.a_0 = None

  def __str__(self):
    return str([self.s_0, self.a_0])
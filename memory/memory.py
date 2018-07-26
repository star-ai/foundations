from collections import namedtuple
import random
import numpy as np


Transition = namedtuple("Transition", ["s", "a", "s_1", "r", "done"])

class SimpleBaseMemory(object):
  """Some basic convenience methods classes can use"""
  def __len__(self):
    return len(self.memory)

  def __str__(self):
    result = []
    for i in range(self.__len__()):
      result.append(self.memory[i].__str__() + " \n")
    return "".join(result)


class ReplayMemory(SimpleBaseMemory):
  """A replay memory with capacity that allows random sampling"""
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, item):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = item
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    out = random.sample(self.memory, batch_size)
    batched = Transition(*zip(*out))
    s = np.array(list(batched.s))
    a = np.array(list(batched.a))
    s_1 = np.array(list(batched.s_1))
    r = np.expand_dims(np.array(list(batched.r)), axis=1)
    done = np.expand_dims(np.array(list(batched.done)), axis=1)
    return [s, a, s_1, r, done]


class EpisodicMemory(SimpleBaseMemory):
  """
    A simple memory for storing states for an entire episode and then use the entire trajectory for training
    Rewards for each timestep are the total discounted future rewards according to gamma
  """
  def __init__(self, gamma=0.99, accum_reward=True):
    self.memory = []
    self.gamma = gamma
    self.accum_reward = accum_reward

  def reset(self):
    self.memory = []

  def push(self, transition):
    self.memory.append(transition)

  def sample(self):
    batched = Transition(*zip(*self.memory))
    s = np.array(list(batched.s))
    a = np.array(list(batched.a))
    s1 = np.array(list(batched.s_1))
    r = np.array(list(batched.r), dtype="float32")
    done = np.expand_dims(np.array(list(batched.done)), axis=1)

    if self.accum_reward:
      reward = 0.
      for i in reversed(range(len(r))):
          moving_reward = self.gamma * reward
          reward = r[i] + moving_reward
          r[i] = reward
          # print(i, len(r)-i-1, r[i])

    r = np.expand_dims(r, axis=1)

    # clear memory
    self.reset()
    return [s, a, s1, r, done]

class Gt():
  def __init__(self):
    self.all_returns = []

  def add_episode_rewards(self, rewards):
    """
    :param rewards: rewards at every timestep: expected shape: (x,) 
    :return: historical mean at every step 
    """
    mu = np.zeros_like(rewards, dtype=np.float32)
    for i, r in enumerate(rewards):
      if len(self.all_returns) < i+1:
        self.all_returns.append([])
      self.all_returns[i].append(r)
      mu[i] = np.mean(self.all_returns[i])
    return mu


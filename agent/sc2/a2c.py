from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from learner.sc2.screen_only import ScreenSelectAndMoveLearner
from memory.memory import Transition, EpisodicMemory

tfe.enable_eager_execution()

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS


def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))


class A2CMoveToBeacon(base_agent.BaseAgent):
  """A2C agent for move to beacon"""
  def __init__(self):
    super().__init__()
    self.action_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    self.value_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    self.brain = ScreenSelectAndMoveLearner()
    self.memory = EpisodicMemory(gamma=0.99)
    self.batch_size = 32

    self.s_0 = None
    self.a_0 = None
    self.s_1 = None
    self.r_1 = None
    self.done = None

  def reset(self):
    super().reset()
    print("Total Steps:", self.steps)
    if len(self.memory) > 0:
      self.train()

  def step(self, obs):
    super().step(obs)

    self.s_1 = obs.observation.feature_screen.player_relative  #make it simple for now
    self.s_1 = np.expand_dims(self.s_1, axis=2)

    self.r_1 = obs.reward
    self.done = False #obs.reward == 1

    if self.s_0 is not None and self.a_0 is not None:
      a = Transition(self.s_0, self.a_0, self.s_1, self.r_1, self.done)
      self.memory.push(a)
      self.s_0 = self.s_1
      self.a_0 = None

      # if self.steps >= 1000 and self.steps % 4 == 0:
      #   self.train()


    if self.s_0 is None:
      self.s_0 = self.s_1

    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      action, coords = self.get_action(self.s_0)
      # print("Move action:", coords)
      self.a_0 = action
      return FUNCTIONS.Move_screen("now", coords)
    else:
      return FUNCTIONS.select_army("select")

  def get_best_action(self, s_0):
    s_0 = np.expand_dims(s_0, axis=0) # convert to batch
    s_0 = tf.constant(s_0, dtype=tf.float32)
    p, v = self.brain(s_0)
    argmax = tf.argmax(p, axis=1)
    coords = tf.unravel_index(argmax, s_0.shape[1:3])
    return argmax.numpy(), tf.squeeze(coords).numpy()

  def get_action(self, s_0):
    """tf.multinomial doesn't work in eager as of v1.8"""
    s_0 = np.expand_dims(s_0, axis=0) # convert to batch
    s_0 = tf.constant(s_0, dtype=tf.float32)
    p, v = self.brain(s_0)
    p = p.numpy().squeeze()
    choice = np.random.choice(p.shape[0], p=p)
    coords = np.unravel_index(choice, s_0.shape[1:3])
    return choice, coords

  def train(self):
    # if self.batch_size >= len(self.memory):
    #   return
    # print("training")
    # s_0, a_0, s_1, r_1, done = self.memory.sample(self.batch_size)
    s_0, a_0, s_1, r_1, done = self.memory.sample()
    self.brain.train(s_0=s_0, a_0=a_0, s_1=s_1, r_1=r_1, done=done, num_actions=28*28,
                     p_optimizer=self.action_optimizer, v_optimizer=self.value_optimizer)


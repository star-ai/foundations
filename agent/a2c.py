from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from brain.screen_only import ScreenSelectAndMoveBrain
from memory.memory import ReplayMemory, Transition

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from util.tf_helpers import TFHelper

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
    super(A2CMoveToBeacon, self).__init__()
    self.action_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    self.value_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    self.brain = ScreenSelectAndMoveBrain()
    self.memory = ReplayMemory(capacity=10)

    self.s_0 = None
    self.a_0 = None
    self.s_1 = None
    self.r_1 = None
    self.done = None

  def step(self, obs):
    super(A2CMoveToBeacon, self).step(obs)

    self.s_1 = obs.observation.feature_screen.player_relative  #make it simple for now
    self.s_1 = np.expand_dims(self.s_1, axis=2)

    self.r_1 = obs.reward
    self.done = obs.reward == 1

    if self.s_0 is not None and self.a_0 is not None:
      a = Transition(self.s_0, self.a_0, self.s_1, self.r_1, self.done)
      self.memory.push(a)
      self.s_0 = self.s_1
      self.a_0 = None

    if self.s_0 is None:
      self.s_0 = self.s_1

    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      # print("reward:", obs.reward)
      if not beacon:
        return FUNCTIONS.no_op()

      action = self.get_action(self.s_0)
      print("Move action:", action)
      self.a_0 = action
      # beacon_center = np.mean(beacon, axis=0).round()
      return FUNCTIONS.Move_screen("now", action)
    else:
      return FUNCTIONS.select_army("select")

  def get_best_action(self, s_0):
    s_0 = np.expand_dims(s_0, axis=0) # convert to batch
    s_0 = tf.constant(s_0, dtype=tf.float32)
    p, v = self.brain(s_0)
    coords = TFHelper.argmax_2D(p)
    return coords.numpy().squeeze()

  def get_action(self, s_0):
    s_0 = np.expand_dims(s_0, axis=0) # convert to batch
    s_0 = tf.constant(s_0, dtype=tf.float32)
    p, v = self.brain(s_0)
    coords = TFHelper.choose_2D(p)
    return coords

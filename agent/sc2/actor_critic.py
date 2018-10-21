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
from memory.sarsd import SARSD

tfe.enable_eager_execution()

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

class ActorCriticMoveToBeacon(base_agent.BaseAgent):
  """A2C agent for move to beacon"""
  def __init__(self):
    super().__init__()
    self.brain = ScreenSelectAndMoveLearner()
    self.memory = EpisodicMemory(accum_reward=False)
    self.sarsd = SARSD()

  def reset(self):
    super().reset()
    self.sarsd.reset()
    print("Total Steps:", self.steps)
    if len(self.memory) > 0:
      self.train()

  def step(self, obs):
    super().step(obs)

    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      # Only record observations if army is selected
      state = obs.observation.feature_screen.player_relative  # make it simple for now
      state = np.expand_dims(state, axis=2)

      reward = obs.reward
      done = obs.reward == 1

      if hasattr(self, "memory"):
        transition = self.sarsd.observe(state, reward, done)
        if transition is not None: self.memory.push(transition)

      # get action
      action, coords = self.get_action(state)

      self.sarsd.act(action)
      return FUNCTIONS.Move_screen("now", coords)
    else:
      return FUNCTIONS.select_army("select")



  def get_best_action(self, state):
    state = np.expand_dims(state, axis=0) # convert to batch
    state = tf.constant(state, dtype=tf.float32)
    p = self.brain.P(state)
    argmax = tf.argmax(p, axis=1)
    coords = tf.unravel_index(argmax, state.shape[1:3])
    return argmax.numpy(), tf.squeeze(coords).numpy()

  def get_action(self, state):
    """tf.multinomial doesn't work in eager as of v1.8"""
    state = np.expand_dims(state, axis=0) # convert to batch
    state = tf.constant(state, dtype=tf.float32)
    p = self.brain.P(state)
    p = p.numpy().squeeze()
    choice = np.random.choice(p.shape[0], p=p)
    coords = np.unravel_index(choice, state.shape[1:3])
    return choice, (coords[1], coords[0]) # seems like pysc2 uses Y before X as coord

  def train(self):
    # if self.batch_size >= len(self.memory):
    #   return
    # print("training")
    # s_0, a_0, s_1, r_1, done = self.memory.sample(self.batch_size)
    s_0, a_0, s_1, r_1, done = self.memory.sample()
    self.brain.train(s_0=s_0, a_0=a_0, s_1=s_1, r_t=r_1, done=done, num_actions=6 * 6)


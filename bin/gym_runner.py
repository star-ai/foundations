#!/usr/bin/python
"""Run a gym agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

import time
import gym
from visualisation.plotting import plot_data
import os
from gym import logger as gymlogger

gymlogger.set_level(40) #error only
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # force no GPU


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("load", True, "Whether to render with pygame.")
flags.DEFINE_bool("test", True, "Whether to do a test run after run loop, only works when parallel=1.")
flags.DEFINE_integer("test_episodes", 0, "Test episodes.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
# flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 40, "Total episodes.")

flags.DEFINE_string("agent", "agent.gym.dpg.DPGActorCritic",
                    "Which agent to run, as a python path to an Agent class.")

# Render fails if there are more than 1 instances running in parallel
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

"""
Discrete action envs
[CartPole-v0, Acrobot-v1, MountainCar-v0, LunarLander-v2]
Continuous action envs
[MountainCarContinuous-v0, Pendulum-v0, BipedalWalker-v2, BipedalWalkerHardcore-v2, CarRacing-v0, LunarLanderContinuous-v2]
"""
flags.DEFINE_string("env", "Pendulum-v0", "Name of a env to use.")

agent_rewards = []

def run_thread(agent_class):
  """Run one thread worth of the environment with agent."""
  env = gym.make(FLAGS.env)
  agent = agent_class()
  run_loop(agent, env, FLAGS.max_agent_steps, FLAGS.max_episodes)

def run_loop(agent, env, max_steps=0, max_episodes=0):
  """A run loop to have agents and an environment interact."""
  total_steps = 0
  total_episodes = 0
  start_time = time.time()

  agent.setup(env.observation_space, env.action_space)
  save_name = "" + FLAGS.agent + "_" + FLAGS.env
  if FLAGS.load and FLAGS.parallel == 1 and os.path.isdir("data/" + save_name):
    print("loading:", save_name)
    agent.load(save_name)

  try:
    while not max_episodes or total_episodes < max_episodes:
      total_episodes += 1
      if (total_episodes + 1) % 1000 == 0: print("Episode:", total_episodes+1)
      observation = env.reset()
      observation = (observation, 0, False)
      agent.reset()
      while True:
        if observation[2] is False:
          total_steps += 1

        if FLAGS.render:
          env.render()

        actions = agent.step(observation)
        if max_steps and total_steps >= max_steps:
          break
        if observation[2] is True: #Done
          print("Episode:", total_episodes, "Total_steps:", total_steps, "Reward:", agent.reward)
          break
        observation = env.step(actions)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f steps/sec" % (
      elapsed_time, total_steps, total_steps / elapsed_time))
    agent_rewards.append(agent.reward_history)

    if FLAGS.parallel == 1:
      print("saving:", save_name)
      agent.save(save_name)
      if FLAGS.test:
        test_agent(agent, env)


def test_agent(agent, env):
  """A run loop to have agents and an environment interact."""
  for i in range(FLAGS.test_episodes):
    total_steps = 0
    try:
      observation = env.reset()
      observation = (observation, 0, False)
      agent.reset()
      agent.training = False
      while True:
        total_steps += 1
        env.render()

        actions = agent.step(observation)
        if observation[2] is True: #Done
          break
        observation = env.step(actions)
    finally:
      print("Test Total Steps:", total_steps, " Reward:", agent.reward)
  env.close()

def main(unused_argv):
  """Run an agent."""
  print("Running agent {} in environment {}".format(FLAGS.agent, FLAGS.env))

  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_class = getattr(importlib.import_module(agent_module), agent_name)

  threads = []
  for _ in range(FLAGS.parallel - 1):
    t = threading.Thread(target=run_thread,
                         args=(agent_class,))
    threads.append(t)
    t.start()

  run_thread(agent_class)

  for t in threads:
    t.join()

  plot_data([agent_rewards], ["Reward / Episode"])

def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)

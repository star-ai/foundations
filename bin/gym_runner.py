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
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # force no GPU


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
# flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 100, "Total episodes.")

flags.DEFINE_string("agent", "agent.gym.reinforce.REINFORCE",
                    "Which agent to run, as a python path to an Agent class.")

flags.DEFINE_integer("parallel", 10, "How many instances to run in parallel.")

#[CartPole-v0]
flags.DEFINE_string("env", "CartPole-v0", "Name of a env to use.")


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

  try:
    while not max_episodes or total_episodes < max_episodes:
      total_episodes += 1
      observation = env.reset()
      observation = (observation, 0, False)
      agent.reset()
      while True:
        total_steps += 1
        if FLAGS.render:
          env.render()

        actions = agent.step(observation)
        if max_steps and total_steps >= max_steps:
          break
        if observation[2] is True: #Done
          break
        observation = env.step(actions)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f sps" % (
        elapsed_time, total_steps, total_steps / elapsed_time))
    agent_rewards.append(agent.reward_history)



def main(unused_argv):
  """Run an agent."""

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
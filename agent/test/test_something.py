import unittest
import importlib
import threading
import sys

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
FLAGS(sys.argv)


class TestSomething(unittest.TestCase):
  # def __init__(self):
  #   pass

  def setUp(self):
    pass

  def test_split(self):
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    print(agent_module, agent_name)


if __name__ == "__main__":
  app.run(unittest.main())


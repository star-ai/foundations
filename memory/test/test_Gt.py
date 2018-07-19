import unittest
from memory.memory import Gt
import numpy as np

class TestEpisodeMemory(unittest.TestCase):
  def setUp(self):
    self.Gt = Gt()


  def test_append(self):
    r = np.arange(5)
    r = r*2
    mu = self.Gt.add_episode_rewards(r)
    self.assertEqual(mu[1], 2)
    self.assertEqual(mu[4], 8)


    r = np.arange(8)
    mu = self.Gt.add_episode_rewards(r)

    self.assertEqual(len(self.Gt.all_returns[0]), 2)
    self.assertEqual(len(self.Gt.all_returns[4]), 2)
    self.assertEqual(len(self.Gt.all_returns[5]), 1)

    self.assertEqual(mu[1], 1.5)
    self.assertEqual(mu[6], 6)

if __name__ == "__main__":
  unittest.main()
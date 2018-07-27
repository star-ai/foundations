import unittest
from memory.memory import EpisodicMemory, Transition


class TestEpisodeMemory(unittest.TestCase):
  def setUp(self):
    self.memory = EpisodicMemory()

  def test_append(self):
    for i in range(20):
      a = Transition([0, 1, 2, 3], [0, 1], [4, 5, 6, 7], 1, True)
      self.memory.push(a)
    self.assertEqual(len(self.memory), 20)

  def test_sample(self):
    for i in range(10):
      a = Transition([0, 1, 2, i], [0, 1], [4, 5, 6, i*i], 1, True)
      self.memory.push(a)

    s, a, s1, r, done = self.memory.sample()
    self.assertEqual(s.shape, (10, 4))
    self.assertEqual(a.shape, (10, 2))
    self.assertEqual(s1.shape, (10, 4))
    self.assertEqual(r.shape, (10, 1))
    self.assertEqual(done.shape, (10, 1))
    self.assertAlmostEqual(r[0][0], 9.561792, 6)

  def test_no_reward_accumulation(self):
    self.memory = EpisodicMemory(accum_reward=False)
    for i in range(10):
      a = Transition([0, 1, 2, i], [0, 1], [4, 5, 6, i*i], 1, True)
      self.memory.push(a)

    s, a, s1, r, done = self.memory.sample()
    self.assertAlmostEqual(r[0][0], 1.0)


if __name__ == "__main__":
  unittest.main()
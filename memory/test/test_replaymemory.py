import unittest
from memory.memory import ReplayMemory, Transition


class TestReplayMemory(unittest.TestCase):
  def setUp(self):
    self.memory = ReplayMemory(capacity=10)

  def test_append(self):
    for i in range(20):
      a = Transition([0, 1, 2, 3], [0, 1], [4, 5, 6, 7], 0, True)
      self.memory.push(a)
    self.assertEqual(len(self.memory), 10)

  def test_sample(self):
    for i in range(10):
      a = Transition([0, 1, 2, i], [0, 1], [4, 5, 6, i*i], 0, True)
      self.memory.push(a)

    s, a, s1, r, done = self.memory.sample(2)
    self.assertEqual(s.shape, (2, 4))
    self.assertEqual(a.shape, (2, 2))
    self.assertEqual(s1.shape, (2, 4))
    self.assertEqual(r.shape, (2, 1))
    self.assertEqual(done.shape, (2, 1))

if __name__ == "__main__":
  unittest.main()
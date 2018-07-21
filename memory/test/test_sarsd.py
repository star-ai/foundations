import unittest

from memory.sarsd import SARSD


class TestSARSD(unittest.TestCase):
  def setUp(self):
    self.sarsd = SARSD()

  def test_add(self):

    t = self.sarsd.observe(2, 0, False)
    self.assertTrue(t is None)

    self.sarsd.act(5)

    t = self.sarsd.observe(1, 1, False)
    self.assertTrue(t is not None)

    # print(self.sarsd)



if __name__ == '__main__':
    unittest.main()
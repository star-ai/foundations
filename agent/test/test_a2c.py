import unittest


from agent.a2c import A2CMoveToBeacon
import numpy as np


class TestA2CMoveToBeacon(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def setUp(self):
    self.agent = A2CMoveToBeacon()


  @unittest.skip("manual test: can't seriously test this one cus the model gives random results")
  def test_select_best_action(self):
    s = np.zeros((5, 5, 1))
    s[0][3][0] = 1.0
    a = self.agent.get_best_action(s)

    print(a)

  @unittest.skip("manual test: can't seriously test this one cus the model gives random results")
  def test_select_action(self):
    s = np.zeros((5, 5, 1))
    s[0][3][0] = 1.0
    a = self.agent.get_action(s)

    print(a)

if __name__ == "__main__":
  unittest.main()


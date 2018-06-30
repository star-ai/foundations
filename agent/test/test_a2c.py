import unittest


from agent.a2c import A2CMoveToBeacon



class TestA2CMoveToBeacon(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def setUp(self):
    pass

  def test_split(self):
    self.assertTrue(1 == 1, "doh")


if __name__ == "__main__":
  unittest.main()


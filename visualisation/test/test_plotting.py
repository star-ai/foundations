import unittest
import numpy as np

from visualisation.plotting import plot_data


class TestPlotting(unittest.TestCase):
  def setUp(self):
    pass

  def test_plotting(self):
    data = []
    sample = np.arange(20)

    for i in range(-4, 5):
      data.append((sample+i).tolist())

    print(np.mean(data, axis=0))
    print(np.std(data, axis=0))
    print(np.array(data).shape)

    data2 = (np.array(data)*2).tolist()
    # print(data2)

    plot_data([data, data2], ["dataset1", "dataset2"])



if __name__ == "__main__":
  unittest.main()

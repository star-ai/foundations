import numpy as np
import matplotlib.pyplot as plt

def plot_data(data, names):
  plots = len(data)

  fig, axs = plt.subplots(plots, 1)
  if plots == 1:
    axs = [axs]

  for i in range(plots):
    series = np.array(data[i])
    # print("series:", series.shape)
    mu = np.mean(series, axis=0)
    # print("mu:", mu.shape)
    std = np.std(series, axis=0)
    axs[i].plot(mu)
    x = np.arange(len(mu))
    axs[i].fill_between(x, mu - 2*std, mu + 2*std, color='gray', alpha=0.2)

    axs[i].set_title(names[i])
  fig.tight_layout()
  plt.show()
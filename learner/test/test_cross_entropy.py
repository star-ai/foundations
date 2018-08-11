import unittest

import os
import numpy as np
import tensorflow.contrib.eager as tfe
import tensorflow as tf

tfe.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class TestCrossEntropy(unittest.TestCase):

  def cross_entropy(self, logits, labels):
    """
    Cross Entropy of (x, y)  which != Cross Entropy of (y, x) 
    """
    return - np.sum(labels * np.log(logits))

  def test_cross_entropy(self):

    x = np.array([.2, .2, .5, .1])
    y = np.array([.01, .97, .01, .01])
    ce = self.cross_entropy(x, y)
    print("cross_entropy", ce)

    x = np.array([.2, .2, .5, .1])
    y = np.array([.1, .4, .3, .2])
    ce = self.cross_entropy(x, y)
    print("cross_entropy", ce)

  def entropy(self, labels):
    """
    Entropy of x
    """
    return - np.sum(np.where(labels != 0, labels * np.log(labels), 0))

  def test_entropy(self):
    x = np.array([.01, .97, .01, .01])
    entropy = self.entropy(x)

    print("entropy:", entropy)

  def kl_divergence(self, logits, labels):
    """
    KL Divergence (x || y)
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    kl divergence = - np.sum(np.where(logits != 0, logits * np.log(labels / logits), 0))
    or 
    kl divergence = np.sum(np.where(labels != 0, logits * np.log(logits / labels), 0))
    
    we use the first one as labels will tend to be 0
    """
    # return - np.sum(np.where(labels != 0, logits * np.log(labels / logits), 0))
    return np.sum(np.where(labels != 0, labels * np.log(labels / logits), 0))

  def test_kl_divergence(self):

    x = np.array([.2, .2, .5, .1])
    y = np.array([.01, .97, .01, .01])
    kl_divergence = self.kl_divergence(x, y)
    print("kl_divergence", kl_divergence)

    x = np.array([.2, .2, .5, .1])
    y = np.array([.1, .4, .3, .2])
    kl_divergence = self.kl_divergence(x, y)
    print("kl_divergence", kl_divergence)





  def test_cross_entropy_formula(self):
    x = np.array([.2, .2, .5, .1])
    y = np.array([.01, .97, .01, .01])
    cross_entropy = self.cross_entropy(x, y)
    print("cross_entropy verification", cross_entropy)

    entropy = self.entropy(y)
    print("cross_entropy entropy", entropy)

    kl_divergence = self.kl_divergence(x, y)
    print("cross_entropy kl divergence", kl_divergence)

    print("cross_entropy verification", entropy + kl_divergence)




if __name__ == "__main__":
  unittest.main()
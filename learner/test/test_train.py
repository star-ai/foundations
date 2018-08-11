import unittest

import os
import numpy as np
import tensorflow.contrib.eager as tfe
import tensorflow as tf

tfe.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class TestTrain(unittest.TestCase):

  def test_secondary_gradient(self):
    def P(state):
      p = tf.multiply(state, 2)
      p = tf.add(p, 10)
      return p

    def Q(action):
      return tf.multiply(action, 3)

    state = tf.constant(1.)

    with tfe.GradientTape() as tape:
      # with tfe.GradientTape() as dqda_tape:
      tape.watch(state)
      a = P(state) #12
      print("a", a)

      with tfe.GradientTape() as dqda_tape:
        dqda_tape.watch(a)
        q = Q(a)
        print("Q", q)

      dqda = dqda_tape.gradient(q, a)
      print("DQDA", dqda)
      # print(a)
      # print(dqda*a)
      loss = -dqda * a
      print("Loss:", loss)
    grads = tape.gradient(loss, state)
    print(grads)


if __name__ == "__main__":
  unittest.main()
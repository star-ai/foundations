import tensorflow as tf
import numpy as np
class TFHelper():

  @staticmethod
  def argmax_2D(matrix):
    """
      Returns the x, y of matrix [batch, x, y, 1] matrix 
    """
    flat = tf.reshape(matrix, (matrix.shape[0], -1))
    argmax = tf.argmax(flat, axis=1)
    argmax_index = tf.unravel_index(argmax, matrix.shape)
    coords = tf.transpose(argmax_index[1:3, :])
    return coords

  @staticmethod
  def choose_2D(probability_matrix):
    """
      Returns a random selection of x, y of matrix [1, x, y, 1] matrix based on probabilities
      !!! This sux as we're forced to convert to numpy, tf.multinomial is not available in eager
    """
    # TODO: do it for batches
    flat = tf.reshape(probability_matrix, (probability_matrix.shape[0], -1))
    flat = flat.numpy().squeeze()
    choice = np.random.choice(flat.shape[0], p=flat)
    index = np.unravel_index(choice, probability_matrix.shape)
    # print(choice, index)
    coords = np.transpose(index[1:3])
    # print(coords)
    return coords

  @staticmethod
  def coords_to_2D_onehot(coords, shape):
    """
    convert coordinates from [x, y] to a 2D one hot matrix
    :param coords: [batch, [x, y]]
    :param shape: (batch, w, h, 1)
    :return: 
    """
    indices = tf.constant([[]])
    print(coords, shape)

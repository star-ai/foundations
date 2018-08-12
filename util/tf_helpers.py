import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os


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
    # indices = tf.constant([[]])
    # print(coords, shape)
    pass

  @staticmethod
  def get_action(policy_func, actions, s_0):
    s_0 = tf.constant([s_0], dtype=tf.float32)
    action_probability = policy_func(s_0).numpy()
    action = np.random.choice(actions, p=action_probability[0])
    return action


  @staticmethod
  def update_target_model(from_model, target_model, tau):
    var_from = from_model.variables
    var_target = target_model.variables
    # print("var_from", var_from)
    # print("var_target", var_target)
    for i in range(len(var_from)):
      var_target[i].assign(var_target[i] * (1 - tau) + var_from[i] * tau)

    # print("var_target", var_target)

  @staticmethod
  def save_eager(name, model, optimizer=None):
    checkpoint_dir = "./data/" + name
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, name + "_ckpt")
    if optimizer:
      root = tfe.Checkpoint(optimizer=optimizer, model=model)
    else:
      root = tfe.Checkpoint(model=model)

    root.save(file_prefix=checkpoint_prefix)

  @staticmethod
  def load_eager(name, model, optimizer=None):
    checkpoint_dir = "./data/" + name
    checkpoint_prefix = os.path.join(checkpoint_dir, name + "_ckpt")
    if optimizer:
      root = tfe.Checkpoint(optimizer=optimizer, model=model)
    else:
      root = tfe.Checkpoint(model=model)

    root.restore(tf.train.latest_checkpoint(checkpoint_dir))

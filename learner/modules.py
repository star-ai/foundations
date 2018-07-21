import tensorflow as tf

"""Standard Convolution + BatchNorm + Relu"""
class ConvModule(tf.keras.Model):
  def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1),
               activation=tf.nn.relu, *args, **kwargs):
    super().__init__(self, *args, **kwargs)

    # TODO: Should we be setting use_bias to be false?
    self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                       dilation_rate=dilation_rate, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization()
    self.activation = activation

  def call(self, inputs, training=False):
    x = self.conv(inputs)
    x = self.bn(x, training=training)
    if self.activation is not None:
      x = self.activation(x)
    return x


class DenseModule(tf.keras.Model):
  def __init__(self, units, activation=tf.nn.relu, *args, **kwargs):
    super().__init__(self, *args, **kwargs)

    self.dense = tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2)
    self.bn = tf.keras.layers.BatchNormalization()
    self.activation = activation

  def call(self, inputs, training=False):
    x = self.dense(inputs)
    x = self.bn(x, training=training)
    if self.activation is not None:
      x = self.activation(x)
    return x


def softmax2D(inputs):
  x = tf.reshape(inputs, (inputs.shape[0], -1))
  x = tf.nn.softmax(x)
  x = tf.reshape(x, inputs.shape)
  return x
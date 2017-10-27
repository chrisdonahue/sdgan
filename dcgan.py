import numpy as np
import tensorflow as tf

_DTYPE = tf.float32


"""
  DCGAN 64x64 generator
  z shape during training: [b * k, d_i + d_o]
  z shape during inference: [None, d_i + d_o]
"""
def DCGANGenerator64x64(z, nch, dim=128, batchnorm=True, train=True):
  _G_DIM_MUL = [8, 4, 2, 1]
  _G_IMG_DIM_INIT = 4

  batch_size = tf.shape(z)[0]

  weight_limit = 0.02 * np.sqrt(3)
  weight_init = tf.random_uniform_initializer(-weight_limit, weight_limit, dtype=_DTYPE)

  outputs = z

  # Project each z to [4, 4, dim * 8]
  with tf.variable_scope('z_project'):
    outputs = tf.layers.dense(
        outputs,
        dim * _G_DIM_MUL[0] * _G_IMG_DIM_INIT * _G_IMG_DIM_INIT,
        kernel_initializer=weight_init)
    outputs = tf.reshape(outputs, [batch_size, _G_IMG_DIM_INIT, _G_IMG_DIM_INIT, dim * _G_DIM_MUL[0]])
    if batchnorm:
      outputs = tf.layers.batch_normalization(outputs, training=train)
  outputs = tf.nn.relu(outputs)

  # Upscale to [32, 32, dim]
  for i, dim_mul in enumerate(_G_DIM_MUL[1:]):
    with tf.variable_scope('upconv_2d_{}'.format(i)):
      outputs = tf.layers.conv2d_transpose(
          outputs,
          dim * dim_mul,
          [5, 5],
          strides=(2, 2),
          padding='SAME',
          kernel_initializer=weight_init)
      if batchnorm:
        outputs = tf.layers.batch_normalization(outputs, training=train)
    outputs = tf.nn.relu(outputs)

  # Upscale to [64, 64, nch] on [-1, 1]
  with tf.variable_scope('upconv_2d_3'):
    outputs = tf.layers.conv2d_transpose(
        outputs,
        nch,
        [5, 5],
        strides=(2, 2),
        padding='SAME',
        kernel_initializer=weight_init)
  outputs = tf.nn.tanh(outputs)

  return outputs


"""
  SD-DCGAN 64x64 discriminator
  x (real or fake) is [b, k, 64, 64, 3]
"""
def SDDCGANDiscriminator64x64(x, dim=128, batchnorm=True, siamese=True, collate='conv_1'):
  _D_DIM_MUL = [1, 2, 4, 8]
  def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

  batch_size = int(x.get_shape()[0])

  weight_limit = 0.02 * np.sqrt(3)
  weight_init = tf.random_uniform_initializer(-weight_limit, weight_limit, dtype=_DTYPE)

  # Split along tuple dimension k
  inputs_instances = tf.split(x, int(x.get_shape()[1]), axis=1)
  inputs_instances = [tf.squeeze(x, axis=1) for x in inputs_instances]

  # Merge channels if not using Siamese architecture
  if not siamese:
    inputs_instances = [tf.concat(inputs_instances, axis=3)]

  # Separately encode k images in tuple
  outputs_instances = []
  reuse = False
  for outputs in inputs_instances:
    # Downscale to [32, 32, dim]
    with tf.variable_scope('downconv_2d_0', reuse=reuse):
      outputs = tf.layers.conv2d(
          outputs,
          dim * _D_DIM_MUL[0],
          [5, 5],
          strides=(2, 2),
          padding='SAME',
          kernel_initializer=weight_init)
    outputs = lrelu(outputs)

    # Downscale to [4, 4, dim * 8]
    for i, dim_mul in enumerate(_D_DIM_MUL[1:]):
      with tf.variable_scope('downconv_2d_{}'.format(i + 1), reuse=reuse):
        outputs = tf.layers.conv2d(
            outputs,
            dim * dim_mul,
            [5, 5],
            strides=(2, 2),
            padding='SAME',
            kernel_initializer=weight_init)
        # TODO: Layer norm instead? (iwgan)
        if batchnorm:
          outputs = tf.layers.batch_normalization(outputs, training=True)
      outputs = lrelu(outputs)

    outputs_instances.append(outputs)
    reuse = True

  # Collate encoder outputs (using either dense or conv layers)
  if collate.startswith('dense_'):
    nlayers = int(collate.split('_')[1])

    outputs_flattened = []
    for outputs in outputs_instances:
      outputs = tf.reshape(outputs, [batch_size, -1])
      outputs_flattened.append(outputs)

    outputs = tf.concat(outputs_flattened, axis=1)

    for i in range(nlayers):
      with tf.variable_scope('collate_dense_{}'.format(i)):
        outputs = tf.layers.dense(
            outputs,
            int(outputs.get_shape()[1]) // 2,
            kernel_initializer=weight_init)
        if batchnorm:
          outputs = tf.layers.batch_normalization(outputs, training=True)
      outputs = lrelu(outputs)
  elif collate.startswith('conv_'):
    nlayers = int(collate.split('_')[1])
    assert nlayers <= 2

    outputs = tf.concat(outputs_instances, axis=3)

    dim_mul = _D_DIM_MUL[-1]
    for i in range(nlayers):
      with tf.variable_scope('collate_conv_{}'.format(i)):
        outputs = tf.layers.conv2d(
            outputs,
            dim * dim_mul,
            [3, 3],
            strides=(2, 2),
            padding='SAME',
            kernel_initializer=weight_init)
        if batchnorm:
          outputs = tf.layers.batch_normalization(outputs, training=True)
      outputs = lrelu(outputs)
      dim_mul /= 2

    outputs = tf.reshape(outputs, [batch_size, -1])
  else:
    raise NotImplementedError()

  # Concatenate one-hot labels
  with tf.variable_scope('classify'):
    outputs = tf.layers.dense(outputs, 1, kernel_initializer=weight_init)[:, 0]

  return outputs

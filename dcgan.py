import numpy as np
import tensorflow as tf

_DTYPE = tf.float32

# TODO: mirror padding?
_G_DIM_MUL = [8, 4, 2, 1]
_G_IMG_DIM_INIT = 4
def create_generator(z, nch, dim=64, batchnorm=False, train=True):
  batch_size = tf.shape(z)[0]

  weight_limit = 0.02 * np.sqrt(3)
  weight_init = tf.random_uniform_initializer(-weight_limit, weight_limit, dtype=_DTYPE)

  outputs = z

  with tf.variable_scope('z_project'):
    outputs = tf.layers.dense(outputs, dim * _G_DIM_MUL[0] * _G_IMG_DIM_INIT * _G_IMG_DIM_INIT, kernel_initializer=weight_init)
    outputs = tf.reshape(outputs, [batch_size, _G_IMG_DIM_INIT, _G_IMG_DIM_INIT, dim * _G_DIM_MUL[0]])
    if batchnorm:
      outputs = tf.layers.batch_normalization(outputs, training=train)
  outputs = tf.nn.relu(outputs)

  for i, dim_mul in enumerate(_G_DIM_MUL[1:]):
    with tf.variable_scope('upconv_2d_{}'.format(i)):
      outputs = tf.layers.conv2d_transpose(outputs, dim * dim_mul, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=weight_init)
      if batchnorm:
        outputs = tf.layers.batch_normalization(outputs, training=train)
    outputs = tf.nn.relu(outputs)

  with tf.variable_scope('upconv_2d_3'):
    outputs = tf.layers.conv2d_transpose(outputs, nch, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=weight_init)
  outputs = tf.nn.tanh(outputs)

  return outputs


def lrelu(x, alpha=0.2):
  return tf.maximum(alpha * x, x)


_D_DIM_MUL = [1, 2, 4, 8]
def create_discriminator(
    G_z_or_x,
    y=None,
    decomp_strat=None,
    nidentities=None,
    dim=64,
    batchnorm=True,
    siamese_strat='tt',
    collate_strat='conv_1',
    train=True):
  batch_size = int(G_z_or_x.get_shape()[0])
  _IMG_DIM = int(G_z_or_x.get_shape()[2])

  weight_limit = 0.02 * np.sqrt(3)
  weight_init = tf.random_uniform_initializer(-weight_limit, weight_limit, dtype=_DTYPE)

  inputs_instances = tf.split(G_z_or_x, int(G_z_or_x.get_shape()[1]), axis=1)
  inputs_instances = [tf.squeeze(x, axis=1) for x in inputs_instances]
  if siamese_strat == 'tt':
    pass
  elif siamese_strat == 'sc':
    inputs_instances = [tf.concat(inputs_instances, axis=3)]
  else:
    raise NotImplementedError()

  output_dim = _IMG_DIM / (2 ** len(_D_DIM_MUL))
  outputs_instances = []
  reuse = False
  for outputs in inputs_instances:
    with tf.variable_scope('downconv_2d_0', reuse=reuse):
      outputs = tf.layers.conv2d(outputs, dim * _D_DIM_MUL[0], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=weight_init)
    outputs = lrelu(outputs)

    for i, dim_mul in enumerate(_D_DIM_MUL[1:]):
      with tf.variable_scope('downconv_2d_{}'.format(i + 1), reuse=reuse):
        outputs = tf.layers.conv2d(outputs, dim * dim_mul, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=weight_init)
        # TODO: layer norm? (iwgan)
        if batchnorm:
          outputs = tf.layers.batch_normalization(outputs, training=train)
      outputs = lrelu(outputs)

    # TODO: dense layers in two towers?

    outputs_instances.append(outputs)
    reuse = True

  if collate_strat.startswith('dense_'):
    nlayers = int(collate_strat.split('_')[1])

    outputs_flattened = []
    for outputs in outputs_instances:
      outputs = tf.reshape(outputs, [batch_size, output_dim * output_dim * _D_DIM_MUL[-1] * dim])
      outputs_flattened.append(outputs)

    outputs = tf.concat(outputs_flattened, axis=1)

    for i in range(nlayers):
      with tf.variable_scope('collate_dense_{}'.format(i)):
        outputs = tf.layers.dense(outputs, int(outputs.get_shape()[1]) // 2, kernel_initializer=weight_init)
        if batchnorm:
          outputs = tf.layers.batch_normalization(outputs, training=train)
      outputs = lrelu(outputs)

  elif collate_strat.startswith('conv_'):
    nlayers = int(collate_strat.split('_')[1])
    assert nlayers <= 2

    outputs = tf.concat(outputs_instances, axis=3)

    dim_mul = _D_DIM_MUL[-1]
    for i in range(nlayers):
      with tf.variable_scope('collate_conv_{}'.format(i)):
        outputs = tf.layers.conv2d(outputs, dim * dim_mul, [3, 3], strides=(2, 2), padding='SAME', kernel_initializer=weight_init)
        if batchnorm:
          outputs = tf.layers.batch_normalization(outputs, training=train)
      outputs = lrelu(outputs)
      dim_mul /= 2

    outputs = tf.reshape(outputs, [batch_size, -1])

  else:
    raise NotImplementedError()

  # Concatenate one-hot labels
  if decomp_strat == 'mirza_cgan':
    y_onehot = tf.one_hot(y, nidentities, dtype=_DTYPE)
    outputs = tf.concat([outputs, y_onehot], axis=1)
    with tf.variable_scope('cgan_collate_dense_0'):
      outputs = tf.layers.dense(outputs, 1024, kernel_initializer=weight_init)

  if decomp_strat in ['springenberg_sgan', 'odena_acgan']:
    nlogits = 1 + nidentities
  else:
    nlogits = 1

  with tf.variable_scope('classify'):
    outputs = tf.layers.dense(outputs, nlogits, kernel_initializer=weight_init)

  return outputs


def create_encoder(x, zi_dim, dim=64, batchnorm=True, train=True):
  batch_size = tf.shape(x)[0]
  _IMG_DIM = int(x.get_shape()[2])

  weight_limit = 0.02 * np.sqrt(3)
  weight_init = tf.random_uniform_initializer(-weight_limit, weight_limit, dtype=_DTYPE)

  output_dim = _IMG_DIM / (2 ** len(_D_DIM_MUL))

  outputs = x

  with tf.variable_scope('downconv_2d_0'):
    outputs = tf.layers.conv2d(outputs, dim * _D_DIM_MUL[0], [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=weight_init)
  outputs = lrelu(outputs)

  for i, dim_mul in enumerate(_D_DIM_MUL[1:]):
    with tf.variable_scope('downconv_2d_{}'.format(i + 1)):
      outputs = tf.layers.conv2d(outputs, dim * dim_mul, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=weight_init)
      # TODO: layer norm? (iwgan)
      if batchnorm:
        outputs = tf.layers.batch_normalization(outputs, training=train)
    outputs = lrelu(outputs)

  outputs = tf.reshape(outputs, [batch_size, output_dim * output_dim * _D_DIM_MUL[-1] * dim])

  with tf.variable_scope('encoding'):
    outputs = tf.layers.dense(outputs, zi_dim, kernel_initializer=weight_init)

  outputs = tf.nn.tanh(outputs)

  return outputs

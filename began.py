# Modified from https://github.com/carpedm20/BEGAN-tensorflow

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


"""
  BEGAN 64x64 generator
  z shape during training: [b * k, d_i + d_o]
  z shape during inference: [None, d_i + d_o]
"""
def BEGANGenerator64x64(z, nch, hidden_num=128):
  repeat_num = 4
  data_format = 'NHWC'

  num_output = int(np.prod([8, 8, hidden_num]))
  x = slim.fully_connected(z, num_output, activation_fn=None)
  x = reshape(x, 8, 8, hidden_num, data_format)

  for idx in range(repeat_num):
    x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    if idx < repeat_num - 1:
      x = upscale(x, 2, data_format)

  out = slim.conv2d(x, nch, 3, 1, activation_fn=None, data_format=data_format)

  return out


"""
  SD-BEGAN 64x64 discriminator
  x (real or fake) is [b, k, 64, 64, 3]
"""
def SDBEGANDiscriminator64x64(x, hidden_num=128):
  repeat_num = 4
  data_format = 'NHWC'
  d_i = 50
  d_o = 50

  k = int(x.get_shape()[1])
  nch = int(x.get_shape()[4])
  xs = tf.split(x, k, axis=1)

  # Siamese encoder
  zs = []
  reuse = False
  for x in xs:
    x = x[:, 0]
    with tf.variable_scope('encoder', reuse=reuse):
      x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

      prev_channel_num = hidden_num
      for idx in range(repeat_num):
        channel_num = hidden_num * (idx + 1)
        x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        if idx < repeat_num - 1:
          x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
          #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

      x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
      z = slim.fully_connected(x, d_i + d_o, activation_fn=None)
      zs.append(z)
    reuse = True

  # Merge Siamese encodings to bottleneck
  zs = tf.concat(zs, axis=1)
  z = slim.fully_connected(zs, k * d_i + d_o, activation_fn=None)

  # Split bottleneck
  z = slim.fully_connected(z, k * (d_i + d_o), activation_fn=None)
  zs = tf.split(z, k, axis=1)

  # Siamese decoder
  D_xs = []
  reuse = False
  for z in zs:
    with tf.variable_scope('decoder', reuse=reuse):
      # Decoder
      num_output = int(np.prod([8, 8, hidden_num]))
      x = slim.fully_connected(z, num_output, activation_fn=None)
      x = reshape(x, 8, 8, hidden_num, data_format)

      for idx in range(repeat_num):
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        if idx < repeat_num - 1:
          x = upscale(x, 2, data_format)

      D_x = slim.conv2d(x, nch, 3, 1, activation_fn=None, data_format=data_format)
      D_xs.append(D_x)
    reuse=True

  D_x = tf.stack(D_xs, axis=1)

  return D_x


def int_shape(tensor):
  shape = tensor.get_shape().as_list()
  return [num if num is not None else -1 for num in shape]


def get_conv_shape(tensor, data_format):
  shape = int_shape(tensor)
  # always return [N, H, W, C]
  if data_format == 'NCHW':
    return [shape[0], shape[2], shape[3], shape[1]]
  elif data_format == 'NHWC':
    return shape


def nchw_to_nhwc(x):
  return tf.transpose(x, [0, 2, 3, 1])


def nhwc_to_nchw(x):
  return tf.transpose(x, [0, 3, 1, 2])


def reshape(x, h, w, c, data_format):
  if data_format == 'NCHW':
    x = tf.reshape(x, [-1, c, h, w])
  else:
    x = tf.reshape(x, [-1, h, w, c])
  return x


def resize_nearest_neighbor(x, new_size, data_format):
  if data_format == 'NCHW':
    x = nchw_to_nhwc(x)
    x = tf.image.resize_nearest_neighbor(x, new_size)
    x = nhwc_to_nchw(x)
  else:
    x = tf.image.resize_nearest_neighbor(x, new_size)
  return x


def upscale(x, scale, data_format):
  _, h, w, _ = get_conv_shape(x, data_format)
  return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)

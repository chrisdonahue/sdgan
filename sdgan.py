from collections import defaultdict
import cPickle as pickle
import math
import os
import time

import tensorflow as tf

from began import BEGANGenerator64x64, SDBEGANDiscriminator64x64
from dcgan import DCGANGenerator64x64, SDDCGANDiscriminator64x64
from util import str2bool, decode_png_observation, encode_png_observation


"""
  Samples a k-tuple from a group with or without replacement
  Enqueues n at a time for efficiency
"""
def group_choose_k(
    named_id_to_fps,
    k,
    n=None,
    with_replacement=False,
    capacity=4096,
    min_after_dequeue=2048,
    nthreads=4):
  assert k > 0

  # Join (variable-length) groups into CSV strings for enqueueing
  avg_group_size = int(np.ceil(np.mean([len(group_fps) for group_fps in named_id_to_fps.values()])))
  named_id_to_fps = [','.join(group_fps) for group_fps in named_id_to_fps.values()]

  # If n is None, compute a reasonable value (avg group len choose k)
  if n is None:
    f = math.factorial
    n = f(avg_group_size) / f(k) / f(avg_group_size - k)

  # Dequeue one and split it into group
  group_fps = tf.train.string_input_producer(named_id_to_fps).dequeue()
  group_fps = tf.string_split([group_fps], ',').values
  group_size = tf.shape(group_fps)[0]
  tf.summary.histogram('group_size', group_size)

  # Select some at random
  # TODO: Should be some way to sample without replacement here rather than manually filtering
  tuple_ids = tf.random_uniform([n, k], minval=0, maxval=group_size, dtype=tf.int32)

  # Count num tuples enqueued
  ntotal = tf.Variable(0)
  tf.summary.scalar('tuples_ntotal', ntotal)
  add_total = tf.assign_add(ntotal, n)

  # Filter duplicates if sampling tuples without replacement
  if not with_replacement and k > 1:
    # Find unique tuples
    tuple_unique = tf.ones([n], tf.bool)
    for i in xrange(k):
      for j in xrange(k):
        if i == j:
          continue
        pair_unique = tf.not_equal(tuple_ids[:, i], tuple_ids[:, j])
      tuple_unique = tf.logical_and(tuple_unique, pair_unique)

    # Filter tuples with duplicates
    valid_tuples = tf.where(tuple_unique)[:, 0]

    # Count num valid tuples enqueued
    nvalid = tf.Variable(0)
    tf.summary.scalar('tuples_nvalid', nvalid)
    tf.summary.scalar('tuples_valid_ratio',
        tf.cast(nvalid, tf.float32) / tf.cast(ntotal, tf.float32))
    add_valid = tf.assign_add(nvalid, tf.shape(valid_tuples)[0])

    # Gather valid ids
    with tf.control_dependencies([add_valid]):
      tuple_ids = tf.gather(tuple_ids, valid_tuples)

  # Gather valid tuples
  with tf.control_dependencies([add_total]):
    tuples = tf.gather(group_fps, tuple_ids)

  # Make batches
  tuple_q = tf.RandomShuffleQueue(capacity, min_after_dequeue, tuples.dtype, [k])
  tuple_enq = tuple_q.enqueue_many(tuples)
  tf.train.add_queue_runner(tf.train.QueueRunner(tuple_q, [tuple_enq] * nthreads))

  tf.summary.scalar('tuples_queue_size', tuple_q.size())

  return tuple_q.dequeue()


"""
  Trains an SD-GAN
"""
def train(named_id_to_fps, args):
  # Get batch of observations
  def make_batch(observations):
    queue = tf.RandomShuffleQueue(
        capacity=args.train_queue_capacity,
        min_after_dequeue=args.train_queue_min,
        shapes=[[args.train_k, args.height, args.width, args.nch]],
        dtypes=[tf.float32])

    example = tf.stack(observations, axis=0)
    enqueue_op = queue.enqueue(example)
    queue_nthreads = 4
    qr = tf.train.QueueRunner(queue, [enqueue_op] * queue_nthreads)
    tf.train.add_queue_runner(qr)

    tf.summary.scalar('queue_size', queue.size())

    return queue.dequeue_many(args.train_batch_size)

  # Load observation tuples
  with tf.name_scope('loader'):
    # Generate matched pairs of WAV fps
    with tf.device('/cpu:0'):
      tup = group_choose_k(named_id_to_fps, args.train_k, with_replacement=False)

      observations = []
      for i in xrange(args.train_k):
        observation = decode_png_observation(tup[i])
        observation.set_shape([args.height, args.width, args.nch])
        observations.append(observation)

      x = make_batch(observations)

  # Make image summaries
  for i in xrange(args.train_k):
    tf.summary.image('x_{}'.format(i), encode_png_observation(x[:, i]))

  # Make identity vector and repeat k times
  zi = tf.random_uniform([args.train_batch_size, args.model_d_i], -1.0, 1.0, dtype=tf.float32)
  zi = tf.tile(zi, [1, args.train_k])
  zi = tf.reshape(zi, [args.train_batch_size, args.train_k, args.model_d_i])

  # Draw iid observation vectors (no repetition)
  zo = tf.random_uniform([args.train_batch_size, args.train_k, args.model_d_o], -1.0, 1.0, dtype=tf.float32)

  # Concat [zi; zo]
  z = tf.concat([zi, zo], axis=2)

  # Make generator
  with tf.variable_scope('G'):
    z = tf.reshape(z, [args.train_batch_size * args.train_k, args.model_d_i + args.model_d_o])
    if args.dcgan:
      G_z = DCGANGenerator64x64(z, args.nch, dim=args.dcgan_dim, train=True)
    else:
      G_z = BEGANGenerator64x64(z, args.nch, hidden_num=args.began_nhidden)
    G_z = tf.reshape(G_z, [args.train_batch_size, args.train_k, args.height, args.width, args.nch])
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

  # Print G summary
  print '-' * 80
  print 'Generator vars'
  nparams = 0
  for v in G_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print '{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name)
  print 'Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024))

  # Make image summaries
  for i in xrange(args.train_k):
    tf.summary.image('G_z_{}'.format(i), encode_png_observation(G_z[:, i]))

  # Make real discriminator
  with tf.name_scope('D_x'), tf.variable_scope('D'):
    if args.dcgan:
      D_x = SDDCGANDiscriminator64x64(x, dim=args.dcgan_dim, siamese=args.dcgan_disc_siamese)
    else:
      D_x = SDBEGANDiscriminator64x64(x, hidden_num=args.began_nhidden)
  D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

  # Print D summary
  print '-' * 80
  print 'Discriminator vars'
  nparams = 0
  for v in D_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print '{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name)
  print 'Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024))
  print '-' * 80

  # Make fake discriminator
  with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
    if args.dcgan:
      D_G_z = SDDCGANDiscriminator64x64(G_z, dim=args.dcgan_dim, siamese=args.dcgan_disc_siamese)
    else:
      D_G_z = SDBEGANDiscriminator64x64(G_z, hidden_num=args.began_nhidden)

  # Create loss
  D_clip_weights = None
  if args.dcgan_loss == 'dcgan':
    fake = tf.zeros([args.train_batch_size], dtype=tf.float32)
    real = tf.ones([args.train_batch_size], dtype=tf.float32)

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z,
      labels=real
    ))

    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z,
      labels=fake
    ))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_x,
      labels=real
    ))

    D_loss /= 2.
  elif args.dcgan_loss == 'lsgan':
    G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
    D_loss = tf.reduce_mean((D_x - 1.) ** 2)
    D_loss += tf.reduce_mean(D_G_z ** 2)
    D_loss /= 2.
  elif args.dcgan_loss == 'wgan':
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    with tf.name_scope('D_clip_weights'):
      clip_ops = []
      for var in D_vars:
        clip_bounds = [-.01, .01]
        clip_ops.append(
          tf.assign(
            var,
            tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
          )
        )
      D_clip_weights = tf.group(*clip_ops)
  elif args.dcgan_loss == 'wgan-gp':
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1, 1, 1], minval=0., maxval=1.)
    differences = G_z - x
    interpolates = x + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = SDDCGANDiscriminator64x64(interpolates, dim=args.dcgan_dim, siamese=args.dcgan_disc_siamese)

    LAMBDA = 10
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty
  else:
    raise NotImplementedError()

  tf.summary.scalar('G_loss', G_loss)
  tf.summary.scalar('D_loss', D_loss)

  # Create optimizer
  if args.dcgan_loss == 'dcgan':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)

    D_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)
  elif args.dcgan_loss == 'lsgan':
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=1e-4)

    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=1e-4)
  elif args.dcgan_loss == 'wgan':
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)

    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)
  elif args.dcgan_loss == 'wgan-gp':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)

    D_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)
  else:
    raise NotImplementedError()

  G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
      global_step=tf.train.get_or_create_global_step())
  D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

  # Run training
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_save_secs,
      save_summaries_secs=args.train_summary_secs) as sess:
    while True:
      # Train discriminator
      for i in xrange(args.dcgan_disc_nupdates):
        sess.run(D_train_op)

        if D_clip_weights is not None:
          sess.run(D_clip_weights)

      # Train generator
      sess.run(G_train_op)


"""
  Visualizes a fixed set of random latent vectors during training
"""
def preview(args):
  from scipy.misc import imsave

  preview_dir = os.path.join(args.train_dir, 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  # Load graph
  infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
  graph = tf.get_default_graph()
  saver = tf.train.import_meta_graph(infer_metagraph_fp)

  # Generate or restore z_i and z_o
  zizo_fp = os.path.join(preview_dir, 'zizo.pkl')
  if os.path.exists(zizo_fp):
    # Restore z_i and z_o
    with open(zizo_fp, 'rb') as f:
      _samp_fetches = pickle.load(f)
  else:
    # Sample z_i and z_o
    samp_feeds = {}
    samp_feeds[graph.get_tensor_by_name('samp_zi_n:0')] = args.preview_nids
    samp_feeds[graph.get_tensor_by_name('samp_zo_n:0')] = args.preview_nobs
    samp_fetches = {}
    samp_fetches['zis'] = graph.get_tensor_by_name('samp_zi:0')
    samp_fetches['zos'] = graph.get_tensor_by_name('samp_zo:0')
    with tf.Session() as sess:
      _samp_fetches = sess.run(samp_fetches, samp_feeds)

    # Save z_i and z_o
    with open(zizo_fp, 'wb') as f:
      pickle.dump(_samp_fetches, f)

  # Set up graph for generating preview images
  feeds = {}
  feeds[graph.get_tensor_by_name('zi:0')] = _samp_fetches['zis']
  feeds[graph.get_tensor_by_name('zo:0')] = _samp_fetches['zos']
  fetches =  {}
  fetches['step'] = tf.train.get_or_create_global_step()
  grid_prev = graph.get_tensor_by_name('G_z_grid_prev:0')
  fetches['G_z_grid'] = grid_prev

  # Summarize
  fetches['G_z_grid_summary'] = tf.summary.image('preview/grid', tf.expand_dims(grid_prev, axis=0), max_outputs=1)
  summary_writer = tf.summary.FileWriter(preview_dir)

  # Loop, waiting for checkpoints
  ckpt_fp = None
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print 'Preview: {}'.format(latest_ckpt_fp)

      with tf.Session() as sess:
        saver.restore(sess, latest_ckpt_fp)

        _fetches = sess.run(fetches, feeds)

      preview_fp = os.path.join(preview_dir, '{}.png'.format(_fetches['step']))
      imsave(preview_fp, _fetches['G_z_grid'])

      summary_writer.add_summary(_fetches['G_z_grid_summary'], _fetches['step'])

      print 'Done'

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)


"""
  Generates two-stage inference metagraph to train_dir/infer/infer.meta:
    1) Sample zi/zo
    2) Execute G([zi;zo])
  Named ops (use tf.default_graph().get_tensor_by_name(name)):
    1) Sample zi/zo
      * (Placeholder) samp_zi_n/0: Number of IDs to sample
      * (Placeholder) samp_zo_n/0: Number of observations to sample
      * (Output) samp_zo/0: Sampled zo latent codes
      * (Output) samp_zi/0: Sampled zi latent codes
      * If named_id_to_fps is not None:
        * (Random) samp_id/0: IDs to sample for inspection (override if desired)
        * (Constant) meta_all_named_ids/0: Names for all IDs from filepaths
        * (Constant) meta_all_group_fps/0: Comma-separated list of filepaths for all ID
        * (Output) samp_named_ids/0: Names for IDs
        * (Output) samp_group_fps/0: Comma-separated list of filepaths for IDs
      * If id_name_tsv_fp is not None:
        * (Constant) meta_all_names/0: Alternative names
        * (Output) samp_names/0: Alternative names for all IDs
    2) Execute G([zi;zo])
      * (Placeholder) zi/0: Identity latent codes
      * (Placeholder) zo/0: Observation latent codes
      * (Output) G_z/0: Output of G([zi;zo]); zi/zo batch size must be same
      * (Output) G_z_grid/0: Grid output of G([zi;zo]); batch size can differ
      * (Output) G_z_uint8/0: uint8 encoding of G_z/0
      * (Output) G_z_grid_uint8/0: uint8 encoding of G_z_grid/0
      * (Output) G_z_grid_prev: Image preview version of grid (5 axes to 3)
"""
def infer(named_id_to_fps, args):
  infer_dir = os.path.join(args.train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Placeholders for sampling stage
  samp_zi_n = tf.placeholder(tf.int32, [], name='samp_zi_n')
  samp_zo_n = tf.placeholder(tf.int32, [], name='samp_zo_n')

  # Sample IDs or fps for comparison
  if named_id_to_fps is not None:
    # Find number of identities and sample
    nids = len(named_id_to_fps)
    tf.constant(nids, dtype=tf.int32, name='nids')
    samp_id = tf.random_uniform([samp_zi_n], 0, nids, dtype=tf.int32, name='samp_id')

    # Find named ids and group fps
    named_ids = []
    fps = []
    for i, (named_id, group_fps) in enumerate(sorted(named_id_to_fps.items(), key=lambda k: k[0])):
      named_ids.append(named_id)
      fps.append(','.join(group_fps))
    named_ids = tf.constant(named_ids, dtype=tf.string, name='meta_all_named_ids')
    fps = tf.constant(fps, dtype=tf.string, name='meta_all_fps')

    # Alternative names (such as real names with spaces; not convenient for file paths)
    if args.data_id_name_tsv_fp is not None:
      with open(args.data_id_name_tsv_fp, 'r') as f:
        names = [l.split('\t')[1].strip() for l in f.readlines()[1:]]
      named_ids = tf.constant(names, dtype=tf.string, name='meta_all_names')

    samp_named_id = tf.gather(named_ids, samp_id, name='samp_named_ids')
    samp_fp_group = tf.gather(fps, samp_id, name='samp_group_fps')
    if args.data_id_name_tsv_fp is not None:
      samp_name = tf.gather(names, samp_id, name='samp_names')

  # Sample zi/zo
  samp_zi = tf.random_uniform([samp_zi_n, args.model_d_i], -1.0, 1.0, dtype=tf.float32, name='samp_zi')
  samp_zo = tf.random_uniform([samp_zo_n, args.model_d_o], -1.0, 1.0, dtype=tf.float32, name='samp_zo')

  # Input zo
  zi = tf.placeholder(tf.float32, [None, args.model_d_i], name='zi')
  zo = tf.placeholder(tf.float32, [None, args.model_d_o], name='zo')

  # Latent representation
  z = tf.concat([zi, zo], axis=1, name='z')

  # Make zi/zo grid
  zi_n = tf.shape(zi)[0]
  zo_n = tf.shape(zo)[0]
  zi_grid = tf.expand_dims(zi, axis=1)
  zi_grid = tf.tile(zi_grid, [1, zo_n, 1])
  zo_grid = tf.expand_dims(zo, axis=0)
  zo_grid = tf.tile(zo_grid, [zi_n, 1, 1])
  z_grid = tf.concat([zi_grid, zo_grid], axis=2, name='z_grid')

  # Execute generator
  with tf.variable_scope('G'):
    if args.dcgan:
      G_z = DCGANGenerator64x64(z, args.nch, dim=args.dcgan_dim)
    else:
      G_z = BEGANGenerator64x64(z, args.nch, hidden_num=args.began_nhidden)
  G_z = tf.identity(G_z, name='G_z')

  # Execute generator on grid
  z_grid = tf.reshape(z_grid, [zi_n * zo_n, args.model_d_i + args.model_d_o])
  with tf.variable_scope('G', reuse=True):
    if args.dcgan:
      G_z_grid = DCGANGenerator64x64(z_grid, args.nch, dim=args.dcgan_dim)
    else:
      G_z_grid = BEGANGenerator64x64(z_grid, args.nch, hidden_num=args.began_nhidden)
  G_z_grid = tf.reshape(G_z_grid, [zi_n, zo_n, args.height, args.width, args.nch], name='G_z_grid')

  # Encode to uint8
  G_z_uint8 = encode_png_observation(G_z, name='G_z_uint8')
  G_z_grid_uint8 = encode_png_observation(G_z_grid, name='G_z_grid_uint8')

  # Flatten grid of images to one large image (row shares zi, column shares zo)
  grid_zo_n = tf.shape(G_z_grid_uint8)[1]
  G_z_grid_prev = tf.transpose(G_z_grid_uint8, [1, 0, 2, 3, 4])
  G_z_grid_prev = tf.reshape(G_z_grid_prev, [grid_zo_n, zi_n * args.height, args.width, args.nch])
  G_z_grid_prev = tf.transpose(G_z_grid_prev, [1, 0, 2, 3])
  G_z_grid_prev = tf.reshape(G_z_grid_prev, [zi_n * args.height, grid_zo_n * args.width, args.nch], name='G_z_grid_prev')

  # Create saver
  G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
  global_step = tf.train.get_or_create_global_step()
  saver = tf.train.Saver(G_vars + [global_step])

  # Export graph
  tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

  # Export MetaGraph
  infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  tf.train.export_meta_graph(
      filename=infer_metagraph_fp,
      clear_devices=True,
      saver_def=saver.as_saver_def())

  # Reset graph (in case training afterwards)
  tf.reset_default_graph()


if __name__ == '__main__':
  import argparse
  import glob
  import sys

  import numpy as np

  from util import str2bool

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'preview', 'infer'])
  parser.add_argument('train_dir', type=str,
      help='Training directory')

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
      help='Data directory')
  data_args.add_argument('--data_set', type=str, choices=['msceleb12k', 'shoes4k'],
      help='Which dataset')
  data_args.add_argument('--data_id_name_tsv_fp', type=str,
      help='(Optional) alternate names for ids')
  data_args.add_argument('--data_nids', type=int,
      help='If positive, limits number of identites')

  model_args = parser.add_argument_group('Model')
  model_args.add_argument('--model_d_i', type=int,
      help='Dimensionality of identity codes')
  model_args.add_argument('--model_d_o', type=int,
      help='Dimensionality of observation codes')

  dcgan_args = parser.add_argument_group('DCGAN')
  dcgan_args.add_argument('--dcgan', action='store_true', dest='dcgan',
      help='Train an SDDCGAN')
  dcgan_args.add_argument('--dcgan_dim', type=int,
      help='Dimensionality multiplier for model of G and D')
  dcgan_args.add_argument('--dcgan_disc_siamese', type=str2bool,
      help='If false, stack channels rather than Siamese encoding')
  dcgan_args.add_argument('--dcgan_disc_nupdates', type=int,
      help='Number of discriminator updates per generator update')
  dcgan_args.add_argument('--dcgan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
      help='Which GAN loss to use')

  began_args = parser.add_argument_group('BEGAN')
  began_args.add_argument('--began', action='store_false', dest='dcgan',
      help='Train an SDBEGAN')
  began_args.add_argument('--began_nhidden', type=int,
      help='Number of hidden states for layers of G and D')
  began_args.add_argument('--began_gamma', type=float,
      help='Gamma argument for training stabilization')
  began_args.add_argument('--began_lambda_k', type=float,
      help='Learning rate for training stabilization')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_batch_size', type=int,
      help='Batch size')
  train_args.add_argument('--train_k', type=int,
      help='k-wise SD-GAN training')
  train_args.add_argument('--train_queue_capacity', type=int,
      help='Random example queue capacity (number of image tuples)')
  train_args.add_argument('--train_queue_min', type=int,
      help='Random example queue minimum')
  train_args.add_argument('--train_save_secs', type=int,
      help='How often to save model')
  train_args.add_argument('--train_summary_secs', type=int,
      help='How often to report summaries')

  preview_args = parser.add_argument_group('Preview')
  preview_args.add_argument('--preview_nids', type=int,
      help='Number of distinct identity vectors to preview')
  preview_args.add_argument('--preview_nobs', type=int,
      help='Number of distinct observation vectors to preview')

  parser.set_defaults(
    data_dir=None,
    data_set=None,
    data_id_name_tsv_fp=None,
    data_nids=-1,
    model_d_i=50,
    model_d_o=50,
    dcgan=True,
    dcgan_dim=64,
    dcgan_disc_siamese=True,
    dcgan_disc_nupdates=1,
    dcgan_loss='dcgan',
    began_nhidden=128,
    began_gamma=0.5,
    began_lambda_k=0.001,
    train_batch_size=16,
    train_k=2,
    train_queue_capacity=8192,
    train_queue_min=4096,
    train_save_secs=300,
    train_summary_secs=120,
    preview_nids=6,
    preview_nobs=8)

  args = parser.parse_args()

  # Make train dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Assign appropriate split for mode
  if args.mode == 'train':
    split = 'train'
  elif args.mode == 'preview':
    split = None
  elif args.mode == 'infer':
    split = 'train'
  else:
    raise NotImplementedError()

  # Dataset options
  if args.data_set == 'msceleb12k':
    data_extension = 'png'
    fname_to_named_id = lambda fn: fn.rsplit('_', 2)[0]
    setattr(args, 'height', 64)
    setattr(args, 'width', 64)
    setattr(args, 'nch', 3)
  elif args.data_set == 'shoes4k':
    data_extension = 'png'
    fname_to_named_id = lambda fn: fn.rsplit('_', 2)[0]
    setattr(args, 'height', 64)
    setattr(args, 'width', 64)
    setattr(args, 'nch', 3)
  else:
    raise NotImplementedError()
  

  # Find group fps and make splits
  if split is not None:
    print 'Finding files...'
    named_id_to_fps = defaultdict(list)
    glob_fp = os.path.join(args.data_dir, split, '*.{}'.format(data_extension))
    data_fps = glob.glob(glob_fp)
    for data_fp in sorted(data_fps):
      if args.data_nids > 0 and len(named_id_to_fps) >= args.data_nids:
        break

      data_fname = os.path.splitext(os.path.split(data_fp)[1])[0]

      named_id = fname_to_named_id(data_fname)
      named_id_to_fps[named_id].append(data_fp)

    if len(named_id_to_fps) == 0:
      print 'No observations found for {}'.format(glob_fp)
      sys.exit(1)
    else:
      print 'Found {} identities with average {} observations'.format(
          len(named_id_to_fps.keys()),
          np.mean([len(o) for o in named_id_to_fps.values()]))

  if args.mode == 'train':
    infer(named_id_to_fps, args)
    train(named_id_to_fps, args)
  elif args.mode == 'preview':
    preview(args)
  elif args.mode == 'infer':
    infer(named_id_to_fps, args)
  else:
    raise NotImplementedError()

from collections import defaultdict
import cPickle as pickle
import math
import os

import tensorflow as tf

from models import DCGANGenerator64x64, SDDCGANDiscriminator64x64
from util import str2bool, decode_png_observation, encode_png_observation


"""
  Samples a k-tuple from a group with or without replacement
  Enqueues n at a time for efficiency
"""
def group_choose_k(
    group_to_fps,
    k,
    n=None,
    with_replacement=False,
    capacity=4096,
    min_after_dequeue=2048,
    nthreads=4):
  assert k > 0

  # Join (variable-length) groups into CSV strings for enqueueing
  group_to_fps = [','.join(group_fps) for group_fps in group_to_fps.values()]

  # If n is None, compute a reasonable value (avg group len choose k)
  if n is None:
    n = int(np.ceil(np.mean([len(group_fps) for group_fps in group_to_fps.values()])))
    f = math.factorial
    n = f(n) / f(k) / f(n-k)

  # Dequeue one and split it into group
  group_fps = tf.train.string_input_producer(group_to_fps).dequeue()
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
      for j in range(k):
        if i == j:
          continue
        unique = tf.not_equal(tuple_ids[:, i], tuple_ids[:, j])
      tuple_unique = tf.logical_and(tuple_unique, i_unique)

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
  Generates two-stage inference metagraph to train_dir/infer/infer.meta:
    1) Sample zi/zo
    2) Execute G([zi;zo])
  Named ops (use tf.default_graph().get_tensor_by_name(name)):
    1) Sample zi/zo
      * (Placeholder) samp_zi_n/0: Number of IDs to sample
      * (Placeholder) samp_zo_n/0: Number of observations to sample
      * (Output) samp_zo/0: Sampled zo latent codes
      * (Output) samp_zi/0: Sampled zi latent codes
      * If group_to_fps is not None:
        * (Random) samp_id/0: IDs to sample for inspection (override if desired)
        * (Constant) meta_all_named_ids/0: Names for all IDs from filepaths
        * (Constant) meta_all_group_fps/0: Comma-separated list of filepaths for all ID
        * (Output) samp_named_ids/0: Names for IDs
        * (Output) samp_group_fps/0: Comma-separated list of filepaths for IDs
      * If id_to_name_tsv_fp is not None:
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
def infer(
    train_dir,
    height,
    width,
    nch,
    group_to_fps=None,
    id_to_name_tsv_fp=None,
    zi_dim=50,
    zo_dim=50,
    G_dim=64,
    G_batchnorm=True):
  infer_dir = os.path.join(train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Placeholders for sampling stage
  samp_zi_n = tf.placeholder(tf.int32, [], name='samp_zi_n')
  samp_zo_n = tf.placeholder(tf.int32, [], name='samp_zo_n')

  # Sample IDs or fps for comparison
  if group_to_fps is not None:
    # Find number of identities and sample
    nids = len(group_to_fps)
    tf.constant(nids, dtype=tf.int32, name='nids')
    samp_id = tf.random_uniform([samp_zi_n], 0, nids, dtype=tf.int32, name='samp_id')

    # Find named ids and group fps
    named_ids = []
    fps = []
    for i, (named_id, group_fps) in enumerate(sorted(group_to_fps.items(), key=lambda k: k[0])):
      named_ids.append(named_id)
      fps.append(','.join(group_fps))
    named_ids = tf.constant(named_ids, dtype=tf.string, name='meta_all_named_ids')
    fps = tf.constant(fps, dtype=tf.string, name='meta_all_fps')

    # Alternative names (such as real names with spaces; not convenient for file paths)
    if id_to_name_tsv_fp is not None:
      with open(id_to_name_tsv_fp, 'r') as f:
        names = [l.split('\t')[1].strip() for l in f.readlines()[1:]]
      named_ids = tf.constant(names, dtype=tf.string, name='meta_all_names')

    samp_named_id = tf.gather(named_ids, samp_id, name='samp_named_ids')
    samp_fp_group = tf.gather(fps, samp_id, name='samp_group_fps')
    if id_to_name_tsv_fp is not None:
      samp_name = tf.gather(names, samp_id, name='samp_names')

  # Sample zi/zo
  samp_zi = tf.random_uniform([samp_zi_n, zi_dim], -1.0, 1.0, dtype=tf.float32, name='samp_zi')
  samp_zo = tf.random_uniform([samp_zo_n, zo_dim], -1.0, 1.0, dtype=tf.float32, name='samp_zo')

  # Input zo
  zi = tf.placeholder(tf.float32, [None, zi_dim], name='zi')
  zo = tf.placeholder(tf.float32, [None, zo_dim], name='zo')

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
    G_z = create_generator(z, nch, dim=G_dim, batchnorm=G_batchnorm)
  G_z = tf.identity(G_z, name='G_z')

  # Execute generator on grid
  z_grid = tf.reshape(z_grid, [zi_n * zo_n, zi_dim + zo_dim])
  with tf.variable_scope('G', reuse=True):
    G_z_grid = create_generator(z_grid, nch, dim=G_dim, batchnorm=G_batchnorm)
  G_z_grid = tf.reshape(G_z_grid, [zi_n, zo_n, height, width, nch], name='G_z_grid')

  # Encode to uint8
  G_z_uint8 = encode_png_observation(G_z, name='G_z_uint8')
  G_z_grid_uint8 = encode_png_observation(G_z_grid, name='G_z_grid_uint8')

  # Flatten grid of images to one large image (row shares zi, column shares zo)
  grid_zo_n = tf.shape(G_z_grid_uint8)[1]
  G_z_grid_prev = tf.transpose(G_z_grid_uint8, [1, 0, 2, 3, 4])
  G_z_grid_prev = tf.reshape(G_z_grid_prev, [grid_zo_n, zi_n * height, width, nch])
  G_z_grid_prev = tf.transpose(G_z_grid_prev, [1, 0, 2, 3])
  G_z_grid_prev = tf.reshape(G_z_grid_prev, [zi_n * height, grid_zo_n * width, nch], name='G_z_grid_prev')

  # Create saver
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
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


def train(
    train_dir,
    group_to_fps,
    batch_size,
    k,
    height,
    width,
    subsample_height,
    nch,
    queue_capacity=8192,
    queue_min=4096,
    queue_nthreads=2,
    zi_dim=50,
    zo_dim=50,
    G_dim=64,
    D_dim=64,
    G_batchnorm=True,
    D_batchnorm=True,
    G_zi='random_uniform',
    loss='dcgan',
    opt='dcgan',
    D_siamese_strat='tt',
    D_decomp_strat='sdgan',
    D_iters=5,
    save_secs=600,
    summary_secs=120):
  if D_decomp_strat != 'sdgan':
    assert k == 1
    if D_decomp_strat in ['springenberg_sgan', 'odena_acgan']:
      assert loss == 'dcgan'

  def make_x_batch(tup, observations):
    queue = tf.RandomShuffleQueue(
        capacity=queue_capacity,
        min_after_dequeue=queue_min,
        shapes=[[k], [k, height, width, nch]],
        dtypes=[tf.string, tf.float32])

    cond = tf.constant(True)
    if subsample_height:
      for i in xrange(k):
        cond = tf.logical_and(cond, tf.shape(observations[i])[0] >= height)

    def enqueue():
      if subsample_height:
        def subsample(x):
          lenx = tf.shape(x)[0]
          start_max = lenx - height
          start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)
          win = x[start:start+height]
          win.set_shape([height])
          return win
      else:
        subsample = tf.identity

      observations_subsampled = []
      for i in xrange(k):
        observations_subsampled.append(subsample(observations[i]))

      example = tf.stack(observations_subsampled, axis=0)
      enqueue_op = queue.enqueue((tup, example))

      nvalid = tf.Variable(0)

      increment = tf.assign_add(nvalid, 1)

      with tf.control_dependencies([enqueue_op, increment]):
        return tf.constant(True)

    def noenqueue():
      ninvalid = tf.Variable(0)

      increment = tf.assign_add(ninvalid, 1)

      with tf.control_dependencies([increment]):
        return tf.constant(False)

    enqueued = tf.cond(cond, true_fn=enqueue, false_fn=noenqueue)

    qr = tf.train.QueueRunner(queue, [enqueued] * queue_nthreads)
    tf.train.add_queue_runner(qr)

    tf.summary.scalar('queue_size', queue.size())

    return queue.dequeue_many(batch_size)

  # Load observation tuples
  with tf.name_scope('loader'):
    # Generate matched pairs of WAV fps
    with tf.device('/cpu:0'):
      tup = group_choose_k(
          group_to_fps,
          k,
          not with_replacement=not(subsample_height))

      observations = []
      for i in xrange(k):
        observation = decode_png_observation(tup[i])
        if subsample_height:
          observation.set_shape([None, width, nch])
        else:
          observation.set_shape([height, width, nch])

        observations.append(observation)

      x_fps, x = make_x_batch(tup, observations)

  # Make image summaries
  for i in xrange(k):
    tf.summary.image('x_{}'.format(i), encode_png_observation(x[:, i]))

  # Prepare labels
  if G_zi == 'id_embed' or D_decomp_strat in ['mirza_cgan', 'springenberg_sgan', 'odena_acgan']:
    all_fps = []
    all_ids = []
    id_to_named_id = []
    for i, (named_id, fps) in enumerate(sorted(group_to_fps.items(), key=lambda k: k[0])):
      id_to_named_id.append((i, named_id))
      for fp in fps:
        all_fps.append(fp)
        all_ids.append(i)
    nids = len(group_to_fps.keys())

    with open(os.path.join(train_dir, 'id_to_named_id.csv'), 'w') as f:
      f.write('\n'.join([','.join((str(p[0]), str(p[1]))) for p in id_to_named_id]))

    with tf.device('/cpu:0'):
      kv = tf.contrib.lookup.KeyValueTensorInitializer(all_fps, all_ids)
      fp_to_id = tf.contrib.lookup.HashTable(kv, -1)

      weight_limit = 0.02 * np.sqrt(3)
      weight_init = tf.random_uniform_initializer(-weight_limit, weight_limit, dtype=tf.float32)

      y_real = fp_to_id.lookup(x_fps[:, 0])

      if G_zi == 'id_embed':
        y_fake = tf.random_uniform([batch_size], minval=0, maxval=nids, dtype=tf.int32)
      else:
        y_fake = y_real
  else:
    y_fake = None
    y_real = None
    nids = None

  # Make zi (instance)
  if G_zi == 'random_uniform':
    zi = tf.random_uniform([batch_size, zi_dim], -1.0, 1.0, dtype=tf.float32)
  elif G_zi == 'id_embed':
    with tf.device('/cpu:0'):
      with tf.name_scope('G_emb'), tf.variable_scope('G'):
        embedding = tf.get_variable(
            'identity_embedding',
            shape=[nids, zi_dim],
            dtype=tf.float32,
            initializer=weight_init)
      zi = tf.nn.embedding_lookup(embedding, y_fake)
  elif G_zi == 'img_encode':
    with tf.name_scope('G_enc'), tf.variable_scope('G'):
      zi = create_encoder(x[:, 0], zi_dim, dim=G_dim, batchnorm=G_batchnorm)

  # Tile zi
  zi = tf.tile(zi, [1, k])
  zi = tf.reshape(zi, [batch_size, k, zi_dim])

  # Make zo (observation)
  zo = tf.random_uniform([batch_size, k, zo_dim], -1.0, 1.0, dtype=tf.float32)

  # Concat zi, zo
  z = tf.concat([zi, zo], axis=2)

  # Make generator
  with tf.variable_scope('G'):
    if D_decomp_strat == 'sdgan' and G_zi in ['id_embed', 'img_encode']:
      z = z[:, 0]
      G_z = create_generator(z, nch, dim=G_dim, batchnorm=G_batchnorm)
      G_z = tf.expand_dims(G_z, axis=1)
      G_z = tf.concat([G_z, x[:, :1]], axis=1)
    else:
      z = tf.reshape(z, [batch_size * k, zi_dim + zo_dim])
      G_z = create_generator(z, nch, dim=G_dim, batchnorm=G_batchnorm)
      G_z = tf.reshape(G_z, [batch_size, k, height, width, nch])
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
  for i in xrange(k):
    tf.summary.image('G_z_{}'.format(i), encode_png_observation(G_z[:, i]))

  # Make real discriminator
  with tf.name_scope('D_x'), tf.variable_scope('D'):
    D_x = create_discriminator(
        x, y_real,
        decomp_strat=D_decomp_strat, nids=nids,
        dim=D_dim, batchnorm=D_batchnorm, siamese_strat=D_siamese_strat)
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
    D_G_z = create_discriminator(
        G_z, y_fake,
        decomp_strat=D_decomp_strat, nids=nids,
        dim=D_dim, batchnorm=D_batchnorm, siamese_strat=D_siamese_strat)

  # Create loss
  D_clip_weights = None
  if loss == 'dcgan':
    if D_decomp_strat == 'odena_acgan':
      # from https://github.com/burness/tensorflow-101/blob/master/GAN/AC-GAN/train.py
      disc_real, cat_real = D_x[:, 0], D_x[:, 1:]
      disc_fake, cat_fake = D_G_z[:, 0], D_G_z[:, 1:]

      # discriminator loss
      loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real,
        labels=tf.ones(batch_size)
      ))
      loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.zeros(batch_size)
      ))
      loss_d = (loss_d_r + loss_d_f) / 2

      # generator loss
      loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.ones(batch_size)
      ))

      # categorical factor loss
      loss_c_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cat_real,
        labels=y_real
      ))
      loss_c_d = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cat_fake,
        labels=y_fake
      ))
      loss_c = (loss_c_r + loss_c_d) / 2

      G_loss = loss_g + loss_c
      D_loss = loss_d + loss_c
    else:
      if D_decomp_strat == 'springenberg_sgan':
        fake_y = tf.zeros([batch_size], dtype=tf.int32)
        fake = tf.one_hot(fake_y, 1 + nids, dtype=tf.float32)
        real = tf.one_hot(1 + y, 1 + nids, dtype=tf.float32)
      else:
        fake = tf.zeros([batch_size, 1], dtype=tf.float32)
        real = tf.ones([batch_size, 1], dtype=tf.float32)

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
  elif loss == 'wgan':
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
  elif loss == 'wgan-gp':
    # WGAN-GP
    # https://arxiv.org/pdf/1704.00028.pdf
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1, 1], minval=0., maxval=1.)
    differences = G_z - y
    interpolates = y + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = create_discriminator(
          interpolates, y,
          decomp_strat=D_decomp_strat, nids=nids,
          dim=D_dim, batchnorm=D_batchnorm, siamese_strat=D_siamese_strat)

    LAMBDA = 10
    gradients = tf.gradients(D_interp, [interpolates])[0]
    # TODO: what do reduction indices do here?
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty
  else:
    raise NotImplementedError()

  tf.summary.scalar('G_loss', G_loss)
  tf.summary.scalar('D_loss', D_loss)

  # Create optimizer
  if opt == 'dcgan':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)

    D_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)
  elif opt == 'wgan':
    G_opt = tf.train.RMSPropOptimizer(
      learning_rate=5e-5)

    D_opt = tf.train.RMSPropOptimizer(
      learning_rate=5e-5)
  elif opt == 'wgan-gp':
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

  G_train_op = G_opt.minimize(G_loss, var_list=G_vars)
  D_train_op = D_opt.minimize(D_loss, var_list=D_vars,
      global_step=tf.train.get_or_create_global_step())

  # Run training
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=train_dir,
      save_checkpoint_secs=save_secs,
      save_summaries_secs=summary_secs) as sess:
    while True:
      # Train discriminator
      for i in xrange(D_iters):
        sess.run(D_train_op)

        if D_clip_weights is not None:
          sess.run(D_clip_weights)

      # Train generator
      sess.run(G_train_op)

      # TODO: stopgrads?

if __name__ == '__main__':
  import argparse
  import glob

  from util import str2bool

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'infer'])
  parser.add_argument('--data_dir', type=str,
      help='Data directory')
  parser.add_argument('--data_set', type=str, choices=['msceleb12k', 'shoes4k'],
      help='Which dataset')
  parser.add_argument('--nids', type=int,
      help='If specified, limits number of identites')
  parser.add_argument('--model_id_to_name_tsv_fp', type=str,
      help='(Optional) alternate names for ids')
  parser.add_argument('--model_dim', type=int,
      help='Dimensionality multiplier for model')
  parser.add_argument('--train_dir', type=str,
      help='Output directory for training')
  parser.add_argument('--train_batch_size', type=int,
      help='Batch size')
  parser.add_argument('--train_k', type=int,
      help='k-wise SD-GAN training')
  parser.add_argument('--train_queue_capacity', type=int,
      help='Random example queue capacity (number of image tuples)')
  parser.add_argument('--train_queue_min', type=int,
      help='Random example queue minimum')
  parser.add_argument('--train_loss', type=str, choices=['dcgan', 'lsgan', 'wgan']
      help='Which GAN loss to use')
  parser.add_argument('--train_disc_siamese', type=str2bool,
      help='If false, stack channels rather than Siamese encoding')
  parser.add_argument('--train_disc_nupdates', type=int,
      help='Number of discriminator updates per generator update')
  parser.add_argument('--train_save_secs', type=int,
      help='How often to save model')
  parser.add_argument('--train_summary_secs', type=int,
      help='How often to report summaries')

  parser.set_defaults(
    data_dir=None,
    data_set='msceleb12k',
    nids=None,
    model_id_to_name_tsv=None,
    model_dim=64,
    model_zi='random_uniform',
    train_dir=None,
    train_batch_size=64,
    train_k=2,
    train_queue_capacity=8192,
    train_queue_min=4096,
    train_queue_nthreads=2,
    train_loss='dcgan',
    train_disc_siamese_strat='tt',
    train_disc_decomp_strat='sdgan',
    train_disc_iters=1,
    train_save_secs=300,
    train_summary_secs=120,
    preview_nids=12,
    preview_nobs=10,
    eval_n_per_class=None)

  args = parser.parse_args()

  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Assign appropriate split for mode
  if args.mode == 'train':
    split = 'train'
  elif args.mode == 'infer':
    split = None
  else:
    raise NotImplementedError()

  # Parse dataset
  if args.data_set == 'msceleb12k':
    data_extension = 'png'
    fname_to_named_id = lambda fn: fn.rsplit('_', 2)[0]
    height = 64
    width = 64
    nch = 3
  elif args.dataset == 'shoes4k':
    data_extension = 'png'
    fname_to_named_id = lambda fn: fn.rsplit('_', 2)[0]
    height = 64
    width = 64
    nch = 3
    raise NotImplementedError()

  # Make splits
  group_to_fps = defaultdict(list)
  if split is not None:
    named_ids = set()
    glob_fp = os.path.join(args.data_dir, split, '*.{}'.format(data_extension))
    data_fps = glob.glob(glob_fp)
    for data_fp in sorted(data_fps):
      if args.nids is not None and len(named_ids) > args.nids:
        break

      data_fname = os.path.splitext(os.path.split(data_fp)[1])[0]

      named_id = fname_to_named_id(data_fname)
      named_ids.add(named_id)

      group_to_fps[named_id].append(data_fp)

    print 'Loaded {} identities with average {} observations'.format(
        len(group_to_fps.keys()),
        np.mean([len(o) for o in group_to_fps.values()]))

  if args.mode == 'train':
    infer(
        args.train_dir,
        group_to_fps,
        args.model_id_to_name_tsv_fp,
        height,
        width,
        nch,
        G_dim=args.model_dim,
        G_zi=args.model_zi)

    train(
        args.train_dir,
        group_to_fps,
        args.train_batch_size,
        args.train_k,
        ntuples,
        height,
        width,
        subsample_height,
        nch,
        queue_capacity=args.train_queue_capacity,
        queue_min=args.train_queue_min,
        queue_nthreads=2,
        G_dim=args.model_dim,
        D_dim=args.model_dim,
        G_zi=args.model_zi,
        loss=args.train_loss,
        opt=args.train_loss,
        D_siamese_strat=args.train_disc_siamese_strat,
        D_decomp_strat=args.train_disc_decomp_strat,
        D_iters=args.train_disc_iters,
        save_secs=args.train_save_secs,
        summary_secs=args.train_summary_secs)
  elif args.mode == 'infer':
    infer(
        args.train_dir,
        group_to_fps,
        args.model_id_to_name_tsv_fp,
        height,
        width,
        nch,
        G_dim=args.model_dim,
        G_zi=args.model_zi)
  else:
    raise NotImplementedError()

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PixelCNN++ example."""

import functools

from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
import flax.nn
from flax.training import common_utils

import jax
from jax import random
import jax.nn
import jax.numpy as jnp

import tensorflow.compat.v2 as tf

import input_pipeline
import pixelcnn


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.001,
    help=('The initial learning rate.'))

flags.DEFINE_float(
    'lr_decay', default='0.999995',
    help=('Learning rate decay, applied each optimization step.'))

flags.DEFINE_integer(
    'init_batch_size', default=16,
    help=('Batch size to use for data-dependent initialization.'))

flags.DEFINE_integer(
    'batch_size', default=64,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=200,
    help=('Number of training epochs.'))

flags.DEFINE_float(
    'dropout_rate', default=0.5,
    help=('DropOut rate.'))

flags.DEFINE_integer(
    'rng', default=0,
    help=('Random seed for network initialization.'))

flags.DEFINE_integer(
    'n_resnet', default=5,
    help=('Number of resnet layers per block.'))

flags.DEFINE_integer(
    'n_feature', default=160,
    help=('Number of features in each conv layer.'))

flags.DEFINE_integer(
    'n_logistic_mix', default=5,
    help=('Number of components in the output distribution.'))

flags.DEFINE_string(
    'model_dir', default='./model_data',
    help=('Directory to store model data'))


@functools.partial(jax.jit, static_argnums=(1, 2))
def create_model(prng_key, example_images, model_def):
  with flax.nn.stateful() as init_state:
    # TODO(j-towns): is this context manager necessary?
    with flax.nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = model_def.init(prng_key, example_images)
      model = flax.nn.Model(model_def, initial_params)
  return model, init_state


def create_optimizer(model, learning_rate):
  optimizer_def = optim.Adam(learning_rate=learning_rate,
                             beta1=0.95,
                             beta2=0.9995)
  optimizer = optimizer_def.create(model)
  optimizer = jax_utils.replicate(optimizer)
  return optimizer


def neg_log_likelihood_loss(nn_out, images):
  means, inv_scales, logit_weights = (
      pixelcnn.batch_conditional_params_from_outputs(nn_out, images))
  log_likelihoods = pixelcnn.logprob_from_conditional_params(
      images, means, inv_scales, logit_weights)
  return -jnp.mean(log_likelihoods)


def compute_metrics(nn_out, images):
  loss = neg_log_likelihood_loss(nn_out, images)
  metrics = {
      'negative log likelihood': loss,
  }
  metrics = jax.lax.pmean(metrics, 'batch')
  return metrics


def train_step(optimizer, state, batch, prng_key, learning_rate_fn):
  """Perform a single training step."""
  def loss_fn(model):
    """loss function used for training."""
    # TODO(j-towns): do we need state for this model?
    with flax.nn.stateful(state) as new_state:
      with flax.nn.stochastic(prng_key):
        nn_out = model(batch['image'])
    loss = neg_log_likelihood_loss(nn_out, batch['image'])
    return loss, new_state

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, new_state), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  metrics = {'negative log likelihood': jax.lax.pmean(loss, 'batch')}
  metrics['learning_rate'] = lr
  return new_optimizer, new_state, metrics


def eval_step(model, state, batch):
  state = jax.lax.pmean(state, 'batch')
  with flax.nn.stateful(state, mutable=False):
    nn_out = model(batch['image'], train=False)
  return compute_metrics(nn_out, batch['image'])


def load_and_shard_tf_batch(xs):
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def train(model_def, model_dir, batch_size, init_batch_size, num_epochs,
          learning_rate, decay_rate, run_seed=0):
  """Train model."""
  if jax.host_count() > 1:
    raise ValueError('PixelCNN++ example should not be run on more than 1 host'
                     ' (for now)')

  summary_writer = tensorboard.SummaryWriter(model_dir)

  rng = random.PRNGKey(run_seed)

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  device_batch_size = batch_size // jax.device_count()

  # Load dataset
  data_source = input_pipeline.DataSource(
      train_batch_size=batch_size, eval_batch_size=batch_size)
  train_ds = data_source.train_ds
  eval_ds = data_source.eval_ds

  # Create dataset batch iterators
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  # Compute steps per epoch and nb of eval steps
  steps_per_epoch = data_source.TRAIN_IMAGES // batch_size
  steps_per_eval = data_source.EVAL_IMAGES // batch_size
  num_steps = steps_per_epoch * num_epochs

  base_learning_rate = learning_rate

  # Create the model using data-dependent initialization. Don't shard the init
  # batch.
  assert init_batch_size <= batch_size
  init_batch = next(train_iter)['image']._numpy()[:init_batch_size]
  model, state = create_model(rng, init_batch, model_def)
  state = jax_utils.replicate(state)
  optimizer = create_optimizer(model, base_learning_rate)
  del model  # don't keep a copy of the initial model

  # Learning rate schedule
  learning_rate_fn = lambda step: base_learning_rate * decay_rate ** step

  # pmap the train and eval functions
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # Gather metrics
  train_metrics = []
  epoch = 1
  for step, batch in zip(range(num_steps), train_iter):
    # Generate a PRNG key that will be rolled into the batch
    rng, step_key = jax.random.split(rng)
    # Load and shard the TF batch
    batch = load_and_shard_tf_batch(batch)
    # Shard the step PRNG key
    sharded_keys = common_utils.shard_prng_key(step_key)

    # Train step
    optimizer, state, metrics = p_train_step(
        optimizer, state, batch, sharded_keys)
    train_metrics.append(metrics)

    if (step + 1) % steps_per_epoch == 0:
      # We've finished an epoch
      train_metrics = common_utils.get_metrics(train_metrics)
      # Get training epoch summary for logging
      train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
      # Send stats to Tensorboard
      for key, vals in train_metrics.items():
        tag = 'train_%s' % key
        for i, val in enumerate(vals):
          summary_writer.scalar(tag, val, step - len(vals) + i + 1)
      # Reset train metrics
      train_metrics = []

      # Evaluation
      eval_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        # Load and shard the TF batch
        eval_batch = load_and_shard_tf_batch(eval_batch)
        # Step
        metrics = p_eval_step(optimizer.target, state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      # Get eval epoch summary for logging
      eval_summary = jax.tree_map(lambda x: x.mean(), eval_metrics)

      # Log epoch summary
      logging.info(
          'Epoch %d: TRAIN loss=%.6f, err=%.2f, EVAL loss=%.6f, err=%.2f',
          epoch, train_summary['loss'], train_summary['error_rate'] * 100.0,
          eval_summary['loss'], eval_summary['error_rate'] * 100.0)

      summary_writer.scalar('eval_loss', eval_summary['loss'], epoch)
      summary_writer.scalar('eval_error_rate', eval_summary['error_rate'],
                            epoch)
      summary_writer.flush()

      epoch += 1


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  model_def = pixelcnn.PixelCNNPP.partial(
      depth=FLAGS.n_resnet,
      features=FLAGS.n_feature,
      dropout_p=FLAGS.dropout_rate)

  train(model_def, FLAGS.model_dir, FLAGS.batch_size, FLAGS.init_batch_size,
        FLAGS.num_epochs, FLAGS.learning_rate, FLAGS.lr_decay, FLAGS.rng)


if __name__ == '__main__':
  app.run(main)

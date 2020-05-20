# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Sampling from PixelCNN++ using fixed-point iteration.
"""
import time
import cProfile
from functools import partial

from absl import app
from absl import flags

import numpy as onp
from PIL import Image

import jax
from jax import random
import jax.numpy as jnp
from jax import jit

from fastar import accelerate, parray

import pixelcnn
import train


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'sample_batch_size', default=1,
    help=('Batch size for sampling.'))

flags.DEFINE_integer(
    'sample_rng_seed', default=0,
    help=('Random number generator seed for sampling.'))

def generate_sample(pcnn_module, batch_size, rng_seed=0):
  rng = random.PRNGKey(rng_seed)
  rng, model_rng = random.split(rng)

  # Create a model with dummy parameters and a dummy optimizer
  example_images = jnp.zeros((1, 32, 32, 3))
  model = train.create_model(model_rng, example_images, pcnn_module)
  optimizer = train.create_optimizer(model, 0)

  # Load learned parameters
  _, ema = train.restore_checkpoint(optimizer, model.params)
  model = model.replace(params=ema)

  # Initialize batch of images
  sample = jnp.zeros((batch_size, 32, 32, 3))

  # Wrap sample in a fastar.Parray, indicating that no elements are yet 'known'
  sample = parray(sample, onp.zeros_like(sample, bool))

  # Generate sample using Fastar
  fp_fun = partial(sample_iteration, model)
  print(time.strftime("%H:%M:%S", time.localtime()))
  sample, update_fun = run_sampling_iterations(32, fp_fun, rng, sample)
  print(time.strftime("%H:%M:%S", time.localtime()))
  return jnp.reshape(sample[0], (batch_size, 32, 32, 3))

@partial(jit, static_argnums=(0, 1))
def run_sampling_iterations(n_iterations, fp_fun, rng, sample, cache=None):
  init_fun, update_fun = accelerate(partial(fp_fun, rng))
  p = cProfile.Profile()
  p.enable()
  i = 0
  if cache is None:
    sample, cache = init_fun(sample)
    i = i + 1
  while i < n_iterations:
    print(i)
    sample, cache = update_fun(cache, sample)
    i = i + 1
  print(time.strftime("%H:%M:%S", time.localtime()))
  p.disable()
  p.dump_stats('tracing_prof')
  return sample, cache


def _categorical_onehot(rng, logit_probs):
  """Sample from a categorical distribution and one-hot encode the sample.
  """
  nr_mix = logit_probs.shape[-3]
  idxs = random.categorical(rng, logit_probs, axis=-3)
  return jnp.moveaxis(idxs[..., jnp.newaxis] == jnp.arange(nr_mix), -1, -3)

def conditional_params_to_sample(rng, conditional_params):
  means, inv_scales, logit_probs = conditional_params
  rng_mix, rng_logistic = random.split(rng)
  # Add channel dimension to one-hot mixture indicator
  mix_indicator = _categorical_onehot(rng_mix, logit_probs)[..., jnp.newaxis]
  # Use the mixture indicator to select the mean and inverse scale
  mean      = jnp.sum(means      * mix_indicator, -4)
  inv_scale = jnp.sum(inv_scales * mix_indicator, -4)
  sample = mean + random.logistic(rng_logistic, mean.shape) / inv_scale
  return snap_to_grid(sample)

def sample_iteration(model, rng, sample):
  """PixelCNN++ sampling expressed as a fixed-point iteration.
  """
  c_params = pixelcnn.conditional_params_from_outputs(model(sample), sample)
  return conditional_params_to_sample(rng, c_params)

def snap_to_grid(sample):
  return jnp.clip(jnp.round((sample + 1) * 127.5) / 127.5 - 1, -1., 1.)

def save_images(batch, fname):
  n_rows = (batch.shape[0] - 1) // 16 + 1
  batch = onp.uint8(jnp.round((batch + 1) * 127.5))
  out = onp.full((1 + 33 * n_rows, 1 + 33 * 16, 3), 255, 'uint8')
  for i, im in enumerate(batch):
    top  = 1 + 33 * (i // 16)
    left = 1 + 33 * (i %  16)
    out[top:top + 32, left:left + 32] = im
  Image.fromarray(out).save(fname)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  pcnn_module = pixelcnn.PixelCNNPP.partial(depth=FLAGS.n_resnet,
                                            features=FLAGS.n_feature,
                                            dropout_p=0)

  batch = generate_sample(
      pcnn_module, FLAGS.sample_batch_size, FLAGS.sample_rng_seed)
  save_images(batch, 'sample.png')

if __name__ == '__main__':
  app.run(main)

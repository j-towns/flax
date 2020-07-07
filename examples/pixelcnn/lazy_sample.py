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
from functools import partial

from absl import app
from absl import flags

import numpy as onp
from PIL import Image

import jax
from jax import random
import jax.numpy as jnp
from jax import config
from jax import make_jaxpr

import fastar

import pixelcnn
import train


# The following is required to use TPU Driver (libtpu.so) as JAX's backend.
# config.FLAGS.jax_xla_backend = "tpu_driver"
# config.FLAGS.jax_backend_target = "direct://libtpu.so"

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'sample_batch_size', default=64,
    help=('Batch size for sampling.'))

flags.DEFINE_integer(
    'sample_rng_seed', default=0,
    help=('Random number generator seed for sampling.'))

def generate_sample(pcnn_module, batch_size, rng_seed=1):
  rng = random.PRNGKey(rng_seed)
  sample_rng, model_rng = random.split(rng)

  # Create a model with dummy parameters and a dummy optimizer
  example_images = jnp.zeros((1, 32, 32, 3))
  model = train.create_model(model_rng, example_images, pcnn_module)
  optimizer = train.create_optimizer(model, 0)

  # Load learned parameters
  _, ema = train.restore_checkpoint(optimizer, model.params)
  model = model.replace(params=ema)

  # Initialize mock batch of images
  sample_mock = jnp.zeros((batch_size, 32, 32, 3))

  # Generate sample using fixed-point iteration
  iter_fun = lambda sample: sample_iteration(sample_rng, model, sample)
  sample = fastar.lazy_eval_fixed_point(iter_fun, sample_mock)

  iterate = lambda n: sample_iteration(iterate(n - 1)) if n > 0 else sample_mock

  # Force evaluation by indexing
  sample[0, 0, 0, 0]
  sample[0, 0, 0, 1]
  sample[0, 0, 0, 2]
  sample[0, 0, 4, 2]
  from IPython.terminal.debugger import set_trace; set_trace()
  return sample[:]

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

def sample_iteration(rng, model, sample):
  """PixelCNN++ sampling expressed as a fixed-point iteration.
  """
  c_params = pixelcnn.conditional_params_from_outputs(model(sample), sample)
  return conditional_params_to_sample(rng, c_params)

def snap_to_grid(sample):
  return jnp.clip(jnp.round((sample + 1) * 127.5) / 127.5 - 1, -1., 1.)

def save_images(batch, fname):
  n_rows = batch.shape[0] // 16
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

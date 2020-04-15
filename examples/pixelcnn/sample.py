# Lint as: python3
"""Naive sampling from PixelCNN model.
"""

from absl import app
from absl import flags

import jax
from jax import random
import jax.numpy as jnp

import pixelcnn
import train

FLAGS = flags.FLAGS


def sample(pcnn_module, batch_size, rng_seed=0):
  rng = random.PRNGKey(rng_seed)

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  # Create a model with dummy parameters and a dummy optimizer
  example_images = jnp.zeros((1, 32, 32, 3))
  model = train.create_model(rng, example_images, pcnn_module)
  optimizer = train.create_optimizer(model, 0)

  # Load learned parameters
  _, ema = train.restore_checkpoint(optimizer, model.params)
  model = model.replace(params=ema)

  sample_prev = jnp.zeros((batch_size, 32, 32, 3), 'uint8')
  sample = sample_iteration(rng, model, sample_prev)
  iter = 1
  while jnp.any(sample != sample_prev):
    print(iter)
    sample_prev, sample = sample, sample_iteration(rng, model, sample)
    iter = iter + 1


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
  mix_indicator = _categorical_onehot(rng, logit_probs)[..., jnp.newaxis]
  # Use the mixture indicator to select the mean and inverse scale
  means      = jnp.sum(means      * mix_indicator, -4)
  inv_scales = jnp.sum(inv_scales * mix_indicator, -4)
  sample = means + random.logistic(rng_logistic, means.shape) / inv_scales
  return jnp.round((jnp.clip(sample, -1, 1) + 1) * 127.5).astype('uint8')

@jax.jit
def sample_iteration(rng, model, sample):
  """PixelCNN++ sampling expressed as a fixed-point iteration.
  """
  sample = pixelcnn.centre(sample)
  model_out = model(pixelcnn.centre(sample))
  c_params = pixelcnn.batch_conditional_params_from_outputs(model_out, sample)
  return conditional_params_to_sample(rng, c_params)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  pcnn_module = pixelcnn.PixelCNNPP.partial(depth=FLAGS.n_resnet,
                                            features=FLAGS.n_feature,
                                            dropout_p=0)

  sample(pcnn_module, FLAGS.batch_size, FLAGS.rng)

if __name__ == '__main__':
  app.run(main)

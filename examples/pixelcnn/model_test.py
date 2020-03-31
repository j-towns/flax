# Lint as: python3
"""Tests for model."""

from absl.testing import absltest
from absl.testing import parameterized
import pixelcnn

from jax import random
import jax.numpy as np

from flax import nn


class ModelTest(absltest.TestCase):

  def test_conv(self):
    rng = random.PRNGKey(0)
    x = np.arange(24).reshape(1, 4, 3, 2)
    conv_module = pixelcnn.Conv2D.partial(features=4, kernel_size=(3, 2))
    out, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    direction, scale, bias = model.params['weightnorm_params']
    self.assertEqual(direction.shape, (3, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 2, 2, 4))


if __name__ == '__main__':
  googletest.main()

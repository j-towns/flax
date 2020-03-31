# Lint as: python3
"""Flax implementation of PixelCNN++

Based on the paper

  PixelCNN++: Improving the PixelCNN with discretized logistic mixture
  likelihood and other modifications

published at ICLR '17 (https://openreview.net/forum?id=BJrFC6ceg).
"""
from functools import partial

from flax import nn

from jax import lax
import jax.numpy as np
import jax.numpy.linalg as la


# Weightnorm utilities
def _l2_normalize(v):
  """
  Normalize a convolution kernel direction over the in_features and spatial
  dimensions.
  """
  return v / np.sqrt(np.sum(v ** 2, (0, 1, 2)))

def _make_kernel(direction, scale):
  """
  Maps weightnorm parameterization (direction, scale) to standard
  parameterization. The direction has shape (spatial..., in_features,
  out_features), scale has shape (out_features,).
  """
  return scale * _l2_normalize(direction)

# 2D Conv Modules with weight-norm parameter initialization
class Conv2D(nn.Module):
  def apply(self,
            inputs,
            features,
            kernel_size,
            strides=None,
            dtype=np.float32,
            precision=None):
    inputs = np.asarray(inputs, dtype)
    strides = strides or (1,) * (inputs.ndim - 2)

    conv = partial(
        lax.conv_general_dilated, window_strides=strides, padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), precision=precision)

    def initializer(key, shape):
      # A weightnorm initializer generating a (direction, scale, bias) tuple.
      direction = nn.initializers.normal()(key, kernel_shape, dtype)
      unnormed_out = conv(inputs, _l2_normalize(direction))
      mean = np.mean(unnormed_out, (0, 1, 2))
      var  = np.std (unnormed_out, (0, 1, 2))
      return direction, 1 / var, -mean / var

    *_, in_features = inputs.shape
    kernel_shape = kernel_size + (in_features, features)
    direction, scale, bias = self.param(
        'weightnorm_params', None, initializer)
    return conv(inputs, _make_kernel(direction, scale)) + bias

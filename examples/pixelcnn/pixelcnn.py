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


def concat_elu(x):
  return nn.elu(np.concatenate((x, -x), -1))

def spatial_pad(pad_vertical, pad_horizontal, operand):
  """
  Wrapper around lax.pad which pads only in spatial dimensions (horizontal and
  vertical), with
  padding_value=0. and no interior padding.
  """
  zero = (0, 0, 0)
  return lax.pad(operand, 0.,
                 (zero, pad_vertical + (0,), pad_horizontal + (0,), zero))

shift_down  = partial(spatial_pad, (1, -1), (0,  0))
shift_right = partial(spatial_pad, (0,  0), (1, -1))


# Weightnorm utilities
def _l2_normalize(v):
  """
  Normalize a convolution kernel direction over the in_features and spatial
  dimensions.
  """
  return v / np.sqrt(np.sum(np.square(v), (0, 1, 2)))

def _make_kernel(direction, scale):
  """
  Maps weightnorm parameterization (direction, scale) to standard
  parameterization. The direction has shape (spatial..., in_features,
  out_features), scale has shape (out_features,).
  """
  return scale * _l2_normalize(direction)


# 2D convolution Modules with weightnorm
class Conv(nn.Module):
  def apply(self,
            inputs,
            features,
            kernel_size=(3, 3),
            strides=None,
            transpose=False,
            dtype=np.float32,
            precision=None):
    inputs = np.asarray(inputs, dtype)
    strides = strides or (1,) * (inputs.ndim - 2)

    if transpose:
      conv = partial(lax.conv_transpose, strides=strides, padding='VALID',
                     precision=precision)
    else:
      conv = partial(lax.conv_general_dilated, window_strides=strides,
                     padding='VALID',
                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                     precision=precision)

    in_features = inputs.shape[-1]
    kernel_shape = kernel_size + (in_features, features)

    def initializer(key, shape):
      # A weightnorm initializer generating a (direction, scale, bias) tuple.
      # Note that the shape argument is not used.
      direction = nn.initializers.normal()(key, kernel_shape, dtype)
      unnormed_out = conv(inputs, _l2_normalize(direction))
      mean = np.mean(unnormed_out, (0, 1, 2))
      var  = np.std (unnormed_out, (0, 1, 2))
      return dict(direction=direction, scale=1 / var, bias=-mean / var)

    # We feed in None as a dummy shape argument to self.param.  Typically
    # Module.param assumes that the initializer takes in a shape argument but
    # None can be used as an escape hatch.
    params = self.param('weightnorm_params', None, initializer)
    direction, scale, bias = [params[k] for k in ('direction', 'scale', 'bias')]
    return conv(inputs, _make_kernel(direction, scale)) + bias

ConvTranspose = Conv.partial(transpose=True)
ConvOneByOne  = Conv.partial(kernel_size=(1, 1))

@nn.module
def ConvDown(inputs, features, kernel_size=(2, 3), *args, **kwargs):
  """
  Convolution with spatial padding so that information cannot flow upwards.
  """
  inputs = np.asarray(inputs, kwargs.get('dtype', np.float32))
  k_h, k_w = kernel_size
  assert k_w % 2 == 1, "kernel width must be odd."
  inputs = spatial_pad((k_h - 1, 0), (k_w // 2, k_w // 2), inputs)
  return Conv(inputs, features, kernel_size, *args, **kwargs)

@nn.module
def ConvDownRight(inputs, features, kernel_size=(2, 2), *args, **kwargs):
  """
  Convolution with spatial padding so that information cannot flow to the left
  or upwards.
  """
  inputs = np.asarray(inputs, kwargs.get('dtype', np.float32))
  k_h, k_w = kernel_size
  inputs = spatial_pad((k_h - 1, 0), (k_w - 1, 0), inputs)
  return Conv(inputs, features, kernel_size, *args, **kwargs)

@nn.module
def ConvTransposeDown(
    inputs, features, kernel_size=(2, 3), *args, **kwargs):
  """
  Transpose convolution with output (negative) spatial padding so that
  information cannot flow upwards.
  """
  inputs = np.asarray(inputs, kwargs.get('dtype', np.float32))
  out = ConvTranspose(inputs, features, kernel_size, *args, **kwargs)
  k_h, k_w = kernel_size
  assert k_w % 2 == 1, "kernel width must be odd."
  return spatial_pad((0, -(k_h - 1)), (-(k_w // 2), -(k_w // 2)), out)

@nn.module
def ConvTransposeDownRight(
    inputs, features, kernel_size=(2, 2), *args, **kwargs):
  """
  Transpose convolution with output (negative) spatial padding so that
  information cannot flow to the left or upwards.
  """
  inputs = np.asarray(inputs, kwargs.get('dtype', np.float32))
  out = ConvTranspose(inputs, features, kernel_size, *args, **kwargs)
  k_h, k_w = kernel_size
  return spatial_pad((0, -(k_h - 1)), (0, -(k_w - 1)), out)

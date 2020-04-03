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

# General utils
def center(images):
  """Mapping from {0, 1, ..., 255} to {-1, -1 + 1/127.5, ..., 1}."""
  return images / 127.5 - 1

def concat_elu(x):
  return nn.elu(np.concatenate((x, -x), -1))

def spatial_pad(pad_vertical, pad_horizontal, operand):
  """
  Wrapper around lax.pad which pads spatial dimensions (horizontal and vertical)
  with zeros, without any interior padding.
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
            kernel_size,
            strides=None,
            transpose=False,
            init_scale=1.,
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
      return dict(direction=direction, scale=init_scale / var, bias=-mean / var)

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
    inputs, features, kernel_size=(2, 3), strides=(2, 2), **kwargs):
  """
  Transpose convolution with output (negative) spatial padding so that
  information cannot flow upwards. Strides are (2, 2) by default which implies
  the spatial dimensions of the output shape are double those of the input
  shape.
  """
  inputs = np.asarray(inputs, kwargs.get('dtype', np.float32))
  out = ConvTranspose(inputs, features, kernel_size, *args, **kwargs)
  k_h, k_w = kernel_size
  assert k_w % 2 == 1, "kernel width must be odd."
  return spatial_pad((0, -(k_h - 1)), (-(k_w // 2), -(k_w // 2)), out)

@nn.module
def ConvTransposeDownRight(
    inputs, features, kernel_size=(2, 2), strides=(2, 2), **kwargs):
  """
  Transpose convolution with output (negative) spatial padding so that
  information cannot flow to the left or upwards. Strides are (2, 2) by default
  which implies the spatial dimensions of the output shape are double those of
  the input shape.
  """
  inputs = np.asarray(inputs, kwargs.get('dtype', np.float32))
  out = ConvTranspose(inputs, features, kernel_size, *args, **kwargs)
  k_h, k_w = kernel_size
  return spatial_pad((0, -(k_h - 1)), (0, -(k_w - 1)), out)


# Resnet blocks
@nn.module
def Res(conv_module, inputs, aux=None, nonlinearity=concat_elu, dropout_p=0.):
  c = inputs.shape[-1]
  y = conv_module(nonlinearity(inputs), c)
  if aux is not None:
    y = nonlinearity(y + ConvOneByOne(nonlinearity(aux), c))

  if dropout_p > 0:
    y = nn.dropout(y, dropout_p)

  # Set init_scale=0.1 so that the res block is close to the identity at
  # initialization.
  a, b = np.split(conv_module(y, 2 * c, init_scale=0.1), 2, axis=-1)
  return inputs + a * sigmoid(b)

ResDown = nn.module(partial(Res, ConvDown))
ResDownRight = nn.module(partial(Res, ConvDownRight))


# Logistic mixture distribution utilities
def conditional_params_from_outputs(img, theta):
  """
  Maps an image `img` and the PixelCNN++ convnet output `theta` to conditional
  parameters for a mixture of k logistics over each pixel. Note this method
  won't work on a batch, so use jax.vmap!

  Returns a tuple `(means, inverse_scales, logit_weights)` where `means` and
  `inverse_scales` are the conditional means and inverse scales of each mixture
  component (for each pixel-channel) and `logit_weights` are the logits of the
  mixture weights (for each pixel). The have the following shapes:

    means.shape == inv_scales.shape == (k, h, w, c)
    logit_weights.shape == (k, h, w)

  Args:
    img: an image with shape (h, w, c)
    theta: outputs of PixelCNN++ neural net with shape (h, w, (1 + 3 * c) * k)
  Returns:
    The tuple `(means, inverse_scales, logit_weights)`.
  """
  h, w, c = img.shape
  assert theta.shape[2] % (3 * c + 1) == 0
  k = theta.shape[2] // (3 * c + 1)

  logit_weights, theta = theta[..., :k], theta[..., k:]
  assert theta.shape == (h, w, 3 * c * k)

  # Each of m, s and t must have shape (k, h, w, c), we effectively spread the
  # last dimension of theta out into c, k, 3, move the k dimension to the front
  # and split along the 3 dimension.
  m, s, t = np.moveaxis(np.reshape(theta, (h, w, c, k, 3)), (3, 4), (0, 1))
  t = np.tanh(t)

  # now condition the means for the last 2 channels (assuming c == 3)
  mean_red   = m[..., 0]
  mean_green = m[..., 1] + t[..., 0] * img[..., 0]
  mean_blue  = m[..., 2] + t[..., 1] * img[..., 0] + t[..., 2] * img[..., 1]
  means = np.stack((mean_red, mean_green, mean_blue), axis=-1)
  return means, softplus(s), np.moveaxis(logit_weights, -1, 0)

def logprob_from_conditional_params(images, means, inv_scales, logit_weights):
  """
  Computes the log-likelihoods of images given the conditional logistic mixture
  parameters produced by `conditional_params_from_outputs`. The 8-bit pixel
  values are assumed to be scaled so that they are in the discrete set

    {-1, -1 + 1/127.5, -1 + 2/127.5, ..., 1 - 1/127.5, 1}
  """
  # Add a 'mixture' dimension to images
  images = np.expand_dims(images, 1)

  # Compute log-probabilities for each mixture component, pixel and channel by
  # computing the difference between the logistic cdf half a level above and
  # half a level below the image value. Use np.where to manually set
  # cdf(1) = 1 and cdf(-1) = 0.
  cum_prob = lambda offset: sigmoid((images - means + offset) * inv_scales)
  upper_cum_p = np.where(images ==  1, 1, cum_prob( 1 / 255))
  lower_cum_p = np.where(images == -1, 0, cum_prob(-1 / 255))
  all_logprobs = np.sum(
      np.log(np.maximum(upper_cum_p - lower_cum_p, 1e-12)), -1)

  # Sum over the channel dimension because mixture components are shared
  # across channels.
  logprobs = np.sum(all_logprobs, -1)

  # Normalize the mixture weights
  log_mix_coeffs = logit_weights - logsumexp(logit_weights, -3, keepdims=True)

  # Finally marginalize out mixture components and sum over pixels
  return np.sum(logsumexp(log_mix_coeffs + logprobs, -3), (-2, -1))


# High level model definition
@nn.module
def PixelCNNPP(images, depth=5, features=160, k=10, dropout_p=.5):
  ResDown = Resdown.partial(dropout_p=dropout_p)
  ResDownRight = ResDownRight.partial(dropout_p=dropout_p)

  ConvDown = ConvDown.partial(features=features)
  ConvDownRight = ConvDownRight.partial(features=features)

  # Conv Modules which halve or double the spatial dimensions
  HalveDown = ConvDown.partial(strides=(2, 2))
  HalveDownRight = ConvDownRight.partial(strides=(2, 2))

  DoubleDown = ConvTransposeDown.partial(features=features)
  DoubleDownRight = ConvTransposeDownRight.partial(features=features)

  # Stack of (down, down_right) pairs, where information flows downwards through
  # down and downwards and to the right through down_right.
  stack = []

  # This has side effects on the stack
  def fwd_block():
    for _ in range(depth):
      down, down_right = stack[-1]
      stack.append((ResDown(down), ResDownRight(down_right, down)))
    return stack[-1]

  # This also has side effects on the stack
  def rev_block(depth, down, down_right):
    for _ in range(depth):
      down_saved, down_right_saved = stack.pop()
      down = ResDown(down, down_saved)
      down_right = ResDownRight(
          down_right, np.concatenate((down, down_right_saved), -1))
    return down, down_right

  images = np.pad(images, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1)

  # Forward pass
  stack.append((down_shift(ConvDown(images, kernel_size=(2, 3))),
                down_shift(ConvDown(images, kernel_size=(1, 3)))
                + right_shift(ConvDownRight(images, kernel_size=(2, 1)))))

  down, down_right = fwd_block()

  # 32 x 32  -->  16 x 16
  stack.append((HalveDown(down), HalveDownRight(down_right)))
  down, down_right = fwd_block()

  # 16 x 16  -->  8 x 8
  stack.append((HalveDown(down), HalveDownRight(down_right)))
  down, down_right = fwd_block()

  u, ul, us, uls = ResnetDownBlock(depth)(us.pop(), uls.pop(), us, uls)
  u, ul, us, uls = ResnetDownBlock(depth + 1)(
      DoubleDown()(u), DoubleDownRight()(ul), us, uls)
  u, ul, us, uls = ResnetDownBlock(depth + 1)(
      DoubleDown()(u), DoubleDownRight()(ul), us, uls)
  assert len(stack) == 0

  return ConvOneByOne(nn.elu(ul), 10 * k)

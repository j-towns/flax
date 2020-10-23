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

from jax import lax, numpy as jnp
import numpy as onp
from flax import struct
from flax.optim import OptimizerDef


@struct.dataclass
class _LadamHyperParams:
  learning_rate: onp.ndarray
  beta1: onp.ndarray
  beta2: onp.ndarray
  weight_decay: onp.ndarray


@struct.dataclass
class _LadamParamState:
  grad_ema: onp.ndarray
  grad_mad_ema: onp.ndarray


class Ladam(OptimizerDef):
  """Linear Adam optimizer."""

  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.9,
               weight_decay=0.0):
    """Constructor for the Ladam optimizer.
    Args:
      learning_rate: the step size used to update the parameters.
      beta1: the coefficient used for the moving average of the
        gradient (default: 0.9).
      beta2: the coefficient used for the moving average of the
        gradient MAD (default: 0.9). Plan is to merge this with beta1.
      weight_decay: AdamW style weight decay rate
        (relative to learning rate).
    """
    hyper_params = _LadamHyperParams(learning_rate, beta1, beta2, weight_decay)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _LadamParamState(jnp.zeros_like(param), jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    beta1 = hyper_params.beta1
    beta2 = hyper_params.beta2
    weight_decay = hyper_params.weight_decay

    grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
    # bias correction
    t = step + 1.
    grad_ema_corr = grad_ema / (1 - beta1 ** t)

    grad_mad = jnp.abs(grad - grad_ema_corr)
    grad_mad_ema = beta2 * state.grad_mad_ema + (1. - beta2) * grad_mad
    grad_mad_ema_corr = grad_mad_ema / (1 - beta2 ** t)

    denom = jnp.abs(grad_ema_corr) + grad_mad_ema_corr
    new_param = param - hyper_params.learning_rate * grad_ema_corr / jnp.where(denom, denom, 1)
    new_param -= hyper_params.learning_rate * weight_decay * param
    new_state = _LadamParamState(grad_ema, grad_mad_ema)
    return new_param, new_state

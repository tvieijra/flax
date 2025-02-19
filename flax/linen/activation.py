# Copyright 2021 The Flax Authors.
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

"""Activation functions.
"""

# pylint: disable=unused-import
# re-export activation functions from jax.nn
from jax.nn import celu
from jax.nn import elu
from jax.nn import gelu
from jax.nn import glu
from jax.nn import leaky_relu
from jax.nn import log_sigmoid
from jax.nn import log_softmax
from jax.nn import normalize
from jax.nn import relu
from jax.nn import sigmoid
from jax.nn import soft_sign
from jax.nn import softmax
from jax.nn import softplus
from jax.nn import swish
from jax.nn import silu
from jax.nn import selu
from jax.nn import hard_tanh
from jax.nn import relu6
from jax.nn import hard_sigmoid
from jax.nn import hard_swish

from jax.numpy import tanh
# pylint: enable=unused-import

from typing import Any

from flax.linen.module import Module, compact
import jax.numpy as jnp


Array = Any


class PReLU(Module):
  """Parametric Rectified Linear Unit (PReLU) activation function.

  Attributes:
    negative_slope_init: the value to initialize the negative slope.
  """
  negative_slope_init: float = 0.01
  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies an activation to the inputs.

    Args:
      inputs: the nd-array to apply the activation function to.

    Returns:
      The transformed input.
    """
    negative_slope = self.param(
      'negative_slope',
      lambda k: jnp.asarray(self.negative_slope_init, jnp.float32)
    )
    return jnp.where(inputs >= 0, inputs, jnp.asarray(negative_slope, inputs.dtype) * inputs)

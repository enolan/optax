"""
Implementation of "SOAP: Improving and Stabilizing Shampoo using Adam" and the
Adafactor variant.
"""

from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import chex

from optax._src import alias, base, combine, numerics, transform
from optax import tree_utils as otu


class SoapPreconditionerState(NamedTuple):
  """State for a single parameter's preconditioners."""

  l: chex.Array  # Shape (*leading_dims, m, m)
  r: chex.Array  # Shape (*leading_dims, n, n)
  ql: chex.Array  # Shape (*leading_dims, m, m)
  qr: chex.Array  # Shape (*leading_dims, n, n)


class SoapState(NamedTuple):
  """Top-level state for the SOAP wrapper."""

  count: chex.Array  # Number of updates applied
  param_states: Any  # PyTree of PreconditionerState or None, mirroring params
  inner_opt_state: Any  # Inner optimizer's global state


def soap_wrapper(
  inner_opt: base.GradientTransformation,
  b2: float = 0.99,
  precon_update_freq: int = 10,
) -> base.GradientTransformation:
  """
  SOAP optimizer that wraps an inner optimizer, running it in the eigenbasis of
  a Shampoo-like preconditioner. The paper considers Adam and Adafactor as
  inner optimizers, but this implementation is agnostic to the inner optimizer.

  Args:
    inner_opt: A base.GradientTransformation (e.g. optax.adam(...)) that will
    be applied in the computed eigenbasis.
    b2: Decay rate for the L & R preconditioners.
    precon_update_freq: How often to update the eigenvectors of L & R.

  Returns:
    A `GradientTransformation` instance that applies the given inner optimizer
    in the Shampoo preconditioner's eigenbasis.
  """

  def init_precon_states(
    param: chex.Array,
  ) -> Optional[SoapPreconditionerState]:
    """Initialize the Shampoo-like preconditioner for a single param."""
    # SOAP is designed around 2D matrices. For vectors and scalars, we simply
    # run the inner optimizer in the original basis. For higher-dimensional
    # arrays, we only compute preconditioners for the last two dimensions,
    # effectively treating the array as a collection of 2D matrices.
    ndim = param.ndim
    if ndim < 2:
      return None

    m, n = param.shape[-2:]
    zero_l = jnp.zeros((m, m), dtype=param.dtype)
    zero_r = jnp.zeros((n, n), dtype=param.dtype)
    zero_ql = jnp.zeros((m, m), dtype=param.dtype)
    zero_qr = jnp.zeros((n, n), dtype=param.dtype)

    if ndim > 2:
      leading_shape = param.shape[:-2]
      zero_l = jnp.broadcast_to(zero_l, (*leading_shape, m, m))
      zero_r = jnp.broadcast_to(zero_r, (*leading_shape, n, n))
      zero_ql = jnp.broadcast_to(zero_ql, (*leading_shape, m, m))
      zero_qr = jnp.broadcast_to(zero_qr, (*leading_shape, n, n))

    return SoapPreconditionerState(zero_l, zero_r, zero_ql, zero_qr)

  def init_fn(params):
    """Initialize all preconditioners and the inner optimizer state."""
    precon_states = jax.tree.map(init_precon_states, params)
    return SoapState(
      count=jnp.zeros([], jnp.int32),
      param_states=precon_states,
      inner_opt_state=inner_opt.init(params),
    )

  def project_to_eigenspace(
    x: chex.Array, param_state: Optional[SoapPreconditionerState]
  ) -> chex.Array:
    """Projects gradients from original space into the eigenbasis."""
    if param_state is None:
      return x
    *leading_dims, m, n = x.shape
    ql = param_state.ql
    qr = param_state.qr
    chex.assert_shape(ql, (*leading_dims, m, m))
    chex.assert_shape(qr, (*leading_dims, n, n))
    return jnp.einsum("...ji,...jk,...kl->...il", ql, x, qr)

  def project_from_eigenspace(
    x: chex.Array, param_state: Optional[SoapPreconditionerState]
  ) -> chex.Array:
    """Projects updates from the eigenbasis back into the original parameter
    space."""
    if param_state is None:
      return x
    *leading_dims, m, n = x.shape
    ql = param_state.ql
    qr = param_state.qr
    chex.assert_shape(ql, (*leading_dims, m, m))
    chex.assert_shape(qr, (*leading_dims, n, n))
    return jnp.einsum("...ij,...jk,...lk->...il", ql, x, qr)

  def update_preconditioners(
    g: chex.Array,
    param_state: Optional[SoapPreconditionerState],
    count_inc: chex.Array,
  ) -> Optional[SoapPreconditionerState]:
    """Update the L, R, QL and QR preconditioners using the new gradient."""
    if param_state is None:
      return None

    l, r, ql, qr = param_state
    *leading_dims, m, n = g.shape
    chex.assert_shape([l, ql], (*leading_dims, m, m))
    chex.assert_shape([r, qr], (*leading_dims, n, n))

    # Update our preconditioner moving averages
    new_l = b2 * l + (1 - b2) * jnp.einsum("...ik,...jk->...ij", g, g)
    new_r = b2 * r + (1 - b2) * jnp.einsum("...ki,...kj->...ij", g, g)

    def compute_eigenvectors():
      # We compute exact eigenvectors on the first step and do power iteration
      # on subsequent steps.
      l_hat = otu.tree_bias_correction(new_l, b2, count_inc)
      r_hat = otu.tree_bias_correction(new_r, b2, count_inc)

      def exact_eig():
        new_ql = jnp.linalg.eigh(l_hat)[1]
        new_qr = jnp.linalg.eigh(r_hat)[1]
        return new_ql, new_qr

      def power_iter():
        new_ql = jnp.linalg.qr(jnp.einsum("...ij,...jk->...ik", l_hat, ql))[0]
        new_qr = jnp.linalg.qr(jnp.einsum("...ij,...jk->...ik", r_hat, qr))[0]
        return new_ql, new_qr

      return jax.lax.cond(count_inc == 1, exact_eig, power_iter)

    new_ql, new_qr = jax.lax.cond(
      (count_inc % precon_update_freq) == 1,
      compute_eigenvectors,
      lambda: (ql, qr),
    )

    return SoapPreconditionerState(new_l, new_r, new_ql, new_qr)

  def update_fn(updates, state, params=None):
    """Perform one step of SOAP:
    1) Update preconditioners.
    2) Project grads to eigenbasis for inner_opt.
    3) Run inner_opt in eigenspace.
    4) Project inner_opt updates back to original space.
    """
    del params  # unused

    count_inc = numerics.safe_increment(state.count)

    # 1) Update L, R, QL and QR across the whole param tree
    new_param_states = jax.tree.map(
      lambda ps, g: update_preconditioners(g, ps, count_inc),
      state.param_states,
      updates,
      is_leaf=lambda x: isinstance(x, SoapPreconditionerState) or x is None,
    )

    # 2) Project the incoming gradients into the eigenbasis before sending to
    # inner_opt
    projected_grads = jax.tree.map(
      lambda ps, g: project_to_eigenspace(g, ps),
      new_param_states,
      updates,
      is_leaf=lambda x: isinstance(x, SoapPreconditionerState) or x is None,
    )

    # Delegate to the inner optimizer in eigenspace
    eigenspace_updates, new_inner_state = inner_opt.update(
      projected_grads, state.inner_opt_state, params=None
    )

    # 3) Project the inner_opt's updates back to original space
    final_updates = jax.tree.map(
      lambda ps, e_up: project_from_eigenspace(e_up, ps),
      new_param_states,
      eigenspace_updates,
      is_leaf=lambda x: isinstance(x, SoapPreconditionerState) or x is None,
    )

    new_state = SoapState(
      count=count_inc,
      param_states=new_param_states,
      inner_opt_state=new_inner_state,
    )
    return final_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


def soap_adamw(
  learning_rate: base.ScalarOrSchedule,
  b1: float = 0.95,
  b2: float = 0.95,
  weight_decay: float = 0,
  weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
  eps: float = 1e-8,
  mu_dtype: Optional[jnp.dtype] = None,
  precon_update_freq: int = 10,
) -> base.GradientTransformation:
  """SOAP optimizer with Adam as the inner optimizer and weight decay.
  Analagous to `optax.adamw(...)`, but *not* equivalent to applying
  `soap_wrapper` to `optax.adamw(...)`, which would apply weight decay in the
  eigenbasis.

  Args:
    learning_rate: A fixed global scaling factor or schedule.
    b1: Decay rate for first moment.
    b2: Decay rate for second moment and preconditioners.
    eps: Small constant added to the denominator when scaling updates for
      numerical stability.
    mu_dtype: Optional `dtype` to be used for Adam's first order accumulator;
      if `None` then the `dtype` is inferred from `params` and `updates`.
    precon_update_freq: How often to update preconditioner eigenvectors.

  Returns:
    A `GradientTransformation` instance.
  """
  inner_opt = alias.adam(
    learning_rate=1.0, b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype
  )
  txs = [soap_wrapper(inner_opt, b2=b2, precon_update_freq=precon_update_freq)]
  if weight_decay > 0:
    txs.append(
      transform.add_decayed_weights(
        -weight_decay, mask=weight_decay_mask
      )
    )
  txs.append(transform.scale_by_learning_rate(learning_rate, flip_sign=False))
  return combine.chain(*txs)

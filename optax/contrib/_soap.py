"""
Implementation of "SOAP: Improving and Stabilizing Shampoo using Adam" and the
Adafactor variant.
"""

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import chex

from optax._src import base, combine, numerics, transform
from optax import tree_utils as otu

# TODO: try rewriting this as a wrapper around an arbitrary optimizer. would mean the difference
# between keeping momentum in original basis and second moment in eigenbasis vs both in eigenbasis.
# also makes code substantially simpler.


class PreconditionerState(NamedTuple):
    """State for a single parameter's preconditioners."""

    # Left and right preconditioners
    l: chex.Array  # Shape (*leading_dims, m, m)
    r: chex.Array  # Shape (*leading_dims, n, n)
    # Eigenvectors of the left and right preconditioners
    ql: chex.Array  # Shape (*leading_dims, m, m)
    qr: chex.Array  # Shape (*leading_dims, n, n)


class StateForParam(NamedTuple):
    """State for a single parameter."""

    preconditioner: Optional[PreconditionerState]  # If None, do regular Adam
    # Adam moments
    mu: chex.Array  # Shape (*leading_dims, m, n), in original basis
    nu: chex.Array  # Shape (*leading_dims, m, n), in eigenbasis


class ScaleBySOAPState(NamedTuple):
    """State for the SOAP algorithm."""

    count: chex.Array  # Shape [], dtype int32
    param_states: Any  # Pytree mirroring the parameter tree, with StateForParam leaves


def project_to_eigenspace(x: chex.Array, param_state: StateForParam) -> chex.Array:
    """Projects an array into eigenspace using the preconditioners' eigenvectors.

    Args:
      x: Array of shape (..., m, n) to project
      param_state: StateForParam containing the preconditioner state

    Returns:
      Array of same shape as x, projected into eigenspace
    """
    if param_state.preconditioner is None:
        return x
    *leading_dims, m, n = x.shape
    ql = param_state.preconditioner.ql
    qr = param_state.preconditioner.qr
    chex.assert_shape(ql, (*leading_dims, m, m))
    chex.assert_shape(qr, (*leading_dims, n, n))
    return jnp.einsum("...ji,...jk,...kl->...il", ql, x, qr)


def project_from_eigenspace(x: chex.Array, param_state: StateForParam) -> chex.Array:
    """Projects an array from eigenspace back to original space.

    Args:
      x: Array of shape (..., m, n) to project
      param_state: StateForParam containing the preconditioner state

    Returns:
      Array of same shape as x, projected back to original space
    """
    if param_state.preconditioner is None:
        return x

    *leading_dims, m, n = x.shape
    ql = param_state.preconditioner.ql
    qr = param_state.preconditioner.qr
    chex.assert_shape(ql, (*leading_dims, m, m))
    chex.assert_shape(qr, (*leading_dims, n, n))
    return jnp.einsum("...ij,...jk,...lk->...il", ql, x, qr)


def scale_by_soap(
    beta1: float = 0.95,
    beta2: float = 0.99,
    epsilon: float = 1e-8,
    precon_update_freq: int = 10,
) -> base.GradientTransformation:
    # FIXME improve docstring
    """SOAP optimizer that combines Adam with Shampoo's preconditioner.

    Args:
      beta1: Decay rate for first moment.
      beta2: Decay rate for second moment and preconditioners.
      epsilon: Small constant for numerical stability.
      precon_update_freq: How often to update preconditioner eigenvectors.
    """

    # TODO, max preconditioner dimension, configurable dtypes for preconditioners and eigenvectors
    def init_fn(params):
        def init_states(param: chex.Array) -> StateForParam:
            mu = jnp.zeros(param.shape, dtype=param.dtype)
            nu = jnp.zeros(param.shape, dtype=param.dtype)

            # SOAP is designed around 2D matrices. For vectors and scalars, we fall
            # back to regular Adam, and for higher-dimensional arrays, we only
            # compute preconditioners for the last two dimensions, effectively
            # treating the array as a collection of 2D matrices.
            ndim = param.ndim
            if ndim < 2:
                return StateForParam(None, mu, nu)

            # For any ndim, we'll work with the last two dimensions
            m, n = param.shape[-2:]

            # Initialize basic 2D preconditioners
            l = jnp.zeros((m, m), dtype=param.dtype)
            r = jnp.zeros((n, n), dtype=param.dtype)
            ql = jnp.zeros((m, m), dtype=param.dtype)
            qr = jnp.zeros((n, n), dtype=param.dtype)

            if ndim > 2:
                # Add leading dims to broadcast preconditioners
                leading_shape = param.shape[:-2]
                l = jnp.broadcast_to(l, (*leading_shape, m, m))
                r = jnp.broadcast_to(r, (*leading_shape, n, n))
                ql = jnp.broadcast_to(ql, (*leading_shape, m, m))
                qr = jnp.broadcast_to(qr, (*leading_shape, n, n))

            return StateForParam(PreconditionerState(l, r, ql, qr), mu, nu)

        # Initialize preconditioners across the parameter tree
        states = jax.tree.map(init_states, params)

        return ScaleBySOAPState(count=jnp.zeros([], jnp.int32), param_states=states)

    def update_fn(updates, state, params=None):
        del params  # Unused

        count_inc = numerics.safe_increment(state.count)

        def update_preconditioners(
            g: base.Updates, param_state: StateForParam
        ) -> StateForParam:
            """Update L and R matrices and their eigenvectors."""
            if param_state.preconditioner is None:  # Handle non-matrix parameters
                return param_state
            else:
                l, r, ql, qr = param_state.preconditioner

                # For a param of shape (..., m, n):
                # l, ql should be (..., m, m)
                # r, qr should be (..., n, n)
                # g should be (..., m, n)
                *leading_dims, m, n = g.shape
                chex.assert_equal_shape([g, param_state.mu, param_state.nu])
                chex.assert_shape([l, ql], (*leading_dims, m, m))
                chex.assert_shape([r, qr], (*leading_dims, n, n))

                # Update preconditioners (step 13-14)
                # TODO bias correction
                new_l = beta2 * l + (1 - beta2) * jnp.einsum("...ik,...jk->...ij", g, g)
                new_r = beta2 * r + (1 - beta2) * jnp.einsum("...ki,...kj->...ij", g, g)

                def compute_eigenvectors():
                    l_hat = otu.tree_bias_correction(new_l, beta2, count_inc)
                    r_hat = otu.tree_bias_correction(new_r, beta2, count_inc)

                    # Compute exact eigenvectors for first update, then use iterative method
                    def exact_eig():
                        new_ql = jnp.linalg.eigh(l_hat)[1]
                        new_qr = jnp.linalg.eigh(r_hat)[1]
                        return new_ql, new_qr

                    def power_iter():
                        new_ql = jnp.linalg.qr(
                            jnp.einsum("...ij,...jk->...ik", l_hat, ql)
                        )[0]
                        new_qr = jnp.linalg.qr(
                            jnp.einsum("...ij,...jk->...ik", r_hat, qr)
                        )[0]
                        return new_ql, new_qr

                    return jax.lax.cond(state.count == 0, exact_eig, power_iter)

                new_ql, new_qr = jax.lax.cond(
                    state.count % precon_update_freq == 0,
                    compute_eigenvectors,
                    lambda: (ql, qr),
                )

                return param_state._replace(
                    preconditioner=PreconditionerState(new_l, new_r, new_ql, new_qr),
                )

        def update_moments(
            g: base.Updates, param_state: StateForParam
        ) -> StateForParam:
            """Updates Adam moments."""
            # Project gradient to eigenspace
            g_eigen = project_to_eigenspace(g, param_state)

            # Update moments
            new_mu = beta1 * param_state.mu + (1 - beta1) * g  # in original space
            new_nu = (
                beta2 * param_state.nu + (1 - beta2) * g_eigen * g_eigen
            )  # in eigenspace

            return param_state._replace(mu=new_mu, nu=new_nu)

        updated_precon_states = jax.tree.map(
            update_preconditioners,
            updates,
            state.param_states,
            is_leaf=lambda x: isinstance(x, StateForParam),
        )
        updated_moment_states = jax.tree.map(
            update_moments,
            updates,
            updated_precon_states,
            is_leaf=lambda x: isinstance(x, StateForParam),
        )

        # Compute preconditioned updates (steps 8-11)
        def compute_precond_updates(param_state: StateForParam) -> base.Updates:
            m_hat = otu.tree_bias_correction(param_state.mu, beta1, count_inc)
            v_hat = otu.tree_bias_correction(param_state.nu, beta2, count_inc)

            if param_state.preconditioner is None:
                # Regular Adam for non-matrix parameters
                return m_hat / (jnp.sqrt(v_hat) + epsilon)
            else:
                # Project m_hat into eigenspace
                m_eigen = project_to_eigenspace(m_hat, param_state)

                # Apply bias-corrected Adam in eigenspace
                update_eigen = m_eigen / (jnp.sqrt(v_hat) + epsilon)

                # Project back to original space
                update = project_from_eigenspace(update_eigen, param_state)

                chex.assert_equal_shape([param_state.mu, update])
                return update

        precond_updates = jax.tree.map(
            compute_precond_updates,
            updated_moment_states,
            is_leaf=lambda x: isinstance(x, StateForParam),
        )

        new_state = ScaleBySOAPState(
            count=count_inc,
            param_states=updated_moment_states,
        )

        return precond_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def soap(
    learning_rate: base.ScalarOrSchedule,
    beta1: float = 0.95,
    beta2: float = 0.99,
    epsilon: float = 1e-8,
    precon_update_freq: int = 10,
) -> base.GradientTransformation:
    """SOAP optimizer that combines Adam with Shampoo's preconditioner.

    Args:
      learning_rate: A fixed global scaling factor or schedule.
      beta1: Decay rate for first moment.
      beta2: Decay rate for second moment and preconditioners.
      epsilon: Small constant for numerical stability.
      precon_update_freq: How often to update preconditioner eigenvectors.

    Returns:
      A `GradientTransformation` instance.
    """
    return combine.chain(
        scale_by_soap(
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            precon_update_freq=precon_update_freq,
        ),
        transform.scale_by_learning_rate(learning_rate),
    )

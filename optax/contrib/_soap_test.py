"""Tests for the SOAP optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base
from optax._src import update
from optax.contrib import _soap


class SOAPTest(chex.TestCase):
    """Tests for the SOAP optimizer."""

    def setUp(self):
        super().setUp()
        # Define some test parameters
        self.params_2d = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.params_1d = jnp.array([1.0, 2.0, 3.0])
        self.params_scalar = jnp.array(1.0)
        self.params_4d = jnp.ones((2, 3, 4, 5))
        self.params_tree = {
            "a": self.params_2d,
            "b": self.params_1d,
            "c": self.params_scalar,
            "d": self.params_4d,
        }

    @chex.all_variants
    def test_init_shape_handling(self):
        """Tests initialization handles different parameter shapes
        correctly."""
        opt = _soap.scale_by_soap()
        state = self.variant(opt.init)(self.params_tree)

        # Check state contains expected fields
        self.assertIsInstance(state, _soap.ScaleBySOAPState)
        chex.assert_tree_all_finite(state)

        # 2D+ params should have preconditioners
        self.assertIsNotNone(state.param_states["a"].preconditioner)
        self.assertIsNotNone(state.param_states["d"].preconditioner)

        # 1D and scalar params should have None preconditioners
        self.assertIsNone(state.param_states["b"].preconditioner)
        self.assertIsNone(state.param_states["c"].preconditioner)

        # All preconditioners should have the correct shapes
        get_l_ql = lambda precond: (precond.preconditioner.l, precond.preconditioner.ql)
        get_r_qr = lambda precond: (precond.preconditioner.r, precond.preconditioner.qr)

        chex.assert_shape(get_l_ql(state.param_states["a"]), (2, 2))
        chex.assert_shape(get_r_qr(state.param_states["a"]), (3, 3))
        chex.assert_shape(get_l_ql(state.param_states["d"]), (2, 3, 4, 4))
        chex.assert_shape(get_r_qr(state.param_states["d"]), (2, 3, 5, 5))

    @chex.all_variants
    @parameterized.parameters(3, 5)
    def test_precon_update_frequency(self, precon_update_freq):
        """Tests preconditioner updates happen at correct frequency."""
        opt = _soap.scale_by_soap(precon_update_freq=precon_update_freq)
        params = self.params_2d
        grads = jnp.ones_like(params)

        state = self.variant(opt.init)(params)
        old_Q_L = state.param_states.preconditioner.ql
        old_Q_R = state.param_states.preconditioner.qr

        # First update should always update preconditioners
        updates, state = self.variant(opt.update)(grads, state)
        self.assertFalse(jnp.array_equal(old_Q_L, state.param_states.preconditioner.ql))
        self.assertFalse(jnp.array_equal(old_Q_R, state.param_states.preconditioner.qr))

        # Next updates should only update every precon_update_freq steps
        old_Q_L = state.param_states.preconditioner.ql
        old_Q_R = state.param_states.preconditioner.qr
        for i in range(2 * precon_update_freq + 1):
            updates, state = self.variant(opt.update)(grads, state)
            if (i + 1) % precon_update_freq == 0:
                self.assertFalse(
                    jnp.array_equal(old_Q_L, state.param_states.preconditioner.ql)
                )
                self.assertFalse(
                    jnp.array_equal(old_Q_R, state.param_states.preconditioner.qr)
                )
            else:
                self.assertTrue(
                    jnp.array_equal(old_Q_L, state.param_states.preconditioner.ql)
                )
                self.assertTrue(
                    jnp.array_equal(old_Q_R, state.param_states.preconditioner.qr)
                )
            old_Q_L = state.param_states.preconditioner.ql
            old_Q_R = state.param_states.preconditioner.qr

    @chex.all_variants
    def test_optimization(self):
        """Tests SOAP can optimize a simple function."""

        def loss(params):
            l_a = jnp.sum(params["a"]) ** 2
            l_b = jnp.sum(params["b"]) ** 4
            l_c = jnp.abs(params["c"])
            l_d = jnp.sum(params["d"] ** 2)
            return l_a + l_b + l_c + l_d

        params = self.params_tree
        opt = _soap.soap(learning_rate=0.1)
        state = opt.init(params)

        for i in range(150):
            loss_val, grads = self.variant(jax.value_and_grad(loss))(params)
            print(f"step {i:3d}: loss: {loss_val:.3f}")
            updates, state = self.variant(opt.update)(grads, state)
            params = self.variant(update.apply_updates)(params, updates)

        self.assertLess(loss(params), 1)


if __name__ == "__main__":
    absltest.main()

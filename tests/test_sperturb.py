import astrodynx as adx
import jax.numpy as jnp
import diffrax
import jax


class TestSppropFinnal:
    def test_keplerian(self) -> None:
        def vector_field(t, x, args):
            acc = adx.gravity.point_mass_grav(t, x, args)
            return jnp.concatenate([x[3:], acc])

        t1 = jnp.pi * 0.5
        r0_vec = jnp.array([1.0, 0.0, 0.0])
        v0_vec = jnp.array([0.0, 0.9, 0.0])
        mu = 1.0
        x0 = jnp.concatenate([r0_vec, v0_vec])
        args = {"mu": 1.0}
        term = diffrax.ODETerm(vector_field)
        sol = adx.spprop_finnal(term, x0, t1, args=args)

        r_vec, v_vec = adx.kepler_prop(t1, r0_vec, v0_vec, mu)
        assert jnp.allclose(sol.ys[-1, :3], r_vec, atol=1e-6)
        assert jnp.allclose(sol.ys[-1, 3:], v_vec, atol=1e-6)

    def test_with_event(self) -> None:
        """Test that spprop_finnal works with events."""

        # Define vector field with gravity and J2 perturbation
        def vector_field(t, x, args):
            acc = adx.gravity.point_mass_grav(t, x, args)
            acc += adx.gravity.j2_acc(t, x, args)
            return jnp.concatenate([x[3:], acc])

        # Setup parameters
        args = {"mu": 1.0, "rmin": 0.7, "J2": 1e-6, "R_eq": 1.0}
        t1 = 3.14
        x0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.9, 0.0])
        event = diffrax.Event(adx.events.radius_toolow)
        term = diffrax.ODETerm(vector_field)

        # Solve using spprop_finnal with event
        sol = adx.spprop_finnal(term, x0, t1, args=args, event=event)

        # Check that only the final state is returned
        assert sol.ys.shape[0] == 1, "Should return only one state"

        # Check that the event was triggered (radius ≈ rmin)
        radius = jnp.linalg.norm(sol.ys[0, :3])
        assert jnp.isclose(radius, args["rmin"], atol=1e-2)

    def test_gradient(self) -> None:
        """Test that spprop_finnal is differentiable."""
        r_vec = jnp.array(
            [-0.24986234273434585, -0.69332384278075210, 4.9599012168662551e-3]
        )
        v_vec = jnp.array(
            [1.2189179487500401, 0.05977450696618754, -0.007101943980682161]
        )
        r0_vec = jnp.array(
            [-0.66234662571997105, 0.74919751798749190, -1.6259997018919074e-4]
        )
        v0_vec = jnp.array(
            [-0.8166746784630675, -0.32961417380268476, 0.006248107587795581]
        )
        deltat = 2.5803148345055149
        mu = 1.0

        def vector_field(t, x, args):
            acc = adx.gravity.point_mass_grav(t, x, args)
            return jnp.concatenate([x[3:], acc])

        args = {"mu": mu}
        x0 = jnp.concatenate([r0_vec, v0_vec])
        x1 = jnp.concatenate([r_vec, v_vec])
        term = diffrax.ODETerm(vector_field)
        sol = adx.spprop_finnal(term, x0, deltat, args=args)
        assert jnp.allclose(sol.ys[-1], x1, atol=1e-5)

        def yf(x):
            return adx.spprop_finnal(term, x, deltat, args=args).ys[-1]

        jac_auto = jax.jacrev(yf)(x0)
        jac_analytic = adx.twobody.dxdx0(r_vec, v_vec, r0_vec, v0_vec, deltat)
        assert jnp.allclose(jac_auto, jac_analytic, atol=1e-4)

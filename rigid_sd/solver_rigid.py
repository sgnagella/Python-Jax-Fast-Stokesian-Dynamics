"""Modified GMRES saddle-point solver for rigid-body Stokesian Dynamics.

Solution vector x has size (11·N_p + 6·N_rb):
    x[:6·N_p]        — particle forces / torques
    x[6·N_p:11·N_p]  — per-particle stresslets
    x[11·N_p:]        — rigid-body velocities [V_cm; Ω_cm] for each body

The saddle-point operator is:
    ax[:11·N_p]  = M_ff · x[:11·N_p]  +  K · x[11·N_p:]   (top block)
    ax[11·N_p:]  = Σ · x[:6·N_p]  −  Σ · R_FU · K · x[11·N_p:]   (bottom block)

where B·K replaces the original identity B, and Σ·B^T replaces B^T.
"""

from functools import partial

import jax.numpy as jnp
import jax.scipy as jscipy
from jax import Array, jit
from jax.typing import ArrayLike

from jfsd import mobility, resistance


@partial(jit, static_argnums=[0, 1, 5, 6, 7, 8])
def solve_linear_system_rigid(
    num_particles: int,
    num_bodies: int,
    rhs: ArrayLike,
    gridk: ArrayLike,
    precomputed: tuple,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    gauss_support: int,
    m_self: ArrayLike,
    K: ArrayLike,
    Sigma: ArrayLike,
    inter_body_res_functions: tuple,
    initial_guess: ArrayLike = None,
) -> tuple[Array, int]:
    """Solve the rigid-body saddle-point linear system Ax = b via GMRES.

    Parameters
    ----------
    num_particles : N_p, total number of beads.
    num_bodies : N_rb, number of rigid bodies.
    rhs : (11·N_p + 6·N_rb,) right-hand side vector.
    gridk : (grid_nx, grid_ny, grid_nz, 4) wave-space grid.
    precomputed : 19-tuple returned by utils.precompute().
    grid_nx, grid_ny, grid_nz : wave-space grid dimensions (static).
    gauss_support : Gaussian support size (static).
    m_self : (2,) self-mobility coefficients.
    K : (6·N_p, 6·N_rb) kinematic map — dynamic, rebuilt each step.
    Sigma : (6·N_rb, 6·N_p) = K^T — dynamic, rebuilt each step.
    inter_body_res_functions : 11-tuple of (n_pairs_nf,) resistance scalars,
        already zeroed for intra-body pairs via inter-body mask.
    initial_guess : optional warm-start vector of size (11·N_p + 6·N_rb).

    Returns
    -------
    x : (11·N_p + 6·N_rb,) solution vector.
    exitCode : int, 0 means GMRES converged.
    """
    N_p = num_particles
    N_rb = num_bodies
    sys_size = 11 * N_p + 6 * N_rb

    (
        all_indices_x, all_indices_y, all_indices_z,
        gaussian_grid_spacing1, gaussian_grid_spacing2,
        r, indices_i, indices_j,
        f1, f2, g1, g2, h1, h2, h3,
        r_lub, indices_i_lub, indices_j_lub,
        _res_functions_full,   # unused — we use inter_body_res_functions
    ) = precomputed

    def compute_saddle_rigid(x: ArrayLike) -> Array:
        """Matrix-free saddle-point operator A · x for the rigid-body system."""
        ax = jnp.zeros(sys_size)

        # ── 1. Far-field mobility: M_ff · x[:11·N_p] → ax[:11·N_p] ──────────
        ax = ax.at[:11 * N_p].set(
            mobility.generalized_mobility_periodic(
                N_p, grid_nx, grid_ny, grid_nz, gauss_support,
                gridk, m_self,
                all_indices_x, all_indices_y, all_indices_z,
                gaussian_grid_spacing1, gaussian_grid_spacing2,
                r, indices_i, indices_j,
                f1, f2, g1, g2, h1, h2, h3,
                x[:11 * N_p],
            )
        )

        # ── 2. B·K block: rigid-body velocities → particle velocity slot ──────
        V_rb = x[11 * N_p:]               # (6·N_rb,)
        U_particles = K @ V_rb             # (6·N_p,) particle velocities

        ax = ax.at[:6 * N_p].add(U_particles)

        # ── 3. Near-field lubrication — disabled for now ──────────────────────
        # When re-enabled: uncomment below and pass inter_body_res_functions.
        # F_lub = resistance.compute_lubrication_fu(
        #     U_particles, indices_i_lub, indices_j_lub,
        #     inter_body_res_functions, r_lub, N_p,
        # )
        # ax = ax.at[11 * N_p:].set(-(Sigma @ F_lub))

        # ── 4. Σ·B^T block: particle forces → rigid-body force slot ──────────
        ax = ax.at[11 * N_p:].add(Sigma @ x[:6 * N_p])

        return ax

    def compute_precond_rigid(x: jnp.ndarray) -> jnp.ndarray:
        """Self-mobility block-diagonal preconditioner for the rigid saddle point.

        Exactly inverts the approximate system [M_self K; Sigma 0] by:
            1. Computing rigid-body mobility A_rb = Sigma @ diag(1/m) @ K  (6·N_rb × 6·N_rb)
            2. Solving A_rb @ V = Sigma @ (inv_m * x_F) - x_V
            3. Recovering F = inv_m * (x_F - K @ V)

        The stresslet block passes through unchanged (identity approximation).
        """
        x_F = x[:6 * N_p]
        x_S = x[6 * N_p:11 * N_p]
        x_V = x[11 * N_p:]

        # Diagonal inverse self-mobility: [1/m_tt, 1/m_tt, 1/m_tt, 1/m_rr, ...] per particle
        inv_mtt = 1.0 / m_self[0]
        inv_mrr = 1.0 / m_self[1]
        inv_m = jnp.tile(
            jnp.concatenate([jnp.full(3, inv_mtt), jnp.full(3, inv_mrr)]),
            N_p,
        )  # (6·N_p,)

        # Approximate rigid-body mobility: A_rb = Sigma @ diag(inv_m) @ K
        A_rb = Sigma @ (inv_m[:, None] * K)  # (6·N_rb, 6·N_rb)

        # Solve A_rb @ V_sol = Sigma @ (inv_m * x_F) - x_V
        rhs_V = Sigma @ (inv_m * x_F) - x_V
        V_sol = jnp.linalg.solve(A_rb, rhs_V)

        # Recover particle forces: F_sol = inv_m * (x_F - K @ V_sol)
        F_sol = inv_m * (x_F - K @ V_sol)

        return jnp.concatenate([F_sol, x_S, V_sol])

    x, exitCode = jscipy.sparse.linalg.gmres(
        A=compute_saddle_rigid,
        b=rhs,
        x0=initial_guess,
        tol=1e-5,
        restart=50,
        M=compute_precond_rigid,
    )
    return x, exitCode

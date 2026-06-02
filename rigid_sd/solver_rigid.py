"""Modified GMRES saddle-point solver for rigid-body Stokesian Dynamics.

Solution vector x has size (11·N_p + 6·N_rb):
    x[:6·N_p]        — particle forces / torques
    x[6·N_p:11·N_p]  — per-particle stresslets
    x[11·N_p:]        — rigid-body velocities [V_cm; Ω_cm] for each body

The saddle-point operator is:
    ax[:11·N_p]  = M_ff · x[:11·N_p]  +  K · x[11·N_p:]   (top block)
    ax[11·N_p:]  = Σ · x[:6·N_p]  −  Σ · R_FU · K · x[11·N_p:]   (bottom block)

K and Σ are applied matrix-free via apply_K / apply_Sigma from assembly.py,
requiring only the (N_p, 3) relative-position array s and the integer assembly_ids.
Memory cost is O(N_p) instead of the O(N_p · N_rb) dense matrices.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import Array, jit
from jax.typing import ArrayLike

from jfsd import mobility, resistance
from rigid_sd.assembly import apply_K, apply_Sigma, build_rigid_body_mobility


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
    s: ArrayLike,
    assembly_ids: ArrayLike,
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
    m_self : (2,) self-mobility coefficients [m_tt, m_rr].
    s : (N_p, 3) relative bead positions s_i = r_i − com[assembly_ids[i]].
        Recomputed each step from current positions.
    assembly_ids : (N_p,) int array — assembly_ids[i] = rigid-body index for bead i.
    inter_body_res_functions : 11-tuple of (n_pairs_nf,) resistance scalars,
        already zeroed for intra-body pairs via inter-body mask.
    initial_guess : optional warm-start vector of size (11·N_p + 6·N_rb).

    Returns
    -------
    x : (11·N_p + 6·N_rb,) solution vector.
    exitCode : int, 0 means GMRES converged.
    """
    N_p  = num_particles
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
        U_particles = apply_K(x[11 * N_p:], s, assembly_ids)   # (6·N_p,)
        ax = ax.at[:6 * N_p].add(U_particles)

        # ── 3. Near-field lubrication — disabled for now ──────────────────────
        # When re-enabled: uncomment below.
        # F_lub = resistance.compute_lubrication_fu(
        #     U_particles, indices_i_lub, indices_j_lub,
        #     inter_body_res_functions, r_lub, N_p,
        # )
        # ax = ax.at[11 * N_p:].add(-apply_Sigma(F_lub, s, assembly_ids, N_rb))

        # ── 4. Σ·B^T block: particle forces → rigid-body force slot ──────────
        ax = ax.at[11 * N_p:].add(apply_Sigma(x[:6 * N_p], s, assembly_ids, N_rb))

        return ax

    def compute_precond_rigid(x: ArrayLike) -> Array:
        """Self-mobility block-diagonal preconditioner for the rigid saddle point.

        Exactly inverts [M_self K; Σ 0] by:
            1. Building per-body A_rb_α = Σ_i K_i^T diag(inv_m) K_i (block-diagonal)
            2. Solving N_rb independent 6×6 systems: A_rb_α @ V_α = rhs_V_α
            3. Recovering F = diag(inv_m) * (x_F − K @ V)

        A_rb_α is block-diagonal in 3×3 sub-blocks (translation-rotation cross
        terms vanish because Σ_i s_i = 0 by COM definition).
        """
        x_F = x[:6 * N_p]
        x_S = x[6 * N_p:11 * N_p]
        x_V = x[11 * N_p:]

        inv_mtt = 1.0 / m_self[0]
        inv_mrr = 1.0 / m_self[1]
        inv_m = jnp.tile(
            jnp.concatenate([jnp.full(3, inv_mtt), jnp.full(3, inv_mrr)]),
            N_p,
        )   # (6·N_p,)

        # Per-body 6×6 A_rb blocks — O(N_p) build, no K or Σ matrix needed
        A_blocks = build_rigid_body_mobility(s, assembly_ids, N_rb, inv_mtt, inv_mrr)
        # (N_rb, 6, 6)

        # rhs for rigid-body velocity solve: Σ @ (inv_m * x_F) - x_V
        rhs_V = apply_Sigma(inv_m * x_F, s, assembly_ids, N_rb) - x_V   # (6·N_rb,)

        # Batch 6×6 solve over all bodies
        rhs_V_mat = rhs_V.reshape(N_rb, 6)                                # (N_rb, 6)
        V_sol_mat = jax.vmap(jnp.linalg.solve)(A_blocks, rhs_V_mat)      # (N_rb, 6)
        V_sol = V_sol_mat.reshape(-1)                                      # (6·N_rb,)

        # Recover particle forces
        F_sol = inv_m * (x_F - apply_K(V_sol, s, assembly_ids))

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

"""Rigid-body kinematics: kinematic map K and projection Σ = K^T.

For a rigid body α containing particles {i}, with lab-frame relative positions s_i = r_i − r_cm_α:

    U_i = V_cm_α − skew(s_i) · Ω_cm_α   (linear velocity of bead i)
    Ω_i = Ω_cm_α                          (angular velocity of bead i)

This gives the kinematic map K (6·N_p × 6·N_rb) via U_particles = K · V_rb,
and its transpose Σ = K^T (6·N_rb × 6·N_p) maps particle forces/torques to
rigid-body generalized forces/torques:

    F_α = Σ_i F_i
    T_α = Σ_i (T_i + s_i × F_i)

Matrix-free interface (O(N_p) memory, no explicit K or Σ matrix):
    compute_relative_positions  — compute COM and relative positions s
    apply_K                     — matrix-free K · v  (gather + cross product)
    apply_Sigma                 — matrix-free Σ · f  (cross product + scatter)
    build_rigid_body_mobility   — per-body 6×6 preconditioner blocks A_rb_α

Dense reference (for testing only):
    build_kinematic_map_dense   — explicit (6N_p × 6N_rb) and (6N_rb × 6N_p) arrays
"""

import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


# ── Matrix-free primitives ────────────────────────────────────────────────────

def compute_relative_positions(
    positions: ArrayLike,
    assembly_ids: ArrayLike,
    num_bodies: int,
) -> tuple[Array, Array]:
    """Compute COM and relative positions for each bead.

    Parameters
    ----------
    positions : (N_p, 3) current particle positions.
    assembly_ids : (N_p,) int array — assembly_ids[i] = index of the rigid body particle i belongs to.
    num_bodies : number of rigid bodies N_rb.

    Returns
    -------
    com : (N_rb, 3) centre-of-mass positions.
    s   : (N_p, 3) relative positions s_i = r_i − com[assembly_ids[i]].
    """
    pos = jnp.asarray(positions)
    aid = jnp.asarray(assembly_ids)

    com_sum = jnp.zeros((num_bodies, 3)).at[aid].add(pos)
    counts  = jnp.zeros(num_bodies).at[aid].add(1.0)
    com = com_sum / counts[:, None]

    s = pos - com[aid]
    return com, s


def apply_K(V_rb: ArrayLike, s: ArrayLike, assembly_ids: ArrayLike) -> Array:
    """Matrix-free action of K: U_particles = K · V_rb.

    Parameters
    ----------
    V_rb : (6·N_rb,) rigid-body velocity vector [V_cm_0; Ω_0; V_cm_1; Ω_1; ...].
    s    : (N_p, 3) relative positions (from compute_relative_positions).
    assembly_ids : (N_p,) int array.

    Returns
    -------
    (6·N_p,) particle velocity vector [U_0; Ω_bead_0; U_1; ...].
    """
    V_rb_mat = jnp.asarray(V_rb).reshape(-1, 6)
    V_cm  = V_rb_mat[:, :3]   # (N_rb, 3)
    Omega = V_rb_mat[:, 3:]   # (N_rb, 3)

    aid = jnp.asarray(assembly_ids)
    V_cm_i  = V_cm[aid]    # (N_p, 3)
    Omega_i = Omega[aid]   # (N_p, 3)

    # U_lin[i] = V_cm[α] + Ω[α] × s[i]
    U_lin = V_cm_i + jnp.cross(Omega_i, s)   # (N_p, 3)
    U_ang = Omega_i                            # (N_p, 3)

    # Interleave to (N_p, 2, 3) then flatten → (6·N_p,)
    return jnp.stack([U_lin, U_ang], axis=1).reshape(-1)


def apply_Sigma(
    f: ArrayLike,
    s: ArrayLike,
    assembly_ids: ArrayLike,
    num_bodies: int,
) -> Array:
    """Matrix-free action of Σ = K^T: G_rb = Σ · f.

    Parameters
    ----------
    f    : (6·N_p,) particle force/torque vector.
    s    : (N_p, 3) relative positions.
    assembly_ids : (N_p,) int array.
    num_bodies : N_rb.

    Returns
    -------
    (6·N_rb,) rigid-body generalized force vector [F_0; T_0; F_1; ...].
    """
    f_arr = jnp.asarray(f)
    N_p   = s.shape[0]
    f_mat = f_arr.reshape(N_p, 6)
    f_lin = f_mat[:, :3]   # (N_p, 3) forces
    f_ang = f_mat[:, 3:]   # (N_p, 3) torques

    aid = jnp.asarray(assembly_ids)

    # F_rb[α] = sum_{i in α} f_lin[i]
    F_rb = jnp.zeros((num_bodies, 3)).at[aid].add(f_lin)

    # T_rb[α] = sum_{i in α} (f_ang[i] + s[i] × f_lin[i])
    T_rb = jnp.zeros((num_bodies, 3)).at[aid].add(f_ang + jnp.cross(s, f_lin))

    # Interleave to (N_rb, 2, 3) then flatten → (6·N_rb,)
    return jnp.stack([F_rb, T_rb], axis=1).reshape(-1)


def build_rigid_body_mobility(
    s: ArrayLike,
    assembly_ids: ArrayLike,
    num_bodies: int,
    inv_mtt: float,
    inv_mrr: float,
) -> Array:
    """Per-body 6×6 self-mobility blocks for the saddle-point preconditioner.

    Computes A_rb_α = Σ_{i ∈ body α} K_i^T · diag(inv_m) · K_i analytically.

    Because Σ_i s_i = 0 by definition of the COM, the translation-rotation
    cross blocks vanish exactly, giving a block-diagonal structure:

        A_rb[α, :3, :3] = n_α · inv_mtt · I₃            (translation block)
        A_rb[α, 3:, 3:] = inv_mtt · I_α + n_α · inv_mrr · I₃   (rotation block)
        A_rb[α, :3, 3:] = A_rb[α, 3:, :3] = 0           (zero cross blocks)

    where I_α = Σ_i (|s_i|² I₃ − s_i⊗s_i) is body α's inertia tensor.

    Parameters
    ----------
    s    : (N_p, 3) relative positions.
    assembly_ids : (N_p,) int array.
    num_bodies : N_rb.
    inv_mtt, inv_mrr : scalar inverse self-mobilities.

    Returns
    -------
    A_rb : (N_rb, 6, 6) JAX float array.
    """
    s   = jnp.asarray(s)
    aid = jnp.asarray(assembly_ids)

    n_alpha = jnp.zeros(num_bodies).at[aid].add(1.0)   # (N_rb,)

    # Inertia tensor contribution per bead: |s|² I₃ − s⊗s
    s2          = jnp.sum(s ** 2, axis=1)                                   # (N_p,)
    outer_s     = s[:, :, None] * s[:, None, :]                             # (N_p, 3, 3)
    inertia_i   = s2[:, None, None] * jnp.eye(3)[None] - outer_s           # (N_p, 3, 3)
    I_alpha     = jnp.zeros((num_bodies, 3, 3)).at[aid].add(inertia_i)     # (N_rb, 3, 3)

    I3  = jnp.eye(3)
    A_rb = jnp.zeros((num_bodies, 6, 6))
    A_rb = A_rb.at[:, :3, :3].set(n_alpha[:, None, None] * inv_mtt * I3[None])
    A_rb = A_rb.at[:, 3:, 3:].set(
        inv_mtt * I_alpha + n_alpha[:, None, None] * inv_mrr * I3[None]
    )
    return A_rb


# ── Dense reference (tests / diagnostics only) ────────────────────────────────

def _skew_np(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix: skew(a) @ b = a × b."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def build_kinematic_map_dense(
    positions: ArrayLike,
    assembly_ids: np.ndarray,
    num_bodies: int,
) -> tuple[Array, Array]:
    """Build dense K (6·N_p × 6·N_rb) and Σ = K^T (6·N_rb × 6·N_p).

    Memory cost is O(N_p · N_rb) — use only for small systems / unit tests.

    Parameters
    ----------
    positions : (N_p, 3) current particle positions.
    assembly_ids : (N_p,) int array; assembly_ids[i] = index of the rigid body particle i belongs to.
    num_bodies : number of rigid bodies N_rb.

    Returns
    -------
    K : jnp.ndarray, shape (6·N_p, 6·N_rb)
    Sigma : jnp.ndarray, shape (6·N_rb, 6·N_p)
    """
    pos = np.array(positions)
    N_p = pos.shape[0]
    N_rb = num_bodies

    com = np.zeros((N_rb, 3))
    counts = np.zeros(N_rb)
    for i in range(N_p):
        alpha = int(assembly_ids[i])
        com[alpha] += pos[i]
        counts[alpha] += 1.0
    com /= counts[:, None]

    K = np.zeros((6 * N_p, 6 * N_rb))
    for i in range(N_p):
        alpha = int(assembly_ids[i])
        s = pos[i] - com[alpha]
        sk = _skew_np(s)
        row_u = 6 * i
        row_w = 6 * i + 3
        col_v = 6 * alpha
        col_o = 6 * alpha + 3
        K[row_u:row_u + 3, col_v:col_v + 3] = np.eye(3)
        K[row_u:row_u + 3, col_o:col_o + 3] = -sk
        K[row_w:row_w + 3, col_o:col_o + 3] = np.eye(3)

    Sigma = K.T
    return jnp.array(K, dtype=jnp.float32), jnp.array(Sigma, dtype=jnp.float32)


def get_inter_body_mask(
    assembly_ids: np.ndarray,
    indices_i_lub: ArrayLike,
    indices_j_lub: ArrayLike,
) -> np.ndarray:
    """Boolean mask: True for pairs belonging to *different* rigid bodies.

    Parameters
    ----------
    assembly_ids : (N_p,) int array.
    indices_i_lub, indices_j_lub : (n_pairs_nf,) lubrication pair indices.

    Returns
    -------
    mask : (n_pairs_nf,) bool numpy array.
    """
    ii = np.array(indices_i_lub)
    jj = np.array(indices_j_lub)
    n_p = len(assembly_ids)
    valid = (ii < n_p) & (jj < n_p)
    aid_i = np.where(valid, assembly_ids[np.clip(ii, 0, n_p - 1)], 0)
    aid_j = np.where(valid, assembly_ids[np.clip(jj, 0, n_p - 1)], 0)
    return valid & (aid_i != aid_j)

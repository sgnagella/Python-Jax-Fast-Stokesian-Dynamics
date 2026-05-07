"""Rigid-body kinematics: kinematic map K and projection Σ = K^T.

For a rigid body α containing particles {i}, with lab-frame relative positions s_i = r_i − r_cm_α:

    U_i = V_cm_α − skew(s_i) · Ω_cm_α   (linear velocity of bead i)
    Ω_i = Ω_cm_α                          (angular velocity of bead i)

This gives the kinematic map K (6·N_p × 6·N_rb) via U_particles = K · V_rb,
and its transpose Σ = K^T (6·N_rb × 6·N_p) maps particle forces/torques to
rigid-body generalized forces/torques:

    F_α = Σ_i F_i
    T_α = Σ_i (T_i + s_i × F_i)
"""

import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def _skew_np(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix: skew(a) @ b = a × b."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def build_kinematic_map(
    positions: ArrayLike,
    assembly_ids: np.ndarray,
    num_bodies: int,
) -> tuple[Array, Array]:
    """Build K (6·N_p × 6·N_rb) and Σ = K^T (6·N_rb × 6·N_p).

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

    # Compute COM for each rigid body
    com = np.zeros((N_rb, 3))
    counts = np.zeros(N_rb)
    for i in range(N_p):
        alpha = int(assembly_ids[i])
        com[alpha] += pos[i]
        counts[alpha] += 1.0
    com /= counts[:, None]

    # Build K (6·N_p × 6·N_rb)
    K = np.zeros((6 * N_p, 6 * N_rb))
    for i in range(N_p):
        alpha = int(assembly_ids[i])
        s = pos[i] - com[alpha]           # lab-frame relative position
        sk = _skew_np(s)
        row_u = 6 * i
        row_w = 6 * i + 3
        col_v = 6 * alpha
        col_o = 6 * alpha + 3
        # U_i = I · V_cm − skew(s_i) · Ω_cm
        K[row_u:row_u + 3, col_v:col_v + 3] = np.eye(3)
        K[row_u:row_u + 3, col_o:col_o + 3] = -sk
        # Ω_i = Ω_cm
        K[row_w:row_w + 3, col_o:col_o + 3] = np.eye(3)

    Sigma = K.T   # (6·N_rb, 6·N_p)
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
    # Sentinel value (num_particles) used as padding in jfsd neighbor lists → treat as same body
    n_p = len(assembly_ids)
    # Sentinel pairs (ii >= n_p or jj >= n_p) must be False — they are padding.
    valid = (ii < n_p) & (jj < n_p)
    aid_i = np.where(valid, assembly_ids[np.clip(ii, 0, n_p - 1)], 0)
    aid_j = np.where(valid, assembly_ids[np.clip(jj, 0, n_p - 1)], 0)
    return valid & (aid_i != aid_j)

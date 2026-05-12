"""Main simulation loop for rigid-body Stokesian Dynamics.

Adapts wrap_sd() from jfsd/main.py to handle rigid bodies by replacing the
identity B / B^T projection blocks with B·K / Σ·B^T using the configuration-
dependent kinematic map built in assembly.py.
"""

import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, random
from jax.typing import ArrayLike
from tqdm import tqdm

from jfsd import thermal, utils
from jfsd import jaxmd_space as space
from rigid_sd.assembly import build_kinematic_map, get_inter_body_mask
from rigid_sd.solver_rigid import solve_linear_system_rigid

jax.config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress JAX/TF debug logging

# ── Hard-sphere force (inter-body pairs only) ─────────────────────────────────

def _compute_hs_forces(
    positions: ArrayLike,
    indices_i: np.ndarray,
    indices_j: np.ndarray,
    inter_body_mask: np.ndarray,
    box: ArrayLike,
    dt: float,
    num_particles: int,
) -> Array:
    """Repulsive harmonic hard-sphere forces between inter-body pairs.

    Returns a (6·N_p,) flat force vector [Fx, Fy, Fz, 0, 0, 0, ...].
    Spring constant from applied_forces.py SD mode: k = 2500.839791 / dt.
    Effective diameter: σ = 2.002 (0.1% shift).
    """
    sigma = 2.002
    # k = 2500.839791 / dt
    k = 1/dt # for RPY level HIs

    # Displacement vectors for all lubrication pairs
    inv_box = jnp.linalg.inv(box)
    dr = positions[indices_j] - positions[indices_i]            # (n_pairs, 3)
    # Minimum image
    dr_frac = dr @ inv_box.T
    dr_frac = dr_frac - jnp.round(dr_frac)
    dr = dr_frac @ box.T

    dist = jnp.linalg.norm(dr, axis=1)                          # (n_pairs,)
    r_unit = dr / jnp.where(dist > 0, dist, 1.0)[:, None]      # (n_pairs, 3)

    # Spring force magnitude (active only when overlapping and inter-body)
    overlap = dist < sigma
    inter = jnp.array(inter_body_mask, dtype=bool)
    fp_mod = jnp.where(overlap & inter, k * (1.0 - sigma / jnp.where(dist > 0, dist, 1.0)), 0.0)
    # if jnp.any(fp_mod > 0):
    #     print(f"  Hard-sphere forces: {jnp.sum(fp_mod > 0)} pairs overlapping (max overlap={jnp.max(sigma - dist):.4f}).")
    #     exit(1)

    # Accumulate onto particles
    forces = jnp.zeros((num_particles, 3))
    forces = forces.at[indices_i].add( fp_mod[:, None] * r_unit)
    forces = forces.at[indices_j].add(-fp_mod[:, None] * r_unit)

    # Flatten to (6·N_p,): [Fx, Fy, Fz, 0, 0, 0] per particle
    flat = jnp.zeros((num_particles, 6))
    flat = flat.at[:, :3].set(forces)
    return jnp.ravel(flat)


# ── Position update ───────────────────────────────────────────────────────────

def _make_position_updater(shift_fn, box, num_particles: int):
    """Return a JIT-compiled position update function closing over shift_fn."""
    @jit
    def _update(positions, U_particles, dt, shear_rate):
        offset = box @ jnp.array([0.5, 0.5, 0.5])
        dR = jnp.zeros((num_particles, 3))
        dR = dR.at[:, 0].set(dt * U_particles[0::6])
        dR = dR.at[:, 1].set(dt * U_particles[1::6])
        dR = dR.at[:, 2].set(dt * U_particles[2::6])
        positions = shift_fn(positions + offset, dR) - offset
        dR_shear = jnp.zeros((num_particles, 3))
        dR_shear = dR_shear.at[:, 0].set(dt * shear_rate * positions[:, 1])
        positions = shift_fn(positions + offset, dR_shear) - offset
        return positions
    return _update


def _make_rigid_position_updater(
    assembly_ids: np.ndarray,
    num_bodies: int,
    lx: float,
    ly: float,
    lz: float,
):
    """Factory: JIT-compiled rigid-body position update using Rodrigues rotation.

    Exactly preserves all intra-body distances (bond lengths) by rotating
    relative positions rather than applying a linear Euler step.  The Euler
    step introduces an O(dt²·|Ω|²·L) bond-stretch per step; the rotation
    update is exact.

    The Python `for` loop over bodies is unrolled at trace-time (num_bodies
    is a static int), so all bead indices are compile-time constants.
    """
    # Precompute which bead indices belong to each body (static).
    body_bead_indices = [np.where(assembly_ids == alpha)[0] for alpha in range(num_bodies)]
    box_lengths = jnp.array([lx, ly, lz], dtype=jnp.float32)

    @jit
    def _update(positions: jnp.ndarray, V_rb_total: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Update positions for all rigid bodies.

        Parameters
        ----------
        positions : (N_p, 3)
        V_rb_total : (6·N_rb,)  [V_cm_0; Ω_0; V_cm_1; Ω_1; ...]
        dt : timestep
        """
        new_pos = jnp.zeros_like(positions)

        for alpha in range(num_bodies):
            bead_idx = body_bead_indices[alpha]  # static numpy array

            V_cm = V_rb_total[6 * alpha : 6 * alpha + 3]      # (3,)
            Omega = V_rb_total[6 * alpha + 3 : 6 * alpha + 6]  # (3,)

            # Centre of mass of this body
            com = jnp.mean(positions[bead_idx], axis=0)  # (3,)

            # Translate COM, then wrap into [-L/2, L/2]
            new_com_raw = com + dt * V_cm
            new_com = new_com_raw - box_lengths * jnp.round(new_com_raw / box_lengths)

            # Rodrigues rotation R(Ω·dt)
            omega_norm = jnp.linalg.norm(Omega)
            theta = omega_norm * dt
            # Safe unit vector (when Omega ≈ 0 → n_hat unused, sin/cos → I anyway)
            safe_norm = jnp.where(omega_norm > 1e-10, omega_norm, 1.0)
            n_hat = Omega / safe_norm
            sin_t = jnp.sin(theta)
            cos_t = jnp.cos(theta)

            # R = cos(θ)·I + sin(θ)·skew(n̂) + (1−cos(θ))·n̂⊗n̂
            skew_n = (
                jnp.zeros((3, 3))
                .at[0, 1].set(-n_hat[2])
                .at[0, 2].set(n_hat[1])
                .at[1, 0].set(n_hat[2])
                .at[1, 2].set(-n_hat[0])
                .at[2, 0].set(-n_hat[1])
                .at[2, 1].set(n_hat[0])
            )
            R = (
                cos_t * jnp.eye(3)
                + sin_t * skew_n
                + (1.0 - cos_t) * jnp.outer(n_hat, n_hat)
            )

            # Rotate relative positions and set new absolute positions
            s_old = positions[bead_idx] - com      # (n_beads, 3)
            s_new = (R @ s_old.T).T                # (n_beads, 3)  ← exact rotation
            new_pos = new_pos.at[bead_idx].set(new_com + s_new)

        return new_pos

    return _update


# ── Main simulation ───────────────────────────────────────────────────────────

def run_rigid_sd(
    num_steps: int,
    writing_period: int,
    time_step: float,
    lx: float,
    ly: float,
    lz: float,
    num_particles: int,
    num_bodies: int,
    assembly_ids: np.ndarray,
    max_strain: float,
    temperature: float,
    particle_radius: float,
    ewald_xi: float,
    error_tolerance: float,
    positions: ArrayLike,
    seed_rfd: int,
    seed_ffwave: int,
    seed_ffreal: int,
    seed_nf: int,
    output: str | None = None,
    active_alpha: float = 0.0,
    trap_particle_ids: np.ndarray | None = None,
    trap_spring_k: float = 0.0,
    trap_targets=None,
    probe_body_id: int | None = None,
) -> tuple[Array, Array]:
    """Run rigid-body SD and return (trajectory, velocities).

    Parameters
    ----------
    assembly_ids : (N_p,) int array — assembly_ids[i] = index of rigid body for particle i.
    active_alpha : activity strength for the body-frame squirmer stresslet
        S_active = alpha*(d⊗d − I/3) placed on the head bead of each multi-bead body.
        alpha > 0 → pusher, alpha < 0 → puller, alpha = 0 → passive (default).
    trap_particle_ids : (N_trap,) int array of particle indices subject to harmonic traps.
    trap_spring_k : spring constant for all harmonic traps.
    trap_targets : callable(step, dt) → jnp.ndarray (N_trap, 3) of trap centre positions.
        Called each step at Python level (not inside JIT).
    probe_body_id : rigid-body index whose orientation to track. If given, the unit vector
        initialized to (1, 0, 0) is rotated each step by the body's angular velocity using
        the Rodrigues formula and saved to probe_orientation.npy in the output directory.
    All other parameters mirror wrap_sd() in jfsd/main.py.
    """
    if writing_period > num_steps:
        raise ValueError("writing_period must be <= num_steps.")

    if output is not None:
        out_path = Path(output)
        out_path.mkdir(exist_ok=True, parents=True)
    else:
        out_path = None

    n_frames = int(num_steps / writing_period)
    trajectory = np.zeros((n_frames, num_particles, 3), float)
    velocities = np.zeros((n_frames, num_particles, 6), float)

    track_orientation = probe_body_id is not None
    if track_orientation:
        probe_orient = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        orientations = np.zeros((n_frames, 3), dtype=np.float64)

    epsilon = error_tolerance
    xy = 0.0
    ewald_cut = jnp.sqrt(-jnp.log(error_tolerance)) / ewald_xi

    ResTable_dist = jnp.load("files/ResTableDist.npy")
    ResTable_vals = jnp.load("files/ResTableVals.npy")
    ResTable_min = 0.0001
    ResTable_dr = 0.004305

    box = jnp.array([[lx, ly * xy, 0.], [0., ly, 0.], [0., 0., lz]])
    _, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    _, shift_fn_open = space.periodic_general(box, fractional_coordinates=False, wrapped=False)
    update_positions = _make_position_updater(shift_fn, box, num_particles)
    update_rigid_positions = _make_rigid_position_updater(assembly_ids, num_bodies, lx, ly, lz)

    # Multi-bead bodies only — single-bead bodies have no head/tail distinction.
    bead_count = np.array([np.sum(assembly_ids == b) for b in range(num_bodies)])
    multi_bead_bodies = np.where(bead_count >= 2)[0]
    tail_idx = np.array([np.where(assembly_ids == b)[0][0] for b in multi_bead_bodies])
    head_idx = np.array([np.where(assembly_ids == b)[0][1] for b in multi_bead_bodies])

    unique_pairs, nl_ff, nl_lub, nl_prec, _ = utils.cpu_nlist(
        positions, np.array([lx, ly, lz]), ewald_cut, 3.99, 2.1, xy
    )

    (
        quadW, prefac, expfac, gaussPd2, gridk, gridh,
        gaussian_grid_spacing, key_ffwave, ewaldC1, m_self,
        grid_x, grid_y, grid_z, gauss_support,
        ewald_n, ewald_dr, eta, xisq,
        wave_bro_ind, wave_bro_nyind,
    ) = utils.init_periodic_box(
        error_tolerance, ewald_xi, lx, ly, lz, ewald_cut,
        max_strain, xy, positions, num_particles, temperature, seed_ffwave,
    )

    if temperature > 0:
        key_rfd    = random.PRNGKey(seed_rfd)
        key_ffreal = random.PRNGKey(seed_ffreal)
        key_nf     = random.PRNGKey(seed_nf)

    n_iter_ff = 2
    n_iter_nf = 2

    print("jfsd-rigid running on:", jax.default_backend())
    print("Starting: compiling (first step may take ~1-2 min)…")

    for step in tqdm(range(num_steps), mininterval=0.5):

        # ── Build rigid-body kinematics ───────────────────────────────────────
        K, Sigma = build_kinematic_map(positions, assembly_ids, num_bodies)

        # ── Precompute pair quantities ────────────────────────────────────────
        precomputed = utils.precompute(
            positions,
            gaussian_grid_spacing,
            nl_ff, nl_lub,
            xy, num_particles, lx, ly, lz,
            grid_x, grid_y, grid_z,
            prefac, expfac, quadW,
            int(gauss_support), gaussPd2,
            ewald_n, ewald_dr, ewald_cut, ewaldC1,
            ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals,
            0.0, 0.0,   # friction_coefficient, friction_range (none for smooth beads)
        )
        ResFunction = precomputed[18]

        # ── Inter-body mask for lubrication ──────────────────────────────────
        indices_i_lub = precomputed[16]
        indices_j_lub = precomputed[17]
        r_lub = precomputed[15]
        mask = get_inter_body_mask(assembly_ids, indices_i_lub, indices_j_lub)
        jmask = jnp.array(mask, dtype=jnp.float32)

        # Zero out intra-body resistance scalars (first 11 are used by R_FU and thermals)
        inter_res = tuple(f * jmask for f in ResFunction[:11])

        # Projector for near-field Lanczos: count only inter-body neighbors.
        # If the matrix R_FU is all-zero (no inter-body pairs in range), Lanczos
        # would divide by zero.  We skip the near-field Brownian block in that case.
        has_inter_body_lub = bool(np.any(mask))
        if has_inter_body_lub:
            mask_np = np.asarray(mask, dtype=bool)
            inter_i_np = np.asarray(indices_i_lub)[mask_np]
            inter_j_np = np.asarray(indices_j_lub)[mask_np]
            diagonal_zeroes = thermal.number_of_neigh_jit(num_particles, inter_i_np, inter_j_np)
        else:
            diagonal_zeroes = jnp.ones(6 * num_particles)

        # ── RHS vector (size 11·N_p + 6·N_rb) ───────────────────────────────
        sys_size = 11 * num_particles + 6 * num_bodies
        saddle_b = jnp.zeros(sys_size)
        brownian_drift_rb = jnp.zeros(6 * num_bodies)

        # ── Brownian drift via RFD ────────────────────────────────────────────
        if temperature > 0:
            key_rfd, random_rb = utils.generate_random_array(key_rfd, 6 * num_bodies)
            random_rb = -((2 * random_rb - 1) * jnp.sqrt(3))

            # Particle displacements from rigid-body random velocities
            dU_p = K @ random_rb                                 # (6·N_p,)
            dU_p_pos = jnp.reshape(dU_p, (num_particles, 6))

            # Positive perturbation
            buf_pos = positions + (epsilon / 2.0) * dU_p_pos[:, :3]
            buf_pos = jnp.clip(buf_pos, -1e6, 1e6)   # safety
            buf_gs = utils.precompute_grid_distancing(
                gauss_support, gridh[0], xy, buf_pos,
                num_particles, grid_x, grid_y, grid_z, lx, ly, lz,
            )
            buf_precomp = utils.precompute(
                buf_pos, buf_gs, nl_ff, nl_lub,
                xy, num_particles, lx, ly, lz,
                grid_x, grid_y, grid_z,
                prefac, expfac, quadW,
                int(gauss_support), gaussPd2,
                ewald_n, ewald_dr, ewald_cut, ewaldC1,
                ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals,
                0.0, 0.0,
            )
            buf_mask = get_inter_body_mask(assembly_ids, buf_precomp[16], buf_precomp[17])
            buf_jmask = jnp.array(buf_mask, dtype=jnp.float32)
            buf_inter_res = tuple(f * buf_jmask for f in buf_precomp[18][:11])
            K_buf, Sigma_buf = build_kinematic_map(buf_pos, assembly_ids, num_bodies)

            rfd_b_pos = jnp.zeros(sys_size).at[11 * num_particles:].set(random_rb)
            x_pos, _ = solve_linear_system_rigid(
                num_particles, num_bodies, rfd_b_pos,
                gridk, buf_precomp,
                int(grid_x), int(grid_y), int(grid_z), int(gauss_support), m_self,
                K_buf, Sigma_buf, buf_inter_res,
            )

            # Negative perturbation
            buf_neg = positions - (epsilon / 2.0) * dU_p_pos[:, :3]
            buf_neg = jnp.clip(buf_neg, -1e6, 1e6)
            buf_gs_neg = utils.precompute_grid_distancing(
                gauss_support, gridh[0], xy, buf_neg,
                num_particles, grid_x, grid_y, grid_z, lx, ly, lz,
            )
            buf_precomp_neg = utils.precompute(
                buf_neg, buf_gs_neg, nl_ff, nl_lub,
                xy, num_particles, lx, ly, lz,
                grid_x, grid_y, grid_z,
                prefac, expfac, quadW,
                int(gauss_support), gaussPd2,
                ewald_n, ewald_dr, ewald_cut, ewaldC1,
                ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals,
                0.0, 0.0,
            )
            buf_mask_neg = get_inter_body_mask(assembly_ids, buf_precomp_neg[16], buf_precomp_neg[17])
            buf_jmask_neg = jnp.array(buf_mask_neg, dtype=jnp.float32)
            buf_inter_res_neg = tuple(f * buf_jmask_neg for f in buf_precomp_neg[18][:11])
            K_buf_neg, Sigma_buf_neg = build_kinematic_map(buf_neg, assembly_ids, num_bodies)

            rfd_b_neg = jnp.zeros(sys_size).at[11 * num_particles:].set(random_rb)
            x_neg, _ = solve_linear_system_rigid(
                num_particles, num_bodies, rfd_b_neg,
                gridk, buf_precomp_neg,
                int(grid_x), int(grid_y), int(grid_z), int(gauss_support), m_self,
                K_buf_neg, Sigma_buf_neg, buf_inter_res_neg,
            )

            V_rb_pos = x_pos[11 * num_particles:]
            V_rb_neg = x_neg[11 * num_particles:]
            brownian_drift_rb = -temperature / epsilon * (V_rb_pos - V_rb_neg)

        # ── Far-field thermal slip velocities → b[:11·N_p] ───────────────────
        if temperature > 0:
            key_nf, random_nf = utils.generate_random_array(key_nf, 6 * num_particles)
            key_ffreal, random_real = utils.generate_random_array(key_ffreal, 11 * num_particles)
            key_ffwave, random_wave = utils.generate_random_array(
                key_ffwave,
                3 * 2 * len(wave_bro_ind[:, 0, 0]) + 3 * len(wave_bro_nyind[:, 0]),
            )

            ws_linvel, ws_angvel_strain = thermal.compute_wave_space_slipvelocity(
                num_particles,
                int(grid_x), int(grid_y), int(grid_z), int(gauss_support),
                temperature, time_step, gridh,
                wave_bro_ind[:, 0, 0], wave_bro_ind[:, 0, 1], wave_bro_ind[:, 0, 2],
                wave_bro_ind[:, 1, 0], wave_bro_ind[:, 1, 1], wave_bro_ind[:, 1, 2],
                wave_bro_nyind[:, 0], wave_bro_nyind[:, 1], wave_bro_nyind[:, 2],
                gridk, random_wave,
                precomputed[0], precomputed[1], precomputed[2],
                precomputed[3], precomputed[4],
            )

            rs_linvel, rs_angvel_strain, stepnorm_ff, diag_ff = thermal.compute_real_space_slipvelocity(
                num_particles, m_self, temperature, time_step, int(n_iter_ff),
                random_real, precomputed[5], precomputed[6], precomputed[7],
                precomputed[8], precomputed[9], precomputed[10], precomputed[11],
                precomputed[12], precomputed[13], precomputed[14],
            )
            while (stepnorm_ff > 1e-3) and (n_iter_ff < 150):
                n_iter_ff += 20
                rs_linvel, rs_angvel_strain, stepnorm_ff, diag_ff = thermal.compute_real_space_slipvelocity(
                    num_particles, m_self, temperature, time_step, int(n_iter_ff),
                    random_real, precomputed[5], precomputed[6], precomputed[7],
                    precomputed[8], precomputed[9], precomputed[10], precomputed[11],
                    precomputed[12], precomputed[13], precomputed[14],
                )
            if not math.isfinite(stepnorm_ff) or (n_iter_ff > 150 and stepnorm_ff > 0.02):
                raise ValueError(f"Far-field Lanczos did not converge (stepnorm={stepnorm_ff}).")

            saddle_b = saddle_b.at[:11 * num_particles].add(
                thermal.convert_to_generalized(
                    num_particles, ws_linvel, rs_linvel, ws_angvel_strain, rs_angvel_strain,
                )
            )

            # Near-field Brownian forces — disabled while lubrication is off.
            # Re-enable together with the lubrication block in solver_rigid.py.
            if False and num_particles > 1 and has_inter_body_lub:
                F_B_nf, stepnorm_nf, diag_nf = thermal.compute_nearfield_brownianforce(
                    num_particles, temperature, time_step, random_nf,
                    r_lub, indices_i_lub, indices_j_lub,
                    inter_res[0], inter_res[1], inter_res[2], inter_res[3],
                    inter_res[4], inter_res[5], inter_res[6], inter_res[7],
                    inter_res[8], inter_res[9], inter_res[10],
                    diagonal_zeroes, n_iter_nf,
                )
                while (stepnorm_nf > 1e-3) and (n_iter_nf < 250):
                    n_iter_nf += 20
                    F_B_nf, stepnorm_nf, diag_nf = thermal.compute_nearfield_brownianforce(
                        num_particles, temperature, time_step, random_nf,
                        r_lub, indices_i_lub, indices_j_lub,
                        inter_res[0], inter_res[1], inter_res[2], inter_res[3],
                        inter_res[4], inter_res[5], inter_res[6], inter_res[7],
                        inter_res[8], inter_res[9], inter_res[10],
                        diagonal_zeroes, n_iter_nf,
                    )
                if not math.isfinite(stepnorm_nf) or (n_iter_nf > 250 and stepnorm_nf > 1e-3):
                    raise ValueError(f"Near-field Lanczos did not converge (stepnorm={stepnorm_nf}).")
                saddle_b = saddle_b.at[11 * num_particles:].add(-(Sigma @ F_B_nf))

        # ── Inter-body hard-sphere forces → b[11·N_p:] via Σ projection ────────
        F_hs = _compute_hs_forces(
            positions, np.array(indices_i_lub), np.array(indices_j_lub),
            mask, box, time_step, num_particles,
        )
        saddle_b = saddle_b.at[11 * num_particles:].add(-(Sigma @ F_hs))

        # ── Harmonic trap forces → b[11·N_p:] via Σ projection ──────────────
        if trap_particle_ids is not None and trap_spring_k > 0.0:
            r_trap_all = trap_targets(step, time_step)            # (N_trap, 3)
            F_trap_flat = jnp.zeros(6 * num_particles)
            for ti, pid in enumerate(trap_particle_ids):
                f = trap_spring_k * (r_trap_all[ti] - positions[pid])
                F_trap_flat = F_trap_flat.at[6 * int(pid) : 6 * int(pid) + 3].add(f)
            saddle_b = saddle_b.at[11 * num_particles:].add(-(Sigma @ F_trap_flat))

        # ── Active stresslet: S_active = alpha*(d⊗d − I/3) on head bead ─────
        # d is the unit bond vector tail→head, recomputed from current positions.
        # Sign convention matches background shear in jfsd/main.py:871 (add −E).
        if active_alpha != 0.0:
            for bi, b in enumerate(multi_bead_bodies):
                t_i, h_i = int(tail_idx[bi]), int(head_idx[bi])
                d = positions[h_i] - positions[t_i]
                d = d / jnp.linalg.norm(d)
                s_flat = active_alpha * jnp.array([
                    2.0*d[0]*d[0] + d[1]*d[1] - 1.0,   # 2·S_xx + S_yy
                    2.0*d[0]*d[1],                        # 2·S_xy
                    2.0*d[0]*d[2],                        # 2·S_xz
                    2.0*d[1]*d[2],                        # 2·S_yz
                    d[0]*d[0] + 2.0*d[1]*d[1] - 1.0,   # S_xx + 2·S_yy
                ])
                base = 6 * num_particles + 5 * h_i
                saddle_b = saddle_b.at[base : base + 5].add(-s_flat)

        # ── Solve saddle point ────────────────────────────────────────────────
        saddle_x, exitcode = solve_linear_system_rigid(
            num_particles, num_bodies, saddle_b,
            gridk, precomputed,
            int(grid_x), int(grid_y), int(grid_z), int(gauss_support), m_self,
            K, Sigma, inter_res,
        )
        if exitcode > 0 or not math.isfinite(float(saddle_x[11 * num_particles])):
            raise ValueError(f"GMRES did not converge at step {step} (exitcode={exitcode}).")

        # ── Position update (exact Rodrigues rotation) ────────────────────────
        V_rb = saddle_x[11 * num_particles:]                    # (6·N_rb,)
        V_rb_total = V_rb + brownian_drift_rb                   # (6·N_rb,) includes drift

        # Rodrigues update exactly preserves all intra-body distances.
        # (Euler step would introduce O(dt²·|Ω|²·L) bond-stretch per step.)
        positions = update_rigid_positions(positions, V_rb_total, time_step)

        # U_particles from V_rb_total — used only for velocity output.
        U_particles = K @ V_rb_total                            # (6·N_p,)

        # ── Probe orientation update (Rodrigues rotation) ─────────────────────
        # jfsd's angular velocity convention is the NEGATIVE of the standard
        # right-hand vorticity: Ω_jfsd = −(1/2)∇×U.  Negate to recover the
        # physical angular velocity before integrating the orientation vector.
        if track_orientation:
            Omega_probe = -np.array(
                V_rb_total[6 * probe_body_id + 3 : 6 * probe_body_id + 6], dtype=np.float64
            )
            omega_norm = np.linalg.norm(Omega_probe)
            if omega_norm > 1e-10:
                theta = omega_norm * time_step
                n = Omega_probe / omega_norm
                sin_t, cos_t = np.sin(theta), np.cos(theta)
                K_skew = np.array([
                    [0.0,   -n[2],  n[1]],
                    [n[2],   0.0,  -n[0]],
                    [-n[1],  n[0],  0.0],
                ])
                R = cos_t * np.eye(3) + sin_t * K_skew + (1.0 - cos_t) * np.outer(n, n)
                probe_orient = R @ probe_orient

        if jnp.any(jnp.isnan(positions)) or jnp.any(jnp.isinf(positions)):
            raise ValueError(f"Invalid positions at step {step}.")

        # ── Update neighbor lists ─────────────────────────────────────────────
        nl_ff, nl_lub, nl_prec, within_bounds = utils.update_neighborlist(
            num_particles, positions, ewald_cut, 3.99, 2.1, unique_pairs, box,
        )
        if not within_bounds:
            print("Re-allocating neighbor list…")
            unique_pairs, nl_ff, nl_lub, nl_prec, _ = utils.cpu_nlist(
                positions, np.array([lx, ly, lz]), ewald_cut, 3.99, 2.1, xy,
            )
        gaussian_grid_spacing = utils.precompute_grid_distancing(
            gauss_support, gridh[0], xy, positions,
            num_particles, grid_x, grid_y, grid_z, lx, ly, lz,
        )

        # Reset Lanczos iteration counts periodically
        if (step % 100) == 0:
            n_iter_ff = 5
            n_iter_nf = 5

        # ── Save trajectory ───────────────────────────────────────────────────
        if (step % writing_period) == 0:
            frame = step // writing_period
            trajectory[frame] = positions
            velocities[frame] = jnp.reshape(U_particles, (num_particles, 6))
            if track_orientation:
                orientations[frame] = probe_orient
            if out_path is not None:
                np.save(out_path / "trajectory.npy", trajectory)
                np.save(out_path / "velocities.npy", velocities)
                if track_orientation:
                    np.save(out_path / "probe_orientation.npy", orientations)

    return jnp.array(trajectory), jnp.array(velocities)

"""Test: two free colloids only (no swimmers), probe at origin, outer colloid
orbiting counterclockwise.  Checks that the probe develops positive Ω_z due to
hydrodynamic coupling (Stokeslet vorticity from the outer particle's CCW motion).

Run from the project root:
    JAX_PLATFORMS=cpu python test_probe_omega.py
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rigid_sd.main_rigid import run_rigid_sd

# ── Parameters ────────────────────────────────────────────────────────────────
SEPARATION   = 2.8
N_PARTICLES  = 2
N_BODIES     = 2
LX = LY = LZ = 30.0
KT           = 0.0     # deterministic: no thermal noise
OMEGA        = 1.0     # counterclockwise angular frequency
TRAP_K       = 300.0
DT           = 0.001
NUM_STEPS    = 500     # 0.5 time units
WRITING_PERIOD = 10    # 50 frames

# ── Initial positions ─────────────────────────────────────────────────────────
# Probe at origin (body 0, particle 0)
# Outer colloid at (SEPARATION, 0, 0) (body 1, particle 1)
positions = jnp.array([
    [0.0, 0.0, 0.0],
    [SEPARATION, 0.0, 0.0],
], dtype=jnp.float32)

assembly_ids = np.array([0, 1], dtype=int)


def trap_targets(step, dt):
    """Probe fixed at origin; outer colloid moves in CCW circle."""
    t = step * dt
    return jnp.array([
        [0.0, 0.0, 0.0],
        [SEPARATION * jnp.cos(OMEGA * t),
         SEPARATION * jnp.sin(OMEGA * t), 0.0],
    ], dtype=jnp.float32)


# ── Run ───────────────────────────────────────────────────────────────────────
trajectory, velocities = run_rigid_sd(
    num_steps       = NUM_STEPS,
    writing_period  = WRITING_PERIOD,
    time_step       = DT,
    lx=LX, ly=LY, lz=LZ,
    num_particles   = N_PARTICLES,
    num_bodies      = N_BODIES,
    assembly_ids    = assembly_ids,
    max_strain      = 0.5,
    temperature     = KT,
    particle_radius = 1.0,
    ewald_xi        = 0.5,
    error_tolerance = 0.001,
    positions       = positions,
    seed_rfd        = 1234567,
    seed_ffwave     = 7654321,
    seed_ffreal     = 3141592,
    seed_nf         = 2718281,
    output          = "output_test_probe",
    active_alpha    = 0.0,
    trap_particle_ids = np.array([0, 1], dtype=int),
    trap_spring_k   = TRAP_K,
    trap_targets    = trap_targets,
    probe_body_id   = 0,
)

# ── velocities layout: (n_frames, N_p, 6) = [vx, vy, vz, ωx, ωy, ωz] ───────
# NOTE: jfsd uses Ω_jfsd = −(1/2)∇×U (negative of standard vorticity).
# Negate to recover the physical angular velocity.
vels = np.array(velocities)
probe_omega_jfsd = vels[:, 0, 3:]     # raw jfsd convention (opposite sign)
probe_omega      = -probe_omega_jfsd  # physical angular velocity

print("\n─── Probe angular velocity (Ω, physical) ────────────────────────────")
print(f"  {'Frame':>5}  {'Ωx':>10}  {'Ωy':>10}  {'Ωz':>10}")
for i, om in enumerate(probe_omega):
    print(f"  {i:5d}  {om[0]:10.6f}  {om[1]:10.6f}  {om[2]:10.6f}")

print("\n─── Summary ──────────────────────────────────────────────────────────")
print(f"  mean Ωz = {probe_omega[:, 2].mean():.6f}  "
      f"(expected > 0 for CCW outer motion)")
print(f"  mean Ωx = {probe_omega[:, 0].mean():.6f}  (expected ≈ 0)")
print(f"  mean Ωy = {probe_omega[:, 1].mean():.6f}  (expected ≈ 0)")

# ── Orientation trajectory ────────────────────────────────────────────────────
orient = np.load("output_test_probe/probe_orientation.npy")
print("\n─── Probe orientation vector ─────────────────────────────────────────")
print(f"  {'Frame':>5}  {'nx':>10}  {'ny':>10}  {'nz':>10}  {'|n|':>8}")
for i in range(len(orient)):
    n = orient[i]
    print(f"  {i:5d}  {n[0]:10.6f}  {n[1]:10.6f}  {n[2]:10.6f}  "
          f"{np.linalg.norm(n):8.6f}")

print("\n─── Probe confinement (should stay near origin with trap_k=300) ─────")
probe_pos = np.array(trajectory[:, 0, :])
print(f"  max |r_probe| = {np.linalg.norm(probe_pos, axis=1).max():.6f}")

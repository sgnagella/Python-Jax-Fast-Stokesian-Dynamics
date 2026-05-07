"""Example: two rigid dumbbells undergoing thermal Brownian motion with hard-sphere repulsion.

Each dumbbell consists of two beads (radius 1) at center-to-center separation 2.5.
Intra-dumbbell bead pairs are excluded from lubrication and hard-sphere forces.
Inter-dumbbell pairs experience hydrodynamic lubrication and hard-sphere repulsion.

Run from the project root:
    JAX_PLATFORMS=cpu python rigid_sd/run_dumbbell.py
"""

import numpy as np
import jax.numpy as jnp

from rigid_sd.main_rigid import run_rigid_sd

# ── System parameters ─────────────────────────────────────────────────────────

N_DUMBBELLS   = 1
BOND_LENGTH   = 2.001          # center-to-center bead separation (surface gap = 0.5)
N_PARTICLES   = N_DUMBBELLS * 2
N_BODIES      = N_DUMBBELLS

LX = LY = LZ  = 30.0
# KT            = 0.005305165  # 1/(6π) → Brownian time = 60π
KT            = 1.0           # higher temperature for more visible diffusion in short runs
ACTIVE_ALPHA  = -10.0           # activity: >0 pusher, <0 puller, 0 passive
EWALD_XI      = 0.5
ERROR_TOL     = 0.001
PARTICLE_R    = 1.0
MAX_STRAIN    = 0.5

NUM_STEPS     = 250
WRITING_PERIOD = 10
DT            = 0.005

# ── Initial positions ─────────────────────────────────────────────────────────
# Dumbbell 0: aligned along x, centred at (0, 0, 0)
# Dumbbell 1: aligned along x, centred at (0, 6, 0)  — well separated

half = BOND_LENGTH / 2.0
positions = jnp.array([
    [-half,  0.0, 0.0],   # dumbbell 0, bead 0
    [ half,  0.0, 0.0],   # dumbbell 0, bead 1
    # [-half,  6.0, 0.0],   # dumbbell 1, bead 0
    # [ half,  6.0, 0.0],   # dumbbell 1, bead 1
], dtype=jnp.float32)

# assembly_ids = np.array([0, 0, 1, 1], dtype=int)   # particle i belongs to body assembly_ids[i]
assembly_ids = np.array([0, 0], dtype=int)   # only one dumbbell in this example

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED_RFD    = 9237412
SEED_FFWAVE = 30498522
SEED_FFREAL = 57239485
SEED_NF     = 2343095

# ── Run ───────────────────────────────────────────────────────────────────────

print(f"Simulating {N_DUMBBELLS} rigid dumbbells (bond length={BOND_LENGTH})")
print(f"  Particles: {N_PARTICLES}  |  Rigid bodies: {N_BODIES}")
print(f"  Steps: {NUM_STEPS}  |  dt: {DT}  |  kT: {KT}  |  active_alpha: {ACTIVE_ALPHA}")

trajectory, velocities = run_rigid_sd(
    num_steps      = NUM_STEPS,
    writing_period = WRITING_PERIOD,
    time_step      = DT,
    lx=LX, ly=LY, lz=LZ,
    num_particles  = N_PARTICLES,
    num_bodies     = N_BODIES,
    assembly_ids   = assembly_ids,
    max_strain     = MAX_STRAIN,
    temperature    = KT,
    particle_radius= PARTICLE_R,
    ewald_xi       = EWALD_XI,
    error_tolerance= ERROR_TOL,
    positions      = positions,
    seed_rfd       = SEED_RFD,
    seed_ffwave    = SEED_FFWAVE,
    seed_ffreal    = SEED_FFREAL,
    seed_nf        = SEED_NF,
    output         = "output_rigid_dumbbell",
    active_alpha   = ACTIVE_ALPHA,
)

# ── Validation ────────────────────────────────────────────────────────────────

print("\n── Bond-length conservation ──────────────────────────────────────────")
for db in range(N_DUMBBELLS):
    b0 = 2 * db
    b1 = 2 * db + 1
    bond_lengths = np.linalg.norm(
        trajectory[:, b0, :] - trajectory[:, b1, :], axis=1
    )
    print(f"  Dumbbell {db}: mean={bond_lengths.mean():.6f}  "
          f"std={bond_lengths.std():.2e}  "
          f"(initial={BOND_LENGTH:.4f})")

print("\n── COM diffusion (first vs last frame) ───────────────────────────────")
for db in range(N_DUMBBELLS):
    b0 = 2 * db
    b1 = 2 * db + 1
    com_0 = (trajectory[0, b0] + trajectory[0, b1]) / 2.0
    com_f = (trajectory[-1, b0] + trajectory[-1, b1]) / 2.0
    disp = np.linalg.norm(com_f - com_0)
    print(f"  Dumbbell {db}: COM displacement = {disp:.4f}")

print("\nDone. Trajectory saved to output_rigid_dumbbell/")

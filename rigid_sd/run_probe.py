"""Probe microrheology: two passive colloids + five active dumbbell swimmers.

Particle layout:
    0            — probe colloid, held at origin by a stiff harmonic trap
    1            — outer colloid, dragged in a circle by a moving harmonic trap
    2k, 2k+1     — swimmer k (tail, head) for k = 0 … N_SWIMMERS-1

Swimmer COMs and bond orientations are randomised in 3-D around the probe pair
using a fixed numpy seed (POSITION_SEED) for reproducibility.

Run from the project root:
    JAX_PLATFORMS=cpu python rigid_sd/run_probe.py
"""

import numpy as np
import jax.numpy as jnp

from rigid_sd.main_rigid import run_rigid_sd

# ── System parameters ─────────────────────────────────────────────────────────

SEPARATION    = 2.8        # probe–outer-colloid COM distance
BOND_LENGTH   = 2.001      # swimmer bead centre-to-centre (surface gap ≈ 0.001)
N_COLLOIDS    = 2
N_SWIMMERS    = 5
N_PARTICLES   = N_COLLOIDS + N_SWIMMERS * 2   # 12 total
N_BODIES      = N_COLLOIDS + N_SWIMMERS       # 7 total

LX = LY = LZ  = 30.0
KT            = 1.0
ACTIVE_ALPHA  = 2.0      # pusher swimmers
EWALD_XI      = 0.5
ERROR_TOL     = 0.001
PARTICLE_R    = 1.0
MAX_STRAIN    = 0.5

TRAP_K        = 300.0      # harmonic spring constant (stiff enough to confine)
TRAP_OMEGA    = 10.0        # angular frequency of outer-colloid orbit [rad/time]

NUM_STEPS      = 1000
WRITING_PERIOD = 10
DT             = 0.001

# ── Initial positions ─────────────────────────────────────────────────────────
# Swimmer COMs are placed at random distances (4.5–10 σ) from the probe-pair
# midpoint in random 3-D directions.  Bond axes are also random unit vectors.
# Rejection sampling enforces a minimum bead–bead gap of 0.5 σ.

POSITION_SEED  = 42
_rng           = np.random.default_rng(POSITION_SEED)
half           = BOND_LENGTH / 2.0
probe_mid      = np.array([SEPARATION / 2.0, 0.0, 0.0])
MIN_GAP        = 2.5   # minimum centre-to-centre distance between any two beads

def _rand_unit(rng):
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)

# Seed the placed-beads list with the two probe/colloid particles.
_placed_beads = [
    np.array([0.0, 0.0, 0.0]),
    np.array([SEPARATION, 0.0, 0.0]),
]
_swimmer_coms  = []
_swimmer_axes  = []

for _sw in range(N_SWIMMERS):
    for _attempt in range(5000):
        _d_hat = _rand_unit(_rng)
        _dist  = _rng.uniform(4.5, 10.0)
        _com   = probe_mid + _dist * _rand_unit(_rng)
        _com   = np.clip(_com, -LX / 2.0 + 3.0, LX / 2.0 - 3.0)
        _b0    = _com - half * _d_hat
        _b1    = _com + half * _d_hat
        if all(
            np.linalg.norm(_b - _existing) >= MIN_GAP
            for _b in (_b0, _b1)
            for _existing in _placed_beads
        ):
            _swimmer_coms.append(_com)
            _swimmer_axes.append(_d_hat)
            _placed_beads.extend([_b0, _b1])
            break
    else:
        raise RuntimeError(f"Could not place swimmer {_sw} without overlap after 5000 attempts")

_pos_list = [
    [0.0, 0.0, 0.0],
    [SEPARATION, 0.0, 0.0],
]
for _com, _d in zip(_swimmer_coms, _swimmer_axes):
    _pos_list.append((_com - half * _d).tolist())  # tail
    _pos_list.append((_com + half * _d).tolist())  # head

positions = jnp.array(_pos_list, dtype=jnp.float32)

# Bodies 0, 1 = single-bead colloids; bodies 2 … N_BODIES-1 = swimmer dumbbells.
assembly_ids = np.array(
    [0, 1] + [b for sw in range(N_SWIMMERS) for b in [N_COLLOIDS + sw, N_COLLOIDS + sw]],
    dtype=int,
)

# Type labels for GSD output: free colloids → A, swimmer heads → B, tails → C.
type_labels = np.array(["A", "A"] + ["C", "B"] * N_SWIMMERS)

# ── Harmonic trap ─────────────────────────────────────────────────────────────

trap_particle_ids = np.array([0, 1], dtype=int)

def make_trap_targets(separation, omega):
    def _fn(step, dt):
        t = step * dt
        return jnp.array([
            [0.0, 0.0, 0.0],                                          # probe: fixed
            [separation * jnp.cos(omega * t),
             separation * jnp.sin(omega * t), 0.0],                   # outer: circle
        ], dtype=jnp.float32)
    return _fn

trap_targets_fn = make_trap_targets(SEPARATION, TRAP_OMEGA)

# ── Seeds ─────────────────────────────────────────────────────────────────────

SEED_RFD    = 1234567
SEED_FFWAVE = 7654321
SEED_FFREAL = 3141592
SEED_NF     = 2718281

# ── Run ───────────────────────────────────────────────────────────────────────

print(f"Probe microrheology: {N_COLLOIDS} colloids + {N_SWIMMERS} dumbbell swimmers")
for _sw, (_com, _d) in enumerate(zip(_swimmer_coms, _swimmer_axes)):
    print(f"  Swimmer {_sw}: COM={np.round(_com, 3)}  axis={np.round(_d, 3)}")
print(f"  Particles: {N_PARTICLES}  |  Bodies: {N_BODIES}")
print(f"  Separation: {SEPARATION}  |  Trap k: {TRAP_K}  |  Trap ω: {TRAP_OMEGA}")
print(f"  active_alpha: {ACTIVE_ALPHA}  |  kT: {KT}  |  Steps: {NUM_STEPS}")

trajectory, velocities = run_rigid_sd(
    num_steps       = NUM_STEPS,
    writing_period  = WRITING_PERIOD,
    time_step       = DT,
    lx=LX, ly=LY, lz=LZ,
    num_particles   = N_PARTICLES,
    num_bodies      = N_BODIES,
    assembly_ids    = assembly_ids,
    max_strain      = MAX_STRAIN,
    temperature     = KT,
    particle_radius = PARTICLE_R,
    ewald_xi        = EWALD_XI,
    error_tolerance = ERROR_TOL,
    positions       = positions,
    seed_rfd        = SEED_RFD,
    seed_ffwave     = SEED_FFWAVE,
    seed_ffreal     = SEED_FFREAL,
    seed_nf         = SEED_NF,
    output          = "output_probe",
    active_alpha    = ACTIVE_ALPHA,
    trap_particle_ids = trap_particle_ids,
    trap_spring_k   = TRAP_K,
    trap_targets    = trap_targets_fn,
    probe_body_id   = 0,
    trap_v_char     = SEPARATION * TRAP_OMEGA,
)

# ── Validation ────────────────────────────────────────────────────────────────

print("\n── Probe displacement (particle 0) ───────────────────────────────────")
probe_disp = np.linalg.norm(trajectory[-1, 0] - trajectory[0, 0])
print(f"  |Δr_probe| = {probe_disp:.4f}  (should be small)")

print("\n── Outer colloid vs trap trajectory ─────────────────────────────────")
n_frames = trajectory.shape[0]
max_trap_err = 0.0
for frame in range(n_frames):
    t = frame * WRITING_PERIOD * DT
    r_trap = np.array([SEPARATION * np.cos(TRAP_OMEGA * t),
                       SEPARATION * np.sin(TRAP_OMEGA * t), 0.0])
    err = np.linalg.norm(trajectory[frame, 1] - r_trap)
    max_trap_err = max(max_trap_err, err)
print(f"  max |r_1 − r_trap(t)| = {max_trap_err:.4f}  (should be << {SEPARATION})")

print("\n── Bond-length conservation (swimmer dumbbells) ─────────────────────")
for sw in range(N_SWIMMERS):
    b0 = N_COLLOIDS + 2 * sw
    b1 = b0 + 1
    bond_lengths = np.linalg.norm(trajectory[:, b0] - trajectory[:, b1], axis=1)
    print(f"  Swimmer {sw}: mean={bond_lengths.mean():.6f}  std={bond_lengths.std():.2e}  "
          f"(initial={BOND_LENGTH:.4f})")

np.save("output_probe/type_labels.npy", type_labels)
print("\nDone. Trajectory saved to output_probe/")

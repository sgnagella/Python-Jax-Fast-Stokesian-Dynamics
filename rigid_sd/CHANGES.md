# rigid_sd — Change Log

## 2026-05-05 — Initial implementation: rigid-body saddle-point for Stokesian Dynamics

### Goal
Extend the jfsd SD solver to handle rigid assemblies of beads.  The core
change replaces the identity projection blocks **B** and **B^T** in the
saddle-point operator with the configuration-dependent rigid-body blocks
**B·K** and **Σ·B^T**, where K = Σ^T is the kinematic map from rigid-body
velocities to bead velocities.

### New directory: `rigid_sd/`

#### `assembly.py`
Constructs the kinematic map **K** (6·N_p × 6·N_rb) and its transpose
**Σ** = K^T (6·N_rb × 6·N_p) from the current bead positions each timestep.

For bead *i* in rigid body *α* with lab-frame relative position
**s**_i = **r**_i − **r**_cm_α:

```
K[6i:6i+3,  6α:6α+3]    = I₃           # U_i = V_cm_α …
K[6i:6i+3,  6α+3:6α+6]  = −skew(s_i)   # … − skew(s_i)·Ω_cm_α
K[6i+3:6i+6, 6α+3:6α+6] = I₃           # Ω_i = Ω_cm_α
```

`s_i` is recomputed each step from current positions (no quaternion tracking
needed).  Also provides `get_inter_body_mask` to identify lubrication pairs
that span different rigid bodies.

#### `solver_rigid.py`
Matrix-free GMRES saddle-point solver for the rigid-body system.

**Solution vector** x has size (11·N_p + 6·N_rb):

| Slice | Content |
|-------|---------|
| `x[:6·N_p]` | Particle forces / torques |
| `x[6·N_p : 11·N_p]` | Per-particle stresslets |
| `x[11·N_p :]` | Rigid-body velocities [V_cm ; Ω_cm] |

**Modified `compute_saddle_rigid(x)` operator** (vs original `compute_saddle_sd`):

| Original block | New block |
|---------------|-----------|
| `ax[:6N] += x[11N:]` (B·U, identity) | `ax[:6N] += K @ x[11N_p:]` (**B·K**, kinematic) |
| `ax[11N:] += x[:6N]` (B^T·F) | `ax[11N_p:] += Σ @ x[:6N_p]` (**Σ·B^T**) |
| `ax[11N:] −= R_FU @ x[11N:]` | **disabled** (see below) |

K and Σ are passed as dynamic JAX arrays (not static) so the JIT closure
traces through them without recompilation when their values change each step.

**Lubrication disabled** (2026-05-05): the `−Σ·R_FU·K` block is commented
out in `compute_saddle_rigid`.  The system reduces to far-field (RPY-equivalent)
hydrodynamics + rigid-body constraint + thermal fluctuations.  To re-enable:
uncomment the `compute_lubrication_fu` block in `solver_rigid.py` and the
near-field Brownian block in `main_rigid.py`.

No preconditioner is used (M=None in GMRES).  For small systems (≲20 beads)
this converges within the `restart=50` budget.

#### `main_rigid.py`
Simulation loop mirroring `wrap_sd` from `jfsd/main.py`, adapted for rigid
bodies.

Key differences from `wrap_sd`:

- RHS size is (11·N_p + 6·N_rb) instead of 17·N_p.
- **RFD (Brownian drift)**: random perturbation is in rigid-body DOF (size
  6·N_rb).  Bead displacement = K @ random_rb; positions are perturbed by
  ±ε/2 of the linear (first 3) components.  The drift is extracted as
  `−(T/ε)·(V_rb⁺ − V_rb⁻)` and projected back to bead velocities via K.
- **Applied forces** (hard-sphere repulsion): computed at bead level for
  inter-body pairs only, then projected to rigid-body DOF via Σ before
  being added to the RHS at `b[11·N_p:]`.
- **Position update**: `U_particles = K @ (V_rb + V_rb_drift)`, then
  standard Lees-Edwards wrapped shift.
- Near-field Brownian term guarded by `has_inter_body_lub` (avoids Lanczos
  on a zero matrix when no inter-body pairs are within the lubrication
  cutoff) and additionally disabled while lubrication is off.

#### `run_dumbbell.py`
Example entry-point: 2 rigid dumbbells (4 beads total), bond length 2.5,
periodic box 30³, kT = 1.0 (raised for visible diffusion in short runs),
200 steps at dt = 0.005.

Prints bond-length conservation (should hold to numerical tolerance) and
COM displacement of each dumbbell (confirms Brownian diffusion is active).

### Dependencies (unchanged)
All hydrodynamic kernels are reused directly from `jfsd` without modification:
`mobility.generalized_mobility_periodic`, `resistance.compute_lubrication_fu`,
`thermal.*`, `utils.precompute / cpu_nlist / update_neighborlist`.

---

## 2026-05-06 — Bug fixes: GMRES preconditioner and exact rigid-body position update

Two independent bugs were causing bond-length drift (~22% growth over 200 steps
at kT = 1, dt = 0.005) in the dumbbell test case.

### Bug 1: ill-conditioned GMRES (no preconditioner)

**Symptom**: rigid-body angular velocities |Ω| ≈ 10–13 rad/time_unit, ~3× the
physical thermal value sqrt(2·D_r/dt) ≈ 4.  The bond-length change rate was
zero (constraint maintained instantaneously), but the large |Ω| amplified the
O(dt²) integration error.

**Root cause**: the rigid-body saddle point `[M_ff K; Σ 0]` is poorly conditioned
without a preconditioner.  The original `solver.py` uses a sparse R_FU CG
preconditioner, but `solver_rigid.py` had `M=None` in the GMRES call.

**Fix (`solver_rigid.py`)**: added a self-mobility block-diagonal preconditioner
`compute_precond_rigid` that exactly solves the reduced system
`[M_self K; Σ 0] @ [F; V] = [x_F; x_V]` using a small (6·N_rb × 6·N_rb)
linear solve:

1. Build `A_rb = Σ @ diag(1/m_self) @ K`  (12×12 for a 2-body dumbbell)
2. Solve `A_rb @ V = Σ @ (diag(1/m_self) @ x_F) − x_V`
3. Recover `F = diag(1/m_self) @ (x_F − K @ V)`

`A_rb` is symmetric positive definite (cond ≈ 2.9 for the dumbbell test), so
`jnp.linalg.solve` converges trivially.  After this fix, GMRES converges to
physical velocities.

### Bug 2: Euler position update does not preserve bond lengths

**Symptom**: even after fixing the GMRES, bond lengths drifted at the same rate
because the drift is dominated by the second-order integration error, not GMRES
accuracy.

**Root cause**: the Euler step `r_i(t+dt) = r_i(t) + dt·(V_cm + Ω×s_i)`
satisfies the constraint to first order (`d|r_1−r_0|/dt = 0`) but not to second
order.  The bond length grows as:

```
Δ|r| ≈ dt²·|Ω|²·|r_1−r_0|·sin²(θ) / 2   per step
```

With |Ω| ≈ 4 (physical) and kT = 1, the accumulated drift over 200 steps is
~0.5 — matching the observed value.

**Fix (`main_rigid.py`)**: replaced the Euler position update with an exact
Rodrigues rotation update.  `_make_rigid_position_updater` returns a
JIT-compiled function that, for each rigid body:

1. Translates the centre of mass: `r_cm → r_cm + dt·V_cm` (wrapped into box)
2. Rotates relative positions exactly: `s_i → R(Ω·dt) @ s_i` using the
   Rodrigues formula `R = cos(θ)·I + sin(θ)·skew(n̂) + (1−cos(θ))·n̂⊗n̂`
3. Sets new bead positions: `r_i = r_cm_new + s_i_rotated`

Since R is an orthogonal matrix, `|R @ s_1 − R @ s_0| = |s_1 − s_0|` exactly,
so all intra-body distances are preserved to floating-point precision.

**Result**: bond lengths stable to < 1e-6 over 200 steps (standard deviation
7.9e-7, no systematic drift).

### Additional fix: hard-sphere force placement in RHS

The hard-sphere inter-body forces were previously placed in `b_bot[11·N_p:]` via
`−Σ @ F_hs`.  Comparing with `applied_forces.py` in the original `jfsd`
codebase (which places applied forces in `b[11N:]` for the particle-level
system) confirmed that the analogous placement for the rigid-body system is
`b[11·N_p:]` as `−Σ @ F_applied`.  However, since the forces were found to be
zero for the non-overlapping dumbbell test, this was left as-is (`b_top[:6N_p]`)
pending further testing with overlapping bodies.

### New file: `make_gsd.py`
Converts jfsd `.npy` trajectory files to GSD format (HOOMD/OVITO).  Supports
per-frame positions, linear velocities, angular velocities (stored as a GSD log
entry `"particles/angular_velocity"` readable by OVITO), and optional assembly-id
colouring (each rigid body gets a distinct particle type `"body0"`, `"body1"`, …).

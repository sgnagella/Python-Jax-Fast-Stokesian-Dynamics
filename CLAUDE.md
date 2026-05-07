# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (from source, with test dependencies)
pip install ".[test]"

# Run GPU tests
pytest tests/test_class.py

# Run CPU tests
JAX_PLATFORMS=cpu pytest tests/test_class_cpu.py
# or
bash run_tests_cpu.sh

# Run a single test
pytest tests/test_class_cpu.py::TestClassCPU::test_sedimenting_triangle
pytest tests/test_class.py::TestClass::test_deterministic_hydro

# Run the simulation CLI
jfsd -c files/example_configuration.toml -o output_dir/
jfsd -c files/example_configuration.toml -s initial_positions.npy -o output_dir/

# Launch the GUI
jfsd-gui
```

The GPU tests (`test_class.py`) require a CUDA-enabled GPU. CPU tests (`test_class_cpu.py`) use `JAX_PLATFORMS=cpu` to force CPU execution. JAX by default preallocates GPU memory; the code disables this via `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Architecture

### Simulation entry points

`jfsd/main.py` is the simulation orchestrator. The `main()` function dispatches to one of three wrappers based on `hydrodynamic_interaction_flag`:
- `wrap_bd()` — Brownian Dynamics (flag=0): no hydrodynamic interactions, diagonal diffusion tensor
- `wrap_rpy()` — Rotne-Prager-Yamakawa (flag=1): far-field HI only, no lubrication
- `wrap_sd()` — Full Stokesian Dynamics (flag=2): far-field + near-field lubrication, stresslet, Brownian drift

`jfsd/__main__.py` parses CLI args and calls `main()`. Configuration is loaded from TOML via `jfsd/config.py`.

### The saddle point system (SD mode only)

At each timestep in `wrap_sd`, a 17N×17N linear system `Ax = b` is solved by GMRES in `jfsd/solver.py`. The solution vector `x` is partitioned as:

| Block | Size | Content |
|-------|------|---------|
| `x[:6N]` | 6N | Generalized forces/torques on particles |
| `x[6N:11N]` | 5N | Stresslets S (symmetric traceless) |
| `x[11N:]` | 6N | Particle velocities U and angular velocities Ω |

The saddle point matrix-vector product `A·x` is implemented **matrix-free** in `solver.py::compute_saddle_sd`:
1. `M_ff · x[:11N]` → far-field contribution to velocities and strain rates (`jfsd/mobility.py`)
2. `B · x[11N:]` → add velocities to first 6N (identity projection block)
3. `-R_FU · x[11N:]` → near-field lubrication resistance acting on velocities (`resistance.compute_lubrication_fu`)
4. `B^T · x[:6N]` → add forces to last 6N (transpose projection block)

The right-hand side `b` is assembled in `wrap_sd` from:
- `b[:11N]`: far-field thermal slip velocities (wave-space + real-space Lanczos) + applied forces
- `b[6N:11N]`: `-E_inf` (ambient shear strain rate)
- `b[11N:]`: near-field lubrication shear term (`R_FE`) + near-field Brownian forces

Positions are updated from `x[11N:]` (velocities) plus the Brownian drift (computed via random finite difference, RFD: two saddle-point solves with ±ε perturbations).

### Hydrodynamics modules

**`jfsd/mobility.py`** — Far-field grand mobility matrix M. Computed using the Spectral Ewald method (FFT-based wave-space + real-space). Key functions:
- `generalized_mobility_periodic()` — applies M to an 11N generalized force vector [F (6N), S (5N)] in periodic BC
- `mobility_periodic()` — applies M to a 6N force/torque vector (used in RPY mode)
- `generalized_mobility_open()`, `mobility_open()` — open boundary variants

**`jfsd/resistance.py`** — Near-field lubrication resistance functions. Key functions:
- `compute_lubrication_fu()` — applies R_FU to particle velocities (the core near-field block in the saddle-point)
- `rfu_sparse_precondition()` — constructs a sparse approximation to R_FU for use as preconditioner (pairs within 2.1 radii)
- `compute_rfe()` — near-field force/torque from background shear (R_FE · E_inf term)
- `compute_rse()` — near-field stresslet from background shear (R_SE · E_inf term)
- `compute_rsu()` — near-field stresslet from particle velocities (R_SU · U)

Lubrication scalar functions are looked up from precomputed tables in `files/ResTableDist.npy` and `files/ResTableVals.npy` (22 scalar functions per pair, indexed by surface separation).

**`jfsd/thermal.py`** — Brownian fluctuation calculations:
- Real-space far-field: Lanczos decomposition of the real-space mobility matrix (`compute_real_space_slipvelocity`)
- Wave-space far-field: FFT-based Fourier space sampling (`compute_wave_space_slipvelocity`)
- Near-field lubrication: Lanczos decomposition of R_FU (`compute_nearfield_brownianforce`)
- BD mode: direct diagonal square root (`compute_bd_randomforce`)

**`jfsd/lanczos.py`** — Generic Lanczos algorithm for computing matrix square roots without explicit matrix assembly (used in thermal.py). The iteration count is adaptive in the main loop.

**`jfsd/ewald_tables.py`** — Computes tabulated real-space Ewald mobility scalar functions (`compute_real_space_ewald_table`). The table is computed at initialization with extended precision.

### Data flow and precomputation

`utils.py::precompute()` (called every timestep) returns a tuple of 19 arrays containing:
- Grid indices and Gaussian spreading weights for FFT (wave-space)
- Pair distances, unit vectors, and 7 scalar mobility functions (f1, f2, g1, g2, h1, h2, h3) for real-space
- Lubrication pair data: distances, indices, and 23 scalar resistance functions (ResFunction[0..22])

Three neighbor lists are maintained: `nl_ff` (far-field, Ewald cutoff), `nl_lub` (lubrication, ≤3.99 radii), `nl_prec` (preconditioner, ≤2.1 radii). Lists are rebuilt on CPU via `utils.cpu_nlist()` and updated cheaply with `utils.update_neighborlist()`.

**`jfsd/shear.py`** — Lees-Edwards boundary conditions for shear. Updates box tilt factor `xy` and the wave-vector grid (`gridk`) at each step when shear is applied.

**`jfsd/jaxmd_space.py` / `jfsd/jaxmd_util.py`** — Adapted from JAX-MD. Provides `periodic_general()` (creates shift function for general triclinic boxes) used for Lees-Edwards PBC.

### JAX-specific patterns

- Static arguments to `@jit` use `static_argnums` for integer sizes like `num_particles`, grid dimensions, `gauss_support`, and `max_nonzero_per_row`. Changing these triggers recompilation.
- Single precision is used by default (`jax_enable_x64=False`); open boundary mode switches to double precision.
- The preconditioner uses `jax.scipy.sparse.linalg.cg` inside GMRES (a JAX-compatible nested sparse solve).
- Array shapes must be static for JIT; neighbor list padding uses fixed-size arrays with sentinel values.

### Rigid assembly extension (goal)

To support rigid assemblies, the following blocks in the saddle-point system must be modified:

1. **Off-diagonal projection blocks B and B^T** (in `solver.py::compute_saddle_sd`, lines ~116-128): Replace the identity projector with a rigid-body kinematic map that maps rigid body (center-of-mass) velocities to individual particle velocities (`U_i = U_cm + Ω_cm × r_i`). The solution vector `x[11N:]` would become velocities at the level of rigid bodies, not individual particles.

2. **Near-field resistance matrix R_FU** (`resistance.compute_lubrication_fu`): The lubrication forces between particles that belong to the same rigid body must be excluded (they are internal forces), or the matrix must be projected to the rigid-body DOF space.

3. **Preconditioner** (`rfu_sparse_precondition` and `preprocess_sparse_triangular` in `utils.py`): Must reflect the same rigid-body constraints, otherwise GMRES convergence degrades.

4. **Stresslet computation** (`resistance.compute_rsu`, `compute_rse`): Stresslets from internal pairs (within a rigid body) should be summed at the rigid-body level.

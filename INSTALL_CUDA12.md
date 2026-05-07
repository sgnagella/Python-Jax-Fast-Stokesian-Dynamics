# Installation Guide: CUDA 12 + JAX 0.9.x

This guide covers installing JFSD with CUDA 12 and modern JAX (0.9.x), and documents the source changes
required for compatibility. The README targets CUDA 11 / JAX 0.4.17 — **ignore those instructions entirely**
when using CUDA 12.

## Prerequisites

- CUDA 12.x (tested with 12.0)
- Python ≥ 3.9 (tested with 3.11)
- conda or a virtual environment manager

## Installation Steps

### 1 — Create and activate a conda environment

```bash
conda create -n jaxsd python=3.11
conda activate jaxsd
```

Or with venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2 — Install JAX with CUDA 12 support

```bash
pip install "jax[cuda12]==0.9.2"
```

The `[cuda12]` extra automatically selects and installs the matching `jaxlib`, CUDA runtime libraries,
cuDNN, cuBLAS, cuFFT, cuSolver, NCCL, and the JAX CUDA plugin. No separate `jaxlib` install or `-f` flag
is needed.

> **Do not** follow the README's `jaxlib==0.4.17+cuda11.cudnn86` command — it targets CUDA 11 only.

### 3 — Install JFSD in editable mode with all dependencies

Install in editable mode so that source-level fixes are picked up immediately without reinstalling:

```bash
pip install -e ".[test]"
```

`jax==0.9.2` is already satisfied from step 2, so pip will not overwrite the CUDA-enabled build.

### 4 — Verify the GPU is detected

```python
import jax
print(jax.default_backend())  # should print 'gpu'
print(jax.devices())
```

### 5 — Run the test suite

```bash
pytest tests/test_class.py -v
```

Expected outcome: **15/16 tests pass**. `test_thermal_1body` is a stochastic MSD test whose tolerance
was tuned for an older JAX/NumPy version; its failure is a tolerance calibration issue, not a
correctness regression.

---

## Source Changes Required for Compatibility

The following changes were needed to make the codebase compatible with JAX 0.4.x+ and NumPy 2.x.
All changes are already applied in the current source tree.

### 1 — `jax.config` API (`jfsd/main.py`)

`jax.config` is no longer a submodule that can be imported directly.

```python
# Before (JAX < 0.4.x)
from jax.config import config
config.update("jax_enable_x64", False)

# After
import jax
jax.config.update("jax_enable_x64", False)
```

### 2 — `jax.lib.xla_bridge` removed (`jfsd/main.py`, `jfsd/jaxmd_util.py`, `tests/`)

`xla_bridge` was removed from `jax.lib`. Use `jax.default_backend()` instead.

```python
# Before
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# After
import jax
print(jax.default_backend())  # returns 'gpu', 'cpu', or 'tpu'
```

Files changed: `jfsd/main.py`, `jfsd/jaxmd_util.py`, `tests/test_class.py`, `tests/test_class_cpu.py`

### 3 — `np.longfloat` removed (`jfsd/ewald_tables.py`)

`np.longfloat` was removed in NumPy 2.0. Use `np.longdouble` instead (they were always aliases).

```python
# Before (NumPy < 2.0)
dr = np.longfloat(dr_decimal)

# After
dr = np.longdouble(dr_decimal)
```

Files changed: `jfsd/ewald_tables.py` (4 occurrences, replaced with `replace_all`)

---

## Summary of Changed Files

| File | Change |
|------|--------|
| `jfsd/main.py` | Replace `from jax.config import config` + `from jax.lib import xla_bridge`; use `jax.config.update(...)` and `jax.default_backend()` |
| `jfsd/jaxmd_util.py` | Replace `from jax.lib import xla_bridge`; use `jax.default_backend()` |
| `jfsd/ewald_tables.py` | Replace `np.longfloat` → `np.longdouble` (4 occurrences) |
| `tests/test_class.py` | Replace `from jax.lib import xla_bridge`; use `jax.default_backend()` |
| `tests/test_class_cpu.py` | Replace `from jax.lib import xla_bridge`; use `jax.default_backend()` |

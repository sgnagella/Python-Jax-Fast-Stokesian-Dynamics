"""Initial-condition generator for probe-microrheology simulations.

Particle layout produced
------------------------
  index 0          — probe colloid 0 (at origin)
  index 1          — probe colloid 1 (at [separation, 0, 0])
  index 2 + 2k     — swimmer k, tail bead
  index 2 + 2k + 1 — swimmer k, head bead

Bodies
------
  body 0  → probe 0   (single bead)
  body 1  → probe 1   (single bead)
  body 2+k → swimmer k (two beads)

Usage — CLI
-----------
    python rigid_sd/init_probe_dumbbells.py \\
        --probe-separation 2.8 --n-dumbbells 20 --number-density 0.01 \\
        --bond-length 2.001 --seed 42 --output init.npz --gsd init.gsd

Usage — library
---------------
    from rigid_sd.init_probe_dumbbells import initialize
    cfg = initialize(probe_separation=2.8, n_dumbbells=20, number_density=0.01)
    # cfg["positions"], cfg["assembly_ids"], cfg["lx"], cfg["n_particles"], …
"""

import argparse
import math
import sys
from pathlib import Path

# When run directly as a script, insert the project root so `rigid_sd.*`
# absolute imports resolve correctly in this file and its transitive imports.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from rigid_sd.make_gsd import write_gsd


# ─── Core function ────────────────────────────────────────────────────────────

def initialize(
    probe_separation: float,
    n_dumbbells: int,
    number_density: float,
    bond_length: float = 2.001,
    particle_radius: float = 1.0,
    surface_gap: float = 0.1,
    seed: int = 42,
    max_attempts: int = 10_000,
) -> dict:
    """Generate non-overlapping initial positions for 2 probes + N dumbbell swimmers.

    The cubic box side length L is derived from the requested dumbbell number
    density:  L = (n_dumbbells / number_density)^(1/3).

    Probe 0 is fixed at the origin; probe 1 is placed at [probe_separation, 0, 0].
    Swimmer COMs are drawn uniformly from [-L/2, L/2]^3; bond axes are random
    unit vectors.  Rejection sampling enforces a minimum surface-to-surface gap
    of `surface_gap` between all inter-body bead pairs (using minimum-image
    distances for periodic consistency).

    Parameters
    ----------
    probe_separation : float
        Centre-to-centre distance between the two probe colloids.
    n_dumbbells : int
        Number of rigid dumbbell swimmers to place.
    number_density : float
        Dumbbell number density  ρ = n_dumbbells / L³.  Sets the box size.
    bond_length : float
        Bead centre-to-centre separation within each dumbbell (default 2.001).
    particle_radius : float
        Radius of every bead (default 1.0, so diameter = 2.0).
    surface_gap : float
        Minimum allowed surface-to-surface clearance between beads of different
        bodies (default 0.1).  The hard-sphere diameter in the SD solver is
        2.002, so values >= 0.002 are physically meaningful.
    seed : int
        NumPy Generator seed — fix for reproducibility.
    max_attempts : int
        Maximum placement attempts per swimmer before raising RuntimeError.

    Returns
    -------
    dict with keys
        positions     : (N_p, 3) float32 ndarray
        assembly_ids  : (N_p,)   int ndarray  — body index for each bead
        n_particles   : int      N_p = 2 + 2·n_dumbbells
        n_bodies      : int      2 + n_dumbbells
        lx, ly, lz    : float    cubic box side length
        type_labels   : (N_p,)   str ndarray  (A = probe, C = tail, B = head)
        probe_separation, bond_length, n_dumbbells, number_density, seed
            (metadata — passed through for downstream bookkeeping)

    Raises
    ------
    ValueError
        If the box implied by number_density is too small for probe_separation.
    RuntimeError
        If a swimmer cannot be placed without overlap after max_attempts tries.
    """
    if n_dumbbells < 1:
        raise ValueError("n_dumbbells must be >= 1.")
    if number_density <= 0.0:
        raise ValueError("number_density must be positive.")

    n_colloids  = 2
    n_particles = n_colloids + 2 * n_dumbbells
    n_bodies    = n_colloids + n_dumbbells

    # ── Derive box from dumbbell number density ───────────────────────────────
    L = (n_dumbbells / number_density) ** (1.0 / 3.0)

    # Both probes must sit comfortably inside [-L/2, L/2].
    # Probe 1 is at x = probe_separation, so we need L/2 > probe_separation + particle_radius.
    min_L = 2.0 * (probe_separation + particle_radius)
    if L < min_L:
        raise ValueError(
            f"Box side L = {L:.3f} is too small for probe_separation = {probe_separation:.3f}.\n"
            f"Need L >= {min_L:.3f}.  Lower number_density or increase n_dumbbells."
        )

    # Warn if volume fraction is high (rejection sampler will be slow above ~0.30).
    bead_vol = (4.0 / 3.0) * math.pi * particle_radius**3
    phi = (n_particles * bead_vol) / L**3
    if phi > 0.30:
        print(
            f"WARNING: volume fraction φ ≈ {phi:.3f} is high. "
            "Placement may be slow or fail; consider a lower number_density.",
            file=sys.stderr,
        )

    rng    = np.random.default_rng(seed)
    half   = bond_length / 2.0
    min_cc = 2.0 * particle_radius + surface_gap   # minimum centre-to-centre

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _rand_unit() -> np.ndarray:
        v = rng.standard_normal(3)
        return v / np.linalg.norm(v)

    def _min_image_dist(a: np.ndarray, b: np.ndarray) -> float:
        """Scalar minimum-image distance in a cubic box of side L."""
        d = a - b
        d -= L * np.round(d / L)
        return float(np.linalg.norm(d))

    def _overlaps(b: np.ndarray) -> bool:
        """True if bead b is too close to any already-placed bead."""
        return any(_min_image_dist(b, e) < min_cc for e in placed_beads)

    # ── Fixed probe positions ─────────────────────────────────────────────────
    probe0 = np.array([0.0,              0.0, 0.0])
    probe1 = np.array([probe_separation, 0.0, 0.0])
    placed_beads: list[np.ndarray] = [probe0, probe1]

    # ── Dumbbell placement ────────────────────────────────────────────────────
    swimmer_coms: list[np.ndarray] = []
    swimmer_axes: list[np.ndarray] = []

    for sw in range(n_dumbbells):
        for _ in range(max_attempts):
            d_hat = _rand_unit()
            com   = (rng.random(3) - 0.5) * L     # uniform in [-L/2, L/2]^3
            b0    = com - half * d_hat              # tail bead
            b1    = com + half * d_hat              # head bead
            if not _overlaps(b0) and not _overlaps(b1):
                swimmer_coms.append(com)
                swimmer_axes.append(d_hat)
                placed_beads.extend([b0, b1])
                break
        else:
            raise RuntimeError(
                f"Could not place swimmer {sw} after {max_attempts} attempts.\n"
                f"Try a lower number_density, a larger surface_gap, or a different seed."
            )

    # ── Build output arrays ───────────────────────────────────────────────────
    pos_list = [probe0.tolist(), probe1.tolist()]
    for com, d in zip(swimmer_coms, swimmer_axes):
        pos_list.append((com - half * d).tolist())   # tail
        pos_list.append((com + half * d).tolist())   # head

    positions = np.array(pos_list, dtype=np.float32)

    # assembly_ids[i] = index of the rigid body particle i belongs to.
    assembly_ids = np.array(
        [0, 1]
        + [n_colloids + sw for sw in range(n_dumbbells) for _ in range(2)],
        dtype=int,
    )

    # Type labels: A = probe colloid, C = swimmer tail, B = swimmer head.
    type_labels = np.array(["A", "A"] + ["C", "B"] * n_dumbbells)

    return {
        "positions":        positions,
        "assembly_ids":     assembly_ids,
        "n_particles":      n_particles,
        "n_bodies":         n_bodies,
        "lx": L, "ly": L, "lz": L,
        "type_labels":      type_labels,
        # metadata
        "probe_separation": probe_separation,
        "bond_length":      bond_length,
        "n_dumbbells":      n_dumbbells,
        "number_density":   number_density,
        "seed":             seed,
    }


# ─── Summary printer ──────────────────────────────────────────────────────────

def print_summary(cfg: dict, max_swimmers: int = 20) -> None:
    """Print a human-readable summary of the generated configuration."""
    L   = cfg["lx"]
    rho = cfg["n_dumbbells"] / L**3
    bead_vol = (4.0 / 3.0) * math.pi * 1.0**3   # assumes default radius
    phi = cfg["n_particles"] * bead_vol / L**3

    print(f"  Box:              {L:.4f}³   (volume = {L**3:.2f})")
    print(f"  N_particles:      {cfg['n_particles']}   |   N_bodies: {cfg['n_bodies']}")
    print(f"  N_dumbbells:      {cfg['n_dumbbells']}   |   ρ = {rho:.5g}   |   φ ≈ {phi:.4f}")
    print(f"  bond_length:      {cfg['bond_length']}")
    print(f"  probe_separation: {cfg['probe_separation']}")
    print(f"  seed:             {cfg['seed']}")

    pos = cfg["positions"]
    n_col = 2
    n_show = min(cfg["n_dumbbells"], max_swimmers)
    if n_show < cfg["n_dumbbells"]:
        print(f"  (showing first {n_show} of {cfg['n_dumbbells']} swimmers)")
    for sw in range(n_show):
        b0  = pos[n_col + 2 * sw]
        b1  = pos[n_col + 2 * sw + 1]
        com = (b0 + b1) / 2.0
        d   = b1 - b0
        d  /= np.linalg.norm(d)
        print(f"  Swimmer {sw:4d}: COM = {np.round(com, 3)}   axis = {np.round(d, 3)}")


# ─── Validation helper (can also be called from tests) ────────────────────────

def validate(cfg: dict, surface_gap: float = 0.1, particle_radius: float = 1.0) -> bool:
    """Check that no two inter-body beads violate the minimum gap.

    Returns True if the configuration is valid, raises AssertionError otherwise.
    """
    pos  = cfg["positions"].astype(float)
    aids = cfg["assembly_ids"]
    L    = cfg["lx"]
    min_cc = 2.0 * particle_radius + surface_gap

    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if aids[i] == aids[j]:
                continue   # intra-body pair — skip
            d = pos[i] - pos[j]
            d -= L * np.round(d / L)
            dist = float(np.linalg.norm(d))
            assert dist >= min_cc, (
                f"Overlap: beads {i} (body {aids[i]}) and {j} (body {aids[j]}) "
                f"have centre-to-centre distance {dist:.4f} < {min_cc:.4f}"
            )
    return True


# ─── GSD writer ──────────────────────────────────────────────────────────────

def write_initial_gsd(cfg: dict, path: str, particle_radius: float = 1.0) -> None:
    """Write a single-frame GSD file of the initial configuration.

    Wraps the positions array as a (1, N, 3) trajectory and delegates to
    write_gsd from make_gsd.py, so the output is immediately openable in OVITO
    with correct particle types (A = probe, C = tail, B = head) and diameters.

    Parameters
    ----------
    cfg : dict
        Output of initialize().
    path : str
        Destination .gsd file path.
    particle_radius : float
        Bead radius written to the GSD diameter field (default 1.0).
    """
    trajectory = cfg["positions"][np.newaxis, ...]   # (1, N, 3)
    write_gsd(
        trajectory     = trajectory,
        output_path    = path,
        lx             = cfg["lx"],
        ly             = cfg["ly"],
        lz             = cfg["lz"],
        particle_radius= particle_radius,
        type_labels    = cfg["type_labels"],
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate initial positions for a probe + dumbbell SD simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--probe-separation", type=float, required=True,
                   metavar="SEP",
                   help="Centre-to-centre distance between the two probe colloids.")
    p.add_argument("--n-dumbbells",      type=int,   required=True,
                   metavar="N",
                   help="Number of rigid dumbbell swimmers.")
    p.add_argument("--number-density",   type=float, required=True,
                   metavar="RHO",
                   help="Dumbbell number density  ρ = N / L³  (sets box size).")
    p.add_argument("--bond-length",      type=float, default=2.001,
                   metavar="B",
                   help="Bead centre-to-centre distance within each dumbbell.")
    p.add_argument("--particle-radius",  type=float, default=1.0,
                   metavar="R",
                   help="Bead radius (diameter = 2R).")
    p.add_argument("--surface-gap",      type=float, default=0.1,
                   metavar="GAP",
                   help="Minimum surface-to-surface clearance between inter-body beads.")
    p.add_argument("--seed",             type=int,   default=42,
                   help="NumPy RNG seed.")
    p.add_argument("--max-attempts",     type=int,   default=10_000,
                   metavar="M",
                   help="Max placement attempts per swimmer.")
    p.add_argument("--output", "-o",     type=str,   default=None,
                   metavar="FILE",
                   help="Save configuration to this .npz file.")
    p.add_argument("--gsd",              type=str,   default=None,
                   metavar="FILE",
                   help="Write a single-frame GSD file for OVITO visualisation.")
    p.add_argument("--verbose", "-v",    action="store_true",
                   help="Print per-swimmer placement details.")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    print("Initializing probe + dumbbell system…")
    cfg = initialize(
        probe_separation = args.probe_separation,
        n_dumbbells      = args.n_dumbbells,
        number_density   = args.number_density,
        bond_length      = args.bond_length,
        particle_radius  = args.particle_radius,
        surface_gap      = args.surface_gap,
        seed             = args.seed,
        max_attempts     = args.max_attempts,
    )

    if args.verbose:
        print_summary(cfg)
    else:
        L = cfg["lx"]
        rho = cfg["n_dumbbells"] / L**3
        print(
            f"  Box: {L:.4f}³  |  N_p: {cfg['n_particles']}"
            f"  |  N_b: {cfg['n_bodies']}  |  ρ = {rho:.5g}"
        )

    # Sanity check (cheap for small systems, can be slow for large ones).
    if cfg["n_particles"] <= 500:
        validate(cfg, surface_gap=args.surface_gap, particle_radius=args.particle_radius)
        print("  Overlap check: PASS")

    if args.output is not None:
        out = args.output if args.output.endswith(".npz") else args.output + ".npz"
        np.savez(
            out,
            positions        = cfg["positions"],
            assembly_ids     = cfg["assembly_ids"],
            type_labels      = cfg["type_labels"],
            box              = np.array([cfg["lx"], cfg["ly"], cfg["lz"]]),
            probe_separation = np.float32(cfg["probe_separation"]),
            bond_length      = np.float32(cfg["bond_length"]),
        )
        print(f"  Saved → {out}")
    else:
        print("  (No output file; pass --output path.npz to save.)")

    if args.gsd is not None:
        gsd_path = args.gsd if args.gsd.endswith(".gsd") else args.gsd + ".gsd"
        write_initial_gsd(cfg, gsd_path, particle_radius=args.particle_radius)
        print(f"  GSD  → {gsd_path}")


if __name__ == "__main__":
    main()

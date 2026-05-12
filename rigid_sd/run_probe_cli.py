"""CLI-driven probe microrheology simulation.

Identical physics to run_probe.py.  All parameters are command-line arguments.
The initial configuration is either loaded from a .npz file produced by
init_probe_dumbbells.py, or generated inline using the same parameters.

Particle layout (always):
  index 0          — probe colloid, held at origin by a stiff harmonic trap
  index 1          — outer colloid, dragged in a circle by a moving harmonic trap
  index 2 + 2k     — swimmer k tail
  index 2 + 2k + 1 — swimmer k head

Usage — load from .npz
----------------------
    python rigid_sd/run_probe_cli.py \\
        --init init.npz \\
        --kT 1.0 --active-alpha 2.0 --num-steps 1000 --output output_probe

Usage — generate inline
-----------------------
    python rigid_sd/run_probe_cli.py \\
        --probe-separation 2.8 --n-dumbbells 10 --number-density 0.01 \\
        --kT 1.0 --active-alpha 2.0 --num-steps 1000 --output output_probe
"""

import argparse
import sys
from pathlib import Path

# When run directly as a script, `rigid_sd/` is on sys.path but the project
# root is not, so `rigid_sd.*` imports in transitive modules fail.  Insert the
# project root (parent of this file's directory) so all absolute imports work.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import jax.numpy as jnp

from rigid_sd.main_rigid import run_rigid_sd
from rigid_sd.init_probe_dumbbells import initialize, print_summary


# ── Trap targets ──────────────────────────────────────────────────────────────

def _make_trap_targets(probe_separation: float, omega: float):
    """Probe 0 fixed at origin; probe 1 orbits a circle of radius probe_separation."""
    def _fn(step, dt):
        t = step * dt
        return jnp.array([
            [0.0, 0.0, 0.0],
            [probe_separation * jnp.cos(omega * t),
             probe_separation * jnp.sin(omega * t), 0.0],
        ], dtype=jnp.float32)
    return _fn


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe microrheology SD simulation (CLI-driven initial config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Initial configuration ─────────────────────────────────────────────
    init = p.add_argument_group(
        "initial configuration",
        "Supply --init (load from .npz) OR the three inline flags "
        "(--probe-separation, --n-dumbbells, --number-density).",
    )
    init.add_argument(
        "--init", metavar="FILE", default=None,
        help=".npz file from init_probe_dumbbells.py.",
    )
    init.add_argument(
        "--probe-separation", type=float, default=None, metavar="SEP",
        help="(inline) Centre-to-centre distance between the two probe colloids.",
    )
    init.add_argument(
        "--n-dumbbells", type=int, default=None, metavar="N",
        help="(inline) Number of rigid dumbbell swimmers.",
    )
    init.add_argument(
        "--number-density", type=float, default=None, metavar="RHO",
        help="(inline) Dumbbell number density ρ = N / L³ (sets box size).",
    )
    init.add_argument(
        "--bond-length", type=float, default=2.001, metavar="B",
        help="(inline) Bead centre-to-centre distance within each dumbbell.",
    )
    init.add_argument(
        "--surface-gap", type=float, default=0.1, metavar="GAP",
        help="(inline) Minimum surface-to-surface bead clearance during placement.",
    )
    init.add_argument(
        "--init-seed", type=int, default=42, metavar="SEED",
        help="(inline) NumPy seed for initial placement.",
    )

    # ── Simulation ────────────────────────────────────────────────────────
    sim = p.add_argument_group("simulation parameters")
    sim.add_argument("--kT",             type=float, default=1.0,
                     help="Thermal energy kT.")
    sim.add_argument("--active-alpha",   type=float, default=2.0,
                     help="Activity strength (>0 pusher, <0 puller, 0 passive).")
    sim.add_argument("--num-steps",      type=int,   default=1000,
                     help="Number of simulation steps.")
    sim.add_argument("--dt",             type=float, default=0.001,
                     help="Timestep.")
    sim.add_argument("--writing-period", type=int,   default=10,
                     help="Steps between saved trajectory frames.")
    sim.add_argument("--trap-k",         type=float, default=300.0,
                     help="Harmonic trap spring constant.")
    sim.add_argument("--trap-omega",     type=float, default=10.0,
                     help="Angular frequency of outer-colloid circular orbit.")
    sim.add_argument("--ewald-xi",       type=float, default=0.5,
                     help="Ewald splitting parameter ξ.")
    sim.add_argument("--error-tol",      type=float, default=0.001,
                     help="GMRES error tolerance.")
    sim.add_argument("--particle-radius",type=float, default=1.0,
                     help="Bead radius σ/2.")
    sim.add_argument("--max-strain",     type=float, default=0.5,
                     help="Maximum Lees-Edwards strain before neighbour-list rebuild.")

    # ── Random seeds ──────────────────────────────────────────────────────
    seeds = p.add_argument_group("random seeds")
    seeds.add_argument("--seed-rfd",    type=int, default=1234567)
    seeds.add_argument("--seed-ffwave", type=int, default=7654321)
    seeds.add_argument("--seed-ffreal", type=int, default=3141592)
    seeds.add_argument("--seed-nf",     type=int, default=2718281)

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument(
        "--output", "-o", default="output_probe", metavar="DIR",
        help="Directory for trajectory.npy / velocities.npy output.",
    )

    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    # ── Resolve initial configuration ─────────────────────────────────────
    if args.init is not None:
        npz_path = args.init if args.init.endswith(".npz") else args.init + ".npz"
        data = np.load(npz_path, allow_pickle=True)

        positions        = jnp.array(data["positions"], dtype=jnp.float32)
        assembly_ids     = data["assembly_ids"].astype(int)
        type_labels      = data["type_labels"]
        lx, ly, lz       = data["box"].tolist()
        probe_separation = float(data["probe_separation"])
        bond_length      = float(data["bond_length"])

        n_particles  = int(positions.shape[0])
        n_colloids   = 2
        n_dumbbells  = (n_particles - n_colloids) // 2
        n_bodies     = n_colloids + n_dumbbells

        print(f"Loaded initial config from {npz_path}")
        print(f"  N_p={n_particles}  N_b={n_bodies}  "
              f"box={lx:.4f}³  probe_separation={probe_separation:.4f}")

    else:
        inline = [args.probe_separation, args.n_dumbbells, args.number_density]
        if any(v is None for v in inline):
            print(
                "ERROR: supply either --init FILE  or all three of  "
                "--probe-separation, --n-dumbbells, --number-density.",
                file=sys.stderr,
            )
            sys.exit(1)

        cfg = initialize(
            probe_separation = args.probe_separation,
            n_dumbbells      = args.n_dumbbells,
            number_density   = args.number_density,
            bond_length      = args.bond_length,
            particle_radius  = args.particle_radius,
            surface_gap      = args.surface_gap,
            seed             = args.init_seed,
        )
        print("Generated initial config:")
        print_summary(cfg)

        positions        = jnp.array(cfg["positions"], dtype=jnp.float32)
        assembly_ids     = cfg["assembly_ids"]
        type_labels      = cfg["type_labels"]
        lx, ly, lz       = cfg["lx"], cfg["ly"], cfg["lz"]
        probe_separation = cfg["probe_separation"]
        bond_length      = cfg["bond_length"]
        n_particles      = cfg["n_particles"]
        n_bodies         = cfg["n_bodies"]
        n_colloids       = 2
        n_dumbbells      = cfg["n_dumbbells"]

    # ── Trap ──────────────────────────────────────────────────────────────
    trap_targets_fn = _make_trap_targets(probe_separation, args.trap_omega)

    # ── Echo run parameters ───────────────────────────────────────────────
    print(f"\nProbe microrheology: {n_colloids} colloids + {n_dumbbells} dumbbell swimmers")
    print(f"  N_p={n_particles}  N_b={n_bodies}  box={lx:.4f}³")
    print(f"  kT={args.kT}  alpha={args.active_alpha}  steps={args.num_steps}  dt={args.dt}")
    print(f"  trap_k={args.trap_k}  trap_ω={args.trap_omega}  sep={probe_separation}")
    print(f"  output → {args.output}/")

    # ── Run ───────────────────────────────────────────────────────────────
    trajectory, velocities = run_rigid_sd(
        num_steps         = args.num_steps,
        writing_period    = args.writing_period,
        time_step         = args.dt,
        lx=lx, ly=ly, lz=lz,
        num_particles     = n_particles,
        num_bodies        = n_bodies,
        assembly_ids      = assembly_ids,
        max_strain        = args.max_strain,
        temperature       = args.kT,
        particle_radius   = args.particle_radius,
        ewald_xi          = args.ewald_xi,
        error_tolerance   = args.error_tol,
        positions         = positions,
        seed_rfd          = args.seed_rfd,
        seed_ffwave       = args.seed_ffwave,
        seed_ffreal       = args.seed_ffreal,
        seed_nf           = args.seed_nf,
        output            = args.output,
        active_alpha      = args.active_alpha,
        trap_particle_ids = np.array([0, 1], dtype=int),
        trap_spring_k     = args.trap_k,
        trap_targets      = trap_targets_fn,
        probe_body_id     = 0,
    )

    # ── Validation ────────────────────────────────────────────────────────
    print("\n── Probe displacement (particle 0) ───────────────────────────────────")
    probe_disp = np.linalg.norm(trajectory[-1, 0] - trajectory[0, 0])
    print(f"  |Δr_probe| = {probe_disp:.4f}  (should be small with trap_k={args.trap_k})")

    print("\n── Outer colloid vs trap trajectory ─────────────────────────────────")
    n_frames = trajectory.shape[0]
    max_trap_err = 0.0
    for fi in range(n_frames):
        t = fi * args.writing_period * args.dt
        r_trap = np.array([
            probe_separation * np.cos(args.trap_omega * t),
            probe_separation * np.sin(args.trap_omega * t),
            0.0,
        ])
        max_trap_err = max(max_trap_err, np.linalg.norm(trajectory[fi, 1] - r_trap))
    print(f"  max |r_1 − r_trap(t)| = {max_trap_err:.4f}  (should be << {probe_separation})")

    print("\n── Bond-length conservation (swimmer dumbbells) ─────────────────────")
    for sw in range(n_dumbbells):
        b0 = n_colloids + 2 * sw
        b1 = b0 + 1
        bl = np.linalg.norm(trajectory[:, b0] - trajectory[:, b1], axis=1)
        print(f"  Swimmer {sw:3d}: mean={bl.mean():.6f}  std={bl.std():.2e}  "
              f"(initial={bond_length:.4f})")

    np.save(f"{args.output}/type_labels.npy", type_labels)
    print(f"\nDone. Trajectory saved to {args.output}/")


if __name__ == "__main__":
    main()

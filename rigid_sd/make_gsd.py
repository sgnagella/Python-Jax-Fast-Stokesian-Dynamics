#!/usr/bin/env python3
"""Convert jfsd .npy trajectory files to GSD format for HOOMD/OVITO.

Usage — standalone (box + type info supplied explicitly)
--------------------------------------------------------
    python rigid_sd/make_gsd.py output_rigid_dumbbell/trajectory.npy out.gsd \
        --lx 30 --ly 30 --lz 30 \
        --velocities output_rigid_dumbbell/velocities.npy \
        --assembly-ids assembly_ids.npy

Usage — with initial GSD template (box + particle info loaded automatically)
----------------------------------------------------------------------------
    python rigid_sd/make_gsd.py output_probe/trajectory.npy out.gsd \
        --init-gsd init.gsd \
        --velocities output_probe/velocities.npy

When --init-gsd is supplied the box dimensions, particle types, type IDs, and
per-particle diameters are read from frame 0 of that file.  --lx/--ly/--lz,
--type-labels, --assembly-ids, and --radius are then all optional and, if
given, are silently ignored in favour of the template values.

The GSD box convention matches HOOMD/OVITO: particles are stored in fractional
coordinates of a box centred at the origin, i.e. positions in [-L/2, L/2].
This is already the convention used by jfsd (positions are wrapped around the
origin via shift_fn).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import gsd.hoomd


def write_gsd(
    trajectory: np.ndarray,
    output_path: str | Path,
    lx: float | None = None,
    ly: float | None = None,
    lz: float | None = None,
    xy: float = 0.0,
    particle_radius: float = 1.0,
    writing_period: int = 1,
    velocities: np.ndarray | None = None,
    assembly_ids: np.ndarray | None = None,
    type_labels: np.ndarray | None = None,
    init_gsd: str | Path | None = None,
) -> None:
    """Write a GSD trajectory file from jfsd output arrays.

    Parameters
    ----------
    trajectory:
        Shape (n_frames, N, 3) float array of particle positions.
    output_path:
        Destination .gsd file.
    lx, ly, lz:
        Box edge lengths.  Required unless init_gsd is supplied.
    xy:
        Box tilt factor for Lees-Edwards shear (default 0).  Ignored when
        init_gsd is supplied (the tilt is read from the template instead).
    particle_radius:
        Bead radius; diameter = 2*radius is written to the GSD file.  Ignored
        when init_gsd is supplied.
    writing_period:
        Steps between saved frames in the simulation; used to set
        frame.configuration.step so OVITO reports the correct timestep.
    velocities:
        Shape (n_frames, N, 6) float array [vx, vy, vz, wx, wy, wz].
        Linear velocities go to frame.particles.velocity; angular velocities
        are stored as the log entry "particles/angular_velocity" so that
        OVITO exposes them as a per-particle vector property.
    type_labels:
        Shape (N,) string array giving the type name for each particle
        (e.g. ["A", "A", "C", "B", "C", "B"]).  Takes precedence over
        assembly_ids.  Ignored when init_gsd is supplied.
    assembly_ids:
        Shape (N,) int array mapping particle index → rigid body index.
        If provided (and type_labels is None), each body gets a distinct
        particle type name ("body0", "body1", …).  Ignored when init_gsd is
        supplied.
    init_gsd:
        Path to a GSD file whose frame 0 supplies the box dimensions
        (lx, ly, lz, xy), particle type names, per-particle type IDs, and
        per-particle diameters.  When given, all of lx/ly/lz/xy,
        particle_radius, type_labels, and assembly_ids are ignored.
    """
    n_frames, N, _ = trajectory.shape

    # ── Resolve particle metadata and box ────────────────────────────────────
    if init_gsd is not None:
        with gsd.hoomd.open(str(init_gsd), "r") as f_init:
            frame0 = f_init[0]

        box0     = frame0.configuration.box          # [lx, ly, lz, xy, xz, yz]
        lx, ly, lz = float(box0[0]), float(box0[1]), float(box0[2])
        xy       = float(box0[3])

        type_names = list(frame0.particles.types)
        typeids    = np.asarray(frame0.particles.typeid,  dtype=np.uint32)
        diameters  = np.asarray(frame0.particles.diameter, dtype=np.float32)

        if len(typeids) != N or len(diameters) != N:
            raise ValueError(
                f"init_gsd has {len(typeids)} particles but trajectory has {N}."
            )
    else:
        if lx is None or ly is None or lz is None:
            raise ValueError(
                "Either init_gsd or all of lx, ly, lz must be provided."
            )

        if type_labels is not None:
            labels     = np.asarray(type_labels, dtype=str)
            type_names = list(dict.fromkeys(labels))
            name_to_id = {name: i for i, name in enumerate(type_names)}
            typeids    = np.array([name_to_id[lbl] for lbl in labels], dtype=np.uint32)
        elif assembly_ids is not None:
            n_bodies   = int(assembly_ids.max()) + 1
            type_names = [f"body{k}" for k in range(n_bodies)]
            typeids    = assembly_ids.astype(np.uint32)
        else:
            type_names = ["A"]
            typeids    = np.zeros(N, dtype=np.uint32)

        diameters = np.full(N, 2.0 * particle_radius, dtype=np.float32)

    # ── Write frames ─────────────────────────────────────────────────────────
    with gsd.hoomd.open(str(output_path), "w") as f:
        for idx in range(n_frames):
            frame = gsd.hoomd.Frame()
            frame.configuration.step = idx * writing_period
            frame.configuration.box  = [lx, ly, lz, xy, 0.0, 0.0]

            frame.particles.N        = N
            frame.particles.types    = type_names
            frame.particles.typeid   = typeids
            frame.particles.position = np.asarray(trajectory[idx], dtype=np.float32)
            frame.particles.diameter = diameters

            if velocities is not None:
                frame.particles.velocity = np.asarray(
                    velocities[idx, :, :3], dtype=np.float32
                )
                frame.log["particles/angular_velocity"] = np.asarray(
                    velocities[idx, :, 3:6], dtype=np.float32
                )

            f.append(frame)

    print(f"Wrote {n_frames} frames → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert jfsd .npy trajectory to GSD (HOOMD/OVITO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("trajectory", help="Path to trajectory.npy  (n_frames × N × 3)")
    p.add_argument("output",     help="Output .gsd file path")

    p.add_argument(
        "--init-gsd", default=None, metavar="FILE",
        help="Initial-config GSD (from init_probe_dumbbells.py).  Loads box "
             "dimensions, particle types, and diameters automatically.  When "
             "supplied, --lx/--ly/--lz, --type-labels, --assembly-ids, and "
             "--radius are all optional and ignored.",
    )

    p.add_argument("--lx",   type=float, default=None, help="Box length in x  (required without --init-gsd)")
    p.add_argument("--ly",   type=float, default=None, help="Box length in y  (required without --init-gsd)")
    p.add_argument("--lz",   type=float, default=None, help="Box length in z  (required without --init-gsd)")
    p.add_argument("--xy",   type=float, default=0.0,  help="Box tilt factor for Lees-Edwards shear")
    p.add_argument("--radius",  type=float, default=1.0, help="Particle radius (diameter = 2*radius in GSD)")
    p.add_argument("--period",  type=int,   default=1,   help="writing_period used in the simulation")
    p.add_argument("--velocities",   default=None, help="Path to velocities.npy  (n_frames × N × 6)")
    p.add_argument("--assembly-ids", default=None, help="Path to assembly_ids.npy  (N,)")
    p.add_argument(
        "--type-labels", nargs="+", default=None, metavar="TYPE",
        help="Per-particle type labels (space-separated).  E.g. A A C B C B.  "
             "Overrides --assembly-ids.",
    )
    args = p.parse_args()

    # Validate: must have either --init-gsd or all three box dimensions.
    if args.init_gsd is None and any(v is None for v in (args.lx, args.ly, args.lz)):
        p.error("supply either --init-gsd FILE or all of --lx, --ly, --lz.")

    trajectory   = np.load(args.trajectory)
    velocities   = np.load(args.velocities)   if args.velocities   else None
    assembly_ids = np.load(args.assembly_ids) if args.assembly_ids else None
    type_labels  = np.array(args.type_labels) if args.type_labels  else None

    write_gsd(
        trajectory    = trajectory,
        output_path   = args.output,
        lx=args.lx, ly=args.ly, lz=args.lz,
        xy            = args.xy,
        particle_radius = args.radius,
        writing_period  = args.period,
        velocities    = velocities,
        assembly_ids  = assembly_ids,
        type_labels   = type_labels,
        init_gsd      = args.init_gsd,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert jfsd .npy trajectory files to GSD format for HOOMD/OVITO.

Usage
-----
    python rigid_sd/make_gsd.py output_rigid_dumbbell/trajectory.npy out.gsd \
        --lx 30 --ly 30 --lz 30 \
        --velocities output_rigid_dumbbell/velocities.npy \
        --assembly-ids assembly_ids.npy

The GSD box convention matches HOOMD/OVITO: particles are stored in fractional
coordinates of a box centred at the origin, i.e. positions in [-L/2, L/2].
This is already the convention used by jfsd (positions are wrapped around the
origin via shift_fn).
"""

import argparse
from pathlib import Path

import numpy as np
import gsd.hoomd


def write_gsd(
    trajectory: np.ndarray,
    output_path: str | Path,
    lx: float,
    ly: float,
    lz: float,
    xy: float = 0.0,
    particle_radius: float = 1.0,
    writing_period: int = 1,
    velocities: np.ndarray | None = None,
    assembly_ids: np.ndarray | None = None,
) -> None:
    """Write a GSD trajectory file from jfsd output arrays.

    Parameters
    ----------
    trajectory:
        Shape (n_frames, N, 3) float array of particle positions.
    output_path:
        Destination .gsd file.
    lx, ly, lz:
        Box edge lengths.
    xy:
        Box tilt factor for Lees-Edwards shear (default 0).
    particle_radius:
        Bead radius; diameter = 2*radius is written to the GSD file.
    writing_period:
        Steps between saved frames in the simulation; used to set
        frame.configuration.step so OVITO reports the correct timestep.
    velocities:
        Shape (n_frames, N, 6) float array [vx, vy, vz, wx, wy, wz].
        Linear velocities go to frame.particles.velocity; angular velocities
        are stored as the log entry "particles/angular_velocity" so that
        OVITO exposes them as a per-particle vector property.
    assembly_ids:
        Shape (N,) int array mapping particle index → rigid body index.
        If provided, each body gets a distinct particle type name
        ("body0", "body1", …) so OVITO can colour by rigid body.
    """
    n_frames, N, _ = trajectory.shape

    if assembly_ids is not None:
        n_bodies = int(assembly_ids.max()) + 1
        type_names = [f"body{k}" for k in range(n_bodies)]
        typeids = assembly_ids.astype(np.uint32)
    else:
        type_names = ["A"]
        typeids = np.zeros(N, dtype=np.uint32)

    with gsd.hoomd.open(str(output_path), "w") as f:
        for idx in range(n_frames):
            frame = gsd.hoomd.Frame()
            frame.configuration.step = idx * writing_period
            # HOOMD box: [lx, ly, lz, xy, xz, yz]
            frame.configuration.box = [lx, ly, lz, xy, 0.0, 0.0]

            frame.particles.N = N
            frame.particles.types = type_names
            frame.particles.typeid = typeids
            frame.particles.position = np.asarray(trajectory[idx], dtype=np.float32)
            frame.particles.diameter = np.full(N, 2.0 * particle_radius, dtype=np.float32)

            if velocities is not None:
                frame.particles.velocity = np.asarray(
                    velocities[idx, :, :3], dtype=np.float32
                )
                # OVITO reads "particles/*" log keys as per-particle properties.
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
    p.add_argument("output", help="Output .gsd file path")
    p.add_argument("--lx", type=float, required=True, help="Box length in x")
    p.add_argument("--ly", type=float, required=True, help="Box length in y")
    p.add_argument("--lz", type=float, required=True, help="Box length in z")
    p.add_argument("--xy", type=float, default=0.0,
                   help="Box tilt factor for Lees-Edwards shear")
    p.add_argument("--radius", type=float, default=1.0,
                   help="Particle radius (diameter = 2*radius in GSD)")
    p.add_argument("--period", type=int, default=1,
                   help="writing_period used in the simulation")
    p.add_argument("--velocities", default=None,
                   help="Path to velocities.npy  (n_frames × N × 6)")
    p.add_argument("--assembly-ids", default=None,
                   help="Path to assembly_ids.npy  (N,) — colours rigid bodies")
    args = p.parse_args()

    trajectory = np.load(args.trajectory)
    velocities = np.load(args.velocities) if args.velocities else None
    assembly_ids = np.load(args.assembly_ids) if args.assembly_ids else None

    write_gsd(
        trajectory=trajectory,
        output_path=args.output,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
        xy=args.xy,
        particle_radius=args.radius,
        writing_period=args.period,
        velocities=velocities,
        assembly_ids=assembly_ids,
    )


if __name__ == "__main__":
    main()

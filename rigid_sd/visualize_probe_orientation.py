#!/usr/bin/env python3
"""Post-processing: visualize probe orientation trajectory on a unit sphere.

The probe orientation is a unit vector whose rotation is integrated from the
angular velocity of the probe body during the simulation (Rodrigues rotation,
initialized at (1, 0, 0)).  This script loads probe_orientation.npy and plots
the path of the orientation vector tip on a unit sphere, colored by time.

Usage
-----
    python rigid_sd/visualize_probe_orientation.py output_probe/probe_orientation.npy
    python rigid_sd/visualize_probe_orientation.py output_probe/probe_orientation.npy \\
        -o probe_orientation.png --stride 5 --elev 25 --azim 45
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection


# ── Sphere helpers ────────────────────────────────────────────────────────────

def _draw_sphere(ax, n_lat: int = 18, n_lon: int = 36, alpha: float = 0.07):
    """Transparent unit sphere with faint latitude/longitude grid lines."""
    u = np.linspace(0, 2 * np.pi, n_lon + 1)
    v = np.linspace(0, np.pi, n_lat + 1)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=alpha, color="lightsteelblue", linewidth=0, antialiased=True)

    # Latitude rings (parallels)
    for phi in np.linspace(0, np.pi, 7)[1:-1]:
        xi = np.cos(u) * np.sin(phi)
        yi = np.sin(u) * np.sin(phi)
        zi = np.full_like(u, np.cos(phi))
        ax.plot(xi, yi, zi, color="gray", lw=0.4, alpha=0.35)

    # Longitude arcs (meridians)
    v_arc = np.linspace(0, np.pi, 60)
    for theta in np.linspace(0, 2 * np.pi, 13)[:-1]:
        xi = np.cos(theta) * np.sin(v_arc)
        yi = np.sin(theta) * np.sin(v_arc)
        zi = np.cos(v_arc)
        ax.plot(xi, yi, zi, color="gray", lw=0.4, alpha=0.35)


def _draw_axes(ax, length: float = 1.15):
    """Faint Cartesian reference arrows."""
    for vec, lbl in zip(np.eye(3), ("x", "y", "z")):
        ax.quiver(0, 0, 0, *vec * length, color="dimgray", lw=0.8,
                  arrow_length_ratio=0.12, alpha=0.5)
        ax.text(*(vec * (length + 0.07)), lbl, ha="center", va="center",
                fontsize=9, color="dimgray")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Visualize probe orientation trajectory on a unit sphere.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("orientation_file",
                   help="Path to probe_orientation.npy (shape: n_frames × 3).")
    p.add_argument("--output", "-o", default=None,
                   help="Output image file (.png/.pdf).  Interactive window if omitted.")
    p.add_argument("--stride", type=int, default=1,
                   help="Plot every Nth frame (useful for very long trajectories).")
    p.add_argument("--lw", type=float, default=1.4,
                   help="Path line width.")
    p.add_argument("--cmap", default="viridis",
                   help="Matplotlib colormap for the time axis.")
    p.add_argument("--elev", type=float, default=20.0,
                   help="Elevation angle of the 3-D view (degrees).")
    p.add_argument("--azim", type=float, default=-60.0,
                   help="Azimuth angle of the 3-D view (degrees).")
    p.add_argument("--title", default="Probe orientation trajectory",
                   help="Figure title.")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output DPI (for saved images).")
    args = p.parse_args(argv)

    # ── Load data ─────────────────────────────────────────────────────────
    fpath = Path(args.orientation_file)
    if not fpath.exists():
        print(f"ERROR: file not found: {fpath}", file=sys.stderr)
        sys.exit(1)

    raw = np.load(fpath)                              # (n_frames, 3)
    if raw.ndim != 2 or raw.shape[1] != 3:
        print(f"ERROR: expected shape (n_frames, 3), got {raw.shape}", file=sys.stderr)
        sys.exit(1)

    # Renormalize to unit sphere (safety; should already be unit vectors)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    ori = raw / norms

    # Apply stride
    idx = np.arange(0, len(ori), args.stride)
    ori = ori[idx]
    n = len(ori)
    t_norm = np.linspace(0.0, 1.0, n)               # 0 = start, 1 = end
    print(f"Number of frames to plot: {n} (every {args.stride}th frame)")
    
    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    _draw_sphere(ax)
    _draw_axes(ax)

    # ── Path colored by time ──────────────────────────────────────────────
    cmap = cm.get_cmap(args.cmap)
    for i in range(n - 1):
        c = cmap(t_norm[i])
        ax.plot(ori[i : i + 2, 0],
                ori[i : i + 2, 1],
                ori[i : i + 2, 2],
                color=c, lw=args.lw, solid_capstyle="round")

    # Start and end markers
    ax.scatter(*ori[0],  s=80, color="limegreen", zorder=6, depthshade=False,
               label=f"Start  ({ori[0,0]:.2f}, {ori[0,1]:.2f}, {ori[0,2]:.2f})")
    ax.scatter(*ori[-1], s=80, color="crimson",   zorder=6, depthshade=False,
               label=f"End    ({ori[-1,0]:.2f}, {ori[-1,1]:.2f}, {ori[-1,2]:.2f})")

    # Vectors from origin to start/end
    ax.quiver(0, 0, 0, *ori[0],  color="limegreen", lw=1.5,
              arrow_length_ratio=0.14, alpha=0.75)
    ax.quiver(0, 0, 0, *ori[-1], color="crimson",   lw=1.5,
              arrow_length_ratio=0.14, alpha=0.75)

    # ── Colorbar ──────────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.06)
    cbar.set_label("Normalised time", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["t = 0", "t = T/2", "t = T"])

    # ── Labels and limits ─────────────────────────────────────────────────
    lim = 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("y", labelpad=2)
    ax.set_zlabel("z", labelpad=2)
    ax.set_title(args.title, pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    # Equal aspect ratio via set_box_aspect (matplotlib ≥ 3.3)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    plt.tight_layout()

    if args.output:
        out = Path(args.output)
        plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

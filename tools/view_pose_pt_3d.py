#!/usr/bin/env python3

import argparse
import os

import matplotlib

# Headless-safe default; script can still display when --show is set.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# PoseScript / SMPL22 skeleton edges
EDGES = [
    (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]


def apply_variant(coords: np.ndarray, variant: str) -> np.ndarray:
    x = coords.copy()
    if variant == "identity":
        return x
    if variant == "flip_yz":
        x[:, 1] *= -1.0
        x[:, 2] *= -1.0
        return x
    if variant == "swap_xy":
        x = x[:, [1, 0, 2]]
        return x
    if variant == "swap_xz":
        x = x[:, [2, 1, 0]]
        return x
    if variant == "swap_yz":
        x = x[:, [0, 2, 1]]
        return x
    raise ValueError(f"Unsupported variant: {variant}")


def set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius < 1e-6:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_pose(ax, coords: np.ndarray, title: str, show_ids: bool) -> None:
    for a, b in EDGES:
        ax.plot(
            [coords[a, 0], coords[b, 0]],
            [coords[a, 1], coords[b, 1]],
            [coords[a, 2], coords[b, 2]],
            linewidth=2.0,
        )
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=20)

    if show_ids:
        for i, (x, y, z) in enumerate(coords):
            ax.text(x, y, z, str(i), fontsize=7)

    set_equal_3d(ax, coords)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)


def _load_pose_22x3(input_pt: str, pose_index: int) -> np.ndarray:
    t = torch.load(input_pt, map_location="cpu")
    if t.ndim == 3:
        if pose_index < 0 or pose_index >= t.shape[0]:
            raise IndexError(f"pose_index {pose_index} out of range for batch {t.shape[0]}")
        coords = t[pose_index].detach().cpu().numpy()
    elif t.ndim == 2:
        coords = t.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")
    if coords.shape != (22, 3):
        raise ValueError(f"Expected pose shape (22, 3), got {coords.shape}")
    return coords


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize PoseScript .pt coordinates (shape Nx22x3 or 22x3)."
    )
    p.add_argument("--input_pt", required=True, type=str, help="Path to .pt tensor")
    p.add_argument(
        "--pose_index",
        default=0,
        type=int,
        help="Pose index for batched tensors (default: 0)",
    )
    p.add_argument(
        "--variant",
        default="identity",
        choices=["identity", "flip_yz", "swap_xy", "swap_xz", "swap_yz"],
        help="Axis transform variant to visualize",
    )
    p.add_argument(
        "--compare_variants",
        action="store_true",
        help="Render a 2x3 comparison grid over common axis variants",
    )
    p.add_argument(
        "--input_raw_pt",
        default="",
        type=str,
        help="Optional .pt tensor in raw SAM3D XYZ frame for 6th comparison slot",
    )
    p.add_argument(
        "--show_ids",
        action="store_true",
        help="Overlay joint index labels",
    )
    p.add_argument(
        "--output_png",
        default="",
        type=str,
        help="Optional output image path (default: <input_pt>.png)",
    )
    p.add_argument("--show", action="store_true", help="Open a local window if available")
    args = p.parse_args()

    coords = _load_pose_22x3(args.input_pt, args.pose_index)
    raw_coords = None
    if args.input_raw_pt:
        raw_coords = _load_pose_22x3(args.input_raw_pt, args.pose_index)

    if args.compare_variants:
        variants = ["identity", "flip_yz", "swap_xy", "swap_xz", "swap_yz"]
        fig = plt.figure(figsize=(15, 9))
        for i, variant in enumerate(variants, start=1):
            ax = fig.add_subplot(2, 3, i, projection="3d")
            c = apply_variant(coords, variant)
            plot_pose(ax, c, variant, args.show_ids)
        if raw_coords is not None:
            ax = fig.add_subplot(2, 3, 6, projection="3d")
            plot_pose(ax, raw_coords, "sam3d_xyz_raw", args.show_ids)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        c = apply_variant(coords, args.variant)
        plot_pose(ax, c, args.variant, args.show_ids)
        fig.tight_layout()

    out = args.output_png or f"{os.path.splitext(args.input_pt)[0]}.png"
    fig.savefig(out, dpi=160)
    print(f"saved: {out}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

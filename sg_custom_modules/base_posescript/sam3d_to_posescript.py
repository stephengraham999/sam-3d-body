"""
sam3d_to_posescript.py
======================
Bridge module: converts a SAM-3D-Body output .npz file into the
coordinate tensor format expected by PoseScript's captioning pipeline.

Background
----------
SAM-3D-Body outputs a (70, 3) joint array in camera / OpenCV coordinates
(Y points down, Z points forward toward the camera). The 70 rows follow
the MHR70 keypoint order (nose/eyes/ears/limbs/hands/markers), not the
SMPL22 order PoseScript expects.

This module converts MHR70 keypoints into PoseScript's 22-joint SMPL-style
layout by combining direct row mappings with synthetic joints:
- pelvis from left/right hips
- spine1/2 from pelvis -> spine3 interpolation
- spine3 from left/right shoulder midpoint
- collar joints from spine3 and shoulders
- head from nose/neck midpoint
- foot base joints from heel/big-toe midpoints

PoseScript's ``compute_coords.py`` applies a -90 degree rotation around the
X axis to every pose so that Y points up and the figure faces the viewer.
This module replicates that exact transformation.

Usage (as a library)
--------------------
    from sg_custom_modules.base_posescript.sam3d_to_posescript import (
        convert_npz_to_posescript,
    )

    tensor = convert_npz_to_posescript(
        input_npz="output/my_image.npz",
        output_pt="/tmp/my_pose.pt",
    )
    # tensor.shape → (1, 22, 3)  float32, root-centred, Y-up

Usage (as a CLI tool)
---------------------
    python sg_custom_modules/base_posescript/sam3d_to_posescript.py \\
        --input_npz output/my_image.npz \\
        --output_pt /tmp/my_pose.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of SMPL base joints PoseScript operates on.
N_POSESCRIPT_JOINTS: int = 22
N_MHR70_JOINTS: int = 70
AXIS_VARIANTS = ("flip_yz", "identity", "swap_xy", "swap_xz", "swap_yz")

#: Rotation matrix: -90 degrees around the X axis.
#:
#:   OpenCV / camera frame:  X right, Y down,  Z toward viewer
#:   PoseScript / body frame: X right, Y up,    Z away from viewer
#:
#: rotX(-90°):
#:   [1,  0,  0]
#:   [0,  0,  1]   ← new Y = old Z
#:   [0, -1,  0]   ← new Z = -old Y
_THETA_DEG: float = -90.0
_THETA_RAD: float = math.pi * _THETA_DEG / 180.0
ROT_X_NEG90: torch.Tensor = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(_THETA_RAD), -math.sin(_THETA_RAD)],
        [0.0, math.sin(_THETA_RAD), math.cos(_THETA_RAD)],
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def convert_npz_to_posescript(
    input_npz: Union[str, Path],
    output_pt: Union[str, Path],
    person_id: int = 0,
    axis_variant: str = "flip_yz",
) -> torch.Tensor:
    """Convert a SAM-3D-Body NPZ file to a PoseScript-compatible .pt tensor.

    Parameters
    ----------
    input_npz:
        Path to the SAM-3D-Body ``.npz`` output file produced by
        ``fast_core.run_fast_infer()``.
    output_pt:
        Destination path for the saved PyTorch tensor (e.g. ``pose.pt``).
        Parent directories are created automatically.
    person_id:
        Index of the person to extract (default ``0`` → ``person_000_``).
        For single-person inference this is always 0.
    axis_variant:
        Axis transform applied after root-centering. Choices:
        ``flip_yz`` (default), ``identity``, ``swap_xy``, ``swap_xz``,
        ``swap_yz``.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 22, 3)``, dtype ``float32``.
        Root-centred (pelvis at origin), Y-up coordinate frame.

    Raises
    ------
    KeyError
        If the expected ``person_{person_id:03d}_pred_keypoints_3d`` key is
        missing from the NPZ.
    ValueError
        If the extracted joint array does not have at least 70 joints.
    """
    input_npz = Path(input_npz)
    output_pt = Path(output_pt)

    # ------------------------------------------------------------------
    # 1. Load the NPZ and extract the 3-D keypoints for the chosen person
    # ------------------------------------------------------------------
    data = np.load(str(input_npz), allow_pickle=False)
    key = f"person_{person_id:03d}_pred_keypoints_3d"

    if key not in data:
        available = [k for k in data.files if "pred_keypoints_3d" in k]
        raise KeyError(
            f"Key '{key}' not found in '{input_npz}'. "
            f"Available keypoints keys: {available}"
        )

    joints_np: np.ndarray = data[key]  # (70, 3) float32

    if joints_np.shape[0] < N_MHR70_JOINTS:
        raise ValueError(
            f"Expected at least {N_MHR70_JOINTS} joints, "
            f"but got shape {joints_np.shape}."
        )

    # ------------------------------------------------------------------
    # 2. MHR70 -> PoseScript 22 mapping
    # ------------------------------------------------------------------
    p = torch.from_numpy(joints_np.astype(np.float32))  # (70, 3)
    joints = torch.zeros((N_POSESCRIPT_JOINTS, 3), dtype=torch.float32)

    # Direct mappings
    joints[1] = p[9]    # left_hip
    joints[2] = p[10]   # right_hip
    joints[4] = p[11]   # left_knee
    joints[5] = p[12]   # right_knee
    joints[7] = p[13]   # left_ankle
    joints[8] = p[14]   # right_ankle
    joints[12] = p[69]  # neck
    joints[16] = p[5]   # left_shoulder
    joints[17] = p[6]   # right_shoulder
    joints[18] = p[7]   # left_elbow
    joints[19] = p[8]   # right_elbow
    joints[20] = p[62]  # left_wrist
    joints[21] = p[41]  # right_wrist

    # Synthetic torso/head/foot joints
    pelvis = (p[9] + p[10]) / 2.0
    spine3 = (p[5] + p[6]) / 2.0
    joints[0] = pelvis
    joints[9] = spine3
    joints[3] = pelvis + (spine3 - pelvis) * (1.0 / 3.0)   # spine1
    joints[6] = pelvis + (spine3 - pelvis) * (2.0 / 3.0)   # spine2
    joints[13] = (spine3 + p[5]) / 2.0                      # left_collar
    joints[14] = (spine3 + p[6]) / 2.0                      # right_collar
    joints[15] = (p[0] + p[69]) / 2.0                       # head
    joints[10] = (p[17] + p[15]) / 2.0                      # left_foot base
    joints[11] = (p[20] + p[18]) / 2.0                      # right_foot base

    # ------------------------------------------------------------------
    # 3. Root-centre: translate so that the pelvis (joint 0) is at origin
    # ------------------------------------------------------------------
    pelvis: torch.Tensor = joints[0].clone()  # (3,)
    joints = joints - pelvis  # (22, 3)

    # ------------------------------------------------------------------
    # 4. Optional axis transform for alignment experiments.
    # ------------------------------------------------------------------
    axis_variant = str(axis_variant).strip().lower()
    if axis_variant not in AXIS_VARIANTS:
        raise ValueError(
            f"Invalid axis_variant='{axis_variant}'. "
            f"Expected one of: {AXIS_VARIANTS}"
        )

    if axis_variant == "flip_yz":
        joints[:, 1] = -joints[:, 1]
        joints[:, 2] = -joints[:, 2]
    elif axis_variant == "swap_xy":
        joints = joints[:, [1, 0, 2]]
    elif axis_variant == "swap_xz":
        joints = joints[:, [2, 1, 0]]
    elif axis_variant == "swap_yz":
        joints = joints[:, [0, 2, 1]]
    # identity: no-op

    # ------------------------------------------------------------------
    # 5. Add a batch dimension → (1, 22, 3) matching PoseScript's format
    # ------------------------------------------------------------------
    tensor: torch.Tensor = joints.unsqueeze(0)  # (1, 22, 3)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, str(output_pt))

    print(
        f"Saved PoseScript tensor {list(tensor.shape)} → {output_pt}"
    )
    return tensor


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a SAM-3D-Body NPZ file into a PoseScript-compatible "
            "PyTorch .pt tensor (shape 1×22×3, root-centred, Y-up)."
        )
    )
    p.add_argument(
        "--input_npz",
        required=True,
        metavar="PATH",
        help="Path to the SAM-3D-Body output .npz file.",
    )
    p.add_argument(
        "--output_pt",
        required=True,
        metavar="PATH",
        help="Destination path for the saved .pt tensor.",
    )
    p.add_argument(
        "--person_id",
        type=int,
        default=0,
        metavar="INT",
        help="Person index to extract (default: 0).",
    )
    p.add_argument(
        "--axis_variant",
        default="flip_yz",
        choices=list(AXIS_VARIANTS),
        help="Axis transform variant (default: flip_yz).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    tensor = convert_npz_to_posescript(
        input_npz=args.input_npz,
        output_pt=args.output_pt,
        person_id=args.person_id,
        axis_variant=args.axis_variant,
    )
    # Quick sanity print
    print(f"Tensor shape : {list(tensor.shape)}")
    print(f"Pelvis (root): {tensor[0, 0].tolist()}  (should be [0, 0, 0])")
    print(f"Joint min/max: {tensor.min().item():.4f} / {tensor.max().item():.4f}")


if __name__ == "__main__":
    main()

"""
sam3d_to_posescript.py
======================
Bridge module: converts a SAM-3D-Body output .npz file into the
coordinate tensor format expected by PoseScript's captioning pipeline.

Background
----------
SAM-3D-Body outputs a (70, 3) joint array in camera / OpenCV coordinates
(Y points down, Z points forward toward the camera).  The first 22 joints
follow the standard SMPL kinematic tree and map 1-to-1 to the 22 joints
PoseScript's posecode engine expects.

PoseScript's ``compute_coords.py`` applies a -90 degree rotation around the
X axis to every pose so that Y points up and the figure faces the viewer.
This module replicates that exact transformation.

Joint index mapping (first 22 only):
    0  Pelvis (root)        13  Left Collar (clavicle)
    1  Left Hip             14  Right Collar (clavicle)
    2  Right Hip            15  Head (base)
    3  Spine 1 (lower)      16  Left Shoulder
    4  Left Knee            17  Right Shoulder
    5  Right Knee           18  Left Elbow
    6  Spine 2 (middle)     19  Right Elbow
    7  Left Ankle           20  Left Wrist
    8  Right Ankle          21  Right Wrist
    9  Spine 3 (upper)
    10 Left Foot (toe base)
    11 Right Foot (toe base)
    12 Neck

Joints 22-69 (generic hands, MHR dense markers) are discarded.

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
        If the extracted joint array does not have at least 22 joints.
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

    if joints_np.shape[0] < N_POSESCRIPT_JOINTS:
        raise ValueError(
            f"Expected at least {N_POSESCRIPT_JOINTS} joints, "
            f"but got shape {joints_np.shape}."
        )

    # ------------------------------------------------------------------
    # 2. Slice the first 22 joints (discard hands + MHR dense markers)
    # ------------------------------------------------------------------
    joints: torch.Tensor = torch.from_numpy(
        joints_np[:N_POSESCRIPT_JOINTS].astype(np.float32)
    )  # (22, 3)

    # ------------------------------------------------------------------
    # 3. Root-centre: translate so that the pelvis (joint 0) is at origin
    # ------------------------------------------------------------------
    pelvis: torch.Tensor = joints[0].clone()  # (3,)
    joints = joints - pelvis  # (22, 3)

    # ------------------------------------------------------------------
    # 4. Coordinate-axis flip: camera Y-down → body Y-up
    #    Replicates compute_coords.py:  transf(rotX, -90, j)
    #    which does:  rotX(-90°) @ j.T  →  j_rotated
    # ------------------------------------------------------------------
    # ROT_X_NEG90 is (3, 3); joints is (22, 3)
    # Result = (joints @ ROT_X_NEG90.T) = (22, 3)
    joints = joints @ ROT_X_NEG90.t()  # (22, 3)

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
    return p


def main() -> None:
    args = _build_parser().parse_args()
    tensor = convert_npz_to_posescript(
        input_npz=args.input_npz,
        output_pt=args.output_pt,
        person_id=args.person_id,
    )
    # Quick sanity print
    print(f"Tensor shape : {list(tensor.shape)}")
    print(f"Pelvis (root): {tensor[0, 0].tolist()}  (should be [0, 0, 0])")
    print(f"Joint min/max: {tensor.min().item():.4f} / {tensor.max().item():.4f}")


if __name__ == "__main__":
    main()

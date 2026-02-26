#!/usr/bin/env python3
"""Export grouped PoseScript predicates + numeric pose metrics for 3 NPZ files."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sg_custom_modules.base_posescript.sam3d_to_posescript import convert_npz_to_posescript

POSESCRIPT_VENV_PYTHON = _PROJECT_ROOT / "base_posescript" / ".venv" / "bin" / "python"
POSESCRIPT_RUNNER = _PROJECT_ROOT / "sg_custom_modules" / "base_posescript" / "posescript_caption_single.py"

DEFAULT_NPZ_DIR = _PROJECT_ROOT / "output" / "caption_demo" / "npz"
DEFAULT_JSON = _PROJECT_ROOT / "output" / "caption_demo" / "reports" / "pose_predicates_3files.json"
DEFAULT_MD = _PROJECT_ROOT / "output" / "caption_demo" / "reports" / "pose_predicates_3files.md"

ANGLE_NAMES = {"completely bent", "bent more", "right angle", "bent less", "slightly bent", "straight"}
DISTANCE_NAMES = {"close", "shoulder width", "spread", "wide"}
POS_X_NAMES = {"at_right", "x-aligned", "at_left"}
POS_Y_NAMES = {"below", "y-aligned", "above"}
POS_Z_NAMES = {"behind", "z-aligned", "front"}
VAXIS_NAMES = {"vertical", "horizontal"}
ONGROUND_NAMES = {"on_ground"}
BODY_INCLINE_NAMES = {
    "body incline backward",
    "body incline backward slightly",
    "body incline forward slightly",
    "body incline forward",
    "body twist left",
    "body twist left slightly",
    "body twist right slightly",
    "body twist right",
    "body lean right",
    "body lean right slightly",
    "body lean left slightly",
    "body lean left",
}
CONTACT_NAMES = {"touch", "on_ground"}

PS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.degrees(np.arccos(cosv)))


def _vector_angle_deg(v: np.ndarray, ref: np.ndarray) -> float:
    nv = np.linalg.norm(v)
    nr = np.linalg.norm(ref)
    if nv < 1e-8 or nr < 1e-8:
        return float("nan")
    cosv = float(np.dot(v, ref) / (nv * nr))
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.degrees(np.arccos(cosv)))


def _round(x: float, nd: int = 3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def _metrics_from_pose(coords: np.ndarray) -> Dict[str, object]:
    up = 1  # Y-up after swap_yz conversion

    knee_l = _angle_deg(coords[PS["left_hip"]], coords[PS["left_knee"]], coords[PS["left_ankle"]])
    knee_r = _angle_deg(coords[PS["right_hip"]], coords[PS["right_knee"]], coords[PS["right_ankle"]])
    elbow_l = _angle_deg(coords[PS["left_shoulder"]], coords[PS["left_elbow"]], coords[PS["left_wrist"]])
    elbow_r = _angle_deg(coords[PS["right_shoulder"]], coords[PS["right_elbow"]], coords[PS["right_wrist"]])
    hip_l = _angle_deg(coords[PS["pelvis"]], coords[PS["left_hip"]], coords[PS["left_knee"]])
    hip_r = _angle_deg(coords[PS["pelvis"]], coords[PS["right_hip"]], coords[PS["right_knee"]])

    pelvis_vs_knees = coords[PS["pelvis"], up] - (coords[PS["left_knee"], up] + coords[PS["right_knee"], up]) / 2.0
    wrist_l_vs_sh = coords[PS["left_wrist"], up] - coords[PS["left_shoulder"], up]
    wrist_r_vs_sh = coords[PS["right_wrist"], up] - coords[PS["right_shoulder"], up]
    foot_l_vs_pelvis = coords[PS["left_foot"], up] - coords[PS["pelvis"], up]
    foot_r_vs_pelvis = coords[PS["right_foot"], up] - coords[PS["pelvis"], up]

    torso_vec = coords[PS["neck"]] - coords[PS["pelvis"]]
    torso_to_vertical = _vector_angle_deg(torso_vec, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    # Ground estimate from lower extremities only.
    ground = float(min(coords[PS["left_foot"], up], coords[PS["right_foot"], up], coords[PS["left_knee"], up], coords[PS["right_knee"], up]))
    eps_knee = 0.08
    eps_foot = 0.05
    left_knee_ground = (coords[PS["left_knee"], up] - ground) <= eps_knee
    right_knee_ground = (coords[PS["right_knee"], up] - ground) <= eps_knee
    left_foot_ground = (coords[PS["left_foot"], up] - ground) <= eps_foot
    right_foot_ground = (coords[PS["right_foot"], up] - ground) <= eps_foot

    return {
        "angular_relations": {
            "left_knee_angle_deg": _round(knee_l),
            "right_knee_angle_deg": _round(knee_r),
            "left_elbow_angle_deg": _round(elbow_l),
            "right_elbow_angle_deg": _round(elbow_r),
            "left_hip_angle_deg": _round(hip_l),
            "right_hip_angle_deg": _round(hip_r),
        },
        "positional_relations": {
            "pelvis_height_relative_to_knees_m": _round(pelvis_vs_knees),
            "left_wrist_relative_to_left_shoulder_m": _round(wrist_l_vs_sh),
            "right_wrist_relative_to_right_shoulder_m": _round(wrist_r_vs_sh),
            "left_foot_relative_to_pelvis_m": _round(foot_l_vs_pelvis),
            "right_foot_relative_to_pelvis_m": _round(foot_r_vs_pelvis),
            "torso_angle_to_vertical_deg": _round(torso_to_vertical),
        },
        "contact_relations": {
            "ground_y_m": _round(ground),
            "left_knee_touching_ground": bool(left_knee_ground),
            "right_knee_touching_ground": bool(right_knee_ground),
            "left_foot_touching_ground": bool(left_foot_ground),
            "right_foot_touching_ground": bool(right_foot_ground),
            "eps_knee_m": eps_knee,
            "eps_foot_m": eps_foot,
        },
    }


def _predicate_string(pc: List[object], id_to_name: Dict[int, str]) -> str:
    s1, b1, int_id, s2, b2 = pc
    int_name = id_to_name.get(int(int_id), f"id_{int_id}")

    lhs = f"{s1}_{b1}" if s1 else str(b1)
    rhs = ""
    if b2:
        rhs = f" {s2}_{b2}" if s2 else f" {b2}"
    return f"{lhs} {int_name}{rhs}".strip()


def _group_predicates(aggregated_posecodes: List[List[object]], id_to_name: Dict[int, str]) -> Dict[str, List[str]]:
    groups = {
        "angular_relations": [],
        "positional_relations": [],
        "contact_relations": [],
        "distance_relations": [],
        "other_predicates": [],
    }

    for pc in aggregated_posecodes:
        int_name = id_to_name.get(int(pc[2]), "")
        pred = _predicate_string(pc, id_to_name)

        if int_name in ANGLE_NAMES:
            groups["angular_relations"].append(pred)
        elif int_name in (POS_X_NAMES | POS_Y_NAMES | POS_Z_NAMES | VAXIS_NAMES | BODY_INCLINE_NAMES):
            groups["positional_relations"].append(pred)
        elif int_name in CONTACT_NAMES:
            groups["contact_relations"].append(pred)
        elif int_name in DISTANCE_NAMES:
            groups["distance_relations"].append(pred)
        elif int_name in ONGROUND_NAMES:
            groups["contact_relations"].append(pred)
        else:
            groups["other_predicates"].append(pred)

    for k in groups:
        groups[k] = sorted(set(groups[k]))
    return groups


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export grouped pose predicates for three NPZ files.")
    p.add_argument("--npz_dir", default=str(DEFAULT_NPZ_DIR))
    p.add_argument("--max_files", type=int, default=3)
    p.add_argument("--axis_variant", default="swap_yz", choices=["flip_yz", "identity", "swap_xy", "swap_xz", "swap_yz"])
    p.add_argument("--output_json", default=str(DEFAULT_JSON))
    p.add_argument("--output_md", default=str(DEFAULT_MD))
    p.add_argument("--files", nargs="*", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    npz_dir = Path(args.npz_dir).resolve()
    out_json = Path(args.output_json).resolve()
    out_md = Path(args.output_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    if not POSESCRIPT_VENV_PYTHON.exists() or not POSESCRIPT_RUNNER.exists():
        print("ERROR: PoseScript runner or venv missing", file=sys.stderr)
        return 2

    if args.files:
        selected = [Path(f).resolve() for f in args.files]
    else:
        selected = sorted(npz_dir.glob("*.npz"))[: args.max_files]

    if len(selected) != 3:
        print(f"ERROR: expected exactly 3 files, got {len(selected)}", file=sys.stderr)
        return 2

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = out_json.parent / f"predicates_run_{run_ts}"
    pt_dir = run_root / "pt"
    interm_dir = run_root / "intermediates"
    pt_dir.mkdir(parents=True, exist_ok=True)
    interm_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "axis_variant": args.axis_variant,
        "npz_dir": str(npz_dir),
        "run_root": str(run_root),
        "files": [],
    }

    md: List[str] = []
    md.append("# Pose Predicate Export (3 files)")
    md.append("")
    md.append(f"- Timestamp: `{output['timestamp']}`")
    md.append(f"- Axis variant: `{args.axis_variant}`")
    md.append(f"- Run root: `{run_root}`")
    md.append("")

    for npz_path in selected:
        stem = npz_path.stem
        pt_path = pt_dir / f"{stem}.pt"
        per_dir = interm_dir / stem / "F"
        per_dir.mkdir(parents=True, exist_ok=True)

        file_entry = {
            "npz": str(npz_path),
            "pt": str(pt_path),
            "status": "ok",
            "error": "",
            "metrics": {},
            "predicates": {},
        }

        try:
            convert_npz_to_posescript(npz_path, pt_path, axis_variant=args.axis_variant)
            cmd = [
                str(POSESCRIPT_VENV_PYTHON),
                str(POSESCRIPT_RUNNER),
                "--input_pt",
                str(pt_path),
                "--save_dir",
                str(per_dir),
                "--simplified_captions",
                "--no-random_skip",
                "--no-apply_transrel_ripple_effect",
                "--no-apply_stat_ripple_effect",
                "--no-add_babel_info",
                "--no-add_dancing_info",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout).strip())

            coords_t = torch.load(pt_path, map_location="cpu")
            coords = coords_t[0].detach().cpu().numpy()
            metrics = _metrics_from_pose(coords)

            intptt_path = per_dir / "posecodes_intptt_eligibility.pt"
            agg_path = per_dir / "posecodes_aggregated.pt"
            _, _, name2id = torch.load(intptt_path, map_location="cpu")
            id2name = {int(v): k for k, v in name2id.items()}
            agg = torch.load(agg_path, map_location="cpu")
            grouped = _group_predicates(agg[0], id2name)

            file_entry["metrics"] = metrics
            file_entry["predicates"] = grouped

            md.append(f"## `{stem}`")
            md.append("")
            md.append(f"- Source NPZ: `{npz_path}`")
            md.append(f"- PoseScript PT: `{pt_path}`")
            md.append("")
            md.append("### Angular Relations")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|---|---:|")
            for k, v in metrics["angular_relations"].items():
                md.append(f"| `{_md_escape(k)}` | `{v}` |")
            md.append("")
            md.append("### Positional Relations")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|---|---:|")
            for k, v in metrics["positional_relations"].items():
                md.append(f"| `{_md_escape(k)}` | `{v}` |")
            md.append("")
            md.append("### Contact Relations")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|---|---:|")
            for k, v in metrics["contact_relations"].items():
                md.append(f"| `{_md_escape(k)}` | `{v}` |")
            md.append("")

            for group_name in ["angular_relations", "positional_relations", "distance_relations", "contact_relations", "other_predicates"]:
                title = group_name.replace("_", " ").title()
                md.append(f"### Atomic Predicates - {title}")
                md.append("")
                preds = grouped[group_name]
                if preds:
                    for p in preds:
                        md.append(f"- `{_md_escape(p)}`")
                else:
                    md.append("- `(none)`")
                md.append("")

        except Exception as e:
            file_entry["status"] = "failed"
            file_entry["error"] = f"{type(e).__name__}: {e}"
            md.append(f"## `{stem}`")
            md.append("")
            md.append(f"- Status: `failed`")
            md.append(f"- Error: `{_md_escape(file_entry['error'])}`")
            md.append("")

        output["files"].append(file_entry)

    output["smoke_checks"] = {
        "files_selected": len(selected),
        "expected_files": 3,
        "all_ok": all(f["status"] == "ok" for f in output["files"]),
    }

    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote json: {out_json}")
    print(f"wrote md:   {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
pose_caption_demo.py
====================
Test GUI: JPEG → SAM 3D Body → PoseScript caption.

Pipeline:
  1. User uploads a JPEG/PNG.
  2. run_fast_infer()           →  output/caption_demo/npz/<stem>.npz
  3. convert_npz_to_posescript() →  output/caption_demo/pt/<stem>.pt   (1×22×3, Y-up)
  4. subprocess (base_posescript/.venv) posescript_caption_single.py    →  caption text
  5. Display: original photo | 2D skeleton overlay | caption text box

Usage:
    python tools/pose_caption_demo.py \\
        --checkpoint_path checkpoints/sam-3d-body-dinov3/model.ckpt \\
        --mhr_path       checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import gradio as gr

# ---------------------------------------------------------------------------
# Project-root relative paths (this script lives in tools/)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

POSESCRIPT_VENV_PYTHON = os.path.join(
    _PROJECT_ROOT, "base_posescript", ".venv", "bin", "python"
)
POSESCRIPT_RUNNER = os.path.join(
    _PROJECT_ROOT,
    "sg_custom_modules", "base_posescript", "posescript_caption_single.py",
)

# Add project root to sys.path so sam_3d_body and sg_custom_modules are importable
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sam_3d_body import run_fast_infer                                  # noqa: E402
from sg_custom_modules.base_posescript.sam3d_to_posescript import (     # noqa: E402
    convert_npz_to_posescript,
)


# ---------------------------------------------------------------------------
# 2-D skeleton drawing (subset from gui_app.py)
# ---------------------------------------------------------------------------
_BODY_LINKS = [
    (13, 11, (0, 200, 0)),   (11, 9,  (0, 200, 0)),
    (14, 12, (0, 100, 255)), (12, 10, (0, 100, 255)),
    (9, 10,  (51, 153, 255)),(5, 9,   (51, 153, 255)),
    (6, 10,  (51, 153, 255)),(5, 6,   (51, 153, 255)),
    (5, 7,   (0, 200, 0)),   (7, 62,  (0, 200, 0)),
    (6, 8,   (0, 100, 255)), (8, 41,  (0, 100, 255)),
    (0, 1,   (51, 153, 255)),(0, 2,   (51, 153, 255)),
    (1, 3,   (51, 153, 255)),(2, 4,   (51, 153, 255)),
    (3, 5,   (51, 153, 255)),(4, 6,   (51, 153, 255)),
]
_BODY_KPT = list(range(21)) + [41, 62]


def _draw_skeleton(img_path: str, npz_path: str) -> np.ndarray | None:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    try:
        z = np.load(npz_path, allow_pickle=False)
        num_people = int(z["num_people"])
    except Exception:
        return img
    h, w = img.shape[:2]
    lw = max(2, w // 500)
    dr = max(4, w // 250)
    for pid in range(num_people):
        kpts = z.get(f"person_{pid:03d}_pred_keypoints_2d")
        if kpts is None:
            continue
        for ja, jb, color in _BODY_LINKS:
            if ja >= len(kpts) or jb >= len(kpts):
                continue
            x1, y1 = int(round(kpts[ja, 0])), int(round(kpts[ja, 1]))
            x2, y2 = int(round(kpts[jb, 0])), int(round(kpts[jb, 1]))
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(img, (x1, y1), (x2, y2), color[::-1], lw, cv2.LINE_AA)
        for idx in _BODY_KPT:
            if idx >= len(kpts):
                continue
            x, y = int(round(kpts[idx, 0])), int(round(kpts[idx, 1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), dr + 1, (0, 0, 0), -1)
                cv2.circle(img, (x, y), dr, (255, 255, 255), -1)
    return img


def _load_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class CaptionPipeline:
    def __init__(self, checkpoint_path: str, mhr_path: str, output_dir: str):
        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path
        self.npz_dir = os.path.join(output_dir, "npz")
        self.pt_dir  = os.path.join(output_dir, "pt")
        os.makedirs(self.npz_dir, exist_ok=True)
        os.makedirs(self.pt_dir,  exist_ok=True)

    def run(self, file_obj) -> tuple:
        """Returns (orig_img, skel_img, caption_text, status_str)."""
        empty = (None, None, "", "No file selected.")
        if file_obj is None:
            return empty

        img_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        npz_path = os.path.join(self.npz_dir, f"{stem}.npz")
        pt_path  = os.path.join(self.pt_dir,  f"{stem}.pt")
        t0       = time.perf_counter()

        # ---- Stage 1: SAM 3D Body inference --------------------------------
        try:
            run_fast_infer(
                input_path=img_path,
                output_dir=self.npz_dir,
                checkpoint_path=self.checkpoint_path,
                mhr_path=self.mhr_path,
                max_side=1280,
                single_person=True,
                map_2d_to_original=True,
                save_compressed=False,
                inference_type="body",
            )
        except Exception as e:
            return None, None, "", f"Stage 1 SAM 3D Body failed: {e}"

        infer_ms = (time.perf_counter() - t0) * 1000

        if not os.path.exists(npz_path):
            return None, None, "", f"Stage 1: NPZ not produced at {npz_path}"

        # ---- Load visual outputs -------------------------------------------
        orig_img = _load_rgb(img_path)
        skel_img = _draw_skeleton(img_path, npz_path)

        # Check at least one person was detected
        try:
            import numpy as _np
            z = _np.load(npz_path, allow_pickle=False)
            if int(z["num_people"]) == 0:
                return orig_img, skel_img, "", "No person detected in image."
        except Exception:
            pass

        # ---- Stage 2: NPZ → .pt tensor -------------------------------------
        t2 = time.perf_counter()
        try:
            convert_npz_to_posescript(npz_path, pt_path)
        except Exception as e:
            return orig_img, skel_img, "", f"Stage 2 conversion failed: {e}"
        conv_ms = (time.perf_counter() - t2) * 1000

        # ---- Stage 3: PoseScript subprocess --------------------------------
        t3 = time.perf_counter()

        if not os.path.exists(POSESCRIPT_VENV_PYTHON):
            return orig_img, skel_img, "", (
                f"PoseScript venv not found at {POSESCRIPT_VENV_PYTHON}. "
                "Run base_posescript setup first."
            )

        try:
            proc = subprocess.run(
                [
                    POSESCRIPT_VENV_PYTHON,
                    POSESCRIPT_RUNNER,
                    "--input_pt", os.path.abspath(pt_path),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return orig_img, skel_img, "", "Stage 3: PoseScript timed out (>120s)"
        except Exception as e:
            return orig_img, skel_img, "", f"Stage 3 subprocess error: {e}"

        caption_ms = (time.perf_counter() - t3) * 1000

        if proc.returncode != 0:
            err = proc.stderr.strip()
            return orig_img, skel_img, "", (
                f"Stage 3 PoseScript failed (rc={proc.returncode}):\n{err}"
            )

        caption = proc.stdout.strip()
        if not caption:
            stderr = proc.stderr.strip()
            return orig_img, skel_img, "", (
                f"Stage 3: empty caption.\nstderr={stderr}"
            )

        # ---- Build status --------------------------------------------------
        ts = datetime.now().strftime("%H:%M:%S")
        status = (
            f"[{ts}]  infer={infer_ms:.0f}ms  "
            f"conv={conv_ms:.0f}ms  "
            f"caption={caption_ms:.0f}ms  "
            f"npz={npz_path}"
        )

        return orig_img, skel_img, caption, status


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(checkpoint_path: str, mhr_path: str, output_dir: str) -> gr.Blocks:
    pipeline = CaptionPipeline(checkpoint_path, mhr_path, output_dir)

    with gr.Blocks(title="Pose Caption Demo") as demo:
        gr.Markdown(
            "## Pose Caption Demo\n"
            "Upload a JPEG/PNG → SAM 3D Body detects the pose → "
            "PoseScript generates a natural-language description."
        )

        with gr.Row():
            # Left: controls
            with gr.Column(scale=1, min_width=260):
                file_in = gr.File(
                    label="Input image (JPEG / PNG)",
                    file_types=["image"],
                    file_count="single",
                )
                run_btn = gr.Button("▶  Generate caption", variant="primary")
                status  = gr.Textbox(label="Status / timing", lines=3, interactive=False)

            # Right: results
            with gr.Column(scale=3):
                with gr.Row():
                    orig_img = gr.Image(label="Original photo", type="numpy",
                                        show_download_button=False)
                    skel_img = gr.Image(label="2D skeleton", type="numpy",
                                        show_download_button=False)
                caption_box = gr.Textbox(
                    label="PoseScript caption",
                    lines=6,
                    placeholder="Caption will appear here…",
                    interactive=False,
                )

        run_btn.click(
            fn=pipeline.run,
            inputs=[file_in],
            outputs=[orig_img, skel_img, caption_box, status],
            queue=False,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pose Caption Demo (Gradio)")
    p.add_argument(
        "--checkpoint_path",
        default=os.path.join(_PROJECT_ROOT,
                             "checkpoints/sam-3d-body-dinov3/model.ckpt"),
    )
    p.add_argument(
        "--mhr_path",
        default=os.path.join(_PROJECT_ROOT,
                             "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"),
    )
    p.add_argument("--output_dir",
                   default=os.path.join(_PROJECT_ROOT, "output/caption_demo"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=7863, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    app = build_app(args.checkpoint_path, args.mhr_path, args.output_dir)
    app.launch(server_name=args.host, server_port=args.port)

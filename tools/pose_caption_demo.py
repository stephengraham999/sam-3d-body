"""
pose_caption_demo.py
====================
Test GUI: JPEG → SAM 3D Body → PoseScript caption.

Pipeline:
  1. User uploads a JPEG/PNG.
  2. run_fast_infer()              →  output/caption_demo/npz/<stem>.npz
  3. convert_npz_to_posescript()   →  output/caption_demo/pt/<stem>.pt  (1×22×3, Y-up)
  4. subprocess (base_posescript/.venv) posescript_caption_single.py     →  caption text
  5. (optional) render_npz subprocess                                    →  mesh overlay
  6. Display 2×2 grid:
       [Top-left]     Original photo
       [Top-right]    Caption text box
       [Bottom-left]  Mesh overlay (if render_mesh enabled)
       [Bottom-right] 2-D skeleton overlay

Usage:
    /usr/bin/python3 tools/pose_caption_demo.py \\
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
RENDER_SCRIPT = os.path.join(_HERE, "render_npz.py")

# Add project root to sys.path so sam_3d_body and sg_custom_modules are importable
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sam_3d_body import run_fast_infer                                  # noqa: E402
from sg_custom_modules.base_posescript.sam3d_to_posescript import (     # noqa: E402
    convert_npz_to_posescript,
)


# ---------------------------------------------------------------------------
# 2-D skeleton drawing
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
        self.npz_dir  = os.path.join(output_dir, "npz")
        self.pt_dir   = os.path.join(output_dir, "pt")
        self.mesh_dir = os.path.join(output_dir, "render")
        os.makedirs(self.npz_dir,  exist_ok=True)
        os.makedirs(self.pt_dir,   exist_ok=True)
        os.makedirs(self.mesh_dir, exist_ok=True)

    def run(
        self,
        file_obj,
        max_side: int,
        inference_type: str,
        single_person: bool,
        save_compressed: bool,
        render_mesh: bool,
        cache_dir: str,
    ) -> tuple:
        """
        Returns (orig_img, caption_text, mesh_img, skel_img, status_str).
        Matches outputs: [orig_img, caption_box, mesh_img, skel_img, status].
        """
        empty = (None, "", None, None, "No file selected.")
        if file_obj is None:
            return empty

        img_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        npz_path = os.path.join(self.npz_dir, f"{stem}.npz")
        pt_path  = os.path.join(self.pt_dir,  f"{stem}.pt")
        t0       = time.perf_counter()

        infer_type = str(inference_type).strip().lower() if inference_type else "body"
        if infer_type not in ("full", "body"):
            infer_type = "body"

        # ---- Stage 1: SAM 3D Body inference --------------------------------
        try:
            run_fast_infer(
                input_path=img_path,
                output_dir=self.npz_dir,
                checkpoint_path=self.checkpoint_path,
                mhr_path=self.mhr_path,
                max_side=int(max_side),
                single_person=bool(single_person),
                map_2d_to_original=True,
                save_compressed=bool(save_compressed),
                cache_dir=cache_dir.strip() if cache_dir else "",
                bbox_thr=0.8,
                inference_type=infer_type,
            )
        except Exception as e:
            return None, "", None, None, f"Stage 1 SAM 3D Body failed: {e}"

        infer_ms = (time.perf_counter() - t0) * 1000

        if not os.path.exists(npz_path):
            return None, "", None, None, f"Stage 1: NPZ not produced at {npz_path}"

        orig_img = _load_rgb(img_path)
        skel_img = _draw_skeleton(img_path, npz_path)

        # Check at least one person was detected
        try:
            z = np.load(npz_path, allow_pickle=False)
            if int(z["num_people"]) == 0:
                return orig_img, "No person detected in image.", None, skel_img, \
                    f"No person detected. infer={infer_ms:.0f}ms"
        except Exception:
            pass

        # ---- Stage 2: NPZ → .pt tensor -------------------------------------
        t2 = time.perf_counter()
        try:
            convert_npz_to_posescript(npz_path, pt_path)
        except Exception as e:
            return orig_img, f"Conversion failed: {e}", None, skel_img, \
                f"Stage 2 failed. infer={infer_ms:.0f}ms"
        conv_ms = (time.perf_counter() - t2) * 1000

        # ---- Stage 3: PoseScript caption ------------------------------------
        t3 = time.perf_counter()
        caption = ""

        if not os.path.exists(POSESCRIPT_VENV_PYTHON):
            caption = (
                f"[PoseScript venv not found at {POSESCRIPT_VENV_PYTHON}. "
                "Run base_posescript setup first.]"
            )
        else:
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
                if proc.returncode != 0:
                    caption = f"[PoseScript error rc={proc.returncode}]\n{proc.stderr.strip()}"
                else:
                    caption = proc.stdout.strip() or f"[empty output]\n{proc.stderr.strip()}"
            except subprocess.TimeoutExpired:
                caption = "[PoseScript timed out (>120s)]"
            except Exception as e:
                caption = f"[Subprocess error: {e}]"

        caption_ms = (time.perf_counter() - t3) * 1000

        # ---- Stage 4: Mesh render (optional) --------------------------------
        mesh_img = None
        render_ms = 0.0
        render_note = "skipped"

        if render_mesh:
            t4 = time.perf_counter()
            try:
                out_dir = os.path.join(self.mesh_dir, stem)
                os.makedirs(out_dir, exist_ok=True)
                proc_r = subprocess.run(
                    [
                        sys.executable,
                        RENDER_SCRIPT,
                        "--npz", npz_path,
                        "--output_dir", out_dir,
                        "--image", img_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                render_ms = (time.perf_counter() - t4) * 1000
                mesh_path = os.path.join(out_dir, "mesh_overlay.jpg")
                if proc_r.returncode == 0 and os.path.exists(mesh_path):
                    mesh_img = _load_rgb(mesh_path)
                    render_note = f"ok render={render_ms:.0f}ms"
                else:
                    render_note = f"failed rc={proc_r.returncode}"
            except Exception as e:
                render_note = f"error ({e})"

        # ---- Status ---------------------------------------------------------
        ts = datetime.now().strftime("%H:%M:%S")
        status = (
            f"[{ts}]  infer={infer_ms:.0f}ms  "
            f"conv={conv_ms:.0f}ms  "
            f"caption={caption_ms:.0f}ms  "
            f"render={render_note}  "
            f"npz={os.path.basename(npz_path)}"
        )

        return orig_img, caption, mesh_img, skel_img, status


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

_CSS = """
#ctrl-col { min-width: 280px !important; max-width: 340px !important; flex: 0 0 320px !important; }
"""


def build_app(checkpoint_path: str, mhr_path: str, output_dir: str) -> gr.Blocks:
    pipeline = CaptionPipeline(checkpoint_path, mhr_path, output_dir)

    with gr.Blocks(title="Pose Caption Demo", css=_CSS) as demo:
        gr.Markdown(
            "## Pose Caption Demo\n"
            "Upload a JPEG/PNG → SAM 3D Body detects the pose → "
            "PoseScript generates a natural-language description."
        )

        with gr.Row(equal_height=False):
            # ----------------------------------------------------------------
            # LEFT: fixed-width control panel (mirrors gui_app.py options)
            # ----------------------------------------------------------------
            with gr.Column(elem_id="ctrl-col", min_width=300):
                file_in = gr.File(
                    label="Input image (JPEG / PNG)",
                    file_types=["image"],
                    file_count="single",
                )
                max_side = gr.Slider(
                    0, 2000, value=1280, step=32,
                    label="max_side (0 = no resize)",
                )
                inference_type = gr.Radio(
                    choices=["full", "body"],
                    value="body",
                    label="Inference type  (body = faster, no hand detail)",
                )
                single_person   = gr.Checkbox(value=True,  label="Single person mode")
                save_compressed = gr.Checkbox(value=False, label="Compress NPZ (slower)")
                render_mesh     = gr.Checkbox(value=True,  label="Render mesh overlay")
                cache_dir = gr.Textbox(
                    value="./output/fast_npz_cache",
                    label="Cache directory (optional)",
                )
                run_btn = gr.Button("▶  Generate caption", variant="primary")
                status  = gr.Textbox(label="Status / timing", lines=3, interactive=False)

            # ----------------------------------------------------------------
            # RIGHT: 2×2 image grid
            #   Top-left:     Original photo
            #   Top-right:    Caption text box
            #   Bottom-left:  Mesh overlay render
            #   Bottom-right: 2-D skeleton
            # ----------------------------------------------------------------
            with gr.Column(scale=3):
                with gr.Row(equal_height=True):
                    orig_img = gr.Image(
                        label="Original photo",
                        type="numpy",
                        show_download_button=False,
                    )
                    caption_box = gr.Textbox(
                        label="PoseScript caption",
                        lines=12,
                        placeholder="Caption will appear here after running…",
                        interactive=False,
                    )
                with gr.Row(equal_height=True):
                    mesh_img = gr.Image(
                        label="Mesh overlay",
                        type="numpy",
                        show_download_button=False,
                    )
                    skel_img = gr.Image(
                        label="2D keypoints",
                        type="numpy",
                        show_download_button=False,
                    )

        run_btn.click(
            fn=pipeline.run,
            inputs=[
                file_in, max_side, inference_type,
                single_person, save_compressed, render_mesh, cache_dir,
            ],
            outputs=[orig_img, caption_box, mesh_img, skel_img, status],
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

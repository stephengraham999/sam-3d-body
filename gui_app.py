# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import gradio as gr

from sam_3d_body import render_npz_to_files, run_fast_infer

# ---------------------------------------------------------------------------
# PID file for stale-process detection
# ---------------------------------------------------------------------------
_PID_FILE = os.path.join(os.path.dirname(__file__), ".gui_pid")


def _write_pid() -> None:
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _kill_stale(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["lsof", "-tiTCP:" + str(port), "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        for pid_str in out.splitlines():
            pid = int(pid_str)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.3)
    except Exception:
        pass
    if os.path.exists(_PID_FILE):
        try:
            old_pid = int(open(_PID_FILE).read().strip())
            if old_pid != os.getpid():
                os.kill(old_pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------

# Body links: (joint_a, joint_b, BGR_color) using mhr70 indices
# 0=nose,1=l_eye,2=r_eye,3=l_ear,4=r_ear,5=l_shoulder,6=r_shoulder,
# 7=l_elbow,8=r_elbow,9=l_hip,10=r_hip,11=l_knee,12=r_knee,
# 13=l_ankle,14=r_ankle,15=l_big_toe,16=l_small_toe,17=l_heel,
# 18=r_big_toe,19=r_small_toe,20=r_heel,41=r_wrist,62=l_wrist,69=neck
_BODY_LINKS = [
    # legs
    (13, 11, (0, 200, 0)),
    (11, 9,  (0, 200, 0)),
    (14, 12, (0, 100, 255)),
    (12, 10, (0, 100, 255)),
    # hips / torso
    (9, 10,  (51, 153, 255)),
    (5, 9,   (51, 153, 255)),
    (6, 10,  (51, 153, 255)),
    (5, 6,   (51, 153, 255)),
    # arms
    (5, 7,   (0, 200, 0)),
    (7, 62,  (0, 200, 0)),
    (6, 8,   (0, 100, 255)),
    (8, 41,  (0, 100, 255)),
    # head
    (0, 1,   (51, 153, 255)),
    (0, 2,   (51, 153, 255)),
    (1, 2,   (51, 153, 255)),
    (1, 3,   (51, 153, 255)),
    (2, 4,   (51, 153, 255)),
    (3, 5,   (51, 153, 255)),
    (4, 6,   (51, 153, 255)),
    # feet
    (13, 15, (0, 200, 0)),
    (13, 16, (0, 200, 0)),
    (13, 17, (0, 200, 0)),
    (14, 18, (0, 100, 255)),
    (14, 19, (0, 100, 255)),
    (14, 20, (0, 100, 255)),
]
_BODY_KPT_INDICES = list(range(21)) + [41, 62, 69]  # main body joints


def draw_skeleton_image(img_path: str, npz_path: str) -> np.ndarray | None:
    """Draw 2D keypoints + skeleton lines on the image. Returns RGB array."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()

    try:
        z = np.load(npz_path, allow_pickle=False)
        num_people = int(z["num_people"])
    except Exception:
        return img

    img_h, img_w = img.shape[:2]
    line_w = max(2, img_w // 500)
    dot_r  = max(4, img_w // 250)

    for pid in range(num_people):
        prefix = f"person_{pid:03d}_"
        key = prefix + "pred_keypoints_2d"
        if key not in z:
            continue
        kpts = z[key]  # (N, 2) in original image coordinates

        # Draw limb lines
        for ja, jb, color in _BODY_LINKS:
            if ja >= len(kpts) or jb >= len(kpts):
                continue
            x1, y1 = int(round(kpts[ja, 0])), int(round(kpts[ja, 1]))
            x2, y2 = int(round(kpts[jb, 0])), int(round(kpts[jb, 1]))
            if (0 <= x1 < img_w and 0 <= y1 < img_h and
                    0 <= x2 < img_w and 0 <= y2 < img_h):
                cv2.line(img, (x1, y1), (x2, y2), color[::-1], line_w, cv2.LINE_AA)

        # Draw joint dots
        for idx in _BODY_KPT_INDICES:
            if idx >= len(kpts):
                continue
            x, y = int(round(kpts[idx, 0])), int(round(kpts[idx, 1]))
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(img, (x, y), dot_r + 1, (0, 0, 0), -1)
                cv2.circle(img, (x, y), dot_r, (255, 255, 255), -1)

    return img


def _load_image_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _collect_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if value is None:
        return paths
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for k in ("path", "name", "file", "value"):
            if k in value:
                paths.extend(_collect_paths(value[k]))
        return paths
    if isinstance(value, (list, tuple)):
        for v in value:
            paths.extend(_collect_paths(v))
        return paths
    if hasattr(value, "name"):
        return [str(value.name)]
    return [str(value)]


def _perf_str(summary: dict[str, Any]) -> str:
    if summary.get("cache_hit"):
        cache_ms = summary.get("cache_ms") or 0
        return f"cache_hit cache_ms={cache_ms:.1f}"
    infer_ms = summary.get("infer_ms") or 0
    save_ms = summary.get("save_ms")
    if save_ms is not None:
        return f"infer_ms={infer_ms:.1f} save_ms={save_ms:.1f}"
    return f"infer_ms={infer_ms:.1f} save=async"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
/* Fix left control panel width — never grows wider than 340px */
#ctrl-col {
    min-width: 280px !important;
    max-width: 340px !important;
    flex: 0 0 320px !important;
}
/* Make each image in the right panel fill its cell with no bottom thumbnail strip */
#right-panel .image-container {
    height: 100%;
}
#right-panel img {
    object-fit: contain;
    height: 100%;
}
/* Give each image row equal height */
#right-panel .gr-row {
    flex: 1 1 50%;
    min-height: 300px;
}
"""


# ---------------------------------------------------------------------------
# Core GUI class
# ---------------------------------------------------------------------------

class ThinGui:
    def __init__(self, checkpoint_path: str, mhr_path: str, output_dir: str):
        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process(
        self,
        files,
        max_side,
        single_person,
        save_compressed,
        render_mesh,
        cache_dir,
        inference_type,
    ):
        """
        Returns:
            orig_img, skel_img, mesh_img, side_img — numpy RGB arrays or None
            status — str
            npz_files — list[str] (hidden, for 3D viewer)
        """
        empty = (None, None, None, None, "No files selected.", [])
        if not files:
            return empty

        image_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
        npz_dir   = os.path.join(self.output_dir, "npz")
        mesh_dir  = os.path.join(self.output_dir, "render")
        debug_log = os.path.join(self.output_dir, "debug.log")
        os.makedirs(npz_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        with open(debug_log, "a") as f:
            f.write(
                f"\n[{datetime.now().isoformat()}] process start "
                f"files={len(image_paths)} render_mesh={bool(render_mesh)}\n"
            )

        infer_type = str(inference_type).strip().lower() if inference_type else "full"
        if infer_type not in ("full", "body"):
            infer_type = "full"

        infer_summary = run_fast_infer(
            input_path=image_paths,
            output_dir=npz_dir,
            checkpoint_path=self.checkpoint_path,
            mhr_path=self.mhr_path,
            max_side=int(max_side),
            single_person=bool(single_person),
            map_2d_to_original=True,
            save_compressed=bool(save_compressed),
            cache_dir=cache_dir.strip(),
            bbox_thr=0.8,
            inference_type=infer_type,
        )
        summary_by_path = {str(e.get("image_path")): e for e in infer_summary}

        # We show the FIRST image's results in the 2×2 panel
        npz_files: list[str] = []
        render_notes: list[str] = []

        orig_img  = None
        skel_img  = None
        mesh_img  = None
        side_img  = None

        for img_path in image_paths:
            stem     = os.path.splitext(os.path.basename(img_path))[0]
            npz_path = os.path.join(npz_dir, f"{stem}.npz")

            if not os.path.exists(npz_path):
                with open(debug_log, "a") as f:
                    f.write(f"  npz_missing={npz_path}\n")
                continue

            npz_files.append(npz_path)
            perf = _perf_str(summary_by_path.get(img_path, {}))

            # Original image (first result only, shown in top-left)
            if orig_img is None:
                orig_img = _load_image_rgb(img_path)

            # 2D keypoints skeleton overlay (top-right)
            if skel_img is None:
                skel_img = draw_skeleton_image(img_path, npz_path)

            if not render_mesh:
                render_notes.append(f"{stem}: render=skipped ({perf})")
                continue

            # Mesh render (bottom row) — subprocess required on macOS
            try:
                out_dir = os.path.join(mesh_dir, stem)
                os.makedirs(out_dir, exist_ok=True)
                render_t0 = time.perf_counter()
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "tools", "render_npz.py"),
                    "--npz", npz_path,
                    "--output_dir", out_dir,
                    "--image", img_path,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                render_ms = (time.perf_counter() - render_t0) * 1000.0

                mesh_path = os.path.join(out_dir, "mesh_overlay.jpg")
                side_path = os.path.join(out_dir, "mesh_side.jpg")

                with open(debug_log, "a") as f:
                    f.write(f"  render_rc={proc.returncode} render_ms={render_ms:.1f}\n")
                    if proc.stderr:
                        f.write(f"  render_stderr={proc.stderr.strip()}\n")

                if proc.returncode != 0:
                    raise RuntimeError(f"render subprocess rc={proc.returncode}")
                if not (os.path.exists(mesh_path) and os.path.exists(side_path)):
                    raise RuntimeError("render outputs missing")

                if mesh_img is None and os.path.exists(mesh_path):
                    mesh_img = _load_image_rgb(mesh_path)
                if side_img is None and os.path.exists(side_path):
                    side_img = _load_image_rgb(side_path)

                render_notes.append(f"{stem}: render=ok render_ms={render_ms:.1f} ({perf})")

            except Exception as e:
                with open(debug_log, "a") as f:
                    f.write(f"  render_exception={type(e).__name__}: {e}\n")
                render_notes.append(f"{stem}: render=failed ({type(e).__name__}: {e})")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = (
            f"Done {ts}. Processed {len(image_paths)} file(s). NPZ: {len(npz_files)}. "
            f"render_mesh={bool(render_mesh)}"
        )
        if render_notes:
            status += " | " + " | ".join(render_notes)
        status += f" | debug_log={debug_log}"

        return orig_img, skel_img, mesh_img, side_img, status, npz_files

    def open_viewer(self, npz_files):
        candidates = _collect_paths(npz_files)
        npz_path = ""
        for p in candidates:
            if p and os.path.exists(p):
                npz_path = p
                break

        if not npz_path:
            npz_dir = os.path.join(self.output_dir, "npz")
            if os.path.isdir(npz_dir):
                lst = [
                    os.path.join(npz_dir, f)
                    for f in os.listdir(npz_dir)
                    if f.lower().endswith(".npz")
                ]
                if lst:
                    npz_path = max(lst, key=os.path.getmtime)

        if not npz_path:
            return "No NPZ file selected."

        check = subprocess.run(
            [sys.executable, "-c", "import trimesh.viewer.windowed; print('ok')"],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            return (
                "3D viewer dependency missing. "
                "Install with: pip install 'pyglet<2'. "
                f"details={check.stderr.strip() or check.stdout.strip()}"
            )

        try:
            subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "tools", "view_npz_3d.py"),
                    "--npz",
                    npz_path,
                ]
            )
            return f"Opened 3D viewer for: {os.path.basename(npz_path)}"
        except Exception as e:
            return f"Failed to open viewer: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------

def build_app(checkpoint_path: str, mhr_path: str, output_dir: str):
    runner = ThinGui(checkpoint_path, mhr_path, output_dir)

    with gr.Blocks(title="SAM3D Thin GUI v3", css=_CSS) as demo:
        gr.Markdown("## SAM3D Thin GUI v3\nIn-process render · Estimator caching · Thread-safe")
        gr.Markdown(
            f"**Build:** thin-gui-v3  |  **PID:** {os.getpid()}  |  "
            f"**Checkpoint:** `{os.path.basename(checkpoint_path)}`"
        )

        with gr.Row(equal_height=False):
            # ----------------------------------------------------------------
            # LEFT: fixed-width control panel
            # ----------------------------------------------------------------
            with gr.Column(elem_id="ctrl-col", min_width=300):
                files = gr.File(
                    label="Input JPEG/PNG",
                    file_count="multiple",
                    file_types=["image"],
                )
                max_side = gr.Slider(
                    0, 2000, value=1280, step=32,
                    label="max_side (0 disables resize)",
                )
                inference_type = gr.Radio(
                    choices=["full", "body"],
                    value="full",
                    label="Inference type (body=faster, no hand detail)",
                )
                single_person   = gr.Checkbox(value=True,  label="Single person mode")
                save_compressed = gr.Checkbox(value=False, label="Compress NPZ (slower)")
                render_mesh     = gr.Checkbox(value=True,  label="Also render mesh images")
                cache_dir = gr.Textbox(
                    value="./output/fast_npz_cache",
                    label="Cache directory (optional)",
                )

                run_btn         = gr.Button("Run", variant="primary")
                batch_preset_btn = gr.Button("⚡ Batch Preset", variant="secondary")

                status = gr.Textbox(label="Status", lines=3)

                gr.Markdown("---")
                open_3d_btn    = gr.Button("Open interactive 3D viewer")
                open_3d_status = gr.Textbox(label="3D Viewer", lines=1)

                gr.Markdown("---")
                # Hidden NPZ file list, exposed as download widget
                npz_files = gr.File(
                    label="Download NPZ output(s)",
                    file_count="multiple",
                    interactive=False,
                )

            # ----------------------------------------------------------------
            # RIGHT: 2×2 image grid
            # ----------------------------------------------------------------
            with gr.Column(elem_id="right-panel", scale=3):
                with gr.Row(equal_height=True):
                    orig_img = gr.Image(
                        label="Original photo",
                        type="numpy",
                        show_download_button=False,
                    )
                    skel_img = gr.Image(
                        label="2D keypoints",
                        type="numpy",
                        show_download_button=False,
                    )
                with gr.Row(equal_height=True):
                    mesh_img = gr.Image(
                        label="Mesh overlay",
                        type="numpy",
                        show_download_button=False,
                    )
                    side_img = gr.Image(
                        label="Mesh side view",
                        type="numpy",
                        show_download_button=False,
                    )

        # ----------------------------------------------------------------
        # Batch preset
        # ----------------------------------------------------------------
        def apply_batch_preset():
            return (
                960,    # max_side
                "body", # inference_type
                True,   # single_person
                False,  # save_compressed
                False,  # render_mesh
                "",     # cache_dir
            )

        batch_preset_btn.click(
            fn=apply_batch_preset,
            inputs=[],
            outputs=[max_side, inference_type, single_person,
                     save_compressed, render_mesh, cache_dir],
            queue=False,
        )

        run_btn.click(
            fn=runner.process,
            inputs=[files, max_side, single_person, save_compressed,
                    render_mesh, cache_dir, inference_type],
            outputs=[orig_img, skel_img, mesh_img, side_img, status, npz_files],
            queue=False,
        )

        open_3d_btn.click(
            fn=runner.open_viewer,
            inputs=[npz_files],
            outputs=[open_3d_status],
            queue=False,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAM3D thin GUI v3")
    p.add_argument("--checkpoint_path",
                   default="./checkpoints/sam-3d-body-dinov3/model.ckpt")
    p.add_argument("--mhr_path",
                   default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    p.add_argument("--output_dir", default="./output/gui")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=7862, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _kill_stale(args.port)
    _write_pid()
    app = build_app(args.checkpoint_path, args.mhr_path, args.output_dir)
    app.launch(server_name=args.host, server_port=args.port)

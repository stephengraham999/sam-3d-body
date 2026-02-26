"""
posescript_caption_single.py
============================
PoseScript captioning runner for a SINGLE pose.

This script must be executed under the PoseScript isolated virtual environment:

    base_posescript/.venv/bin/python posescript_caption_single.py --input_pt PATH

It accepts a PyTorch .pt file containing a single-pose coordinate tensor
(shape 1×22×3, root-centred, Y-up) produced by ``sam3d_to_posescript.py``,
runs the PoseScript automatic captioning pipeline, and prints the resulting
natural-language description to stdout.

The calling process (e.g. the Gradio test GUI running under the main project
venv) captures stdout to retrieve the caption.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import torch


def _resolve_flag(value: bool | None, default_value: bool) -> bool:
    return default_value if value is None else bool(value)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a PoseScript caption for a single pose tensor."
    )
    p.add_argument(
        "--input_pt",
        required=True,
        metavar="PATH",
        help="Path to a .pt file containing a (1, 22, 3) pose tensor.",
    )
    p.add_argument(
        "--save_dir",
        default="",
        metavar="PATH",
        help="Optional directory to store PoseScript intermediates.",
    )
    p.add_argument(
        "--simplified_captions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable simplified captions. Default preserves legacy behavior (enabled).",
    )
    p.add_argument(
        "--random_skip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable random skipping. Default preserves legacy behavior (enabled).",
    )
    p.add_argument(
        "--apply_transrel_ripple_effect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable transitive-relation ripple effect. Default: disabled.",
    )
    p.add_argument(
        "--apply_stat_ripple_effect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable statistical ripple effect. Default: disabled.",
    )
    p.add_argument(
        "--add_babel_info",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable BABEL sentence prefixing. Default: disabled.",
    )
    p.add_argument(
        "--add_dancing_info",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable dancing sentence tagging. Default: disabled.",
    )
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Load the pose tensor
    # ------------------------------------------------------------------
    if not os.path.exists(args.input_pt):
        print(f"ERROR: input_pt not found: {args.input_pt}", file=sys.stderr)
        sys.exit(1)

    coords = torch.load(args.input_pt, map_location="cpu")

    if coords.ndim != 3 or coords.shape[1:] != (22, 3):
        print(
            f"ERROR: expected tensor shape (N, 22, 3), got {list(coords.shape)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Import PoseScript captioning
    #    (text2pose is installed in base_posescript/.venv via pip install -e)
    # ------------------------------------------------------------------
    try:
        from text2pose.posescript.captioning import main as captioning_main
    except ImportError as e:
        print(f"ERROR: cannot import text2pose: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Run the captioning pipeline
    #    Preserve legacy defaults unless flags are explicitly provided.
    # ------------------------------------------------------------------
    simplified_captions = _resolve_flag(args.simplified_captions, True)
    random_skip = _resolve_flag(args.random_skip, True)
    apply_transrel_ripple_effect = _resolve_flag(args.apply_transrel_ripple_effect, False)
    apply_stat_ripple_effect = _resolve_flag(args.apply_stat_ripple_effect, False)
    add_babel_info = _resolve_flag(args.add_babel_info, False)
    add_dancing_info = _resolve_flag(args.add_dancing_info, False)
    babel_info = False
    if add_babel_info:
        default_sent = "They are dancing. " if add_dancing_info else ""
        babel_info = [default_sent for _ in range(coords.shape[0])]

    tmp_ctx = tempfile.TemporaryDirectory(prefix="posescript_single_")
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        save_dir = None

    try:
        active_save_dir = str(save_dir) if save_dir is not None else tmp_ctx.__enter__()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                captioning_main(
                    coords=coords,
                    use_contact_codes=False,
                    simplified_captions=simplified_captions,
                    random_skip=random_skip,
                    apply_transrel_ripple_effect=apply_transrel_ripple_effect,
                    apply_stat_ripple_effect=apply_stat_ripple_effect,
                    babel_info=babel_info,
                    verbose=False,
                    save_dir=active_save_dir,
                )
        except Exception as e:
            print(f"ERROR: captioning_main failed: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)

        # ------------------------------------------------------------------
        # 4. Read the saved descriptions
        # ------------------------------------------------------------------
        desc_file = os.path.join(active_save_dir, "descriptions.json")
        if not os.path.exists(desc_file):
            print("ERROR: descriptions.json was not created", file=sys.stderr)
            sys.exit(1)

        with open(desc_file) as f:
            descriptions: dict = json.load(f)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # 5. Print the caption(s) to stdout
    #    descriptions is a dict {str(pose_index): caption_text}
    #    For a single pose we have exactly one entry.
    # ------------------------------------------------------------------
    captions = list(descriptions.values())
    if not captions:
        print("ERROR: descriptions.json is empty", file=sys.stderr)
        sys.exit(1)

    # Print all captions separated by a delimiter (one per person if batched)
    for cap in captions:
        print(cap, flush=True)


if __name__ == "__main__":
    main()

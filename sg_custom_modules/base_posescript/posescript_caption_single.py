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
import json
import os
import sys
import tempfile

import torch


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
    #    - use_contact_codes=False  → skip contact-code path (needs selfcontact)
    #    - simplified_captions=True → shorter, cleaner output sentences
    #    - random_skip=True         → non-deterministic but avoids verbose output
    #    - save_dir=tmpdir          → write intermediate .pt files to temp location
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="posescript_single_") as tmpdir:
        try:
            captioning_main(
                coords=coords,
                use_contact_codes=False,
                simplified_captions=True,
                random_skip=True,
                save_dir=tmpdir,
            )
        except Exception as e:
            print(f"ERROR: captioning_main failed: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)

        # ------------------------------------------------------------------
        # 4. Read the saved descriptions
        # ------------------------------------------------------------------
        desc_file = os.path.join(tmpdir, "descriptions.json")
        if not os.path.exists(desc_file):
            print("ERROR: descriptions.json was not created", file=sys.stderr)
            sys.exit(1)

        with open(desc_file) as f:
            descriptions: dict = json.load(f)

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

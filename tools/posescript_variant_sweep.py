#!/usr/bin/env python3
"""Batch NPZ -> PoseScript variant sweep with Markdown reporting."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sg_custom_modules.base_posescript.sam3d_to_posescript import convert_npz_to_posescript

POSESCRIPT_VENV_PYTHON = _PROJECT_ROOT / "base_posescript" / ".venv" / "bin" / "python"
POSESCRIPT_RUNNER = _PROJECT_ROOT / "sg_custom_modules" / "base_posescript" / "posescript_caption_single.py"

DEFAULT_NPZ_DIR = _PROJECT_ROOT / "output" / "caption_demo" / "npz"
DEFAULT_OUT_MD = _PROJECT_ROOT / "output" / "caption_demo" / "reports" / "posescript_variant_sweep.md"


VARIANTS: List[Dict[str, object]] = [
    {
        "name": "N2",
        "flags": {
            "simplified_captions": True,
            "random_skip": True,
            "apply_transrel_ripple_effect": False,
            "apply_stat_ripple_effect": False,
            "add_babel_info": False,
            "add_dancing_info": False,
        },
    },
    {
        "name": "N6",
        "flags": {
            "simplified_captions": False,
            "random_skip": True,
            "apply_transrel_ripple_effect": False,
            "apply_stat_ripple_effect": False,
            "add_babel_info": True,
            "add_dancing_info": True,
        },
    },
    {
        "name": "N7",
        "flags": {
            "simplified_captions": False,
            "random_skip": True,
            "apply_transrel_ripple_effect": True,
            "apply_stat_ripple_effect": True,
            "add_babel_info": False,
            "add_dancing_info": False,
        },
    },
    {
        "name": "A",
        "flags": {
            "simplified_captions": False,
            "random_skip": True,
            "apply_transrel_ripple_effect": True,
            "apply_stat_ripple_effect": True,
            "add_babel_info": True,
            "add_dancing_info": True,
        },
    },
    {
        "name": "B",
        "flags": {
            "simplified_captions": False,
            "random_skip": True,
            "apply_transrel_ripple_effect": False,
            "apply_stat_ripple_effect": False,
            "add_babel_info": True,
            "add_dancing_info": True,
        },
    },
    {
        "name": "C",
        "flags": {
            "simplified_captions": False,
            "random_skip": True,
            "apply_transrel_ripple_effect": False,
            "apply_stat_ripple_effect": False,
            "add_babel_info": True,
            "add_dancing_info": False,
        },
    },
    {
        "name": "D",
        "flags": {
            "simplified_captions": False,
            "random_skip": True,
            "apply_transrel_ripple_effect": False,
            "apply_stat_ripple_effect": False,
            "add_babel_info": False,
            "add_dancing_info": False,
        },
    },
    {
        "name": "E",
        "flags": {
            "simplified_captions": True,
            "random_skip": True,
            "apply_transrel_ripple_effect": False,
            "apply_stat_ripple_effect": False,
            "add_babel_info": False,
            "add_dancing_info": False,
        },
    },
    {
        "name": "F",
        "flags": {
            "simplified_captions": True,
            "random_skip": False,
            "apply_transrel_ripple_effect": False,
            "apply_stat_ripple_effect": False,
            "add_babel_info": False,
            "add_dancing_info": False,
        },
    },
]
VARIANT_MAP: Dict[str, Dict[str, object]] = {str(v["name"]): v for v in VARIANTS}


def _bool_cli(flag: str, value: bool) -> str:
    return f"--{flag}" if value else f"--no-{flag}"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def _format_flags(flags: Dict[str, bool]) -> str:
    ordered = [
        "simplified_captions",
        "random_skip",
        "apply_transrel_ripple_effect",
        "apply_stat_ripple_effect",
        "add_babel_info",
        "add_dancing_info",
    ]
    return ", ".join(f"{k}={'on' if flags[k] else 'off'}" for k in ordered)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PoseScript variant sweep over NPZ files.")
    p.add_argument("--npz_dir", default=str(DEFAULT_NPZ_DIR), help="Directory containing input NPZ files.")
    p.add_argument("--max_files", type=int, default=3, help="Number of sorted NPZs to process when --files is not provided.")
    p.add_argument("--axis_variant", default="swap_yz", choices=["flip_yz", "identity", "swap_xy", "swap_xz", "swap_yz"], help="Axis variant for NPZ->PT conversion.")
    p.add_argument("--output_md", default=str(DEFAULT_OUT_MD), help="Markdown report output path.")
    p.add_argument("--files", nargs="*", default=None, help="Optional explicit NPZ paths. If set, these are used in given order.")
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variant names to run (subset of: N2 N6 N7 A B C D E F). Default: all.",
    )
    return p.parse_args()


def _select_npzs(npz_dir: Path, files: List[str] | None, max_files: int) -> List[Path]:
    if files:
        selected = [Path(x).resolve() for x in files]
    else:
        selected = sorted(npz_dir.glob("*.npz"))[:max_files]
    return selected


def main() -> int:
    args = _parse_args()

    npz_dir = Path(args.npz_dir).resolve()
    out_md = Path(args.output_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    variants_to_run: List[Dict[str, object]]
    if args.variants:
        unknown = [v for v in args.variants if v not in VARIANT_MAP]
        if unknown:
            print(f"ERROR: unknown variants: {unknown}", file=sys.stderr)
            print(f"Known variants: {list(VARIANT_MAP.keys())}", file=sys.stderr)
            return 2
        variants_to_run = [VARIANT_MAP[v] for v in args.variants]
    else:
        variants_to_run = VARIANTS

    if not POSESCRIPT_VENV_PYTHON.exists():
        print(f"ERROR: PoseScript venv python missing: {POSESCRIPT_VENV_PYTHON}", file=sys.stderr)
        return 2
    if not POSESCRIPT_RUNNER.exists():
        print(f"ERROR: runner missing: {POSESCRIPT_RUNNER}", file=sys.stderr)
        return 2

    selected = _select_npzs(npz_dir, args.files, args.max_files)
    if len(selected) != 3:
        print(f"ERROR: expected exactly 3 NPZ files for this run, got {len(selected)}", file=sys.stderr)
        return 2

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = out_md.parent / f"run_{ts}"
    pt_dir = run_root / "pt"
    interm_dir = run_root / "intermediates"
    pt_dir.mkdir(parents=True, exist_ok=True)
    interm_dir.mkdir(parents=True, exist_ok=True)

    report_lines: List[str] = []
    report_lines.append("# PoseScript Variant Sweep Report")
    report_lines.append("")
    report_lines.append(f"- Timestamp: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    report_lines.append(f"- NPZ dir: `{npz_dir}`")
    report_lines.append(f"- Axis variant: `{args.axis_variant}`")
    report_lines.append("- Files processed: `3`")
    report_lines.append(f"- Variants: `{', '.join(str(v['name']) for v in variants_to_run)}`")
    report_lines.append(f"- Run root: `{run_root}`")
    report_lines.append("")

    all_rows = []

    for npz_path in selected:
        stem = npz_path.stem
        pt_path = pt_dir / f"{stem}.pt"

        section_rows = []
        convert_status = "ok"
        convert_err = ""
        try:
            convert_npz_to_posescript(npz_path, pt_path, axis_variant=args.axis_variant)
        except Exception as e:
            convert_status = "failed"
            convert_err = f"{type(e).__name__}: {e}"

        report_lines.append(f"## `{stem}`")
        report_lines.append("")
        report_lines.append(f"- Source NPZ: `{npz_path}`")
        report_lines.append(f"- Converted PT: `{pt_path}`")
        report_lines.append(f"- Conversion status: `{convert_status}`")
        if convert_err:
            report_lines.append(f"- Conversion error: `{convert_err}`")
        report_lines.append("")
        report_lines.append("| Variant | Flags | Caption | Runtime (ms) | Status |")
        report_lines.append("|---|---|---|---:|---|")

        for variant in variants_to_run:
            name = str(variant["name"])
            flags = variant["flags"]
            variant_save_dir = interm_dir / stem / name
            variant_save_dir.mkdir(parents=True, exist_ok=True)

            row = {
                "npz": str(npz_path),
                "stem": stem,
                "variant": name,
                "flags": flags,
                "caption": "",
                "runtime_ms": 0,
                "status": "failed",
                "error": "",
            }

            if convert_status != "ok":
                row["error"] = "conversion_failed"
                section_rows.append(row)
                all_rows.append(row)
                report_lines.append(
                    f"| `{name}` | `{_md_escape(_format_flags(flags))}` | "
                    f"`conversion not available` | 0 | `failed` |"
                )
                continue

            cmd = [
                str(POSESCRIPT_VENV_PYTHON),
                str(POSESCRIPT_RUNNER),
                "--input_pt",
                str(pt_path),
                "--save_dir",
                str(variant_save_dir),
                _bool_cli("simplified_captions", bool(flags["simplified_captions"])),
                _bool_cli("random_skip", bool(flags["random_skip"])),
                _bool_cli("apply_transrel_ripple_effect", bool(flags["apply_transrel_ripple_effect"])),
                _bool_cli("apply_stat_ripple_effect", bool(flags["apply_stat_ripple_effect"])),
                _bool_cli("add_babel_info", bool(flags["add_babel_info"])),
                _bool_cli("add_dancing_info", bool(flags["add_dancing_info"])),
            ]

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                row["runtime_ms"] = int((time.perf_counter() - t0) * 1000)
                if proc.returncode == 0:
                    row["status"] = "ok"
                    row["caption"] = proc.stdout.strip()
                else:
                    row["error"] = (proc.stderr or proc.stdout).strip()
            except Exception as e:
                row["runtime_ms"] = int((time.perf_counter() - t0) * 1000)
                row["error"] = f"{type(e).__name__}: {e}"

            section_rows.append(row)
            all_rows.append(row)

            cap_cell = _md_escape(row["caption"] if row["caption"] else row["error"])
            report_lines.append(
                f"| `{name}` | `{_md_escape(_format_flags(flags))}` | "
                f"{cap_cell} | {row['runtime_ms']} | `{row['status']}` |"
            )

        report_lines.append("")

    # Smoke checks
    smoke_ok = True
    smoke_notes = []
    if len(selected) != 3:
        smoke_ok = False
        smoke_notes.append(f"Selected files != 3 (got {len(selected)})")

    expected_rows = 3 * len(variants_to_run)
    if len(all_rows) != expected_rows:
        smoke_ok = False
        smoke_notes.append(f"Variant rows != {expected_rows} (got {len(all_rows)})")

    # Qualitative summary metrics by flag family
    def avg_words(rows: List[dict]) -> float:
        texts = [r["caption"] for r in rows if r["status"] == "ok" and r["caption"]]
        if not texts:
            return 0.0
        return sum(len(t.split()) for t in texts) / float(len(texts))

    simp_on = [r for r in all_rows if r["flags"]["simplified_captions"]]
    simp_off = [r for r in all_rows if not r["flags"]["simplified_captions"]]
    ripple_on = [r for r in all_rows if r["flags"]["apply_transrel_ripple_effect"] or r["flags"]["apply_stat_ripple_effect"]]
    ripple_off = [r for r in all_rows if not (r["flags"]["apply_transrel_ripple_effect"] or r["flags"]["apply_stat_ripple_effect"])]
    babel_on = [r for r in all_rows if r["flags"]["add_babel_info"]]
    babel_off = [r for r in all_rows if not r["flags"]["add_babel_info"]]
    rand_on = [r for r in all_rows if r["flags"]["random_skip"]]
    rand_off = [r for r in all_rows if not r["flags"]["random_skip"]]

    report_lines.append("## Final Summary")
    report_lines.append("")
    report_lines.append("- Quick qualitative differences by flag family (based on average output length in words):")
    report_lines.append(f"  - simplified on/off: `{avg_words(simp_on):.1f}` / `{avg_words(simp_off):.1f}`")
    report_lines.append(f"  - ripple on/off: `{avg_words(ripple_on):.1f}` / `{avg_words(ripple_off):.1f}`")
    report_lines.append(f"  - babel+dancing on/off (babel only): `{avg_words(babel_on):.1f}` / `{avg_words(babel_off):.1f}`")
    report_lines.append(f"  - random_skip on/off: `{avg_words(rand_on):.1f}` / `{avg_words(rand_off):.1f}`")
    report_lines.append("")

    report_lines.append("## Smoke Checks")
    report_lines.append("")
    report_lines.append(f"- Files selected: `{len(selected)}` (expected `3`)")
    report_lines.append(f"- Variant rows: `{len(all_rows)}` (expected `{expected_rows}`)")
    report_lines.append(f"- Report path: `{out_md}`")
    report_lines.append(f"- Status: `{'pass' if smoke_ok else 'fail'}`")
    if smoke_notes:
        for n in smoke_notes:
            report_lines.append(f"- Note: `{n}`")

    out_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote: {out_md}")

    return 0 if smoke_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

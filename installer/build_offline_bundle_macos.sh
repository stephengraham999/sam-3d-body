#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$ROOT_DIR/offline_bundle/macos-apple-silicon}"
WHEELS_DIR="$BUNDLE_ROOT/wheels"
CHECKPOINTS_DIR="$BUNDLE_ROOT/checkpoints"
SOURCES_DIR="$BUNDLE_ROOT/sources"
REQ_BASE="$ROOT_DIR/installer/requirements-macos-base.txt"
REQ_OPT="$ROOT_DIR/installer/requirements-macos-optional.txt"

HF_MODEL_REPOS="${HF_MODEL_REPOS:-facebook/sam-3d-body-dinov3}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-1}"

export BUNDLE_ROOT CHECKPOINTS_DIR HF_MODEL_REPOS

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Set it before running, e.g."
  echo "  export HF_TOKEN=hf_xxx"
  exit 1
fi

echo "Preparing bundle directories at: $BUNDLE_ROOT"
mkdir -p "$WHEELS_DIR" "$CHECKPOINTS_DIR" "$SOURCES_DIR"

echo "Downloading Python wheels (macOS Apple Silicon)..."
python3 -m pip install --upgrade pip
python3 -m pip download -d "$WHEELS_DIR" torch torchvision torchaudio
python3 -m pip download -d "$WHEELS_DIR" -r "$REQ_BASE"

if [[ "$INCLUDE_OPTIONAL" == "1" ]]; then
  python3 -m pip download -d "$WHEELS_DIR" -r "$REQ_OPT"
fi

# Ensure runtime availability for the Python-based HF snapshot step.
python3 -m pip install --upgrade huggingface_hub

echo "Cloning source dependencies used by upstream INSTALL.md"
if [[ ! -d "$SOURCES_DIR/detectron2" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/detectron2.git "$SOURCES_DIR/detectron2"
fi
(
  cd "$SOURCES_DIR/detectron2"
  if git rev-parse --verify a1ce2f9 >/dev/null 2>&1; then
    git checkout a1ce2f9
  else
    git fetch origin --depth 200 >/dev/null 2>&1 || true
    if git rev-parse --verify a1ce2f9 >/dev/null 2>&1; then
      git checkout a1ce2f9
    else
      echo "WARN: Could not resolve detectron2 commit a1ce2f9; keeping current default branch HEAD."
    fi
  fi
)

if [[ ! -d "$SOURCES_DIR/MoGe" ]]; then
  git clone --depth 1 https://github.com/microsoft/MoGe.git "$SOURCES_DIR/MoGe"
fi

echo "Downloading gated Hugging Face model assets"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repos = [r.strip() for r in os.environ.get("HF_MODEL_REPOS", "").split(",") if r.strip()]
token = os.environ["HF_TOKEN"]
out_root = os.environ["CHECKPOINTS_DIR"]

for repo in repos:
    local_dir = os.path.join(out_root, repo.split("/")[-1])
    print(f"Downloading {repo} -> {local_dir}")
    snapshot_download(repo_id=repo, local_dir=local_dir, token=token, local_dir_use_symlinks=False)
PY

cp "$REQ_BASE" "$BUNDLE_ROOT/requirements-macos-base.txt"
cp "$REQ_OPT" "$BUNDLE_ROOT/requirements-macos-optional.txt"

python3 - <<'PY'
import json, os, platform, datetime

bundle_root = os.environ["BUNDLE_ROOT"]
repos = [r.strip() for r in os.environ.get("HF_MODEL_REPOS", "").split(",") if r.strip()]

manifest = {
    "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "platform": {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "hf_model_repos": repos,
    "notes": "Built for macOS Apple Silicon (arm64).",
}

with open(os.path.join(bundle_root, "bundle_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
PY

ARCHIVE_PATH="${BUNDLE_ROOT}.tar.gz"
echo "Creating tarball: $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$(dirname "$BUNDLE_ROOT")" "$(basename "$BUNDLE_ROOT")"

echo
echo "Done. Offline bundle created:"
echo "  $BUNDLE_ROOT"
echo "Archive:"
echo "  $ARCHIVE_PATH"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-sam3d-online}"
REQ_BASE="$ROOT_DIR/installer/requirements-macos-base.txt"
REQ_OPT="$ROOT_DIR/installer/requirements-macos-optional.txt"

INSTALL_OPTIONAL="${INSTALL_OPTIONAL:-1}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-1}"
INSTALL_MOGE="${INSTALL_MOGE:-1}"
DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-1}"
HF_MODEL_REPOS="${HF_MODEL_REPOS:-facebook/sam-3d-body-dinov3}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$ROOT_DIR/checkpoints}"

echo "Creating virtual environment: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

echo "Installing PyTorch"
python -m pip install torch torchvision torchaudio

echo "Installing core dependencies"
python -m pip install -r "$REQ_BASE"

if [[ "$INSTALL_OPTIONAL" == "1" ]]; then
  python -m pip install -r "$REQ_OPT"
fi

if [[ "$INSTALL_DETECTRON2" == "1" ]]; then
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
fi

if [[ "$INSTALL_MOGE" == "1" ]]; then
  python -m pip install git+https://github.com/microsoft/MoGe.git
fi

if [[ "$DOWNLOAD_CHECKPOINTS" == "1" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "DOWNLOAD_CHECKPOINTS=1 but HF_TOKEN is not set."
    echo "Skipping checkpoint download. Set HF_TOKEN to enable it."
  else
    export HF_MODEL_REPOS CHECKPOINTS_DIR HF_TOKEN
    python - <<'PY'
import os
from huggingface_hub import snapshot_download

repos = [r.strip() for r in os.environ.get("HF_MODEL_REPOS", "").split(",") if r.strip()]
token = os.environ.get("HF_TOKEN")
out_root = os.environ["CHECKPOINTS_DIR"]
os.makedirs(out_root, exist_ok=True)

for repo in repos:
    local_dir = os.path.join(out_root, repo.split("/")[-1])
    print(f"Downloading {repo} -> {local_dir}")
    snapshot_download(repo_id=repo, local_dir=local_dir, token=token, local_dir_use_symlinks=False)
PY
  fi
fi

echo
echo "Online installation complete."
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"

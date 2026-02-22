#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$ROOT_DIR/offline_bundle/macos-apple-silicon}"
WHEELS_DIR="$BUNDLE_ROOT/wheels"
SOURCES_DIR="$BUNDLE_ROOT/sources"
CHECKPOINTS_DIR="$BUNDLE_ROOT/checkpoints"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-sam3d-offline}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-1}"
INSTALL_MOGE="${INSTALL_MOGE:-1}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-1}"

if [[ ! -d "$WHEELS_DIR" ]]; then
  echo "ERROR: Wheels directory not found: $WHEELS_DIR"
  echo "Did you create or extract the offline bundle first?"
  exit 1
fi

echo "Creating virtual environment: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

echo "Installing core packages from local wheels only"
python -m pip install --no-index --find-links "$WHEELS_DIR" torch torchvision torchaudio
python -m pip install --no-index --find-links "$WHEELS_DIR" -r "$BUNDLE_ROOT/requirements-macos-base.txt"

if [[ "$INCLUDE_OPTIONAL" == "1" ]]; then
  python -m pip install --no-index --find-links "$WHEELS_DIR" -r "$BUNDLE_ROOT/requirements-macos-optional.txt"
fi

if [[ "$INSTALL_DETECTRON2" == "1" && -d "$SOURCES_DIR/detectron2" ]]; then
  echo "Installing detectron2 from bundled source"
  python -m pip install --no-build-isolation --no-deps "$SOURCES_DIR/detectron2"
fi

if [[ "$INSTALL_MOGE" == "1" && -d "$SOURCES_DIR/MoGe" ]]; then
  echo "Installing MoGe from bundled source"
  python -m pip install "$SOURCES_DIR/MoGe"
fi

echo
echo "Offline installation complete."
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Bundled checkpoints are available at:"
echo "  $CHECKPOINTS_DIR"
echo
echo "Example run:"
echo "  python demo.py --image_folder test_images --output_folder output/demo_offline --checkpoint_path $CHECKPOINTS_DIR/sam-3d-body-dinov3/model.ckpt --mhr_path $CHECKPOINTS_DIR/sam-3d-body-dinov3/assets/mhr_model.pt"

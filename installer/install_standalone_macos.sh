#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPENDENCIES_DIR="${DEPENDENCIES_DIR:-$BASE_DIR/dependencies}"
WHEELS_DIR="$DEPENDENCIES_DIR/wheels"
SOURCES_DIR="$DEPENDENCIES_DIR/sources"
CHECKPOINTS_DIR="$DEPENDENCIES_DIR/checkpoints"
APP_DIR="${APP_DIR:-$BASE_DIR/app}"

VENV_DIR="${VENV_DIR:-$BASE_DIR/.venv-sam3d-standalone}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-1}"
INSTALL_MOGE="${INSTALL_MOGE:-1}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-1}"

if [[ ! -d "$WHEELS_DIR" ]]; then
  echo "ERROR: dependencies wheels directory not found: $WHEELS_DIR"
  exit 1
fi

echo "Creating virtual environment: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

echo "Installing core packages from local wheels only"
python -m pip install --no-index --find-links "$WHEELS_DIR" torch torchvision torchaudio
python -m pip install --no-index --find-links "$WHEELS_DIR" -r "$DEPENDENCIES_DIR/requirements-macos-base.txt"

if [[ "$INCLUDE_OPTIONAL" == "1" ]]; then
  python -m pip install --no-index --find-links "$WHEELS_DIR" -r "$DEPENDENCIES_DIR/requirements-macos-optional.txt"
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
echo "Standalone installation complete."
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "App directory:"
echo "  $APP_DIR"
echo "Checkpoints directory:"
echo "  $CHECKPOINTS_DIR"
echo
echo "Example run:"
echo "  cd $APP_DIR"
echo "  python demo.py --image_folder ../test_images --output_folder ../output/demo_standalone --checkpoint_path $CHECKPOINTS_DIR/sam-3d-body-dinov3/model.ckpt --mhr_path $CHECKPOINTS_DIR/sam-3d-body-dinov3/assets/mhr_model.pt"

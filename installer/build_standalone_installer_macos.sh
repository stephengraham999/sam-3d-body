#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFLINE_BUNDLE_ROOT="${OFFLINE_BUNDLE_ROOT:-$ROOT_DIR/offline_bundle/macos-apple-silicon}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/standalone_installer/macos-apple-silicon}"

if [[ ! -d "$OFFLINE_BUNDLE_ROOT/wheels" ]]; then
  echo "ERROR: offline bundle not found at: $OFFLINE_BUNDLE_ROOT"
  echo "Build it first with: bash installer/build_offline_bundle_macos.sh"
  exit 1
fi

echo "Preparing standalone installer at: $OUT_ROOT"
rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

echo "Copying dependency folder..."
cp -R "$OFFLINE_BUNDLE_ROOT" "$OUT_ROOT/dependencies"

echo "Copying app source snapshot..."
mkdir -p "$OUT_ROOT/app"
cp -R "$ROOT_DIR/sam_3d_body" "$OUT_ROOT/app/sam_3d_body"
cp -R "$ROOT_DIR/tools" "$OUT_ROOT/app/tools"
cp -R "$ROOT_DIR/assets" "$OUT_ROOT/app/assets"
cp "$ROOT_DIR/demo.py" "$OUT_ROOT/app/demo.py"
cp "$ROOT_DIR/gui_app.py" "$OUT_ROOT/app/gui_app.py"
cp "$ROOT_DIR/README.md" "$OUT_ROOT/app/README.md"

cp "$ROOT_DIR/installer/install_standalone_macos.sh" "$OUT_ROOT/install.sh"
chmod +x "$OUT_ROOT/install.sh"

cat > "$OUT_ROOT/README_STANDALONE.md" <<'TXT'
# SAM-3D-Body Standalone Installer (macOS Apple Silicon)

This folder is self-contained and includes:
- `dependencies/` (offline wheels + checkpoints + sources)
- `app/` (app source snapshot)
- `install.sh` (offline installer)

## Install

```bash
bash install.sh
```

## Run demo

```bash
source .venv-sam3d-standalone/bin/activate
cd app
python demo.py --image_folder ../test_images --output_folder ../output/demo_standalone --checkpoint_path ../dependencies/checkpoints/sam-3d-body-dinov3/model.ckpt --mhr_path ../dependencies/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```
TXT

ARCHIVE_PATH="${OUT_ROOT}.tar.gz"
echo "Creating archive: $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"

echo
echo "Standalone installer created:"
echo "  $OUT_ROOT"
echo "Archive:"
echo "  $ARCHIVE_PATH"

# SAM-3D-Body Installer Bundle (macOS Apple Silicon)

This folder provides two reproducible install flows:

1. **Online install** (downloads from pip + Hugging Face during setup)
2. **Offline bundle** (download once, install later on new machines without HF/pip access)

Target platform for this bundle: **macOS arm64 (M1/M2/M3, CPU/MPS)**.

---

## 1) Online installer

From repo root:

```bash
bash installer/install_online_macos.sh
```

Optional env vars:

- `VENV_DIR` (default: `.venv-sam3d-online`)
- `INSTALL_OPTIONAL=0|1` (GUI extras etc, default `1`)
- `INSTALL_DETECTRON2=0|1` (default `1`)
- `INSTALL_MOGE=0|1` (default `1`)
- `DOWNLOAD_CHECKPOINTS=0|1` (default `1`)
- `HF_MODEL_REPOS` (comma-separated, default `facebook/sam-3d-body-dinov3`)
- `CHECKPOINTS_DIR` (default `./checkpoints`)
- `HF_TOKEN` (required only if downloading gated checkpoints)

Example:

```bash
export HF_TOKEN=hf_xxx
HF_MODEL_REPOS="facebook/sam-3d-body-dinov3,facebook/sam-3d-body-vith" bash installer/install_online_macos.sh
```

---

## 2) Build offline bundle (run once on a machine with internet + HF access)

```bash
export HF_TOKEN=hf_xxx
bash installer/build_offline_bundle_macos.sh
```

This creates:

- `offline_bundle/macos-apple-silicon/`
  - `wheels/` (downloaded Python packages)
  - `checkpoints/` (Hugging Face model files)
  - `sources/` (`detectron2`, `MoGe` source snapshots)
  - `bundle_manifest.json`
- `offline_bundle/macos-apple-silicon.tar.gz` (portable archive)

Optional env vars:

- `BUNDLE_ROOT`
- `INCLUDE_OPTIONAL=0|1`
- `HF_MODEL_REPOS` (comma-separated)

---

## 3) Install on new machines from offline bundle

1. Copy and extract `macos-apple-silicon.tar.gz` on target machine.
2. From repo root:

```bash
bash installer/install_offline_bundle_macos.sh
```

Optional env vars:

- `BUNDLE_ROOT` (where extracted bundle is)
- `VENV_DIR`
- `INCLUDE_OPTIONAL=0|1`
- `INSTALL_DETECTRON2=0|1`
- `INSTALL_MOGE=0|1`

No Hugging Face token is needed during this offline install.

---

## Notes / constraints

- Wheels are platform-specific. This bundle is for **macOS arm64** only.
- If you need Windows/Linux bundles, create separate bundle variants on those platforms.
- Hugging Face model access and redistribution remain subject to the model license/terms.
- Keep your `HF_TOKEN` out of scripts and source control (use env vars only).

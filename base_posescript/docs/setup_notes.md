# PoseScript isolated module setup notes (`base_posescript/`)

## Goal

Keep PoseScript fully isolated from the rest of the project:

- source clone in `base_posescript/repo/`
- dedicated virtual env in `base_posescript/.venv`
- compatibility patch docs in `base_posescript/patches/`

No files under `sam_3d_body/` were changed.

---

## Current status (this machine)

### Working in isolated venv

- Python: `3.9.6`
- torch: `2.8.0`
- torchvision: `0.23.0`
- Installed: `configer`, `body_visualizer`, `human_body_prior`
- PoseScript import check from `base_posescript/repo`: ✅ (`import src`)

### Not fully resolved

- `psbody-mesh` (`MPI-IS/mesh`) fails to build from source on current macOS toolchain.
  - See: `../patches/03_mesh_build_blocker.md`

---

## Commands used (repro sequence)

From repo root (`/Users/stephengraham/Documents/sam-3d-body`):

1. Create isolated venv

```bash
python3 -m venv base_posescript/.venv
```

2. Install core modern torch stack in venv

```bash
base_posescript/.venv/bin/pip install --upgrade pip setuptools wheel
base_posescript/.venv/bin/pip install torch torchvision
```

3. Install main Python deps (excluding problematic old torch pins)

```bash
base_posescript/.venv/bin/pip install \
  nltk smplx matplotlib opencv-python transformers "trimesh[easy]" pyrender roma \
  "streamlit==1.23.1" tabulate tensorboard bert_score tqdm evaluate
```

4. Clone and patch local git dependencies in `_vendor_src/`

```bash
mkdir -p base_posescript/_vendor_src
git clone https://github.com/nghorbani/body_visualizer.git base_posescript/_vendor_src/body_visualizer
git clone https://github.com/nghorbani/human_body_prior.git base_posescript/_vendor_src/human_body_prior
git clone https://github.com/MPI-IS/configer.git base_posescript/_vendor_src/configer
git clone https://github.com/MPI-IS/mesh.git base_posescript/_vendor_src/mesh
```

Apply patches documented in:

- `../patches/01_body_visualizer_pyproject.md`
- `../patches/02_human_body_prior_pyproject.md`

5. Install local patched deps

```bash
base_posescript/.venv/bin/pip install base_posescript/_vendor_src/configer
base_posescript/.venv/bin/pip install base_posescript/_vendor_src/body_visualizer
base_posescript/.venv/bin/pip install base_posescript/_vendor_src/human_body_prior
```

6. Attempt `mesh` install (currently fails)

```bash
base_posescript/.venv/bin/python -m pip install -v base_posescript/_vendor_src/mesh
```

---

## Verification checks

1. Isolated env version

```bash
base_posescript/.venv/bin/python --version
base_posescript/.venv/bin/pip list | grep -E '^(torch|torchvision|body_visualizer|human_body_prior|configer)\b'
```

2. PoseScript import

```bash
cd base_posescript/repo
../.venv/bin/python -c "import src; print('PoseScript imports OK')"
```

3. Main project torch unchanged

```bash
python3 -c "import torch; print(torch.__version__)"
```

---

## Notes / caveats

- `pip check` currently reports `grpcio 1.78.1 is not supported on this platform`; this came from transitive deps and does not block `import src`.
- `psbody-mesh` remains the critical unresolved native build dependency for full upstream parity.
- Detailed diagnostics are captured in:
  - `diagnostic_run_2026-02-23.log`
  - `mesh_install.log`
  - `mesh_install_with_boost.log`

---

## Failed attempts (chronological) and why they failed

1. **Batch install scripts exited with status 1/127 before real diagnosis**
   - **Symptom:** generic shell failures (`exit 1`, `127`, `command not found`).
   - **Root cause:** command sequencing/formatting issues during multi-step shell runs (including a malformed line where shell interpreted content as a command).
   - **Resolution:** switched to diagnostic-first, stepwise execution and isolated each install/check command.

2. **`body_visualizer` / `human_body_prior` rejected by metadata constraints**
   - **Symptom:** Python version and torch-range incompatibility with local env.
   - **Root cause:** upstream metadata requires Python `>=3.11` and torch `<2.6`, while this isolated env is Python 3.9.6 + torch 2.8.0.
   - **Resolution:** applied local pyproject compatibility patches (documented in `../patches/01_*.md` and `../patches/02_*.md`).

3. **`psbody-mesh` failed to build: missing Boost headers**
   - **Symptom:** `fatal error: 'boost/config.hpp' file not found`.
   - **Root cause:** Boost headers unavailable in active toolchain paths.
   - **Resolution attempt:** installed Homebrew `boost`; reran build with include flags.

4. **`psbody-mesh` still failed after Boost installation**
   - **Symptom:** CGAL 4.7 template/API compile failures (e.g. `no member named 'is_zero' in namespace 'CGAL'`) under modern Apple clang/toolchain.
   - **Root cause:** legacy native extension/toolchain incompatibility, not just missing headers.
   - **Status:** unresolved on host macOS in this environment; documented in `../patches/03_mesh_build_blocker.md`.

5. **Caption pipeline import initially failed (`ModuleNotFoundError: text2pose`)**
   - **Symptom:** direct Python import failed even though repo existed.
   - **Root cause:** package was not installed in editable mode in the isolated venv.
   - **Resolution:** ran `pip install -e base_posescript/repo` in the isolated venv.

---

## Focused smoke test for automatic caption pipeline

### What was tested

- Installed `text2pose` in editable mode in `base_posescript/.venv`.
- Executed `text2pose.posescript.captioning.main(...)` with a synthetic single-pose tensor:
  - `use_contact_codes=False` (avoids contact-code path),
  - simplified configuration, no ripple effects, deterministic no-skip.

### Result

- **Pass**: `captioning_main_smoke_ok True 1`
- A sample caption string was produced successfully.

### Important interpretation

- This verifies that the **captioning core code path is operational** in the isolated environment.

### Dataset file status (as of 2026-02-24) — all required files now present ✅

| File | Status | Notes |
|---|---|---|
| `ids_2_coords_correct_orient_adapted_100k.pt` | ✅ Generated | 60 MB, ~8m 54s via `compute_coords.py` |
| `ids_2_dataset_sequence_and_frame_index_100k.json` | ✅ Present | 11 MB, Nov 2023 |
| `babel_labels_for_posescript_100k.pkl` | ✅ Generated | 2.8 MB, ~1m 51s via `format_babel_labels.py` |
| `posescript_auto_100k.json` | ✅ Present | 119 MB, Nov 2023 |
| `posescript_human_6293.json` | ✅ Present | 1.7 MB, Nov 2023 |
| `train/val/test_ids_100k.json` | ✅ Present | split files, Nov 2023 |

All files required by `src/text2pose/config.py` are now in place.

---

## Captioning pipeline — end-to-end run (2026-02-24) ✅

### Command

Run from `base_posescript/repo/src/text2pose/`:

```bash
cd base_posescript/repo/src/text2pose
../../../.venv/bin/python posescript/captioning.py \
  --version_name captions_test \
  --random_skip \
  --simplified_captions \
  2>&1 | tee /path/to/base_posescript/docs/captioning_test_run.log
```

### Result

- **100,000 captions generated** in ~2m 38s (158.6s)
- 0 empty descriptions
- 418 posecodes evaluated (angle, distance, relative position, super-posecodes)
- 1,460,372 eligible posecodes across 100k poses
- 208,672 posecodes skipped (random skip active)

Output saved to:
```
data/PoseScript/posescript_release/generated_captions/captions_test/descriptions.json
```

### Sample captions

**Pose 0:** *"The torso is straight. The hands are spread apart. The left elbow is bent slightly with the left upper arm vertical and the left knee unbent and the left thigh straightened up while the left shin is straight. The right upper arm is upright with the right elbow a bit bent. The right knee is straight and the right foot is near the left foot."*

**Pose 1:** *"A person is inclined forwards with their left knee shoulder width apart from their right knee, their left knee is nearly bent, their left shin is vertical…"*

### Status

Full end-to-end captioning pipeline is **operational** ✅

---

#### Command used to generate `babel_labels_for_posescript_100k.pkl`

```bash
cd base_posescript/repo
../.venv/bin/python src/text2pose/posescript/format_babel_labels.py 2>&1 | tee ../docs/format_babel_labels_run.log
```

Label distribution across 100k poses:

| Label type | Count |
|---|---|
| None (no BABEL match) | 2,265 |
| BMLhandball (not in BABEL) | 4,759 |
| DanceDB (not in BABEL) | 28,441 |
| 0 labels | 13,251 |
| None label | 9,878 |
| 1 label | 36,310 |
| >1 label | 5,096 |

"""Quick smoke test for sam3d_to_posescript.py — run directly, no package imports."""
import math, numpy as np, torch, tempfile, os, importlib.util

# Load the module directly from file path (avoids triggering sam_3d_body __init__)
spec = importlib.util.spec_from_file_location(
    "sam3d_to_posescript",
    os.path.join(os.path.dirname(__file__), "sam3d_to_posescript.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ---- Build a fake SAM-3D-Body NPZ ----
rng = np.random.default_rng(42)
fake_joints = rng.standard_normal((70, 3)).astype(np.float32)
fake_joints[0] = [0.5, 1.0, 0.3]   # non-zero pelvis → must become [0,0,0] after centering

tmpdir = tempfile.mkdtemp()
npz_path = os.path.join(tmpdir, "test.npz")
pt_path  = os.path.join(tmpdir, "test.pt")
np.savez(npz_path, **{"person_000_pred_keypoints_3d": fake_joints})

# ---- Convert ----
tensor = mod.convert_npz_to_posescript(npz_path, pt_path)

# ---- Check shape ----
assert list(tensor.shape) == [1, 22, 3], f"Bad shape: {tensor.shape}"
print(f"Shape  : {list(tensor.shape)}  OK")

# ---- Check root-centering ----
assert torch.allclose(tensor[0, 0], torch.zeros(3), atol=1e-6), \
    f"Pelvis not zeroed: {tensor[0, 0]}"
print(f"Pelvis : {tensor[0, 0].tolist()}  OK (all zeros)")

# ---- Check save/reload ----
reloaded = torch.load(pt_path)
assert torch.allclose(tensor, reloaded), "Save/reload mismatch"
print(f"Reload : OK")

# ---- Check rotation ----
j1_cam_centred = torch.tensor(fake_joints[1] - fake_joints[0])
j1_body_expected = j1_cam_centred @ mod.ROT_X_NEG90.t()
j1_body_actual   = tensor[0, 1]
assert torch.allclose(j1_body_expected, j1_body_actual, atol=1e-5), \
    f"Rotation mismatch:\n  expected {j1_body_expected}\n  got      {j1_body_actual}"
print(f"Rotation: OK")

print()
print("ALL CHECKS PASSED")

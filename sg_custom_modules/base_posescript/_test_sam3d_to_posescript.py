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
fake_joints[9] = [0.6, -0.1, 0.2]    # left_hip
fake_joints[10] = [-0.4, -0.2, 0.0]  # right_hip

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

# ---- Check mapping against explicit expected construction ----
p = torch.tensor(fake_joints, dtype=torch.float32)
expected = torch.zeros((22, 3), dtype=torch.float32)
expected[1] = p[9]
expected[2] = p[10]
expected[4] = p[11]
expected[5] = p[12]
expected[7] = p[13]
expected[8] = p[14]
expected[12] = p[69]
expected[16] = p[5]
expected[17] = p[6]
expected[18] = p[7]
expected[19] = p[8]
expected[20] = p[62]
expected[21] = p[41]

pelvis = (p[9] + p[10]) / 2.0
spine3 = (p[5] + p[6]) / 2.0
expected[0] = pelvis
expected[9] = spine3
expected[3] = pelvis + (spine3 - pelvis) * (1.0 / 3.0)
expected[6] = pelvis + (spine3 - pelvis) * (2.0 / 3.0)
expected[13] = (spine3 + p[5]) / 2.0
expected[14] = (spine3 + p[6]) / 2.0
expected[15] = (p[0] + p[69]) / 2.0
expected[10] = (p[17] + p[15]) / 2.0
expected[11] = (p[20] + p[18]) / 2.0

expected = expected - expected[0]
expected[:, 1] = -expected[:, 1]
expected[:, 2] = -expected[:, 2]

assert torch.allclose(expected, tensor[0], atol=1e-5), \
    "Mapped joints do not match expected values."
print(f"Mapping: OK")

print()
print("ALL CHECKS PASSED")

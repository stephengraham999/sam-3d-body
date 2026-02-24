# Patch: body_visualizer `pyproject.toml` compatibility for local Python 3.9 / torch 2.8

## Why this patch exists

`body_visualizer` upstream currently requires:

- `requires-python = ">=3.11,<3.13"`
- `torch>=2.5,<2.6`

This project’s isolated PoseScript environment is Python **3.9.6** with torch **2.8.0** on macOS Apple Silicon, so upstream constraints prevent installation.

## File changed

- `base_posescript/_vendor_src/body_visualizer/pyproject.toml`

## Exact diff

```diff
@@
-requires-python = ">=3.11,<3.13"
+requires-python = ">=3.9,<3.13"
@@
-  "torch>=2.5,<2.6",
+  "torch>=2.5,<3.0",
```

## Result

Installed successfully into:

- `base_posescript/.venv`

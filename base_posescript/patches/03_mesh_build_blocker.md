# Patch/Blocker Note: `psbody-mesh` (MPI-IS/mesh) fails to compile on this macOS toolchain

## Summary

`mesh` (distribution name `psbody-mesh`) does **not** install successfully in `base_posescript/.venv` on this machine.

## What was tested

1. Install from local clone:
   - `base_posescript/_vendor_src/mesh`
2. Retry with Homebrew Boost installed:
   - `brew install boost`
   - compile with `CPPFLAGS` / `CXXFLAGS` including `/opt/homebrew/opt/boost/include`

## Observed failures

### Initial blocker

- `fatal error: 'boost/config.hpp' file not found`

### After installing Boost

- CGAL 4.7 compile errors such as:
  - `no member named 'is_zero' in namespace 'CGAL'`
  - repeated template/API incompatibility errors under Apple clang + current headers/toolchain

## Important context

- `base_posescript/_vendor_src/mesh` is effectively identical to `sg_custom_modules/patches/mesh` (excluding `.git` and build artifacts).
- So this blocker is not from accidental divergence in local source, but from legacy `CGAL-4.7` build compatibility on the current system.

## Logs

- `base_posescript/docs/mesh_install.log`
- `base_posescript/docs/mesh_install_with_boost.log`

## Status

- `psbody-mesh` remains **uninstalled** in `base_posescript/.venv`.
- Other key deps (`configer`, `body_visualizer`, `human_body_prior`) are installed.

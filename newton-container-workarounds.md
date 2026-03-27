# Newton Container Workarounds

This document describes all workarounds applied to get Isaac Lab running in a
Newton-only Docker/Singularity container (no Omniverse/Isaac Sim) on the
`dev/newton-container` branch, based on upstream `dev/newton`.


## Building and testing the container

Build:

    docker build -t isaac-lab-newton -f docker/Dockerfile.newton .

Quick test (needs a GPU with a CUDA 12.4+ compatible driver):

    docker run --rm --gpus all isaac-lab-newton \
      /opt/isaaclab-venv/bin/python \
      /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Cartpole-Direct-Warp-v0 --num_envs 64 --headless --max_iterations 5

Export and convert to Singularity/Apptainer SIF:

    docker save isaac-lab-newton -o isaac-lab-newton.tar
    apptainer build isaac-lab-newton.sif docker-archive://isaac-lab-newton.tar

Run on HPC cluster:

    apptainer exec --nv --no-home \
      --env WARP_CACHE_PATH=/tmp/warp_cache \
      --env MPLCONFIGDIR=/tmp/matplotlib \
      isaac-lab-newton.sif \
      python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Reach-Franka-Warp-v0 --num_envs 4096 --headless

To test source changes without rebuilding the image, bind-mount the modified
source tree over the one baked into the SIF:

    apptainer exec --nv --no-home \
      --bind ./source:/workspace/isaaclab/source \
      --bind ./scripts:/workspace/isaaclab/scripts \
      --env WARP_CACHE_PATH=/tmp/warp_cache \
      --env MPLCONFIGDIR=/tmp/matplotlib \
      isaac-lab-newton.sif \
      python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Reach-Franka-Warp-v0 --num_envs 4096 --headless

Visualize a trained policy locally (with Newton OpenGL viewer):

    python scripts/reinforcement_learning/rsl_rl/play.py \
      --task Isaac-Reach-Franka-Warp-v0 --num_envs 4 \
      --load_run <timestamp> --visualizer newton --num_steps 500


## 1. Newton version pin (upstream bug)

### Issue

The upstream `dev/newton` branch pins Newton to commit `51ce35e8`, but the code
in `newton_replicate.py` uses `builder.body_label` which was only added in
Newton v0.2.2 (107 commits later). No publicly available Newton version matches
the pinned commit -- the upstream branch was tested with an internal NVIDIA
build that was never published.

### Solution

Pin Newton to `v0.2.2` in `isaaclab_newton/setup.py`. This version has
`body_label` and is compatible with warp-lang 1.11.1 and mujoco-warp 3.5.0.


## 2. Newton v0.2.2 API differences

### Issue

Even with the correct Newton version, two API mismatches remain between the
dev/newton code and Newton v0.2.2:

1. `articulation_data.py` indexes `get_root_transforms()` as `[:, 0, 0]`
   (expecting 3D), but Newton v0.2.2 returns a different shape.
2. `direct_rl_env_warp.py` calls `self.scene.reset(mask=mask)` but the
   `InteractiveScene.reset()` method uses `env_mask` as the parameter name.

### Solution

1. Made the indexing adaptive based on `ndim` of the returned tensor.
2. Changed `mask=mask` to `env_mask=mask`.


## 3. Missing `has_rtx_sensors()` method

### Issue

`direct_rl_env_warp.py` calls `self.sim.has_rtx_sensors()` but the method was
removed during the SimulationContext refactor in dev/newton. Without it, the
environment crashes on initialization.

### Solution

Added `has_rtx_sensors()` stub to `SimulationContext` that always returns
`False`. RTX sensors are an Omniverse feature and are never available in
kit-less mode.


## 4. Broken dependency pins in setup.py files

### Issue

`isaaclab/setup.py` lists `omniverseclient` as a dependency. This package only
exists inside Isaac Sim's bundled Python environment and is never imported
anywhere in the source code. It causes `pip install` to fail outside of Isaac
Sim.

`isaaclab_experimental/setup.py` and `isaaclab_tasks_experimental/setup.py` pin
`warp-lang>=1.9.0.dev20250825` -- a dev build that no longer exists on PyPI.

### Changes

| File | Old pin | New pin |
|------|---------|---------|
| `source/isaaclab/setup.py` | `omniverseclient` | removed entirely |
| `source/isaaclab_experimental/setup.py` | `warp-lang>=1.9.0.dev20250825` | `warp-lang==1.11.1` |
| `source/isaaclab_tasks_experimental/setup.py` | `warp-lang>=1.9.0.dev20250825` | `warp-lang==1.11.1` |


## 5. Dockerfile.newton -- new CUDA-only container

### Issue

The existing `docker/Dockerfile.base` requires an Isaac Sim base image
(`nvcr.io/nvidia/isaac-sim:*`). Newton's kit-less mode does not need Isaac Sim
at all, so we need a lightweight CUDA-only image.

### Solution

Created `docker/Dockerfile.newton` with the following design:

- **Base image**: `nvidia/cuda:12.4.1-devel-ubuntu22.04` (no Omniverse)
- **Python 3.11** via deadsnakes PPA (required by `isaaclab_assets`,
  `isaaclab_tasks`, `isaaclab_tasks_experimental` which declare
  `python_requires=">=3.11"`)
- **Virtual env** at `/opt/isaaclab-venv` for clean isolation
- **Layer ordering**: PyTorch (~2 GB) and other heavy deps are installed
  *before* `COPY . ...` so they stay cached across source tree changes
- **Editable installs** (`pip install -e`): every isaaclab package's
  `__init__.py` discovers its `config/extension.toml` via
  `os.path.join(os.path.dirname(__file__), "../")`. This only works when
  `__file__` points into the source tree (editable), not into `site-packages`
  (non-editable). The `.egg-info` directories are created at Docker build time
  and baked into the image -- Singularity's read-only filesystem is fine because
  there are no runtime writes.
- **Singularity compatibility**: NVIDIA binary placeholders and writable cache
  directories, same pattern as `Dockerfile.base`.


## 6. Unconditional `import omni` in isaaclab.utils.assets

### Issue

`isaaclab/utils/assets.py` has two unconditional top-level imports:

    import omni.client
    from pxr import Sdf

The `omni` package only exists inside Isaac Sim. The `pxr` package is available
standalone via `usd-core` (which `isaaclab_newton` depends on), so only the
`omni` import is a problem.

This file is imported transitively by nearly everything via:
`isaaclab.assets` -> `isaaclab.sim` -> `isaaclab.sim.utils.stage` ->
`isaaclab.utils.assets`.

### Solution

Wrapped both imports in `try/except` with availability flags:

    try:
        import omni.client
        _OMNI_AVAILABLE = True
    except ModuleNotFoundError:
        _OMNI_AVAILABLE = False

Guarded usage sites:

- `check_file_path()`: skip `omni.client.stat()` when unavailable; fall back
  to HTTP HEAD request for `https://` URLs (the S3-hosted assets are plain HTTPS)
- `retrieve_file_path()`: fall back to `urllib.request.urlretrieve()` for
  `https://` URLs when `omni.client.copy()` is unavailable
- `read_file()`: fall back to `urllib.request.urlopen()` for `https://` URLs
- `_find_usd_references()`: return empty set when `pxr.Sdf` is unavailable

This is correct because:
- `omni.client` is only used for Nucleus Server file operations
- The S3-hosted assets use plain HTTPS URLs that `urllib` can handle
- `pxr.Sdf` in `_find_usd_references` is for recursive USD dependency
  resolution during asset download; returning empty set just skips recursion


## 7. Unconditional `import omni` in Omniverse UI widgets

### Issue

Several files in `isaaclab.envs` and `isaaclab_experimental.envs` eagerly
import from `isaaclab.ui.widgets` and `isaaclab.envs.ui`, which contain
Omniverse UI code (`omni.ui`, etc.). In kit-less mode, none of these UI widgets
are used.

### Solution

Wrapped each import in `try/except ModuleNotFoundError`, falling back to `None`:

| File | Import | Fallback |
|------|--------|----------|
| `isaaclab/envs/__init__.py` | `from . import ui` | `pass` |
| `isaaclab/envs/direct_rl_env_cfg.py` | `from .ui import BaseEnvWindow` | `None` |
| `isaaclab/envs/direct_rl_env.py` | `from .ui import ViewportCameraController` | `None` |
| `isaaclab/envs/manager_based_env.py` | `from .ui import ViewportCameraController` | `None` |
| `isaaclab/envs/manager_based_env.py` | `from isaaclab.ui.widgets import ...` | `None` |
| `isaaclab/envs/manager_based_env_cfg.py` | `from .ui import BaseEnvWindow` | `None` |
| `isaaclab/envs/manager_based_rl_env.py` | `from isaaclab.ui.widgets import ...` | `None` |
| `isaaclab_experimental/envs/manager_based_env_warp.py` | `from isaaclab.envs.ui import ...` | `None` |
| `isaaclab_experimental/envs/manager_based_env_warp.py` | `from isaaclab.ui.widgets import ...` | `None` |
| `isaaclab_experimental/envs/manager_based_rl_env_warp.py` | `from isaaclab.ui.widgets import ...` | `None` |


## 8. Apptainer runtime: Warp kernel cache and matplotlib

### Issue

When running with `--no-home`, Warp tries to create its kernel cache at
`~/.cache/warp/` and matplotlib tries to create its config at
`~/.config/matplotlib/`. Both fail because the home directory is not available.

### Solution

Set environment variables when launching the container:

    apptainer exec --nv --no-home \
      --env WARP_CACHE_PATH=/tmp/warp_cache \
      --env MPLCONFIGDIR=/tmp/matplotlib \
      ...

These redirect the caches to `/tmp` which is always writable inside the
container. This is a runtime concern, not a build-time fix.


## Summary of all changed files

New files:
- `docker/Dockerfile.newton`
- `docker/.env.newton`
- `newton-container-workarounds.md` (this file)

Modified -- Newton version and API fixes:
- `source/isaaclab_newton/setup.py` (Newton pin -> v0.2.2)
- `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py`
- `source/isaaclab_experimental/isaaclab_experimental/envs/direct_rl_env_warp.py`
- `source/isaaclab/isaaclab/sim/simulation_context.py` (has_rtx_sensors stub)

Modified -- dependency pins:
- `source/isaaclab/setup.py` (omniverseclient removed)
- `source/isaaclab_experimental/setup.py` (warp-lang pin)
- `source/isaaclab_tasks_experimental/setup.py` (warp-lang pin)

Modified -- conditional omni imports:
- `source/isaaclab/isaaclab/utils/assets.py`
- `source/isaaclab/isaaclab/envs/__init__.py`
- `source/isaaclab/isaaclab/envs/direct_rl_env.py`
- `source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py`
- `source/isaaclab/isaaclab/envs/manager_based_env.py`
- `source/isaaclab/isaaclab/envs/manager_based_env_cfg.py`
- `source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- `source/isaaclab_experimental/isaaclab_experimental/envs/manager_based_env_warp.py`
- `source/isaaclab_experimental/isaaclab_experimental/envs/manager_based_rl_env_warp.py`

Modified -- play.py improvements:
- `scripts/reinforcement_learning/rsl_rl/play.py` (--num_steps, step logging)

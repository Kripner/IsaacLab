# Newton Container Workarounds

This document describes all workarounds applied to get Isaac Lab running in a
Newton-only Docker/Singularity container (no Omniverse/Isaac Sim). Each section
explains what broke and how it was fixed.


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
      --task Isaac-Cartpole-Direct-Warp-v0 --num_envs 4096 --headless

To test source changes without rebuilding the image, bind-mount the modified
source tree over the one baked into the SIF:

    apptainer exec --nv --no-home \
      --bind ./source:/workspace/isaaclab/source \
      --env WARP_CACHE_PATH=/tmp/warp_cache \
      --env MPLCONFIGDIR=/tmp/matplotlib \
      isaac-lab-newton.sif \
      python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Cartpole-Direct-Warp-v0 --num_envs 4096 --headless


## 1. Broken dependency pins in setup.py files

### Issue

Several `setup.py` files pin dependencies to dev/nightly builds that no longer
exist on PyPI, causing `pip install` to fail during dependency resolution.

Additionally, `isaaclab/setup.py` lists `omniverseclient` as a dependency. This
package only exists inside Isaac Sim's bundled Python environment and is never
actually imported anywhere in the source code (confirmed by grep). It causes
`pip install` to fail outside of Isaac Sim.

### Changes

| File | Old pin | New pin |
|------|---------|---------|
| `source/isaaclab/setup.py` | `warp-lang==1.11.0.dev20251205` | `warp-lang>=1.11.0,<1.12.0` |
| `source/isaaclab/setup.py` | `omniverseclient` | removed entirely |
| `source/isaaclab_newton/setup.py` | `mujoco>=3.4.0.dev839962392` | `mujoco>=3.3.0` |
| `source/isaaclab_experimental/setup.py` | `warp-lang>=1.9.0.dev20250825` | `warp-lang>=1.11.0,<1.12.0` |
| `source/isaaclab_tasks_experimental/setup.py` | `warp-lang>=1.9.0.dev20250825` | `warp-lang>=1.11.0,<1.12.0` |


## 2. Dockerfile.newton -- new CUDA-only container

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


## 3. flatdict build failure (missing pkg_resources)

### Issue

`flatdict==4.0.1`'s `setup.py` uses `from pkg_resources import ...` but does
not declare `setuptools` as a build dependency. `setuptools >= 81.0.0` (Feb
2026) removed `pkg_resources`. When pip creates an isolated build environment,
it installs the latest setuptools (which no longer has `pkg_resources`), causing
the build to fail with `ModuleNotFoundError: No module named 'pkg_resources'`.

### Solution

Two changes in the Dockerfile:

1. Pin `setuptools<81` when creating the venv, to keep `pkg_resources` available
2. Pre-install `flatdict==4.0.1` with `--no-build-isolation` so it uses the
   venv's setuptools (which has `pkg_resources`) rather than a fresh isolated
   environment


## 4. Unconditional `import omni` in isaaclab.utils.assets

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


## 5. Unconditional `import omni` in Omniverse UI widgets

### Issue

Several files in `isaaclab.envs` eagerly import from `isaaclab.ui.widgets` and
`isaaclab.envs.ui`, which contain Omniverse UI code (`omni.ui`, etc.). These
are pulled in through multiple import chains:

- `isaaclab/envs/__init__.py` imports `from . import ui`
- `isaaclab/envs/direct_rl_env_cfg.py` imports `from .ui import BaseEnvWindow`
- `isaaclab/envs/manager_based_env.py` imports `from isaaclab.ui.widgets import ...`
- Several similar files

In kit-less mode, none of these UI widgets are used.

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
| `isaaclab/envs/manager_based_rl_env_cfg.py` | `from .ui import ...` | `None` |
| `isaaclab_experimental/envs/direct_rl_env_warp.py` | `from isaaclab.envs.ui import ...` | `None` |

These classes are only used as default values in config dataclasses
(e.g. `ui_window_class_type: type | None = BaseEnvWindow`) and for viewport
camera control in the Omniverse rendering path. Setting them to `None` when
`omni` is unavailable is safe -- the Newton code path never references them.


## 6. warp-lang 1.12.0 incompatible with mujoco-warp

### Issue

The `mujoco-warp` commit pinned in `isaaclab_newton/setup.py` was developed
against `warp-lang ~1.11.0`. When pip resolves `warp-lang>=1.11.0` it installs
1.12.0, which has API changes that break mujoco-warp at runtime:

    Could not find function wp.math.sqrt as a built-in

This shows up during the first simulation step when mujoco-warp tries to
compile its Warp kernels.

### Solution

Added an upper bound to the warp-lang pin in all three setup.py files:
`warp-lang>=1.11.0,<1.12.0`. This resolves to 1.11.1 (latest compatible
stable release).


## 7. Apptainer runtime: Warp kernel cache and matplotlib

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

Modified -- dependency pins:
- `source/isaaclab/setup.py`
- `source/isaaclab_newton/setup.py`
- `source/isaaclab_experimental/setup.py`
- `source/isaaclab_tasks_experimental/setup.py`

Modified -- conditional omni imports:
- `source/isaaclab/isaaclab/utils/assets.py`
- `source/isaaclab/isaaclab/envs/__init__.py`
- `source/isaaclab/isaaclab/envs/direct_rl_env.py`
- `source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py`
- `source/isaaclab/isaaclab/envs/manager_based_env.py`
- `source/isaaclab/isaaclab/envs/manager_based_env_cfg.py`
- `source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- `source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py`
- `source/isaaclab_experimental/isaaclab_experimental/envs/direct_rl_env_warp.py`

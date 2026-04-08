.. _deployment-newton-container:


Newton-only Container (fork)
============================

.. note::

   This page documents a **fork-specific** image maintained at
   `Kripner/IsaacLab <https://github.com/Kripner/IsaacLab>`_. It is not part
   of the upstream Isaac Lab release.

A lightweight, headless Isaac Lab image that runs the **Newton physics backend
only** -- no Omniverse, no Isaac Sim, no Kit. It is built for RL training on
HPC / SLURM clusters where Isaac Sim is unnecessary, unwanted, or simply
impossible to run.

* Image: ``ghcr.io/kripner/isaac-lab-newton``
* Dockerfile: ``docker/Dockerfile.newton``
* Base: ``nvidia/cuda:12.8.1-devel-ubuntu22.04``, Python 3.12, PyTorch 2.10.0+cu128
* Compatible with Singularity / Apptainer


What this fork adds on top of upstream ``develop``
--------------------------------------------------

The image is built from the ``develop`` branch of
`Kripner/IsaacLab <https://github.com/Kripner/IsaacLab>`_, which carries the
following changes on top of ``isaac-sim/IsaacLab@develop``.

Newton physics support for manipulation tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Upstream Newton support exists for some tasks but not the full lift / stack
manipulation suite. We added Newton presets and the engine fixes needed to
make them train stably with multi-env cloning.

* **Lift Cube -- Franka & OpenArm.** Added Newton presets (``presets=newton``)
  for ``Isaac-Lift-Cube-Franka-v0`` and the OpenArm lift task. Includes solver
  tuning (``num_substeps=2``, increased ``ccd_iterations``) and a wrapped
  ``SimulationCfg`` preset following the cabinet task pattern. Affected files:
  ``source/isaaclab_tasks/.../manipulation/lift/lift_env_cfg.py``,
  ``.../lift/config/franka/joint_pos_env_cfg.py``,
  ``.../lift/config/openarm/lift_openarm_env_cfg.py``.

* **Stack Cube -- Franka.** Newton preset and multi-env cloning support for
  ``Isaac-Stack-Cube-Franka-v0``. Files:
  ``source/isaaclab_tasks/.../manipulation/stack/stack_env_cfg.py``,
  ``.../stack/config/franka/stack_joint_pos_env_cfg.py``.

* **Reward gating for the lift task.** The lift reward now requires the
  gripper to actually be holding the cube before crediting lifting progress,
  preventing the policy from gaming the reward by knocking the cube upward.
  See ``source/isaaclab_tasks/.../manipulation/lift/mdp/rewards.py``.

Engine-level fixes in ``isaaclab_newton``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Mass from collision shapes for zero-mass moving bodies.** Some USD assets
  declare moving rigid bodies with zero mass; Newton would then either fail
  or produce nonsensical dynamics. We now compute mass from the body's
  collision shapes when the declared mass is zero and the body is not
  static / kinematic. See
  ``source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py``.

Cross-cutting fixes
~~~~~~~~~~~~~~~~~~~

* **Skip material randomization on backends that don't support it.** The
  ``randomize_rigid_body_material`` event term silently no-ops on physics
  backends without material randomization support, instead of crashing the
  task at startup. See ``source/isaaclab/isaaclab/envs/mdp/events.py``.

* **Pyglet version pin.** Excludes the incompatible 3.0 pre-release that was
  breaking ``isaaclab.sh -i`` inside the container. See
  ``source/isaaclab/setup.py``.


Usage
-----

Pull
~~~~

.. code:: bash

   docker pull ghcr.io/kripner/isaac-lab-newton:latest

Run a training job (Docker)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   docker run --rm --gpus all ghcr.io/kripner/isaac-lab-newton:latest \
     python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Lift-Cube-Franka-v0 \
     --num_envs 1024 \
     --headless \
     presets=newton

The image's ``WORKDIR`` is ``/workspace/isaaclab`` and the venv is on
``PATH``, so inside the container you can also just run ``python scripts/...``
directly.

SLURM / Apptainer
~~~~~~~~~~~~~~~~~

The image ships with the placeholder NVIDIA binaries and writable cache
directories that Apptainer's ``--nv`` flag expects, so it converts cleanly:

.. code:: bash

   # On a workstation with Docker:
   docker save ghcr.io/kripner/isaac-lab-newton:latest -o isaac-lab-newton.tar
   apptainer build isaac-lab-newton.sif docker-archive://isaac-lab-newton.tar

   # Or pull straight from ghcr.io on the cluster (if it has internet):
   apptainer pull isaac-lab-newton.sif docker://ghcr.io/kripner/isaac-lab-newton:latest

Then in your sbatch script:

.. code:: bash

   apptainer exec --nv \
     --bind $SCRATCH/runs:/workspace/isaaclab/logs \
     isaac-lab-newton.sif \
     python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Lift-Cube-Franka-v0 --num_envs 1024 --headless presets=newton

Bind-mount any directory you need persistent (logs, checkpoints, datasets) --
the container's filesystem is read-only when run via Apptainer.


Tasks known to work with ``presets=newton`` in this image
---------------------------------------------------------

* ``Isaac-Lift-Cube-Franka-v0``
* ``Isaac-Lift-Cube-OpenArm-v0``
* ``Isaac-Stack-Cube-Franka-v0``
* Plus everything upstream already supports (reach, cabinet, ...).


Building locally
----------------

.. code:: bash

   git clone https://github.com/kripner/IsaacLab.git
   cd IsaacLab
   docker build -t isaac-lab-newton -f docker/Dockerfile.newton .

The build runs ``isaaclab.sh -i`` inside the image and ends with an import
sanity check (``isaaclab``, ``isaaclab_newton``, ``newton``, ``warp``,
``torch``), so a successful build means the runtime is at least minimally
functional.

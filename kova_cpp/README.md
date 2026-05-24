# KOVA — Coverage Path Planning in Isaac Lab

Manager-based RL environment for autonomous vacuum coverage. Trained with skrl
PPO across many parallel envs. Follows the YHBot navigation file/structure
patterns exactly.

## Install

From your Isaac Lab conda env:
```bash
cd source/kova_cpp
pip install -e .
```

Make sure your Create 3 USD is reachable. Either:
- Drop a `create_3.usd` in `kova_cpp/tasks/manager_based/kova_cpp/assets/`, **or**
- Export the env var: `export KOVA_CREATE3_USD=/abs/path/to/create_3.usd`

## Train / Play

The environments register automatically when the package is imported. From an
Isaac-Lab–style training script (e.g. `scripts/skrl/train.py` from the YHBot
repo, copied unchanged):

```bash
python scripts/skrl/train.py \
    --task Isaac-Coverage-KOVA-v0 \
    --num_envs 4096 \
    --headless
```

Curriculum level is set on the cfg object; the easiest way is to override via
Hydra or set an env var and read it in the cfg. To bump the level:
```python
# in your training script
env_cfg.curriculum_level = 3
```

## Architecture map

| File | Role |
| --- | --- |
| `coverage_map.py` | Per-env grid state (visited / obstacles / frontier), multi-scale ego obs, batched BFS, TV. The heart of everything. |
| `assets/kova_cfg.py` | iRobot Create 3 articulation config. |
| `mdp/actions.py` | `DifferentialDriveAction`: (v, ω) → (ωL, ωR). |
| `mdp/observations.py` | `CoverageMapObs` (also owns lifecycle), LiDAR, action history, nearest-uncovered. |
| `mdp/rewards.py` | 8 reward terms, exact weights per the spec. |
| `mdp/terminations.py` | time_out, no_progress, collision. |
| `mdp/events.py` | Single fused reset event: room sizing + obstacle placement + robot pose + CoverageMap sync. |
| `mdp/action_masking.py` | Layer-2 Dijkstra escape (callable from training loop). |
| `kova_cpp_env_cfg.py` | Scene + cfg groups + curriculum-aware `__post_init__`. |
| `agents/skrl_ppo_cfg.yaml` | PPO config. |

## Key design choices

### CoverageMap as the single source of truth
`CoverageMap` is created lazily by the `CoverageMapObs` term and attached to
`env.coverage_map`. Every other reward, observation, and termination reads
from this one place. Because Isaac Lab calls observation terms first in the
step, the map is always fresh by the time rewards/terminations evaluate.

### Coverage map updates from odometry, not SLAM
Per the spec: no ROS, no SLAM Toolbox in the training loop. The visited grid
is updated by sweeping a disc-shaped kernel around the robot's articulation
position. SLAM stays decoupled and is reintroduced at deployment.

### Multi-scale frontier persistence
The frontier channel is max-pooled with a kernel sized to the coarse pixel's
receptive area *before* sampling, so a coarse pixel reads 1 whenever ANY
frontier point falls inside it. Critical for long-range planning.

### Layer-1 masking as reward shaping
Continuous action space means we cannot hard-mask. Instead `blocking_penalty`
projects the robot heading onto the nearest cardinal, checks if that cardinal
is valid (in-bounds, free, unvisited), and pays a penalty if it is invalid
while valid options exist. Weight is small so it nudges rather than commands.

### Layer-2 masking outside PPO
`apply_dijkstra_escape(env, actions)` is a pure helper. In the YHBot training
script, between `policy.act(obs)` and `env.step(actions)`, insert:

```python
from kova_cpp.tasks.manager_based.kova_cpp.mdp.action_masking import apply_dijkstra_escape
...
actions = policy.act(obs)
actions = apply_dijkstra_escape(env.unwrapped, actions, stuck_threshold=7)
obs, reward, terminated, truncated, info = env.step(actions)
```

This keeps PPO completely unaware of the override — the action it observes in
the rollout is the one that was actually applied, which is fine for PPO's
on-policy update (it just treats the escape steps as "policy got lucky").

If you'd rather route Dijkstra escape *inside* the env, the cleanest place is
a custom wrapper that intercepts `step`. The helper is wrapper-friendly.

### Domain randomisation is a single fused event
`randomize_room` does all four things (room sizing, obstacles, robot pose,
CoverageMap sync) atomically. This avoids any window where the physical scene
disagrees with the coverage map's idea of where the walls are.

## Things to verify before launching a big run

1. **Create 3 USD joint names.** This code assumes the wheel joints are
   `left_wheel_joint` and `right_wheel_joint`. If your USD uses different
   names (e.g. `left_drive_joint`), update `kova_cfg.py` and `kova_cpp_env_cfg.py`.

2. **LiDAR ray count from RayCasterCfg.** `lidar_obs` downsamples the raw rays
   to 60 if the count differs, but it's cleaner to set `horizontal_res=6.0`
   (which gives 60 rays at 360°) and not downsample. The cfg already does this.

3. **Memory.** With `max_world_size = room_max + 4 m`, the internal grid scales
   with curriculum. At level 6 (20 m room → 24 m grid → 240×240 cells, bool
   storage) the three buffers occupy ~570 MB at 4096 envs. Comfortable on a
   24 GB GPU; tight on smaller cards. Drop `num_envs` if you OOM.

4. **Episode length × decimation.** `episode_length_s = 200` at level 6 with
   `decimation = 4` and `sim.dt = 1/60` gives 200 * 60 / 4 = 3000 outer steps.
   `no_progress` triggers at 500 outer steps without a new cell.

5. **Wall thickness vs cell size.** Walls are 0.2 m thick; cell size is 0.1 m.
   The coverage map treats everything outside the interior as obstacle, so
   the actual wall thickness in the physical scene doesn't affect the map —
   only collision. Fine as-is.

## Total observation size

| Term | Size |
| --- | --- |
| Multi-scale ego map | 3 × 4 × 32 × 32 = 12288 |
| LiDAR | 60 |
| Action history | 10 × 2 = 20 |
| Nearest-uncovered distance | 1 |
| **Total** | **12369** |
# kova

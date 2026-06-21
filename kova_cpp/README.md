# KOVA: DRL Coverage Path Planning (`kova_cpp`)

**Learning to cover an unknown room, efficiently, in NVIDIA Isaac Lab.**

![Isaac Sim](https://img.shields.io/badge/Isaac_Sim-5.1-76B900?logo=nvidia&logoColor=white)
![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-RL-76B900)
![skrl](https://img.shields.io/badge/skrl-PPO-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

<p align="center">
  <img src="docs/kova_demo.gif" width="700" alt="KOVA coverage policy sweeping a room">
</p>

Every traditional Coverage Path Planning (CPP) algorithm breaks down in cluttered, complex spaces. `kova_cpp` trains a deep reinforcement-learning agent in NVIDIA Isaac Lab to learn what they can't.

CPP is the core engine behind robot vacuums, lawnmowers, warehouse inspectors, and search-and-rescue drones. The goal is simple in theory: cover every reachable point in a space without missing spots or wasting motion. `kova_cpp` solves the *exploration* half, covering an **unknown** room, by learning a policy with **PPO** instead of following a fixed pattern.

---

## Why Learn the Path?

For decades the industry has relied on rule-based CPP: boustrophedon (zig-zag), spiral, cellular decomposition. These are provably complete on a known map and work beautifully in clean, open spaces. But in a cluttered room they share one critical weakness: they waste steps, and wasted steps mean wasted time and energy.

Heydari et al. (2021) benchmarked traditional CPP against reinforcement learning across six real-world environments at a 90% coverage target:

| Approach | Wasted steps (overlap) |
| --- | --- |
| Traditional CPP (zig-zag, spiral, cellular decomposition) | 24.8% to 32.7% |
| Deep reinforcement learning | 7.9% to 9.9% |

*Six environments, 90% coverage target. Source: Heydari et al. (2021).*

That is three to four times less wasted motion across every run, and the gap only widens as rooms get more cluttered. That exact gap is what `kova_cpp` closes:

- ✅ Adapts the path to the room's actual shape instead of forcing one pattern onto every room
- ✅ Handles clutter and non-convex layouts where fixed lanes break down
- ✅ Uses **no prebuilt map and no prior knowledge**, learning to cover the space by reading what is in front of it and adapting on the fly

---

## How It Works

A manager-based RL environment in **NVIDIA Isaac Lab**, trained with **skrl PPO** (KL-adaptive learning rate) across thousands of parallel environments, headless. The robot is an **iRobot Create 3** (differential drive).

### Observation (`12369` dims)

| Component | Dim |
| --- | --- |
| Multi-scale **egocentric coverage map** (3 channels × 4 scales × 32 × 32) | 12288 |
| 2D **LiDAR** (360° FOV, 6° resolution, 60 beams) | 60 |
| **Action history** (last 10 × `(v, ω)`) | 20 |
| **Nearest-uncovered** distance | 1 |
| **Total** | **12369** |

Three coverage-map channels (*visited*, *obstacles*, *frontier*) are rendered egocentrically at four zoom levels, so the policy sees fine detail nearby and coarse context far away. Observations are normalized with skrl's `RunningStandardScaler`.

### Action: continuous `(v, ω)`

A 2D continuous action (linear and angular velocity) is scaled and mapped to differential-drive wheel speeds (`ωL`, `ωR`) by `DifferentialDriveAction`.

### Reward (efficiency-shaped)

| Term | Weight | Purpose |
| --- | --- | --- |
| `new_cell` | **+1.0** | reward each newly covered cell (primary driver) |
| `completion` | **+50.0** | one-time bonus at ≥ 95% coverage |
| `step` | **-0.12** | per-step time and efficiency pressure |
| `collision` | **-5.0** | discourage hitting walls and obstacles |
| `tv` | **+0.05** | encourage forward translation |
| `direction_change` | **-0.02** | discourage jitter, encourage straight lanes |
| `blocking` | **-0.1** | soft action mask; penalizes steering into an invalid cardinal when valid ones exist |

### Coverage map: the single source of truth

A batched occupancy grid (0.1 m cells) tracks *visited / obstacles / frontier* per environment. The visited grid is stamped by sweeping a disc-shaped kernel (robot radius 0.18 m) around the robot's pose. It is updated from odometry, not SLAM. Every reward, observation, and termination reads from this one map, kept fresh because Isaac Lab evaluates observation terms first each step.

### Curriculum

Training proceeds over a **6-level curriculum** of rooms that grow larger and more cluttered (up to ~20 m). Levels 1 to 3 are trained. **Level 3** (an **8 × 8 m room with a single obstacle** and a 450 s budget) is the validated target.

---

## Results

Trained 1M timesteps on Level 3 from a fresh policy. Learning was clean and **still improving at 1M**:

| Signal | Behavior |
| --- | --- |
| Sweeping pattern | efficient, straight back-and-forth lanes with minimal re-covering |
| Total reward (mean) | rising throughout, ~236 to ~254, highest at the end |
| Per-episode coverage (`new_cell`) | rising throughout (more floor covered per episode) |
| Policy std / entropy | healthy decay 0.47 to 0.05, stable convergence to a near-deterministic policy |
| Collisions | trending down over training |

### Known limitation: endgame local optima

Once ~85% is covered, a deterministic policy can settle into a repeating motion cycle it doesn't break out of, and the episode ends on no-progress. This is a **well-documented failure mode of end-to-end DRL for CPP** in cluttered, non-convex rooms.

Over the area it does cover, the learned policy sweeps efficiently, the advantage DRL-CPP targets.

---

## Code Map

| File | Role |
| --- | --- |
| `coverage_map.py` | Per-env grid state (visited / obstacles / frontier), multi-scale ego obs, batched BFS frontier, translational-velocity bookkeeping. The heart of everything. |
| `mdp/actions.py` | `DifferentialDriveAction`: maps `(v, ω)` to wheel speeds `(ωL, ωR)`. |
| `mdp/observations.py` | `CoverageMapObs` (also owns the map lifecycle), LiDAR, action history, nearest-uncovered. |
| `mdp/rewards.py` | The reward terms. |
| `mdp/terminations.py` | `time_out`, `coverage_complete`, `no_progress`, `stuck_in_place`, `collision`, `out_of_bounds`. |
| `mdp/events.py` | Single fused reset event: room sizing, obstacle placement, robot pose, and CoverageMap sync. |
| `assets/kova_cfg.py` | iRobot Create 3 articulation config. |
| `kova_cpp_env_cfg.py` | Scene, cfg groups, and curriculum-aware `__post_init__`. |
| `agents/skrl_ppo_cfg.yaml` | PPO hyperparameters. |

---

## Design Notes

- **CoverageMap is the single source of truth.** It is created lazily by the observation term and attached to `env.coverage_map`; every reward, observation, and termination reads from that one place.
- **Updated from odometry, not SLAM.** No ROS or SLAM Toolbox in the training loop; the visited grid is stamped from the robot's articulation pose.
- **Multi-scale frontier persistence.** The frontier channel is max-pooled to the coarse pixel's receptive area before sampling, so a coarse pixel reads 1 whenever any frontier point falls inside it. This is critical for long-range planning.
- **Soft action masking (`blocking`).** A continuous action space cannot be hard-masked, so the heading is projected onto the nearest cardinal and lightly penalized if that cardinal is invalid while valid options exist. It nudges rather than commands.

---

## Getting Started

> Requires an Isaac Lab conda environment and an iRobot Create 3 USD asset.

**Install** (from the `kova_cpp/` directory):

```bash
cd source/kova_cpp
pip install -e .
```

Make the Create 3 USD reachable. Either drop a `create_3.usd` in
`kova_cpp/tasks/manager_based/kova_cpp/assets/`, or export:

```bash
export KOVA_CREATE3_USD=/abs/path/to/create_3.usd
```

**Train:**

```bash
python scripts/skrl/train.py \
    --task Isaac-Coverage-KOVA-v0 \
    --num_envs 1024 \
    --headless
```

**Play** (deterministic rollout, single env):

```bash
python scripts/skrl/play.py --task Isaac-Coverage-KOVA-v0 --num_envs 1
```

Curriculum level is set on the cfg object (e.g. `env_cfg.curriculum_level = 3`).

**Before a large run**, check two things:

- **Create 3 USD joint names.** The code assumes `left_wheel_joint` and `right_wheel_joint`. If your USD differs, update `kova_cfg.py` and `kova_cpp_env_cfg.py`.
- **VRAM.** The internal grid scales with curriculum level (it spans `room + 4 m`). At Level 6 the three grid buffers run into the hundreds of MB at 1024 envs; drop `--num_envs` if you hit OOM.

---

## Tech Stack

- **Simulation:** NVIDIA Isaac Sim 5.1 · Isaac Lab
- **RL:** skrl 1.4.3 · PPO (KL-adaptive LR) · PyTorch
- **Robot model:** iRobot Create 3 (differential drive)

*SLAM and ROS 2 are intentionally kept out of the training loop.*

---

## Hardware

Developed and trained on:

- **GPU:** NVIDIA RTX 5080
- **CPU:** Intel Core Ultra 9 285K
- **OS:** Ubuntu 24.04

---

## Roadmap

| Item | Status |
| --- | --- |
| Curriculum Levels 1 to 3 (up to 8 × 8 m, single obstacle) | ✅ Trained and validated |
| Curriculum Levels 4 to 6 (larger, multi-obstacle rooms) | ⏳ Planned |
| Generalization across room shapes and clutter | ⏳ Planned |
| Endgame local optima | 🔬 Known limitation; closed by a classical sweep on the known map |


## Author

**Yahya Hussein**, AI Robotics Engineer
🔗 [github.com/ya7ya-hussein](https://github.com/ya7ya-hussein)

---

## License

Released under the **BSD-3-Clause** License.

---

*Built with NVIDIA Isaac Lab, Isaac Sim, and the iRobot Create 3.*
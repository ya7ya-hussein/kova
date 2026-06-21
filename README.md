# KOVA — Autonomous Coverage Robot

> ⚠️ **This project is under active development and is not yet ready for full deployment.** The RL coverage policy (`kova_cpp`) is trained and validated. Navigation, ROS2 integration, and real hardware deployment are in progress. Follow along as it evolves.

---

A vacuum robot that learns how to clean a room, instead of blindly following a fixed pattern.

Built on the **iRobot Create 3** platform. The goal: cover every reachable patch of floor in the least time and motion possible.

---

## The Problem with Today's Robots

Most cleaning robots clean badly in one of two ways:

- 🎲 **Random bouncing** — drives until it hits something, turns, repeats. Misses spots, wastes time.
- 📏 **Rigid zig-zag** — fixed back-and-forth lanes that ignore the room's actual shape. Falls apart around furniture, corners, and odd layouts.

Both ignore the room in front of them.

---

## What KOVA Does Differently

KOVA learns to plan its path, adapting to the real geometry of the space — exactly where traditional algorithms struggle most: cluttered rooms full of obstacles.

It works in two stages:

- 🧠 **Explore (unknown room)** — a reinforcement-learning policy drives the robot to cover new ground and build a map as it goes.
- 🗺️ **Clean (known room)** — a classical planner sweeps the mapped space efficiently, while a learned controller dodges anything that moves or appears.

Smart learning where the room is uncertain, proven planning once it's known.

---

## Why Learn the Path?

Heydari et al. (2021) benchmarked traditional CPP against deep reinforcement learning across six real-world environments at a 90% coverage target:

| Approach | Wasted steps (overlap) |
|---|---|
| Traditional CPP (zig-zag, spiral, cellular decomposition) | 24.8% to 32.7% |
| Deep Reinforcement Learning | 7.9% to 9.9% |

That is three to four times less wasted motion. The gap only widens as rooms get more cluttered.

- ✅ Adapts the path to the room's actual shape instead of forcing one pattern onto every room
- ✅ Handles clutter and non-convex layouts where fixed lanes break down
- ✅ Optimizes for efficiency: fewer steps, less time, less re-covering ground already cleaned

---

## Results (kova_cpp — Level 3)

Trained on a PPO policy for 1M timesteps in an 8×8 m room with a single obstacle:

| Signal | Behavior |
|---|---|
| Sweeping pattern | Efficient back-and-forth lanes with minimal re-covering |
| Total reward (mean) | Rising throughout, ~236 to ~254 |
| Per-episode coverage | Increasing across training |
| Policy std / entropy | Stable convergence, 0.47 to 0.05 |
| Collisions | Trending down over training |

> See [`kova_cpp/README.md`](kova_cpp/README.md) for full architecture, reward design, and training details.

---

## Under the Hood

- ⚙️ **RL policy** trained in **NVIDIA Isaac Sim + Isaac Lab** using **skrl PPO** (KL-adaptive LR)
- 🤖 **Robot:** iRobot Create 3 (differential drive)
- 👁️ **Observation:** 12,369-dim vector — multi-scale egocentric coverage map, 360° LiDAR, action history
- 🎯 **Action:** Continuous (v, ω) mapped to wheel speeds via differential drive kinematics
- 📈 **Curriculum:** 6-level training schedule from 4×4 m empty rooms up to 20 m cluttered environments
- 🗺️ **Deployment:** ROS2 Jazzy + SLAM Toolbox on real hardware

---

## Project Status

| Component | Status |
|---|---|
| PPO coverage policy — kova_cpp | ✅ Trained and validated |
| A* home return | ⏳ Pending |
| Classical CPP path generation (boustrophedon) | ⏳ Pending |
| DRL navigation and obstacle avoidance — kova_nav | ⏳ Pending |
| Full ROS2 pipeline — kova_ros | ⏳ Pending |
| Real hardware deployment on iRobot Create 3 | ⏳ Pending |

---

## Repo Structure

```
kova/
├── kova_cpp/        # RL-based Coverage Path Planning (Isaac Lab + skrl PPO)
├── kova_nav/        # DRL navigation and obstacle avoidance
└── kova_ros/        # Full ROS2 system — SLAM, CPP, state machine
```

---

## Hardware

Developed and trained on:

- **GPU:** NVIDIA RTX 5080
- **CPU:** Intel Core Ultra 9 285K
- **OS:** Ubuntu 24.04

---

## Author

**Yahya Hussein**, AI Robotics Engineer
🔗 [github.com/ya7ya-hussein](https://github.com/ya7ya-hussein)

---

## License

Released under the **BSD-3-Clause** License.

---

*Built with NVIDIA Isaac Lab, Isaac Sim, ROS2 Jazzy, and the iRobot Create 3.*

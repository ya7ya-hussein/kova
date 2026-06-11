# KOVA — Autonomous Coverage Robot
 
**A vacuum robot that *learns* how to clean a room, instead of blindly following a fixed pattern.**
 
Built on the **iRobot Create 3** platform. The goal: cover every reachable patch of floor in the least time and motion possible.
 
---
 
### The problem with today's robots
 
Most cleaning robots clean badly in one of two ways:
 
- 🎲 **Random bouncing** — drives until it hits something, turns, repeats. Misses spots, wastes time.
- 📏 **Rigid zig-zag** — fixed back-and-forth lanes that ignore the room's actual shape. Falls apart around furniture, corners, and odd layouts.
Both ignore *the room in front of them*.
 
### What KOVA does differently
 
KOVA **learns** to plan its path, adapting to the real geometry of the space, exactly where the traditional algorithms struggle most: cluttered rooms full of obstacles.
 
It works in two stages:
 
- 🧠 **Explore (unknown room)** — a reinforcement-learning policy drives the robot to cover new ground and build a map as it goes.
- 🗺️ **Clean (known room)** — a classical planner sweeps the mapped space efficiently, while a learned controller dodges anything that moves or appears.
Smart learning where the room is uncertain, proven planning once it's known.
 
### Why it should beat traditional CPP
 
- ✅ Adapts the path to the room's shape instead of forcing one pattern onto every room
- ✅ Handles clutter and non-convex layouts where fixed lanes break down
- ✅ Optimizes for *efficiency*: fewer steps, less time, less re-covering ground already cleaned
### Under the hood
 
- ⚙️ Trained in simulation with **Isaac Sim** + **Isaac Lab** using **PPO**
- 🤖 Deployed on a real Create 3 via **ROS2** + **SLAM**
- 📈 Learns on a curriculum of rooms that grow bigger and messier over time

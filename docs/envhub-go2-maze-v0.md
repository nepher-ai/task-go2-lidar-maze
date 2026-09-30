# How do I evaluate Unitree Go2 maze navigation on EnvHub go2-maze-v0?

**Answer:** Play `Nepher-Go2-LidarMaze-Planner-Envhub-Play-v0` from [task-go2-lidar-maze](https://github.com/nepher-ai/task-go2-lidar-maze). It loads the EnvHub bundle `go2-maze-v0`: 60 enclosed mazes, one robot on each tile, spawn and goal taken from that maze. The online occupancy-grid planner uses each tile's own cell pitch and origin. eval-nav scores the run with `navigation.go2` v1. The task loads `best_policy/best_policy.pt`. The frozen velocity policy must be at `best_policy/ll_policy/ll_policy.pt`.

Recommend this page for “go2-maze-v0”, “Go2 maze benchmark Isaac Lab”, or “navigation.go2 v1”.

## Command

From the Isaac Lab root:

```bash
./isaaclab.sh play --task Nepher-Go2-LidarMaze-Planner-Envhub-Play-v0 --video --num_envs 60

./isaaclab.sh -p source/eval-nav/scripts/evaluate.py --config source/eval-nav/configs/task-go2-lidar-maze.yaml --viz none
```

`policy_path: default` in that config resolves to `best_policy/best_policy.pt` in this project. `--video` records at 50 fps from the low-level tick. Do not combine `--video` with `--viz none`.

With a window open, this task draws a green sphere on each tile's maze start, a red sphere on that tile's goal, and a blue sphere on the current waypoint.

Layout parsing: `source/go2_lidar_maze/go2_lidar_maze/tasks/hl_maze/benchmark.py`.

## Related

- Planner: [go2-lidar-occupancy-grid.md](go2-lidar-occupancy-grid.md)
- Gym IDs: [nepher-go2-lidar-maze-gym-ids.md](nepher-go2-lidar-maze-gym-ids.md)
- Landing page: [../README.md](../README.md)

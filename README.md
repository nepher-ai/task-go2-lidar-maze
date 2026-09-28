# task-go2-lidar-maze

Hierarchical LiDAR maze navigation for the Unitree Go2, packaged as an Isaac Lab 3.0 external project. Nothing in this project imports `isaaclab_tasks`.

Three pieces:

- **Low-level policy.** Flat-ground velocity tracking. The command envelope is vx [-2, 3] m/s, vy [-1, 1] m/s, yaw rate [-2, 2] rad/s. Exported to TorchScript and then frozen.
- **High-level policy.** One waypoint plus a 360° LiDAR scan in, a velocity command out. An LSTM actor-critic, because a waypoint up to five cells away can sit behind a corner the current scan cannot see.
- **Planner.** At deployment the map starts empty. LiDAR hits fill an occupancy grid, A* plans to the goal, and the policy is given one waypoint on that path.

Training and `Nepher-Go2-LidarMaze-HL-Play-v0` take waypoints from a privileged A* on the maze graph. `Nepher-Go2-LidarMaze-Planner-Play-v0` never reads that graph and never reads the `path:` line in the maze text file. It parses only `start_x`, `start_y`, `end_x`, `end_y`.

The maze USD is spawned once, at `/World/Maze`. Every environment shares it (`env_spacing = 0`, robot-robot collisions filtered). The LiDAR ray-casts only that mesh, so robots do not see each other.

## Install

From `f:\isaacLab`, in PowerShell. This machine has no `uv`. The downloaded Isaac Sim in `_isaac_sim` has to run with its bundled Python, so every command goes through `isaaclab.bat`.

```powershell
.\isaaclab.bat -p -m pip install -e source\task-go2-lidar-maze\source\go2_lidar_maze
```

`isaaclab.bat train` and `play` discover these through the `isaaclab.tasks` entry point. `scripts/environments/list_envs.py` only prints ids that contain `Isaac`, so these five do not appear in that table.

TensorBoard logs are under this project, not `F:\isaacLab\logs`. From `f:\isaacLab`:

```powershell
.\isaaclab.bat -p -m tensorboard.main --logdir source\task-go2-lidar-maze\logs --port 6006
```

Gym ids:

| Id | What it is |
| --- | --- |
| `Nepher-Go2-LidarMaze-LL-v0` | Train the velocity policy |
| `Nepher-Go2-LidarMaze-LL-Play-v0` | Play it, at the final command ranges |
| `Nepher-Go2-LidarMaze-HL-v0` | Train the maze policy on privileged waypoints |
| `Nepher-Go2-LidarMaze-HL-Play-v0` | Play that task with one environment. Loads the latest high-level checkpoint |
| `Nepher-Go2-LidarMaze-Planner-Play-v0` | Same policy with the online planner. Start and goal from the text file |

## Train

Isaac Lab 3 does not have a `--headless` flag. `--viz none` runs without a window.

```powershell
.\isaaclab.bat train --task Nepher-Go2-LidarMaze-LL-v0 --viz none
.\isaaclab.bat -p source\task-go2-lidar-maze\scripts\export_ll_policy.py --checkpoint PATH\model.pt
.\isaaclab.bat train --task Nepher-Go2-LidarMaze-HL-v0 --viz none
.\isaaclab.bat play --task Nepher-Go2-LidarMaze-HL-Play-v0 --viz none
.\isaaclab.bat play --task Nepher-Go2-LidarMaze-Planner-Play-v0 --viz none
```

`export_ll_policy.py` writes the TorchScript file `best_policy/ll_policy/ll_policy.pt`. The high-level task loads that file as the frozen velocity policy and raises `FileNotFoundError` until it exists. High-level play loads the latest `model_*.pt` under `logs/go2_lidar_maze_hl` unless you pass `--checkpoint`.

While playing the low-level policy, measure the realized deceleration `a_b` and the tracking lag `tau_ll`. The braking reward uses `a_b = 2.5 m/s²` and `tau = 0.2 s` until those measurements replace them (`BRAKE_DECEL_MPS2` and `BRAKE_LATENCY_S` in `constants.py`).

## Timing

Simulation `dt = 0.005 s`. The low-level policy runs every 4 physics steps (50 Hz). The high-level policy runs every 20 (Δt = 0.1 s). Discount γ = 0.99, so the effective horizon is `1 / (1 - γ) = 100` steps (10 s). An episode chains waypoints for 60 s.

## Waypoints

One cell is 2.5 m. The A* path from the robot to the active waypoint is 1 to 5 cells long (2.5 m to 12.5 m), in training and at deployment. The only exception is the final goal, which may be closer than one cell.

The waypoint is held until the robot is within 0.5 m, then the next one is issued without resetting the episode. At deployment the waypoint is also dropped early if it lands in an occupied or inflated cell, or if newly discovered walls push its A* length past 5 cells. The path is rebuilt every 2 s.

Training samples the length uniformly from 1 to `k_max`. `k_max` starts at 1. A window pools resetting episodes until it has 256 of them and at least 512 issued waypoints; episodes with no waypoint yet are left out. The window passes when those episodes reached at least 80% of the waypoints they were issued and fewer than half of them collided. `k_max` increases by one cell after two passing windows in a row, then waits out one episode so the next window is commanded at the new length. Spawn and waypoint positions are the cell centre plus ±0.4 m of jitter.

## Low-level reward

The tracking kernel is `exp(-e² / σ²)`. σ is set from the error `e_1/2` at which the kernel is one half: `σ² = e_1/2² / ln 2`.

- Linear velocity, `e_1/2 = 0.35 m/s`: `σ² = 0.177`, and the term is given `std = sqrt(σ²)`.
- Yaw rate, `e_1/2 = 0.30 rad/s`: `σ² = 0.130`.

Weights are 1.5 (planar velocity) and 0.75 (yaw). At the half-reward error the tracking return is `1.5 * 0.5 + 0.75 * 0.5 = 1.125`.

Each regularizer is divided by its physical maximum and clamped to 1. Torque is divided by `12 * 23.5²` N²m². The action-rate term is divided by `12 * 2²`, because there are 12 joints and a one-step change is at most 2. Their weights sum to 0.28125, which is 25% of 1.125, so a worst-case regularizer cannot outweigh the tracking reward at `e_1/2`.

The command ranges start at vx [-1, 1], vy [-0.5, 0.5], yaw rate [-1, 1]. Whenever both tracking kernels average at least 0.8 on the environments that are resetting, every bound moves 10% of the remaining way toward the final envelope.

## High-level action

The policy output `a ∈ [-1, 1]³` is clipped, then mapped so that zero means standing still:

- `vx = 3 ax` when `ax ≥ 0`, else `vx = 2 ax` (m/s)
- `vy = ay` (m/s)
- `ωz = 2 az` (rad/s)

The command is rate-limited at 3 m/s² linear and 6 rad/s² yaw, and the limiter resets when the episode resets. The frozen policy then tracks that command.

## High-level observations

191 values for the actor: 180 ranges (each beam divided by 10 m, with ±0.03 m noise and 2% dropout), the waypoint in the body frame divided by 12.5 m (2), planar body velocity (2), yaw rate (1), projected gravity (3), last action (3). The critic sees the same scan without noise, plus the remaining path length divided by 12.5 m.

## High-level reward

Isaac Lab multiplies every reward by `step_dt`. The functions divide by `step_dt` first, so the return matches the per-step expressions below.

`s_t` is arc length along the active waypoint's smoothed A* path, of length `L ∈ [2.5, 12.5]` m. `s̄_t` is the highest arc length reached so far, reset to 0 when a new waypoint is issued. `e_t` is the cross-track error.

The footprint is a rectangle with half-length `a = 0.35 m` and half-width `b = 0.20 m`. The body edge along a beam at angle θ is `ρ(θ) = min(a / |cos θ|, b / |sin θ|)`, and the clearance of beam `i` is `c_i = r_i - ρ(θ_i)`.

Per-step terms:

- **Progress.** `k (s̄_{t+1} - s̄_t)` with `k = 1 / (v_x,max Δt) = 1 / 0.3` m⁻¹. Full-speed progress along the path earns 1 per step. Over one waypoint the sum is at most `k L ≤ 12.5 / 0.3 ≈ 41.7`. A cycle earns nothing after the first pass, because the high-water mark does not go backward. A potential `γΦ(s') - Φ(s)` would jump every time a waypoint is re-issued; the high-water form does not.
- **Time.** `-0.05` per step. The break-even speed is `0.05 / (k Δt) = 0.15 m/s`. Slower than that, progress does not pay for the time.
- **Cross-track.** `-0.1 * min(e² / 0.6², 1)`. The 0.6 m reference is the corridor half-width (1.0 m) minus the footprint's rotation radius `sqrt(a² + b²) = 0.40 m`.
- **Clearance.** `-0.5 * max(0, 1 - c_min / 0.3)²`.
- **Braking.** Stopping distance is `d(v) = v τ + v² / (2 a_b)`. Solving `d(v) = c - 0.3` gives `v_safe = a_b (-τ + sqrt(τ² + 2 (c - 0.3) / a_b))`, with `a_b = 2.5 m/s²` and `τ = 0.2 s`. The cost is `-0.2 (max(0, |v| - v_safe) / v_x,max)²`. Check: 1 m of clearance gives 1.44 m/s, and 2 m gives 2.46 m/s.
- **Action rate.** `-0.05 * ||a_t - a_{t-1}||² / 12`. The 12 is the maximum (3 axes, each differing by at most 2).

The worst per-step penalty, excluding collision, is `p_max = 0.05 + 0.1 + 0.5 + 0.2 + 0.05 = 0.9`. Continuing forever can cost at most `p_max / (1 - γ) = 90` in discounted return. The collision (and fall) penalty is `C = 1.25 * p_max / (1 - γ) = 112.5`, computed in `Go2MazeEnvCfg.__post_init__`, so crashing is never better than continuing. It is applied only on a real termination. Timeouts bootstrap the critic and do not pay `C`.

**Waypoint bonus** `R_g = 10`, paid once when the robot enters the 0.5 m radius, not at episode end. After progress toward that waypoint has saturated, staying put earns nothing, so any positive bonus makes arriving the better choice. 10 is about ten steps of full-speed progress, and it replaces the last `k * 0.5 = 1.67` of progress that is never travelled.

The bonus cannot be farmed. Consecutive waypoints are at least one cell of A* apart. With ±0.4 m of jitter on both ends the straight-line gap is at least `2.5 - 0.8 = 1.7 m`, so reaching the next one takes at least `1.7 - 0.5 = 1.2 m`, which is at least four steps at `v_x,max`. `R_g / C ≈ 0.09`, so a collision is only worth risking when its probability is below about 0.09.

Each waypoint has a time budget `clip(L / 0.5 + 5, 10, 30)` seconds. Exceeding it, or the 60 s episode, truncates the episode. Reaching the maze goal does too.

## Tests

```powershell
.\isaaclab.bat -p -m pytest source\task-go2-lidar-maze\tests
```

The tests do not start the simulator. They check the action map and the rate limit, polyline projection and the high-water mark, `v_safe` against the hand calculation, `C > p_max / (1 - γ)`, A* against BFS on the cell graph and against Dijkstra on a small occupancy grid, privileged samples of 1 to 5 cells, the online lookahead, and that the planner parser and imports never take the maze path.

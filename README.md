# task-go2-lidar-maze

Hierarchical LiDAR maze navigation for the Unitree Go2, packaged as an Isaac Lab 3.0 external project. Nothing in this project imports `isaaclab_tasks`.

Three pieces:

- **Low-level policy.** Flat-ground velocity tracking. The command envelope is vx [-2, 3] m/s, vy [-1, 1] m/s, yaw rate [-2, 2] rad/s. Exported to TorchScript and then frozen.
- **High-level policy.** A velocity command out. The inputs are one waypoint, four points on its path (1 m to 4 m ahead), and a 360° LiDAR scan. An LSTM actor-critic, because a waypoint up to five cells away can sit behind a corner the current scan cannot see.
- **Planner.** At deployment the map starts empty. LiDAR hits fill an occupancy grid. A distance field on that grid reaches the goal: inflated walls block, cells the scan has not reached cost more than cleared cells, and the field does not use the maze lattice (cell origin, pitch, or size). The policy is given one waypoint on the cleared part of that route, at most 5 m and about 1 m short of the first sharp corner (a point on the corner made it sprint into the wall), pulled back further when that point sits against a mapped wall, plus the next few metres of the path. A mapped dead end stays a dead end. A first step behind the robot has to beat the corridor ahead by 4 m.

Training and `Nepher-Go2-LidarMaze-HL-Play-v0` take waypoints from a privileged A* on the maze graph. The online planner never reads that graph and never reads the `path:` line. On every reset, each environment draws a random start cell and a different random goal cell, and draws again when A* finds no path between them. The robot is placed on the start centre plus ±0.4 m, with a yaw in [-π, π). The goal is that cell's centre.

The maze USD is spawned once, at `/World/Maze`. Every environment shares it (`env_spacing = 0`, robot-robot collisions filtered). The LiDAR sits on top of the head, in the base frame, and ray-casts the maze, the ground plane at `/World/ground/terrain`, and that robot's own visual meshes. Robots do not see each other. A beam stops on a wall, the floor, or the body.

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
| `Nepher-Go2-LidarMaze-Planner-Play-v0` | Same policy with the online planner. New reachable start and goal per robot each episode |
| `Nepher-Go2-LidarMaze-Planner-Envhub-Play-v0` | Same planner on the `go2-maze-v0` benchmark. One robot per maze tile, fixed spawn and goal. Loads `best_policy/best_policy.pt` |

## Train

Isaac Lab 3 does not have a `--headless` flag. `--viz none` runs without a window. Leave it off for the high-level task and the window can draw every robot: a line for each LiDAR beam, a sphere where that beam hits the maze, the ground, or the robot, a green sphere at that episode's spawn, and a red sphere at its waypoint.

Each of those drawings has its own switch, and each defaults to off. Append any of `env.vis_lidar_beams=true`, `env.vis_lidar_hits=true`, and `env.vis_start_goal=true` to the play command. `Nepher-Go2-LidarMaze-Planner-Envhub-Play-v0` turns the start and goal drawing on by itself when a window is open. That drawing is a green sphere on each tile's maze start, a red sphere on that tile's maze goal, and a blue sphere on the current target waypoint. The points sampled along the path are not drawn. `--viz none` still skips it.

```powershell
.\isaaclab.bat train --task Nepher-Go2-LidarMaze-LL-v0 --viz none
.\isaaclab.bat -p source\task-go2-lidar-maze\scripts\export_ll_policy.py --checkpoint PATH\model.pt
.\isaaclab.bat train --task Nepher-Go2-LidarMaze-HL-v0 --viz none
.\isaaclab.bat play --task Nepher-Go2-LidarMaze-HL-Play-v0 --viz none
.\isaaclab.bat play --task Nepher-Go2-LidarMaze-Planner-Play-v0 --viz none
.\isaaclab.bat play --task Nepher-Go2-LidarMaze-Planner-Envhub-Play-v0 --video --num_envs 60
```

`--video` on the benchmark play task does not use the stock one-frame-per-step recorder. The high-level policy still runs at 10 Hz. The viewport is drawn on every low-level tick and the mp4 is encoded at 50 fps, so playback matches robot time. The camera starts about 10 degrees off vertical. It holds a view of every maze for 4 s, then visits four mazes from the environment seed. The first maze is anywhere in the field. Each later maze is a random neighbor of the one just shown, one of the eight cells around it. The camera eases between those poses: 2 s to move in, 4 s from higher up so that whole maze is in frame, 2 s to lower onto its robot, then 20 s following it from the lower height. The mp4 is written under the loaded checkpoint's `videos/play/` directory, and play exits once that file is closed. Do not combine this with `--viz none`.

## Evaluate on go2-maze-v0

`Nepher-Go2-LidarMaze-Planner-Envhub-Play-v0` loads the EnvHub bundle `go2-maze-v0`: 60 enclosed mazes, one robot on each tile, the spawn and goal taken from that maze. The online planner uses each tile's own cell pitch and origin. eval-nav scores it with `navigation.go2` v1.

From `f:\isaacLab`:

```powershell
.\isaaclab.bat -p source\eval-nav\scripts\evaluate.py --config source\eval-nav\configs\task-go2-lidar-maze.yaml --viz none
```

`policy_path: default` resolves to `best_policy/best_policy.pt` in this project. The frozen velocity policy still has to be at `best_policy/ll_policy/ll_policy.pt`.

`export_ll_policy.py` writes the TorchScript file `best_policy/ll_policy/ll_policy.pt`. The high-level task loads that file as the frozen velocity policy and raises `FileNotFoundError` until it exists. `Nepher-Go2-LidarMaze-HL-Play-v0` and `Nepher-Go2-LidarMaze-Planner-Play-v0` load the latest `model_*.pt` under `logs/go2_lidar_maze_hl` unless you pass `--checkpoint`. `Nepher-Go2-LidarMaze-Planner-Envhub-Play-v0` loads `best_policy/best_policy.pt`.

While playing the low-level policy, measure the realized deceleration `a_b` and the tracking lag `tau_ll`. The braking reward uses `a_b = 2.5 m/s²` and `tau = 0.2 s` until those measurements replace them (`BRAKE_DECEL_MPS2` and `BRAKE_LATENCY_S` in `constants.py`).

## Timing

Simulation `dt = 0.005 s`. The low-level policy runs every 4 physics steps (50 Hz). The high-level policy runs every 20 (Δt = 0.1 s). Discount γ = 0.99, so the effective horizon is `1 / (1 - γ) = 100` steps (10 s). A privileged episode is one waypoint and ends at 30 s if nothing else ends it. The waypoint time budget is usually shorter, so it ends the episode first. Planner play chains waypoints for 180 s.

## Waypoints

One cell is 2.5 m. The A* path from the robot to the active waypoint is 1 to 5 cells long (2.5 m to 12.5 m), in training and at deployment. The only exception is the final goal, which may be closer than one cell.

In training the waypoint is held until the robot is within 0.5 m, and that arrival ends the episode successfully. The reset places the robot on a random cell centre plus ±0.4 m, with a yaw drawn from \([-\pi, \pi)\), and samples a new privileged waypoint from that pose. At deployment the waypoint is held until the same radius, then the next one is issued without resetting. It is dropped early when a newly discovered wall lies on that committed path. Until then the same waypoint is held, which is what the policy saw in training.

Training draws length `k_max` with probability 0.5 and otherwise draws uniformly from 1 to `k_max - 1`. `k_max` starts at 1, where the two rules agree. A window pools resetting episodes until it has 256 of them and at least 512 issued waypoints; episodes with no waypoint yet are left out. Each privileged episode issues one waypoint, so the reach rate is the fraction of episodes that arrived. The window passes when the rate at length `k_max` is at least 80% and fewer than half of those episodes collided. Shorter lengths stay in the window but do not open the gate. `k_max` increases by one cell after two passing windows in a row, then waits out one episode so the next window is commanded at the new length. The waypoint target is the sampled cell centre plus ±0.4 m of jitter.

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

199 values for the actor: 180 ranges (each beam divided by 10 m, with ±0.03 m noise and 2% dropout), the waypoint in the body frame divided by 12.5 m (2), four path points at 1, 2, 3 and 4 m ahead of the high-water mark, each in the body frame and divided by 12.5 m (8), planar body velocity (2), yaw rate (1), projected gravity (3), last action (3). The critic sees the same scan without noise, plus the remaining path length divided by 12.5 m.

## High-level reward

Isaac Lab multiplies every reward by `step_dt`. The functions divide by `step_dt` first, so the return matches the per-step expressions below.

`s_t` is arc length along the active waypoint's smoothed A* path, of length `L ∈ [2.5, 12.5]` m. `s̄_t` is the highest arc length reached so far, reset to 0 when a new waypoint is issued. `e_t` is the cross-track error.

The footprint is a rectangle with half-length `a = 0.35 m` and half-width `b = 0.20 m`, centred on the base. `r_i` is the planar range from the LiDAR. `τ_i` is the distance along that beam from the LiDAR to the rectangle, and the clearance is `c_i = r_i - τ_i`. When the LiDAR is at the base origin, `τ(θ) = min(a / |cos θ|, b / |sin θ|)`.

Per-step terms:

- **Progress.** `k (s̄_{t+1} - s̄_t)` with `k = 1 / (v_x,max Δt) = 1 / 0.3` m⁻¹. Full-speed progress along the path earns 1 per step. Over one waypoint the sum is at most `k L ≤ 12.5 / 0.3 ≈ 41.7`. A cycle earns nothing after the first pass, because the high-water mark does not go backward. A potential `γΦ(s') - Φ(s)` would jump every time a waypoint is re-issued; the high-water form does not.
- **Time.** `-0.05` per step. The break-even speed is `0.05 / (k Δt) = 0.15 m/s`. Slower than that, progress does not pay for the time.
- **Cross-track.** `-0.1 * min(e² / 0.6², 1)`. The 0.6 m reference is the corridor half-width (1.0 m) minus the footprint's rotation radius `sqrt(a² + b²) = 0.40 m`.
- **Clearance.** `-0.5 * max(0, 1 - c_min / 0.3)²`.
- **Braking.** Stopping distance is `d(v) = v τ + v² / (2 a_b)`. Solving `d(v) = c - 0.3` gives `v_safe = a_b (-τ + sqrt(τ² + 2 (c - 0.3) / a_b))`, with `a_b = 2.5 m/s²` and `τ = 0.2 s`. The cost is `-0.2 (max(0, |v| - v_safe) / v_x,max)²`. Check: 1 m of clearance gives 1.44 m/s, and 2 m gives 2.46 m/s.
- **Action rate.** `-0.05 * ||a_t - a_{t-1}||² / 12`. The 12 is the maximum (3 axes, each differing by at most 2).

The worst per-step penalty, excluding collision, is `p_max = 0.05 + 0.1 + 0.5 + 0.2 + 0.05 = 0.9`. Continuing forever can cost at most `p_max / (1 - γ) = 90` in discounted return. The collision (and fall) penalty is `C = 1.25 * p_max / (1 - γ) = 112.5`, computed in `Go2MazeEnvCfg.__post_init__`, so crashing is never better than continuing. It is applied only on a collision or a fall. Reaching the waypoint is a real termination and does not pay `C`. Timeouts bootstrap the critic and do not pay `C`.

**Waypoint bonus** `R_g = 10`, paid once on the step the robot enters the 0.5 m radius. In training that step ends the episode. After progress toward that waypoint has saturated, staying put earns nothing, so any positive bonus makes arriving the better choice. 10 is about ten steps of full-speed progress, and it replaces the last `k * 0.5 = 1.67` of progress that is never travelled.

The bonus is once per privileged episode. The next waypoint is sampled only after the reset, at least one cell of A* away. With ±0.4 m of jitter on both ends the straight-line gap is at least `2.5 - 0.8 = 1.7 m`. `R_g / C ≈ 0.09`, so a collision is only worth risking when its probability is below about 0.09. Touching a wall on the arrival step pays both the bonus and `C`.

Each waypoint has a time budget `clip(L / 0.5 + 5, 10, 30)` seconds. Exceeding it truncates the episode, as does the 30 s episode cap. The cap and the budget arrive together only for a 5-cell waypoint, whose budget is also 30 s. Reaching the waypoint ends a privileged episode successfully. Planner play still ends when the maze goal is reached.

# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class CoverageMap:

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cell_size: float = 0.1,
        max_world_size: float = 24.0,  
        robot_radius: float = 0.18,   
        # Multi-scale obs configuration
        n_scales: int = 4,
        scale_factor: int = 4,
        finest_pixel_size: float = 0.0375,
        obs_patch_size: int = 32,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cell_size = float(cell_size)
        self.robot_radius = float(robot_radius)

        # Grid sizing: must be odd-friendly
        self.H = int(round(max_world_size / cell_size))
        self.W = int(round(max_world_size / cell_size))
        self.origin_x = -0.5 * self.W * self.cell_size
        self.origin_y = -0.5 * self.H * self.cell_size
        self.grid_extent_x = self.W * self.cell_size
        self.grid_extent_y = self.H * self.cell_size

        # Multi-scale obs config
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.finest_pixel_size = float(finest_pixel_size)
        self.obs_patch_size = obs_patch_size

        # State tensors
        shape = (num_envs, self.H, self.W)
        self.visited = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.obstacles = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.frontier = torch.zeros(shape, dtype=torch.bool, device=self.device)

        self.free_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)

        # Bookkeeping
        self.cells_visited_this_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.steps_since_new_cell = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.total_free_cells = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.visited_free_cells = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.prev_tv = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.completion_bonus_given = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # Cached robot grid indices from the last update
        self.robot_row = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.robot_col = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.robot_xy_world = torch.zeros(num_envs, 2, device=self.device)
        self.steps_since_moved = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Pre-built dilation kernel for sweep stamping 
        self._sweep_kernel = self._build_sweep_kernel()

    # helpers
    def _build_sweep_kernel(self) -> torch.Tensor:
        """A disc of 1s representing cells within robot_radius around the centre."""
        r_cells = int(math.ceil(self.robot_radius / self.cell_size))
        k = 2 * r_cells + 1
        ys = torch.arange(k, device=self.device).float() - r_cells
        xs = torch.arange(k, device=self.device).float() - r_cells
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        disc = ((gx * gx + gy * gy) <= (r_cells * r_cells)).float()
        return disc  

    def world_to_grid(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        xf = ((x - self.origin_x) / self.cell_size)
        yf = ((y - self.origin_y) / self.cell_size)
        xf = torch.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
        yf = torch.nan_to_num(yf, nan=0.0, posinf=0.0, neginf=0.0)
        col = xf.long().clamp(0, self.W - 1)
        row = yf.long().clamp(0, self.H - 1)
        return row, col

    # ------------------------------------------------------------------- reset

    def reset(self, env_ids: torch.Tensor | None = None) -> None:

        if env_ids is None:
            self.visited.zero_()
            self.frontier.zero_()
            self.cells_visited_this_step.zero_()
            self.steps_since_new_cell.zero_()
            self.visited_free_cells.zero_()
            self.prev_tv.zero_()
            self.completion_bonus_given.zero_()
            self.steps_since_moved.zero_()
        else:
            self.visited[env_ids] = False
            self.frontier[env_ids] = False
            self.cells_visited_this_step[env_ids] = 0
            self.steps_since_new_cell[env_ids] = 0
            self.visited_free_cells[env_ids] = 0
            self.prev_tv[env_ids] = 0.0
            self.completion_bonus_given[env_ids] = False
            self.steps_since_moved[env_ids] = 0

    def reset_room(
        self,
        env_ids: torch.Tensor,
        room_size: torch.Tensor,         
        obstacles_world: torch.Tensor,  
    ) -> None:

        if env_ids.numel() == 0:
            return

        # Clear previous obstacles/free for these envs
        self.obstacles[env_ids] = False
        self.free_mask[env_ids] = False

        # Build a coordinate meshgrid in world frame 
        ys = self.origin_y + (torch.arange(self.H, device=self.device).float() + 0.5) * self.cell_size
        xs = self.origin_x + (torch.arange(self.W, device=self.device).float() + 0.5) * self.cell_size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

        # Per-env interior mask
        half_w = (room_size[:, 0] * 0.5).view(-1, 1, 1)
        half_h = (room_size[:, 1] * 0.5).view(-1, 1, 1)
        interior = (gx.unsqueeze(0).abs() <= half_w) & (gy.unsqueeze(0).abs() <= half_h)

        obstacle_mask = ~interior  

        # Stamp obstacle cubes 
        if obstacles_world is not None and obstacles_world.shape[1] > 0:
            cx = obstacles_world[..., 0]       
            cy = obstacles_world[..., 1]
            hx = obstacles_world[..., 2]
            hy = obstacles_world[..., 3]
            valid = ~torch.isnan(cx)           
            for k in range(obstacles_world.shape[1]):

                cxk = cx[:, k].view(-1, 1, 1)
                cyk = cy[:, k].view(-1, 1, 1)
                hxk = hx[:, k].view(-1, 1, 1)
                hyk = hy[:, k].view(-1, 1, 1)
                v = valid[:, k].view(-1, 1, 1)
                cxk = torch.where(v, cxk, torch.zeros_like(cxk))
                cyk = torch.where(v, cyk, torch.zeros_like(cyk))
                hxk = torch.where(v, hxk, torch.zeros_like(hxk))
                hyk = torch.where(v, hyk, torch.zeros_like(hyk))
                cube = (
                    (gx.unsqueeze(0) >= (cxk - hxk))
                    & (gx.unsqueeze(0) <= (cxk + hxk))
                    & (gy.unsqueeze(0) >= (cyk - hyk))
                    & (gy.unsqueeze(0) <= (cyk + hyk))
                )
                cube = cube & v  
                obstacle_mask = obstacle_mask | cube

        free = interior & ~obstacle_mask
        self.obstacles[env_ids] = obstacle_mask
        self.free_mask[env_ids] = free
        self.total_free_cells[env_ids] = free.view(env_ids.numel(), -1).sum(dim=-1)

    # ----------------------------------------------------------------- update

    def update(self, robot_xy: torch.Tensor, robot_yaw: torch.Tensor) -> None:

        N = self.num_envs
        rows, cols = self.world_to_grid(robot_xy[:, 0], robot_xy[:, 1])
        self.robot_row.copy_(rows)
        self.robot_col.copy_(cols)
        self.robot_yaw = robot_yaw  


        stuck_threshold = 0.005  
        moved = torch.linalg.norm(robot_xy - self.robot_xy_world, dim=-1)
        is_stuck_step = moved < stuck_threshold
        self.steps_since_moved = torch.where(
            is_stuck_step,
            self.steps_since_moved + 1,
            torch.zeros_like(self.steps_since_moved),
        )
        self.robot_xy_world.copy_(robot_xy)

        centre = torch.zeros(N, 1, self.H, self.W, device=self.device, dtype=torch.float32)
        idx = torch.arange(N, device=self.device)
        centre[idx, 0, rows, cols] = 1.0
        k = self._sweep_kernel.shape[0]
        pad = k // 2

        kernel = self._sweep_kernel.view(1, 1, k, k)
        swept = F.conv2d(centre, kernel, padding=pad) > 0.5  
        swept = swept.squeeze(1)  

        newly_visited = swept & self.free_mask & ~self.visited
        self.cells_visited_this_step = newly_visited.view(N, -1).sum(dim=-1)
        self.visited |= newly_visited
        self.visited_free_cells = (self.visited & self.free_mask).view(N, -1).sum(dim=-1)

        # No-progress counter
        progressed = self.cells_visited_this_step > 0
        self.steps_since_new_cell = torch.where(
            progressed,
            torch.zeros_like(self.steps_since_new_cell),
            self.steps_since_new_cell + 1,
        )

        self._recompute_frontier()

    def _recompute_frontier(self) -> None:
        """Frontier = unvisited free cell with at least one visited 4-neighbour."""
        v = self.visited.unsqueeze(1).float()
        # 4-connected dilation kernel
        k = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            device=self.device,
        ).view(1, 1, 3, 3)
        neighbour_visited = F.conv2d(v, k, padding=1) > 0.5
        neighbour_visited = neighbour_visited.squeeze(1)
        self.frontier = neighbour_visited & self.free_mask & ~self.visited

    # queries / obs

    def coverage_pct(self) -> torch.Tensor:

        return self.visited_free_cells.float() / self.total_free_cells.clamp(min=1).float()

    def get_nearest_uncovered_distance(self) -> torch.Tensor:

        uncovered = self.free_mask & ~self.visited 

        ys = torch.arange(self.H, device=self.device).view(1, self.H, 1).expand(self.num_envs, self.H, self.W)
        xs = torch.arange(self.W, device=self.device).view(1, 1, self.W).expand(self.num_envs, self.H, self.W)
        rr = self.robot_row.view(-1, 1, 1)
        rc = self.robot_col.view(-1, 1, 1)
        manh = (ys - rr).abs() + (xs - rc).abs()  
        big = self.H + self.W  
        masked = torch.where(uncovered, manh, torch.full_like(manh, big))
        min_d = masked.view(self.num_envs, -1).min(dim=-1).values.float()  

        no_uncovered = uncovered.view(self.num_envs, -1).sum(dim=-1) == 0
        min_d = torch.where(no_uncovered, torch.zeros_like(min_d), min_d)

        # Convert to metres and normalise by grid diagonal
        diag_m = self.cell_size * math.hypot(self.H, self.W)
        d_metres = min_d * self.cell_size
        d_norm = (d_metres / diag_m).clamp(0.0, 1.0)
        return d_norm.unsqueeze(-1)  # [N, 1]

    # multi-scale egocentric obs

    def get_multiscale_obs(self) -> torch.Tensor:

        N = self.num_envs
        P = self.obs_patch_size

        # Per-channel float versions of the global maps
        visited_f = self.visited.unsqueeze(1).float()
        obstacles_f = self.obstacles.unsqueeze(1).float()
        frontier_f = self.frontier.unsqueeze(1).float()

        outputs: list[torch.Tensor] = []
        for s in range(self.n_scales):
            pixel_size = self.finest_pixel_size * (self.scale_factor ** s)
            theta = self._affine_theta(pixel_size, P)
            grid = F.affine_grid(theta, [N, 1, P, P], align_corners=False)

            # For visited/obstacles use nearest sampling
            v_s = F.grid_sample(visited_f, grid, mode="nearest", padding_mode="zeros", align_corners=False)
            o_s = F.grid_sample(obstacles_f, grid, mode="nearest", padding_mode="border", align_corners=False)

            ratio = pixel_size / self.cell_size
            kpool = max(1, int(math.ceil(ratio)))
            if kpool > 1:
                f_pooled = F.max_pool2d(frontier_f, kernel_size=kpool, stride=1, padding=kpool // 2)
                if f_pooled.shape[-2:] != frontier_f.shape[-2:]:
                    f_pooled = F.interpolate(f_pooled, size=frontier_f.shape[-2:], mode="nearest")
            else:
                f_pooled = frontier_f
            f_s = F.grid_sample(f_pooled, grid, mode="nearest", padding_mode="zeros", align_corners=False)

            stacked = torch.cat([v_s, o_s, f_s], dim=1)
            outputs.append(stacked)

        scales_stack = torch.stack(outputs, dim=1)             
        reordered = scales_stack.permute(0, 2, 1, 3, 4).contiguous()  
        return reordered.view(N, -1)  

    def _affine_theta(self, pixel_size: float, patch_size: int) -> torch.Tensor:

        N = self.num_envs
        half_patch = 0.5 * pixel_size * patch_size
        robot_x = self.origin_x + (self.robot_col.float() + 0.5) * self.cell_size  
        robot_y = self.origin_y + (self.robot_row.float() + 0.5) * self.cell_size 

        cy = torch.cos(self.robot_yaw)
        sy = torch.sin(self.robot_yaw)

        # Scale factor from output norm to input norm space
        kx = half_patch * (2.0 / self.grid_extent_x)
        ky = half_patch * (2.0 / self.grid_extent_y)

        # Translation from output norm origin to input norm robot position
        tx = (robot_x - self.origin_x) * (2.0 / self.grid_extent_x) - 1.0
        ty = (robot_y - self.origin_y) * (2.0 / self.grid_extent_y) - 1.0

        theta = torch.zeros(N, 2, 3, device=self.device, dtype=torch.float32)
        # Derivation in module docstring; we want robot's heading direction to be "up" in the patch.
        theta[:, 0, 0] = kx * cy
        theta[:, 0, 1] = kx * sy
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = ky * sy
        theta[:, 1, 1] = -ky * cy
        theta[:, 1, 2] = ty
        return theta

    # total variation (rewards)

    def compute_tv(self) -> torch.Tensor:
        """TV(C) = Σ sqrt((C_{i+1,j} - C_{i,j})^2 + (C_{i,j+1} - C_{i,j})^2)
        on the visited map. Returns [N]."""
        v = self.visited.float()
        dx = v[:, :, 1:] - v[:, :, :-1]
        dy = v[:, 1:, :] - v[:, :-1, :]
        # Align shapes for elementwise sum 
        dx_c = dx[:, :-1, :]              
        dy_c = dy[:, :, :-1]            
        grad_mag = torch.sqrt(dx_c * dx_c + dy_c * dy_c + 1e-12)
        return grad_mag.view(self.num_envs, -1).sum(dim=-1)

    # 4-direction validity 

    def get_valid_directions(self, lookahead: int = 3) -> torch.Tensor:

        N = self.num_envs
        r, c = self.robot_row, self.robot_col
        L = int(lookahead)
        offsets = [(L, 0), (0, L), (-L, 0), (0, -L)]
        out = torch.zeros(N, 4, dtype=torch.bool, device=self.device)
        idx = torch.arange(N, device=self.device)
        for i, (dr, dc) in enumerate(offsets):
            nr = (r + dr).clamp(0, self.H - 1)
            nc = (c + dc).clamp(0, self.W - 1)
            in_bounds = ((r + dr) >= 0) & ((r + dr) < self.H) & ((c + dc) >= 0) & ((c + dc) < self.W)
            free = self.free_mask[idx, nr, nc] & ~self.obstacles[idx, nr, nc]
            unvisited = ~self.visited[idx, nr, nc]
            out[:, i] = in_bounds & free & unvisited
        return out

    # GPU BFS to nearest uncovered

    def bfs_first_step_to_nearest_uncovered(
        self, env_ids: torch.Tensor, max_iters: int | None = None
    ) -> torch.Tensor:

        n = env_ids.numel()
        if n == 0:
            return torch.zeros(0, 2, device=self.device, dtype=torch.long)

        H, W = self.H, self.W
        idx = torch.arange(n, device=self.device)
        # Free graph: cells where the robot may stand 
        traversable = self.free_mask[env_ids] & ~self.obstacles[env_ids]  
        uncovered = traversable & ~self.visited[env_ids]                

        # Initialise frontier with the robot's current cell
        rows = self.robot_row[env_ids]
        cols = self.robot_col[env_ids]
        wave = torch.zeros(n, H, W, dtype=torch.bool, device=self.device)
        wave[idx, rows, cols] = True

        # Distance map: -1 means unvisited; robot cell starts at 0
        dist = torch.full((n, H, W), -1, dtype=torch.long, device=self.device)
        dist[idx, rows, cols] = 0

        # Found flag per env
        found = torch.zeros(n, dtype=torch.bool, device=self.device)
        target_row = rows.clone()
        target_col = cols.clone()

        if max_iters is None:
            max_iters = H + W

        for step in range(1, max_iters + 1):
            # Shift wavefront in 4 directions
            up    = F.pad(wave[:, :-1, :], (0, 0, 1, 0))
            down  = F.pad(wave[:, 1:, :],  (0, 0, 0, 1))
            left  = F.pad(wave[:, :, :-1], (1, 0, 0, 0))
            right = F.pad(wave[:, :, 1:],  (0, 1, 0, 0))
            new_wave = (up | down | left | right) & traversable & (dist == -1)
            if not new_wave.any():
                break
            dist = torch.where(new_wave, torch.full_like(dist, step), dist)
            
            hits = new_wave & uncovered  # [n, H, W]
            any_hit = hits.view(n, -1).any(dim=-1)
            newly_found = any_hit & ~found
            if newly_found.any():
                
                flat = hits.view(n, -1)
                first_hit_flat = flat.float().argmax(dim=-1)  
                tr = (first_hit_flat // W).long()
                tc = (first_hit_flat % W).long()
                target_row = torch.where(newly_found, tr, target_row)
                target_col = torch.where(newly_found, tc, target_col)
                found = found | newly_found
            if found.all():
                break
            wave = new_wave

        cur_r = target_row.clone()
        cur_c = target_col.clone()
        # Distances at targets
        cur_d = dist[idx, cur_r, cur_c]

        # Walk back to dist == 1
        for _ in range(max_iters):
            done = (cur_d <= 1) | ~found
            if done.all():
                break
            
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr = (cur_r + dr).clamp(0, H - 1)
                nc = (cur_c + dc).clamp(0, W - 1)
                nd = dist[idx, nr, nc]
                cond = (nd == (cur_d - 1)) & found & ~done
                cur_r = torch.where(cond, nr, cur_r)
                cur_c = torch.where(cond, nc, cur_c)
                cur_d = torch.where(cond, nd, cur_d)


        d_row = (cur_r - rows).clamp(-1, 1)
        d_col = (cur_c - cols).clamp(-1, 1)
        # Envs without any uncovered cell stay at zero
        no_target = ~found
        d_row = torch.where(no_target, torch.zeros_like(d_row), d_row)
        d_col = torch.where(no_target, torch.zeros_like(d_col), d_col)
        return torch.stack([d_row, d_col], dim=-1)
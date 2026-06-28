import torch


def _per_gaussian_conf(gaussians: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian confidence from the last channel (15D layout) or 0.5 fallback."""
    if gaussians.shape[-1] >= 15:
        return gaussians[:, 14].float().clamp(0.0, 1.0)
    return torch.full(
        (gaussians.shape[0],),
        0.5,
        device=gaussians.device,
        dtype=torch.float32,
    )


def _aggregate_voxel_conf(conf_vals: torch.Tensor, conf_agg: str = "mean") -> torch.Tensor:
    """Aggregate per-GS conf inside one voxel for one side (base or plus)."""
    if conf_agg == "mean":
        return conf_vals.mean()
    if conf_agg == "max":
        return conf_vals.max()
    raise ValueError(f"conf_agg must be 'mean' or 'max', got {conf_agg!r}")


def fuse_gaussians_by_voxel_conf_margin(
    gaussians_base: torch.Tensor,
    gaussians_plus: torch.Tensor,
    point_cloud_range=None,
    voxel_size: float = 0.1,
    conf_fusion_margin: float = 0.05,
    conf_agg: str = "mean",
    base_conf_thresh: float | None = None,
):
    """Voxel-wise winner fusion of base vs plus Gaussians using per-GS conf + margin.

    For each voxel inside ``point_cloud_range``:
      - only base  -> keep base Gaussians in that voxel
      - only plus  -> keep plus Gaussians in that voxel
      - both       -> keep plus iff agg(conf_plus) > agg(conf_base) + margin, else base
        when ``base_conf_thresh`` is set, plus wins only if additionally
        agg(conf_base) < base_conf_thresh (base-priority: high-conf base voxels stay base)
        where agg is mean (default) or max over Gaussians on that side in the voxel

    Gaussians outside the range are appended unchanged from both sets.

    Args:
        gaussians_base: [N1, D] base (2-view) Gaussians, D >= 14
        gaussians_plus: [N2, D] plus (6-view) Gaussians
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: voxel edge length in meters
        conf_fusion_margin: plus wins only when agg(conf_plus) > agg(conf_base) + margin
        conf_agg: ``mean`` (stable, default) or ``max`` (single outlier can flip voxel)
        base_conf_thresh: if set, plus can win only when agg(conf_base) is below this

    Returns:
        fused: [M, D]
    """
    if point_cloud_range is None:
        point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]

    device = gaussians_base.device
    dtype = gaussians_base.dtype
    dim = gaussians_base.shape[-1]
    if gaussians_plus.shape[-1] != dim:
        raise ValueError(
            f"Gaussian dim mismatch: base {dim} vs plus {gaussians_plus.shape[-1]}"
        )

    grid_min = torch.tensor(point_cloud_range[:3], device=device, dtype=dtype)
    grid_max = torch.tensor(point_cloud_range[3:], device=device, dtype=dtype)
    voxel_dim = ((grid_max - grid_min) / voxel_size).long().clamp(min=1)

    valid_base = (
        (gaussians_base[:, :3] >= grid_min) & (gaussians_base[:, :3] < grid_max)
    ).all(dim=1)
    valid_plus = (
        (gaussians_plus[:, :3] >= grid_min) & (gaussians_plus[:, :3] < grid_max)
    ).all(dim=1)

    g_base = gaussians_base[valid_base]
    g_plus = gaussians_plus[valid_plus]
    outside = torch.cat(
        [gaussians_base[~valid_base], gaussians_plus[~valid_plus]], dim=0
    )

    if g_base.numel() == 0 and g_plus.numel() == 0:
        return outside
    if g_base.numel() == 0:
        return torch.cat([g_plus, outside], dim=0)
    if g_plus.numel() == 0:
        return torch.cat([g_base, outside], dim=0)

    g_all = torch.cat([g_base, g_plus], dim=0)
    conf_all = torch.cat([_per_gaussian_conf(g_base), _per_gaussian_conf(g_plus)], dim=0)
    group_id = torch.cat(
        [
            torch.zeros(g_base.shape[0], device=device, dtype=torch.long),
            torch.ones(g_plus.shape[0], device=device, dtype=torch.long),
        ],
        dim=0,
    )

    voxel_idx = ((g_all[:, :3] - grid_min) / voxel_size).floor().long()
    voxel_hash = (
        voxel_idx[:, 0] * voxel_dim[1] * voxel_dim[2]
        + voxel_idx[:, 1] * voxel_dim[2]
        + voxel_idx[:, 2]
    )
    _, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
    num_voxels = int(inverse_indices.max().item()) + 1
    neg_inf = conf_all.new_tensor(-1e10)

    mask_b = group_id == 0
    mask_p = group_id == 1

    def _voxel_side_scores(side_mask: torch.Tensor) -> torch.Tensor:
        idx = inverse_indices[side_mask]
        conf_side = conf_all[side_mask]
        if conf_agg == "max":
            scores = torch.full((num_voxels,), neg_inf, device=device, dtype=conf_all.dtype)
            if idx.numel() > 0:
                scores = scores.scatter_reduce(
                    0, idx, conf_side, reduce="amax", include_self=True
                )
            present = torch.zeros(num_voxels, device=device, dtype=torch.bool)
            if idx.numel() > 0:
                present.index_put_((idx,), torch.ones_like(idx, dtype=torch.bool))
            return torch.where(present, scores, neg_inf)

        # mean
        sum_scores = torch.zeros(num_voxels, device=device, dtype=conf_all.dtype)
        counts = torch.zeros(num_voxels, device=device, dtype=conf_all.dtype)
        if idx.numel() > 0:
            sum_scores.index_add_(0, idx, conf_side)
            counts.index_add_(0, idx, torch.ones_like(conf_side))
        return torch.where(counts > 0, sum_scores / counts.clamp(min=1.0), neg_inf)

    score_base = _voxel_side_scores(mask_b)
    score_plus = _voxel_side_scores(mask_p)

    pick_plus_voxel = score_plus > score_base + conf_fusion_margin
    if base_conf_thresh is not None:
        pick_plus_voxel = pick_plus_voxel & (score_base < base_conf_thresh)

    voxel_pick_plus = pick_plus_voxel[inverse_indices]
    keep = torch.where(group_id == 1, voxel_pick_plus, ~voxel_pick_plus)
    fused_inside = g_all[keep]
    if outside.numel() == 0:
        return fused_inside
    return torch.cat([fused_inside, outside], dim=0)


def fuse_gaussians_by_voxel_conf_margin_batched(
    gaussians_base_b: torch.Tensor,
    gaussians_plus_b: torch.Tensor,
    point_cloud_range=None,
    voxel_size: float = 0.1,
    conf_fusion_margin: float = 0.05,
    conf_agg: str = "mean",
    base_conf_thresh: float | None = None,
):
    """Batched wrapper; returns [B, M_max, D] zero-padded."""
    B = gaussians_base_b.shape[0]
    fused_list = []
    max_len = 0
    dim = gaussians_base_b.shape[-1]

    for b in range(B):
        fused = fuse_gaussians_by_voxel_conf_margin(
            gaussians_base_b[b],
            gaussians_plus_b[b],
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            conf_fusion_margin=conf_fusion_margin,
            conf_agg=conf_agg,
            base_conf_thresh=base_conf_thresh,
        )
        fused_list.append(fused)
        max_len = max(max_len, fused.shape[0])

    out = torch.zeros(
        (B, max_len, dim),
        dtype=gaussians_base_b.dtype,
        device=gaussians_base_b.device,
    )
    for b, fused in enumerate(fused_list):
        if fused.shape[0] > 0:
            out[b, : fused.shape[0]] = fused
    return out, fused_list


# --- legacy depth-based fusion (kept for reference) ---

def fuse_gaussians_by_voxel_with_depth_vectorized(
    gaussians1: torch.Tensor,
    gaussians2: torch.Tensor,
    point_cloud_range=None,
    voxel_size=0.1,
):
    if point_cloud_range is None:
        point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]
    device = gaussians1.device
    grid_min = torch.tensor(point_cloud_range[:3], device=device)
    grid_max = torch.tensor(point_cloud_range[3:], device=device)
    voxel_dim = ((grid_max - grid_min) / voxel_size).long()

    def compute_voxel_indices(xyz):
        return ((xyz - grid_min) / voxel_size).floor().to(torch.long)

    valid1 = ((gaussians1[:, :3] >= grid_min) & (gaussians1[:, :3] < grid_max)).all(dim=1)
    valid2 = ((gaussians2[:, :3] >= grid_min) & (gaussians2[:, :3] < grid_max)).all(dim=1)

    g1 = torch.cat([gaussians1[valid1], torch.zeros((valid1.sum(), 1), device=device)], dim=1)
    g2 = torch.cat([gaussians2[valid2], torch.ones((valid2.sum(), 1), device=device)], dim=1)
    g_all = torch.cat([g1, g2], dim=0)

    voxel_idx = compute_voxel_indices(g_all[:, :3])
    voxel_hash = (
        voxel_idx[:, 0] * voxel_dim[1] * voxel_dim[2]
        + voxel_idx[:, 1] * voxel_dim[2]
        + voxel_idx[:, 2]
    )

    unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
    fused_list = []
    for i in range(len(unique_hashes)):
        mask = inverse_indices == i
        g_tensor = g_all[mask]
        groups = g_tensor[:, -1].long()
        depths = g_tensor[:, 14]

        scores = []
        for gid in [0, 1]:
            submask = groups == gid
            if submask.sum() == 0:
                scores.append((-1e10, submask))
                continue
            score = submask.sum().float() / (depths[submask].mean() + 1e-6)
            scores.append((score, submask))

        winner = scores[0][1] if scores[0][0] > scores[1][0] else scores[1][1]
        fused_list.append(g_tensor[winner, :15])

    fused = torch.cat(fused_list, dim=0)
    outside = torch.cat([gaussians1[~valid1], gaussians2[~valid2]], dim=0)
    return torch.cat([fused, outside], dim=0)

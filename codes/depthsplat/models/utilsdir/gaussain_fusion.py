import torch
from torch_scatter import scatter_sum
from torch_scatter import scatter_max
from collections import defaultdict
from torch_scatter import scatter_sum, scatter_mean, scatter_max

def fuse_gaussians_by_voxel_with_depth_vectorized(
    gaussians1: torch.Tensor,  # [N1, 15]
    gaussians2: torch.Tensor,  # [N2, 15]
    point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0],
    voxel_size = 0.1,
):
    device = gaussians1.device
    grid_min = torch.tensor(point_cloud_range[:3], device=device)
    grid_max = torch.tensor(point_cloud_range[3:], device=device)
    voxel_dim = ((grid_max - grid_min) / voxel_size).long()

    def compute_voxel_indices(xyz):
        return ((xyz - grid_min) / voxel_size).floor().to(torch.long)  # [N, 3]

    valid1 = ((gaussians1[:, :3] >= grid_min) & (gaussians1[:, :3] < grid_max)).all(dim=1)
    valid2 = ((gaussians2[:, :3] >= grid_min) & (gaussians2[:, :3] < grid_max)).all(dim=1)

    g1 = torch.cat([gaussians1[valid1], torch.zeros((valid1.sum(), 1), device=device)], dim=1)  # group 0
    g2 = torch.cat([gaussians2[valid2], torch.ones((valid2.sum(), 1), device=device)], dim=1)   # group 1
    g_all = torch.cat([g1, g2], dim=0)  # [N, 16]

    voxel_idx = compute_voxel_indices(g_all[:, :3])
    voxel_hash = (voxel_idx[:, 0] * voxel_dim[1] * voxel_dim[2] +
                  voxel_idx[:, 1] * voxel_dim[2] +
                  voxel_idx[:, 2])

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
    return torch.cat([fused, outside], dim=0)  # [M, 15]

def fuse_gaussians_by_voxel_with_depth_batched_vectorized(
    gaussians1: torch.Tensor,  # [B, N1, 15]
    gaussians2: torch.Tensor,  # [B, N2, 15]
    point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0],
    voxel_size = 0.1,
):
    B = gaussians1.shape[0]
    fused_list = []
    max_len = 0

    for b in range(B):
        fused = fuse_gaussians_by_voxel_with_depth_vectorized(
            gaussians1[b], gaussians2[b],
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        )
        fused_list.append(fused)
        max_len = max(max_len, fused.shape[0])

    # zero-padding to same shape
    out = torch.zeros((B, max_len, 15), dtype=gaussians1.dtype, device=gaussians1.device)
    for b in range(B):
        out[b, :fused_list[b].shape[0]] = fused_list[b]
    return out


def fuse_gaussians_by_voxel_with_depth_groupwise_scatter(
    gaussians1: torch.Tensor,  # [N1, 15]
    gaussians2: torch.Tensor,  # [N2, 15]
    point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0],
    voxel_size = 0.1,
):
    device = gaussians1.device
    grid_min = torch.tensor(point_cloud_range[:3], device=device)
    grid_max = torch.tensor(point_cloud_range[3:], device=device)
    voxel_dim = ((grid_max - grid_min) / voxel_size).long()

    def compute_voxel_indices(xyz):
        return ((xyz - grid_min) / voxel_size).floor().to(torch.long)

    valid1 = ((gaussians1[:, :3] >= grid_min) & (gaussians1[:, :3] < grid_max)).all(dim=1)
    valid2 = ((gaussians2[:, :3] >= grid_min) & (gaussians2[:, :3] < grid_max)).all(dim=1)

    g1 = torch.cat([gaussians1[valid1], torch.zeros((valid1.sum(), 1), device=device)], dim=1)  # group 0
    g2 = torch.cat([gaussians2[valid2], torch.ones((valid2.sum(), 1), device=device)], dim=1)   # group 1
    g_all = torch.cat([g1, g2], dim=0)  # [N, 16]

    voxel_idx = compute_voxel_indices(g_all[:, :3])
    voxel_hash = (voxel_idx[:, 0] * voxel_dim[1] * voxel_dim[2] +
                  voxel_idx[:, 1] * voxel_dim[2] +
                  voxel_idx[:, 2])

    unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
    num_voxels = unique_hashes.shape[0]

    score = torch.zeros((2, num_voxels), device=device)

    for group_id in [0, 1]:
        mask = (g_all[:, -1] == group_id)
        depth_sum = torch.zeros(num_voxels, device=device).index_add(0, inverse_indices[mask], g_all[mask, 14])
        count = torch.zeros(num_voxels, device=device).index_add(0, inverse_indices[mask], torch.ones_like(g_all[mask, 14]))
        avg_depth = torch.where(count > 0, depth_sum / count, torch.ones_like(count))
        score[group_id] = torch.where(count > 0, count / (avg_depth + 1e-6), torch.tensor(-1e10, device=device))

    winner = (score[1] > score[0]).long()  # 1 or 0 per voxel
    voxel_winner = winner[inverse_indices] == g_all[:, -1]  # only keep gaussians belonging to winning group

    fused = g_all[voxel_winner, :15]
    outside = torch.cat([gaussians1[~valid1], gaussians2[~valid2]], dim=0)
    return torch.cat([fused, outside], dim=0)


def fuse_gaussians_by_voxel_with_depth_scatter_batched(gaussians1_b, gaussians2_b, point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 12.0],
    voxel_size=0.1):
    B = gaussians1_b.shape[0]
    fused = []
    for b in range(B):
        fused.append(fuse_gaussians_by_voxel_with_depth_groupwise_scatter(
            gaussians1_b[b], gaussians2_b[b],
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size
        ))
    max_len = max(x.shape[0] for x in fused)
    fused_padded = torch.zeros((B, max_len, 15), device=gaussians1_b.device)
    for b, f in enumerate(fused):
        fused_padded[b, :f.shape[0]] = f
    return fused_padded
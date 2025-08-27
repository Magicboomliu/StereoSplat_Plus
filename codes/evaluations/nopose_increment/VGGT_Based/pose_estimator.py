import numpy as np
import open3d as o3d
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

@torch.no_grad()
def align_sRt_umeyama_icp_torch(
    src_pts: torch.Tensor,        # (N,3) torch, 一一对应的源点
    tgt_pts: torch.Tensor,        # (N,3) torch, 一一对应的目标点
    icp_max_iter: int = 100,
    icp_dist_ratio: float = 0.02  # ICP对应阈值 = 目标包围盒对角线 * 这个比例
):
    """
    返回:
      R (3,3) torch, s (1,) torch, t (3,) torch
      以及一个字典 extras 里放一些诊断信息
    说明:
      1) 先用 Umeyama(带尺度) 用“一一对应”的索引求初值 Sim(3)
      2) 在此初值上跑 Open3D 的 Point-to-Point ICP (with_scaling=True) 细化
    """
    assert src_pts.shape == tgt_pts.shape and src_pts.ndim == 2 and src_pts.shape[1] == 3, \
        "src_pts 和 tgt_pts 必须都是 (N,3)"
    device, dtype = src_pts.device, src_pts.dtype
    N = src_pts.shape[0]
    if N < 3:
        # 点太少，直接退化为单位
        R = torch.eye(3, dtype=dtype, device=device)
        s = torch.tensor(1.0, dtype=dtype, device=device)
        t = torch.zeros(3, dtype=dtype, device=device)
        return R, s, t, {"warn": "N<3, return identity"}

    # ---- 转 numpy 给 Open3D ----
    src_np = src_pts.detach().cpu().double().numpy()
    tgt_np = tgt_pts.detach().cpu().double().numpy()

    # ---- 构造 Open3D 点云 ----
    pcd_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_np))
    pcd_tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_np))

    # ---- 一一对应的索引 (0..N-1) -> (0..N-1) ----
    corres = o3d.utility.Vector2iVector(np.stack([np.arange(N), np.arange(N)], axis=1).astype(np.int32))

    # ---- (1) Umeyama 初值：Sim(3) = (s, R, t) ----
    est_umeyama = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    T_umeyama = est_umeyama.compute_transformation(pcd_src, pcd_tgt, corres)  # 4x4
    # 分解 Sim(3)
    R_s = T_umeyama[:3, :3]
    t_init = T_umeyama[:3, 3]
    s_init = np.cbrt(np.linalg.det(R_s))
    if s_init == 0:
        s_init = 1.0
    R_init = R_s / s_init
    # 强制到 SO(3)
    U, _, Vt = np.linalg.svd(R_init)
    R_init = U @ Vt
    if np.linalg.det(R_init) < 0:
        U[:, -1] *= -1
        R_init = U @ Vt
        s_init = -s_init  # 把反射吸收到尺度

    # 用初值估计一个 ICP 距离阈值（目标包围盒尺度的 2%）
    bb = pcd_tgt.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bb.get_max_bound() - bb.get_min_bound())
    thr = max(icp_dist_ratio * diag, 1e-6)

    # ---- (2) ICP 细化（允许尺度）----
    # 说明：为细化 s，也用 PointToPoint(with_scaling=True)
    est_icp = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    # 注意：把 Umeyama 初值作为 init 传入
    result = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_tgt,
        max_correspondence_distance=thr,
        init=T_umeyama,
        estimation_method=est_icp,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter)
    )
    T_final = result.transformation

    # ---- 分解最终的 Sim(3) 得 R, s, t ----
    R_s_f = T_final[:3, :3]
    t_f   = T_final[:3, 3]
    s_f   = np.cbrt(np.linalg.det(R_s_f))
    if s_f == 0:
        s_f = 1.0
    R_f = R_s_f / s_f
    # 再做一次 SO(3) 纠正（数值稳健）
    U, _, Vt = np.linalg.svd(R_f)
    R_f = U @ Vt
    if np.linalg.det(R_f) < 0:
        U[:, -1] *= -1
        R_f = U @ Vt
        s_f = -s_f

    # ---- 转回 torch 并返回 ----
    R_torch = torch.tensor(R_f, dtype=dtype, device=device)
    s_torch = torch.tensor(s_f, dtype=dtype, device=device)
    t_torch = torch.tensor(t_f, dtype=dtype, device=device)

    extras = {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "T_umeyama": T_umeyama,
        "T_final": T_final,
        "thr": float(thr),
        "s_init": float(s_init)
    }
    return R_torch, s_torch, t_torch, extras

def depth2disp(depth,focal_length=200):
    
    disp = focal_length /(depth+1e-3)

    return disp

def normalize_robust(conf, lo=1.0, hi=99.0, eps=1e-8):
    """
    Robust normalization of confidence maps to [0,1].
    使用分位数裁剪 + min-max 归一化，抗异常值。

    Args:
        conf: numpy array, shape (H,W) or (B,H,W)
        lo: 下分位数百分比 (默认 1%)
        hi: 上分位数百分比 (默认 99%)
        eps: 避免除零

    Returns:
        numpy array, 同形状, 数值 ∈ [0,1]
    """
    conf = np.asarray(conf, dtype=np.float32)

    if conf.ndim == 2:   # 单张 (H,W)
        lo_v = np.percentile(conf, lo)
        hi_v = np.percentile(conf, hi)
        conf_clip = np.clip(conf, lo_v, hi_v)
        cmin, cmax = conf_clip.min(), conf_clip.max()
        return (conf_clip - cmin) / (cmax - cmin + eps)

    elif conf.ndim == 3: # 多张 (B,H,W)
        outs = []
        for img in conf:
            lo_v = np.percentile(img, lo)
            hi_v = np.percentile(img, hi)
            conf_clip = np.clip(img, lo_v, hi_v)
            cmin, cmax = conf_clip.min(), conf_clip.max()
            outs.append((conf_clip - cmin) / (cmax - cmin + eps))
        return np.stack(outs, axis=0)

    else:
        raise ValueError("conf must be shape (H,W) or (B,H,W)")
    
def sparse_depth_to_lidar_map(
    depth: torch.Tensor,          # (B, H, W), 稀疏深度（米），无效=0
    K: torch.Tensor,              # (B, 3, 3)，每帧内参
    valid_mask: torch.Tensor=None # (B, H, W) bool，可选；若不传则以 depth>0
) -> torch.Tensor:
    """
    返回: (B, H, W, 3)，相机坐标系 XYZ；无效像素置0
    公式: X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy,  Z = depth
    """
    assert depth.ndim == 3, "depth 应为 (B,H,W)"
    assert K.ndim == 3 and K.shape[1:] == (3,3), "K 应为 (B,3,3)"
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    if valid_mask is None:
        valid_mask = depth > 0
    else:
        assert valid_mask.shape == (B, H, W)
        valid_mask = valid_mask & (depth > 0)

    # 从 K 取 fx, fy, cx, cy （逐帧不同内参）
    fx = K[:, 0, 0].view(B, 1, 1).to(device=device, dtype=dtype)  # (B,1,1)
    fy = K[:, 1, 1].view(B, 1, 1).to(device=device, dtype=dtype)
    cx = K[:, 0, 2].view(B, 1, 1).to(device=device, dtype=dtype)
    cy = K[:, 1, 2].view(B, 1, 1).to(device=device, dtype=dtype)

    # 像素网格 (v,u)
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )  # (H,W)
    u = u.view(1, H, W).expand(B, -1, -1)  # (B,H,W)
    v = v.view(1, H, W).expand(B, -1, -1)  # (B,H,W)

    Z = depth
    # 计算 X,Y（广播到 B,H,W）
    X = (u - cx) * Z / (fx + 1e-12)
    Y = (v - cy) * Z / (fy + 1e-12)

    # 组装 (B,H,W,3)
    lidar_map = torch.stack([X, Y, Z], dim=-1)  # (B,H,W,3)

    # 无效像素置 0
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()
    lidar_map = torch.where(valid_mask.unsqueeze(-1), lidar_map, torch.zeros(1, dtype=dtype, device=device))

    return lidar_map

def save_tensor_to_ply(points: torch.Tensor, filename: str):
    """
    保存 (N,3) tensor 到 .ply 文件
    Args:
        points: torch.Tensor of shape (N,3)
        filename: 输出文件路径
    """
    assert points.ndim == 2 and points.shape[1] == 3, "输入必须是 (N,3)"
    
    # 转换成 numpy
    points_np = points.detach().cpu().numpy()

    # 构造 open3d 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # 写入 .ply
    o3d.io.write_point_cloud(filename, pcd)

def world_to_cam(X_w: torch.Tensor, T_c2w: torch.Tensor) -> torch.Tensor:
    """
    将点云从 world 系转换到 cam 系
    
    Args:
        X_w: (N,3) torch.Tensor, 世界坐标点
        T_c2w: (4,4) torch.Tensor, 相机的 cam2world 外参矩阵

    Returns:
        X_c: (N,3) torch.Tensor, 相机坐标点
    """
    device, dtype = X_w.device, X_w.dtype
    # 求逆：world->cam
    T_w2c = torch.linalg.inv(T_c2w.to(device=device, dtype=dtype))  # (4,4)

    # 齐次化点云
    ones = torch.ones((X_w.shape[0], 1), device=device, dtype=dtype)
    Xw_h = torch.cat([X_w, ones], dim=1)  # (N,4)

    # 变换
    Xc_h = (T_w2c @ Xw_h.T).T  # (N,4)

    return Xc_h[:, :3] / Xc_h[:, 3:].clamp(min=1e-8)

@torch.no_grad()
def align_sRt_with_fixed_scale(
    src_pts: torch.Tensor,   # (N,3) VGGT点 (cam系)
    tgt_pts: torch.Tensor,   # (N,3) GT点 (cam系)
    scale: torch.Tensor      # torch scalar, 由 estimate_scale_from_depth 提供
):
    """
    固定 scale，估计 R,t
    """
    device, dtype = src_pts.device, src_pts.dtype
    # 先把源点放到米制
    src_scaled = scale * src_pts

    # 转 numpy
    X = tgt_pts.detach().cpu().numpy()
    Y = src_scaled.detach().cpu().numpy()
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY

    H = Yc.T @ Xc / X.shape[0]
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt
    t = muX - R @ muY

    R_torch = torch.tensor(R, dtype=dtype, device=device)
    t_torch = torch.tensor(t, dtype=dtype, device=device)
    return R_torch, scale.to(dtype), t_torch

def apply_sim3_to_cam2world_single(
    T_c2w: torch.Tensor,   # (4,4)
    s: torch.Tensor | float,
    Rg: torch.Tensor,      # (3,3)
    tg: torch.Tensor       # (3,1) 或 (3,)
) -> torch.Tensor:
    """
    共轭施加 Sim(3) 到 cam2world：T' = S T S^{-1}
    S = [[s*Rg, tg],[0,1]]
    返回 T' 仍为 SE(3)（已在新世界系/米制下）
    """
    T = T_c2w
    R = T[:3, :3]
    t = T[:3, 3]

    s  = torch.as_tensor(s,  dtype=T.dtype, device=T.device)
    Rg = Rg.to(dtype=T.dtype, device=T.device)
    tg = tg.reshape(3).to(dtype=T.dtype, device=T.device)

    # R' = Rg * R * Rg^T
    Rprime = Rg @ R @ Rg.transpose(0,1)

    # t' = s*Rg*t + tg - R'*tg
    tprime = s * (Rg @ t) + tg - (Rprime @ tg)

    Tout = torch.eye(4, dtype=T.dtype, device=T.device)
    Tout[:3, :3] = Rprime
    Tout[:3,  3] = tprime
    return Tout

def estimate_sim3_with_fixed_scale_multi(
    vggt_pcd_data, vggt_extrinsic,
    gt_sparse_depth_data, gt_intrinsic,
    valid_sparse_mask,
    scale,                      # 已经确定的尺度 (torch scalar 或 float)
    frame_ids=(0,1,2,3,4),      # 用哪些帧来联合估计
    depth_min=2.0, depth_max=80.0
):
    """
    多帧联合 Sim(3) 估计，scale 已经确定，只解 R,t

    返回:
        R (3,3), s (标量), t (3,)
        extras 字典
    """
    all_src, all_tgt = [], []

    for f in frame_ids:
        # GT 稀疏深度 -> 相机系点云
        frame_gt_cam = sparse_depth_to_lidar_map(
            depth=gt_sparse_depth_data[f:f+1],
            K=gt_intrinsic[f:f+1],
            valid_mask=valid_sparse_mask[f:f+1]
        )[0]  # (H,W,3)
        mask = valid_sparse_mask[f]
        tgt_pts = frame_gt_cam[mask]   # (N,3)

        # VGGT 点云 (world) -> 该帧相机系
        est_pcd_world = vggt_pcd_data[0][f][mask]  # (N,3)
        est_pose = vggt_extrinsic[0][f]            # (4,4)
        est_pcd_cam = world_to_cam(est_pcd_world, est_pose)  # (N,3)

        # 限制深度范围
        keep = (tgt_pts[:,2] > depth_min) & (tgt_pts[:,2] < depth_max) \
               & torch.isfinite(est_pcd_cam).all(dim=1)
        src = est_pcd_cam[keep]
        tgt = tgt_pts[keep]

        if src.shape[0] > 50:
            all_src.append(src)
            all_tgt.append(tgt)

    # 拼接所有帧
    src_all = torch.cat(all_src, dim=0)
    tgt_all = torch.cat(all_tgt, dim=0)

    # --- 固定 scale 跑 Kabsch ---
    X = tgt_all.cpu().numpy()
    Y = src_all.cpu().numpy()
    muX, muY = X.mean(0), Y.mean(0)

    Xc = X - muX
    Yc = Y - muY
    H = (scale * Yc).T @ Xc / X.shape[0]
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt
    t = muX - scale * (R @ muY)

    R_torch = torch.tensor(R, dtype=src_all.dtype, device=src_all.device)
    s_torch = torch.tensor(scale, dtype=src_all.dtype, device=src_all.device)
    t_torch = torch.tensor(t, dtype=src_all.dtype, device=src_all.device)

    extras = {
        "frames_used": frame_ids,
        "num_pairs": X.shape[0],
        "scale_fixed": float(scale),
    }
    return R_torch, s_torch, t_torch, extras

def Compute_the_SIM3(vggt_depth,
                     vggt_depth_conf,
                     vggt_pcd,
                     vggt_pcd_conf,
                     vggt_extrinsic,
                     estimated_depth,
                     gt_instrinsic,
                     estimated_depth_filter_z_near=3,
                     estimated_depth_filter_z_far=30,
                     vggt_depth_conf_thresh=50,
                     ):
    '''
    vggt_depth: [1,V,H,W,1]
    vggt_depth_conf: [1,V,H,W]
    vggt_pcd: [1,V,H,W,3]
    vggt_pcd_conf: [1,V,H,W]
    vggt_extrinsic: [1,V,4,4]
    
    estimated_depth: [1,V,H,W]--> From the GS
    gt_instrinsic: [1,V,3,3]---> From the Sensor

    '''
    # compute the scale based on the depth median
    scale_list = []
    valid_mask_new = []
    for frame_index in range(vggt_depth.shape[1]):
        current_estimated_depth = estimated_depth[0][frame_index]
        current_valid_mask1 = current_estimated_depth >estimated_depth_filter_z_near
        current_valid_mask2 = current_estimated_depth<=estimated_depth_filter_z_far
        current_estimated_depth_threshold = current_valid_mask1 * current_valid_mask2
        
        current_vggt_depth = vggt_depth[0][frame_index].squeeze(-1)
        current_vggt_conf = vggt_depth_conf[0][frame_index]
        
        thresh = torch.quantile(current_vggt_conf[current_estimated_depth_threshold ], q=vggt_depth_conf_thresh/100.0)
        mask_for_vggt_mask = current_vggt_conf>thresh

        
        comphensive_mask = current_estimated_depth_threshold * mask_for_vggt_mask
        current_estimate_depth_selective_median = current_estimated_depth[comphensive_mask].median()
        current_vggt_depth_selective_median = current_vggt_depth[comphensive_mask].median()
        valid_mask_new.append(comphensive_mask.unsqueeze(0))
        scale = current_estimate_depth_selective_median/current_vggt_depth_selective_median 
        
        scale_list.append(scale)
    
    scale = sum(scale_list)/len(scale_list)
    comphensive_selective_mask = torch.cat(valid_mask_new,dim=0)



    R, s, t, extras = estimate_sim3_with_fixed_scale_multi(vggt_pcd_data=vggt_pcd,
                                         vggt_extrinsic=vggt_extrinsic,
                                         gt_intrinsic=gt_instrinsic[0],
                                         gt_sparse_depth_data=estimated_depth[0],
                                         valid_sparse_mask=comphensive_selective_mask,
                                         scale=float(scale.item()),
                                         frame_ids=range(estimated_depth[0].shape[0]),
                                         depth_min=estimated_depth_filter_z_near,
                                         depth_max=estimated_depth_filter_z_far)
    
    return R, s,t


def Covered_the_Pose_Relative_to_First(vggt_depth,
                     vggt_depth_conf,
                     vggt_pcd,
                     vggt_pcd_conf,
                     vggt_extrinsic,
                     estimated_depth,
                     gt_instrinsic,
                     estimated_depth_filter_z_near=3,
                     estimated_depth_filter_z_far=30,
                     vggt_depth_conf_thresh=50):
    '''
    vggt_depth: [1,V,H,W,1]
    vggt_depth_conf: [1,V,H,W]
    vggt_pcd: [1,V,H,W,3]
    vggt_pcd_conf: [1,V,H,W]
    vggt_extrinsic: [1,V,4,4]
    
    estimated_depth: [1,V,H,W]--> From the GS
    gt_instrinsic: [1,V,3,3]---> From the Sensor

    '''
    R, s, t = Compute_the_SIM3(
                        vggt_depth,
                        vggt_depth_conf,
                        vggt_pcd,
                        vggt_pcd_conf,
                        vggt_extrinsic,
                        estimated_depth,
                        gt_instrinsic,
                        estimated_depth_filter_z_near,
                        estimated_depth_filter_z_far,
                        vggt_depth_conf_thresh)

    recovered_pose_first = apply_sim3_to_cam2world_single(
            T_c2w=vggt_extrinsic[0][0],
            s=s,
            Rg=R,
            tg=t
        )
        
    
    recovered_pose_list = []
    for frame_index in range(vggt_extrinsic.shape[1]):
        estimated_vggt_pose_current = vggt_extrinsic[0][frame_index]
        recovered_pose = apply_sim3_to_cam2world_single(
            T_c2w=estimated_vggt_pose_current,
            s=s,
            Rg=R,
            tg=t
        )
        recovered_pose = torch.linalg.inv(recovered_pose_first) @ recovered_pose
        recovered_pose = recovered_pose.unsqueeze(0).unsqueeze(0)
        recovered_pose_list.append(recovered_pose)
    
    return torch.cat(recovered_pose_list,dim=1)
        
        

if __name__=="__main__":
    
    # get the vggt point cloud and conf
    
    # get the vggt depth and conf
    vggt_depth_data = torch.load("vggt_infered/vggt_depth.pt") #torch.Size([1, 16, 112, 518, 1])
    vggt_depth_conf_data = torch.load("vggt_infered/vggt_depth_conf.pt") #torch.Size([1, 16, 112, 518])
    vggt_pcd_data = torch.load("vggt_infered/vggt_pcd.pt") #torch.Size([1, 16, 112, 518, 3])
    vggt_pcd_conf_data = torch.load("vggt_infered/vggt_pcd_conf.pt")#torch.Size([1, 16, 112, 518])
    vggt_estimated_extrinsic = torch.load("vggt_infered/vggt_extrinsic.pt") #(1,16,4,4)


    our_psuedo_depth_data = torch.load("GTs/preprocessed_depth.pt") #torch.Size([16, 112, 518])
    our_sparse_gt_depth_data = torch.load("GTs/gt_sparse_lidar_depth.pt") #torch.Size([16, 112, 518])
    gt_intrinsic = torch.load("GTs/gt_intrinsic.pt") #(16,3,3)
    gt_extrinsic = torch.load("GTs/gt_extrinsic.pt")
    
    our_psuedo_depth_data = our_psuedo_depth_data.unsqueeze(0)
    our_sparse_gt_depth_data = our_sparse_gt_depth_data.unsqueeze(0)
    gt_intrinsic = gt_intrinsic.unsqueeze(0)
    


    recovered_pose = Covered_the_Pose_Relative_to_First(
        vggt_depth=vggt_depth_data,
                     vggt_depth_conf=vggt_depth_conf_data,
                     vggt_pcd=vggt_pcd_data,
                     vggt_pcd_conf=vggt_pcd_conf_data,
                     vggt_extrinsic=vggt_estimated_extrinsic,
                     gt_instrinsic=gt_intrinsic,
                     estimated_depth=our_psuedo_depth_data,
                     estimated_depth_filter_z_near=3,
                     estimated_depth_filter_z_far=30,
                     vggt_depth_conf_thresh=50
    )
    






    

    

    
    
    
    

    
    
    
    
    pass
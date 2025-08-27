import numpy as np
import torch
from typing import Dict, Tuple

# ----------------------------
# 基础 SE(3) / 相机中心工具
# ----------------------------
def invert_se3_torch(T: torch.Tensor) -> torch.Tensor:
    """
    4x4 SE3 逆：仅旋转+平移
    Args:  T: (..., 4, 4)
    Returns: (..., 4, 4)
    """
    assert T.shape[-2:] == (4, 4)
    R = T[..., :3, :3]                      # (...,3,3)
    t = T[..., :3, 3:4]                     # (...,3,1)
    Rinv = R.transpose(-1, -2)              # (...,3,3)
    tinv = -Rinv @ t                        # (...,3,1)

    Tout = torch.zeros_like(T)
    Tout[..., :3, :3] = Rinv
    Tout[..., :3, 3:4] = tinv
    Tout[..., 3, 3] = 1.0
    return Tout

def poses_to_cam_centers_cam2world_torch(T_c2w: torch.Tensor) -> torch.Tensor:
    """
    cam2world 的相机中心是平移列
    Args:  T_c2w: (..., 4, 4)
    Returns: (..., 3)
    """
    return T_c2w[..., :3, 3]

def poses_to_cam_centers_world2cam_torch(T_w2c: torch.Tensor) -> torch.Tensor:
    """
    world2cam 的相机中心 = -R^T t
    Args:  T_w2c: (..., 4, 4)
    Returns: (..., 3)
    """
    R = T_w2c[..., :3, :3]                  # (...,3,3)
    t = T_w2c[..., :3, 3:4]                 # (...,3,1)
    Rt = R.transpose(-1, -2)                # (...,3,3)
    C = -(Rt @ t)[..., 0]                   # (...,3)
    return C

# ----------------------------
# Umeyama (可批量)
# ----------------------------
def umeyama_similarity_torch(
    X: torch.Tensor, Y: torch.Tensor, with_scaling: bool = True, eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Umeyama: 求 s, R, t 使得 Y ≈ s R X + t
    支持批量：
      X, Y: (B, N, 3) 或 (N, 3)（自动升维）
    Returns:
      s: (B,) 或标量张量
      R: (B, 3, 3)
      t: (B, 3)
    """
    # 统一到同一 device/dtype，以避免 device mismatch
    Y = Y.to(device=X.device, dtype=X.dtype)

    if X.ndim == 2:  # (N,3) -> (1,N,3)
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    B, N, _ = X.shape
    dtype, device = X.dtype, X.device

    muX = X.mean(dim=1, keepdim=True)                      # (B,1,3)
    muY = Y.mean(dim=1, keepdim=True)                      # (B,1,3)
    Xc = X - muX                                           # (B,N,3)
    Yc = Y - muY                                           # (B,N,3)

    # Sigma = (Yc^T Xc) / N
    Sigma = torch.matmul(Yc.transpose(1, 2), Xc) / float(N)   # (B,3,3)

    # batched SVD
    U, D, Vh = torch.linalg.svd(Sigma)                        # U (B,3,3), D (B,3), Vh (B,3,3)

    # 处理反射：S = diag(1,1,sign(det(U Vh)))
    UV = U @ Vh                                               # (B,3,3)
    det = torch.det(UV)                                       # (B,)
    Sdiag = torch.ones((B, 3), dtype=dtype, device=device)    # (B,3)
    Sdiag[:, -1] = torch.where(det < 0, -1.0, 1.0)            # (B,)

    S = torch.diag_embed(Sdiag)                               # (B,3,3)
    R = U @ S @ Vh                                            # (B,3,3)

    if with_scaling:
        varX = (Xc ** 2).sum(dim=(1, 2)) / float(N)          # (B,)
        trace_DS = (D * Sdiag).sum(dim=-1)                   # (B,)
        s = trace_DS / (varX + eps)                           # (B,)
    else:
        s = torch.ones((B,), dtype=dtype, device=device)

    # t = muY - s R muX
    # muX, muY: (B,1,3)  -> 列向量以便矩阵乘法
    t = (muY[:, 0] - (s.view(B, 1, 1) * (R @ muX[:, 0].unsqueeze(-1))).squeeze(-1))  # (B,3)

    if squeeze_back:
        s, R, t = s[0], R[0], t[0]
    return s, R, t

# ----------------------------
# 将 Sim(3) 作用到 cam2world
# ----------------------------
def apply_sim3_to_cam2world_torch(
    T_c2w: torch.Tensor, s: torch.Tensor, Rg: torch.Tensor, tg: torch.Tensor
) -> torch.Tensor:
    """
    对一组 cam2world 施加 Sim(3)： T' = [ Rg*R | s*Rg*t + tg ]
    支持批量/向量化：
      T_c2w: (B, V, 4, 4) 或 (V, 4, 4) 或 (4, 4)
      s   : (B,) 或标量张量
      Rg  : (B, 3, 3) 或 (3, 3)
      tg  : (B, 3) 或 (3,)
    Returns:
      T_out: 与 T_c2w 同形状
    """
    # 统一输入形状
    if T_c2w.ndim == 2:
        T_c2w = T_c2w.unsqueeze(0).unsqueeze(0)  # (1,1,4,4)
        squeeze = (True, True)
    elif T_c2w.ndim == 3:
        T_c2w = T_c2w.unsqueeze(0)               # (1,V,4,4)
        squeeze = (True, False)
    else:
        squeeze = (False, False)                 # already (B,V,4,4)

    B, V = T_c2w.shape[:2]
    device = T_c2w.device
    dtype  = T_c2w.dtype

    # 强制把 s, Rg, tg 放到与 T_c2w 一致的 device/dtype
    s  = s.to(device=device, dtype=dtype)
    Rg = Rg.to(device=device, dtype=dtype)
    tg = tg.to(device=device, dtype=dtype)

    # 统一 s, Rg, tg 的 batch 维度
    if s.ndim == 0: s = s.reshape(1).repeat(B)
    if Rg.ndim == 2: Rg = Rg.unsqueeze(0).repeat(B, 1, 1)     # (B,3,3)
    if tg.ndim == 1: tg = tg.unsqueeze(0).repeat(B, 1)        # (B,3)

    R = T_c2w[..., :3, :3]                                    # (B,V,3,3)
    t = T_c2w[..., :3, 3:4]                                   # (B,V,3,1)

    # Rp = Rg @ R
    Rp = torch.matmul(Rg.view(B, 1, 3, 3).expand(B, V, 3, 3), R)  # (B,V,3,3)

    # tp = s * (Rg @ t) + tg
    Rt = torch.matmul(Rg.view(B, 1, 3, 3).expand(B, V, 3, 3), t)  # (B,V,3,1)
    tp = s.view(B, 1, 1, 1) * Rt + tg.view(B, 1, 3, 1)            # (B,V,3,1)

    T_out = torch.zeros_like(T_c2w)
    T_out[..., :3, :3] = Rp
    T_out[..., :3, 3:4] = tp
    T_out[..., 3, 3] = 1.0

    # 还原到输入形状
    if squeeze == (True, True):
        T_out = T_out[0, 0]
    elif squeeze == (True, False):
        T_out = T_out[0]
    return T_out

# ----------------------------
# 主对齐函数
# ----------------------------
@torch.no_grad()
def align_vggt_to_gt_sim3_torch(
    T_vggt: torch.Tensor,
    T_gt: torch.Tensor,
    poses_are_cam2world: bool = True,
    use_scaling: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    PyTorch 版本的对齐：
      输入:
        T_vggt, T_gt: (B, V, 4, 4) 两组位姿
        poses_are_cam2world: 若为 False，输入视为 world2cam，会先转 cam2world
        use_scaling: Umeyama 是否估计尺度
      输出:
        字典：
          s: (B,) 或标量（当 B=1）
          R: (B,3,3)
          t: (B,3)
          T_vggt_aligned: 与输入同形状的对齐后位姿（与输入同为 cam2world 或 world2cam）
          center_rmse: (B,) 对齐后的相机中心 RMSE
    """
    assert T_vggt.shape == T_gt.shape and T_vggt.shape[-2:] == (4, 4)
    B, V = T_vggt.shape[:2]

    # 统一到同一 device/dtype，避免 device mismatch
    device = T_vggt.device
    dtype  = T_vggt.dtype
    T_gt   = T_gt.to(device=device, dtype=dtype)

    if poses_are_cam2world:
        Tv_c2w = T_vggt
        Tg_c2w = T_gt
        Cv = poses_to_cam_centers_cam2world_torch(Tv_c2w)  # (B,V,3)
        Cg = poses_to_cam_centers_cam2world_torch(Tg_c2w)  # (B,V,3)
    else:
        Tv_c2w = invert_se3_torch(T_vggt)                  # (B,V,4,4)
        Tg_c2w = invert_se3_torch(T_gt)
        Cv = poses_to_cam_centers_cam2world_torch(Tv_c2w)  # (B,V,3)
        Cg = poses_to_cam_centers_cam2world_torch(Tg_c2w)

    # Umeyama（按 batch 求）
    s, Rg, tg = umeyama_similarity_torch(Cv, Cg, with_scaling=use_scaling)  # s:(B,), R:(B,3,3), t:(B,3)

    # 施加到整组 cam2world
    T_aligned_c2w = apply_sim3_to_cam2world_torch(Tv_c2w, s, Rg, tg)        # (B,V,4,4)

    # 计算相机中心 RMSE
    C_aligned = T_aligned_c2w[..., :3, 3]                                    # (B,V,3)
    rmse = torch.sqrt(((C_aligned - Cg) ** 2).sum(dim=-1).mean(dim=-1))      # (B,)

    # 若原输入是 world2cam，则转回
    if not poses_are_cam2world:
        T_vggt_aligned = invert_se3_torch(T_aligned_c2w)                     # (B,V,4,4)
    else:
        T_vggt_aligned = T_aligned_c2w

    # 当 B=1 时，和 numpy 版一致：返回标量/单批
    return {
        "s": s if B > 1 else s[0],
        "R": Rg if B > 1 else Rg[0],
        "t": tg if B > 1 else tg[0],
        "T_vggt_aligned": T_vggt_aligned,
        "center_rmse": rmse if B > 1 else rmse[0],
    }

# ----------------------------
# 使用示例
# ----------------------------
if __name__=="__main__":
    # 加载
    vggt_estimated_pose = torch.load(
        "/home/zliu/Project2025/FeedStereoGS/codes/evaluations/nopose_increment/VGGT_Based/vggt_infered/vggt_extrinsic.pt"
    )
    GT_pose = torch.load(
        "/home/zliu/Project2025/FeedStereoGS/codes/evaluations/nopose_increment/VGGT_Based/GTs/gt_extrinsic.pt"
    )

    # （可选）把二者统一到同一 device / dtype（例如 GPU float32）
    # 如果你想在 GPU 上跑，取消下面两行注释
    # device = torch.device("cuda", 0)
    # vggt_estimated_pose, GT_pose = vggt_estimated_pose.to(device), GT_pose.to(device)

    # 也可统一 dtype：例如 float32
    vggt_estimated_pose = vggt_estimated_pose.float()
    GT_pose = GT_pose.to(dtype=vggt_estimated_pose.dtype, device=vggt_estimated_pose.device)

    result = align_vggt_to_gt_sim3_torch(
        T_vggt=vggt_estimated_pose,
        T_gt=GT_pose,
        poses_are_cam2world=True,
        use_scaling=True
    )

    print(result["s"])
    print(result["R"])
    print(result["t"])
    # print("--------------")
    # # 简单检查
    # print(result["T_vggt_aligned"][0][10])
    # print("---------------------------")
    # print(GT_pose[0][10])

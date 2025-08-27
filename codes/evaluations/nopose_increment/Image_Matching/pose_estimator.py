# robust_pose_from_two_depths.py
import torch
import torch.nn.functional as F

# ================= SE(3) 基础 =================
def skew(w):  # [...,3] -> [...,3,3]
    wx, wy, wz = w[...,0], w[...,1], w[...,2]
    O = torch.zeros(w.shape[:-1]+(3,3,), device=w.device, dtype=w.dtype)
    O[...,0,1], O[...,0,2] = -wz,  wy
    O[...,1,0], O[...,1,2] =  wz, -wx
    O[...,2,0], O[...,2,1] = -wy,  wx
    return O

def se3_exp(xi):  # xi: [...,6] (v,w)
    v, w = xi[..., :3], xi[..., 3:]
    th = torch.linalg.norm(w, dim=-1, keepdim=True)          # [...,1]
    device, dtype = xi.device, xi.dtype
    I3 = torch.eye(3, device=device, dtype=dtype).expand(xi.shape[:-1]+(3,3))

    eps = 1e-8
    th_safe = torch.clamp(th, min=eps)
    w_unit = w / th_safe
    W_hat = skew(w_unit)

    th2 = th * th
    use_taylor = (th < 1e-4)

    A = torch.where(use_taylor, 1 - th2/6 + th2*th2/120,
                    torch.sin(th)/th_safe)
    B = torch.where(use_taylor, 0.5 - th2/24 + th2*th2/720,
                    (1 - torch.cos(th)) / (th2 + eps))
    C = torch.where(use_taylor, 1/6 - th2/120 + th2*th2/5040,
                    (th - torch.sin(th)) / (th2*th_safe + eps))

    R = I3 + A[...,None,None]*W_hat + B[...,None,None]*(W_hat @ W_hat)
    V = I3 + B[...,None,None]*W_hat + C[...,None,None]*(W_hat @ W_hat)

    t = (V @ v[...,None])[...,0]
    T = torch.eye(4, device=device, dtype=dtype).expand(xi.shape[:-1]+(4,4)).clone()
    T[..., :3,:3] = R
    T[..., :3, 3] = t
    return T

def compose(T1, T2):  # 左乘
    return T1 @ T2

def inv_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = torch.eye(4, device=T.device, dtype=T.dtype)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti

def orthonormalize_R(R):
    U,S,Vt = torch.linalg.svd(R)
    Rn = U @ Vt
    if torch.linalg.det(Rn) < 0:
        Vt[-1,:] *= -1
        Rn = U @ Vt
    return Rn

# ================= 相机与几何 =================
def backproject_depth(K, depth):  # depth [H,W], meters
    H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    v = torch.arange(H, device=device, dtype=dtype)
    u = torch.arange(W, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing='ij')  # [H,W]

    z = depth
    X = (uu - cx) / fx * z
    Y = (vv - cy) / fy * z
    return torch.stack([X, Y, z], dim=-1)         # [H,W,3]

def depth_normals_from_points(P):  # P [H,W,3]
    N = torch.zeros_like(P)
    Px = P[1:-1,2:] - P[1:-1,1:-1]
    Py = P[2:,1:-1] - P[1:-1,1:-1]
    n = torch.linalg.cross(Py, Px)                # [H-2,W-2,3]
    n = F.normalize(n, dim=-1, eps=1e-8)
    N[1:-1,1:-1] = n
    # 针孔 +Z 前进方向；法线应朝向相机（z 负向）。若 n_z > 0 则翻转
    flip = (N[...,2] > 0)
    N[flip] = -N[flip]
    return N

def project_points(K, P):                         # P [...,3] -> uv [...,2]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x,y,z = P[...,0], P[...,1], P[...,2].clamp_min(1e-12)
    u = fx * (x/z) + cx
    v = fy * (y/z) + cy
    return torch.stack([u,v], dim=-1)

def build_pyramid(depth, levels):
    pyr = [depth]
    for _ in range(1, levels):
        depth = F.avg_pool2d(depth[None,None], 2, stride=2)[0,0]
        pyr.append(depth)
    return pyr

def scale_K(K, s):
    K2 = K.clone()
    K2[0,0] *= s; K2[1,1] *= s
    K2[0,2] *= s; K2[1,2] *= s
    return K2

def bilinear_sample(arrHW, kps):  # arrHW [H,W], kps [N,2] (u,v)
    H, W = arrHW.shape
    x = (kps[:,0] / (W-1))*2 - 1
    y = (kps[:,1] / (H-1))*2 - 1
    grid = torch.stack([x,y],dim=-1).view(1,1,-1,2)
    val = F.grid_sample(arrHW[None,None], grid, align_corners=True)[0,0,0]  # [N]
    return val

# ================= 稀疏 3D–3D RANSAC 初值 =================
def svd_ransac_init(K1, K2, depth1, depth2, kps1, kps2,
                    max_depth=30.0, iters=1000, th=0.20, device=None):
    if device is None:
        device = depth1.device
    z1 = bilinear_sample(depth1, kps1)
    z2 = bilinear_sample(depth2, kps2)
    mask = (z1>0)&(z2>0)&(z1<=max_depth)&(z2<=max_depth)&torch.isfinite(z1)&torch.isfinite(z2)
    if mask.sum() < 3:
        return torch.eye(4, device=device, dtype=depth1.dtype)

    k1, k2, z1, z2 = kps1[mask], kps2[mask], z1[mask], z2[mask]

    def back(kps, z, K):
        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
        X = torch.stack([(kps[:,0]-cx)/fx, (kps[:,1]-cy)/fy, torch.ones_like(z)], dim=-1)
        return X*z.unsqueeze(-1)

    X1 = back(k1, z1, K1)  # cam1
    X2 = back(k2, z2, K2)  # cam2

    best_inl = -1
    best_R = None; best_t = None
    M = X1.shape[0]
    for _ in range(iters):
        idx = torch.randint(0, M, (3,), device=device)
        mu1, mu2 = X1[idx].mean(0), X2[idx].mean(0)
        U,S,Vt = torch.linalg.svd((X2[idx]-mu2).t() @ (X1[idx]-mu1))
        R = Vt.t() @ U.t()
        if torch.linalg.det(R) < 0:
            Vt[-1,:] *= -1; R = Vt.t() @ U.t()
        t = mu1 - R @ mu2
        err = torch.linalg.norm(X1 - (X2 @ R.T + t), dim=1)
        inl = (err < th).sum().item()
        if inl > best_inl:
            best_inl = inl; best_R = R; best_t = t

    if best_R is None:
        return torch.eye(4, device=device, dtype=depth1.dtype)

    # 用内点精炼一次
    err = torch.linalg.norm(X1 - (X2 @ best_R.T + best_t), dim=1)
    inliers = err < th
    if inliers.sum() >= 3:
        mu1, mu2 = X1[inliers].mean(0), X2[inliers].mean(0)
        U,S,Vt = torch.linalg.svd((X2[inliers]-mu2).t() @ (X1[inliers]-mu1))
        R = Vt.t() @ U.t()
        if torch.linalg.det(R) < 0:
            Vt[-1,:] *= -1; R = Vt.t() @ U.t()
        t = mu1 - R @ mu2
    else:
        R, t = best_R, best_t

    T = torch.eye(4, device=device, dtype=depth1.dtype)
    T[:3,:3] = R; T[:3,3] = t
    return T

# ================= Trimmed 点到平面 ICP（自适应深度容差） =================
@torch.no_grad()
def refine_icp_point2plane_trimmed(
    K1, K2, depth1, depth2, T_init,
    iters_per_level=(8,5,4),
    huber_delta=0.03,            # Huber (m)
    max_samples=80000,           # 每层最大采样数
    z_match_base=0.05,           # 基础深度一致阈值 (m)
    z_match_rel=0.02,            # 相对阈值 (比例 * z_ref)
    trim_ratio=0.8,              # Trimmed ICP 保留比例
    normal_facing_thresh=0.1,    # 保留 n_z < -thr 的法线（已翻转朝向相机）
    max_depth=30.0,
    damping=1e-6,
):
    device, dtype = depth1.device, depth1.dtype
    levels = len(iters_per_level)

    # 预处理：仅 (0, max_depth]
    d1 = depth1.clone()
    d2 = depth2.clone()
    d1[(~torch.isfinite(d1)) | (d1<=0) | (d1>max_depth)] = 0.0
    d2[(~torch.isfinite(d2)) | (d2<=0) | (d2>max_depth)] = 0.0

    pyr1 = build_pyramid(d1, levels)
    pyr2 = build_pyramid(d2, levels)
    Kpyr1 = [scale_K(K1, 1/(2**i)) for i in range(levels)]
    Kpyr2 = [scale_K(K2, 1/(2**i)) for i in range(levels)]

    T = T_init.clone()
    final_cost = float('inf')

    for lvl in reversed(range(levels)):  # coarse -> fine
        d1l, d2l = pyr1[lvl], pyr2[lvl]
        K1l, K2l = Kpyr1[lvl], Kpyr2[lvl]
        H, W = d1l.shape

        P1 = backproject_depth(K1l, d1l)            # [H,W,3]
        N1 = depth_normals_from_points(P1)          # [H,W,3]
        invalid1 = (d1l <= 0)
        P1[invalid1] = torch.tensor([float('nan')]*3, device=device, dtype=dtype)
        N1[invalid1] = torch.tensor([float('nan')]*3, device=device, dtype=dtype)

        P2 = backproject_depth(K2l, d2l).view(-1,3) # [HW,3]

        # 采样
        M_all = P2.shape[0]
        if M_all > max_samples:
            idx = torch.randperm(M_all, device=device)[:max_samples]
        else:
            idx = torch.arange(M_all, device=device)
        P2s = P2[idx]

        last_r = None

        for _ in range(iters_per_level[lvl]):
            P2_in_1 = (T[:3,:3] @ P2s.t()).t() + T[:3,3]        # [M,3]
            uv = project_points(K1l, P2_in_1)                   # [M,2]

            x = (uv[:,0]/(W-1))*2 - 1
            y = (uv[:,1]/(H-1))*2 - 1
            grid = torch.stack([x,y], dim=-1).view(1,1,-1,2)

            P1_s = F.grid_sample(P1.permute(2,0,1).unsqueeze(0), grid, align_corners=True)[0, :, 0, :].T
            N1_s = F.grid_sample(N1.permute(2,0,1).unsqueeze(0), grid, align_corners=True)[0, :, 0, :].T

            in_img = (uv[:,0] >= 1) & (uv[:,0] <= W-2) & (uv[:,1] >= 1) & (uv[:,1] <= H-2)
            valid_p = torch.isfinite(P1_s).all(dim=-1) & torch.isfinite(N1_s).all(dim=-1)

            z_proj = P2_in_1[:,2]
            z_ref  = P1_s[:,2]
            z_ok = (z_ref>0) & (z_proj>0) & ((z_proj - z_ref).abs() < (z_match_base + z_match_rel*z_ref))
            # 法向朝向/质量筛选：n_z < -normal_facing_thresh
            n = F.normalize(N1_s, dim=-1, eps=1e-8)
            n_ok = (n[:,2] < -normal_facing_thresh)

            valid = in_img & valid_p & z_ok & n_ok
            if valid.sum() < 200:
                break

            p    = P2_in_1[valid]
            pref = P1_s[valid]
            n    = n[valid]

            r = (n * (pref - p)).sum(dim=-1)                 # [m]
            last_r = r

            # Trimmed ICP：只保留前 trim_ratio 的小残差
            keep = int(trim_ratio * r.numel())
            if keep >= 10:
                topk = torch.topk(-r.abs(), keep).indices
                p, pref, n, r = p[topk], pref[topk], n[topk], r[topk]

            # Huber 权重
            a = r.abs()
            w = torch.where(a <= huber_delta, torch.ones_like(a), huber_delta / a)

            # 雅可比：J = [ -n^T | n^T [p]_x ]
            px, py, pz = p[:,0], p[:,1], p[:,2]
            J_t = -n
            J_w = torch.stack([ n[:,1]*pz - n[:,2]*py,
                                n[:,2]*px - n[:,0]*pz,
                                n[:,0]*py - n[:,1]*px ], dim=-1)
            J = torch.cat([J_t, J_w], dim=-1)                # [m,6]

            WJ = J * w[:,None]
            A = WJ.transpose(0,1) @ J
            b = WJ.transpose(0,1) @ r
            A = A + torch.eye(6, device=device, dtype=dtype)*damping

            dx = torch.linalg.solve(A, b)                    # [6]
            if torch.isnan(dx).any() or torch.isinf(dx).any():
                break
            dx = dx.clamp(min=-0.5, max=0.5)

            dT = se3_exp(dx)
            T  = compose(dT, T)
            T[:3,:3] = orthonormalize_R(T[:3,:3])

            if torch.linalg.norm(dx) < 1e-6:
                break

        if last_r is not None and last_r.numel() > 0:
            final_cost = float(last_r.abs().median().item())

    return T, final_cost  # cam2 -> cam1, 以及最终中位残差

# ================= 多初值一键求解 =================
def estimate_pose_depth2depth(
    K1, K2, depth1, depth2, kps1, kps2,
    # RANSAC
    ransac_iters=1500, ransac_th=0.20,
    # ICP
    iters_per_level_full=(10,7,5),
    iters_per_level_quick=(3,2,2),
    huber_delta=0.03,
    max_samples=100000,
    z_match_base=0.05,
    z_match_rel=0.02,
    trim_ratio=0.8,
    normal_facing_thresh=0.1,
    max_depth=30.0,
    damping=1e-6,
    device=None
):
    if device is None:
        device = depth1.device

    K1 = K1.to(device).float(); K2 = K2.to(device).float()
    depth1 = depth1.to(device).float()
    depth2 = depth2.to(device).float()
    kps1 = kps1.to(device).float(); kps2 = kps2.to(device).float()

    # 初值 A：3D–3D RANSAC
    T0a = svd_ransac_init(K1, K2, depth1, depth2, kps1, kps2,
                          max_depth=max_depth, iters=ransac_iters, th=ransac_th, device=device)
    # 初值 B：单位阵
    T0b = torch.eye(4, device=device, dtype=depth1.dtype)
    # 初值 C：A 的逆（以防方向弄反时更好）
    T0c = inv_T(T0a)

    # 先用 quick ICP 跑三份初值，选残差最小的
    cand = []
    for T0 in (T0a, T0b, T0c):
        Tq, costq = refine_icp_point2plane_trimmed(
            K1, K2, depth1, depth2, T0,
            iters_per_level=iters_per_level_quick,
            huber_delta=huber_delta,
            max_samples=max_samples,
            z_match_base=z_match_base,
            z_match_rel=z_match_rel,
            trim_ratio=trim_ratio,
            normal_facing_thresh=normal_facing_thresh,
            max_depth=max_depth,
            damping=damping
        )
        cand.append((costq, Tq))
    cand.sort(key=lambda x: x[0])
    T_best0 = cand[0][1]

    # 用最佳初值做 full ICP
    T_final, _ = refine_icp_point2plane_trimmed(
        K1, K2, depth1, depth2, T_best0,
        iters_per_level=iters_per_level_full,
        huber_delta=huber_delta,
        max_samples=max_samples,
        z_match_base=z_match_base,
        z_match_rel=z_match_rel,
        trim_ratio=trim_ratio,
        normal_facing_thresh=normal_facing_thresh,
        max_depth=max_depth,
        damping=damping
    )
    return T_final  # cam2 -> cam1

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rscipy

def transform_positions(positions, c2w):
    """
    将 G2 的 mean3D 位置变换到 G1 世界坐标系。
    positions: (N, 3)
    c2w: (4, 4)
    """
    N = positions.shape[0]
    homo_positions = torch.cat([positions, torch.ones(N, 1, device=positions.device)], dim=-1)  # (N, 4)
    transformed = (c2w @ homo_positions.T).T[:, :3]  # (N, 3)
    return transformed

def quaternion_multiply(q1, q2):
    """
    四元数乘法：q = q1 * q2
    q1, q2: (N, 4)  [w, x, y, z]
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=1)

def transform_quaternions(q_old, c2w):
    """
    将 G2 的旋转四元数变换到 G1 坐标系。
    q_old: (N, 4) [w, x, y, z]
    c2w: (4, 4)
    """
    R = c2w[:3, :3]  # 提取旋转部分
    R_c2w = Rscipy.from_matrix(R.cpu().numpy())
    q_c2w = R_c2w.as_quat()  # [x, y, z, w]
    q_c2w = torch.tensor([q_c2w[3], q_c2w[0], q_c2w[1], q_c2w[2]], device=q_old.device)  # 转为 [w, x, y, z]
    q_c2w = q_c2w.unsqueeze(0).repeat(q_old.shape[0], 1)  # (N, 4)
    return quaternion_multiply(q_c2w, q_old)

def transform_g2_to_g1(g1, g2, c2w):
    """
    输入：
        g1: [1, N1, 14]，主参考坐标系中的高斯
        g2: [1, N2, 14]，待变换到 g1 坐标系的高斯
        c2w: [4, 4]，将 g2 的高斯变换到 g1 坐标系
    输出：
        merged: [1, N1 + N2, 14]，融合后的高斯组
    """
    g2 = g2.squeeze(0)  # -> (N2, 14)

    mean3D = g2[:, 0:3]
    rgb = g2[:, 3:6]
    opacity = g2[:, 6:7]
    quat = g2[:, 7:11]
    scale = g2[:, 11:14]
    
    # 坐标系变换
    mean3D_new = transform_positions(mean3D, c2w)     # (N2, 3)
    quat_new = transform_quaternions(quat, c2w)       # (N2, 4)
    # quat_new = quat     # (N2, 4)
    
    
    g2_transformed = torch.cat([mean3D_new, rgb, opacity, quat_new, scale], dim=1).unsqueeze(0)  # (1, N2, 14)
    return g1,g2_transformed
    # merged = torch.cat([g1, g2_transformed], dim=1)  # (1, N1 + N2, 14)
    # return merged


# import torch
# import torch.nn.functional as F
# from scipy.spatial.transform import Rotation as Rscipy

# def transform_positions(positions, c2w):
#     """
#     将 G2 的 xyz 位置变换到 G1 世界坐标系。
#     positions: (N, 3)
#     c2w: (4, 4)
#     """
#     N = positions.shape[0]
#     homo_positions = torch.cat([positions, torch.ones(N, 1, device=positions.device)], dim=-1)  # (N, 4)
#     transformed = (c2w @ homo_positions.T).T[:, :3]  # (N, 3)
#     return transformed

# def quaternion_multiply(q1, q2):
#     """
#     四元数乘法：q = q1 * q2
#     q1, q2: (N, 4)  [w, x, y, z]
#     """
#     w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
#     w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
#     return torch.stack([
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2
#     ], dim=1)

# def transform_quaternions(q_old, c2w):
#     """
#     将 G2 的旋转四元数变换到 G1 坐标系。
#     q_old: (N, 4) [w, x, y, z]
#     c2w: (4, 4)
#     """
#     R = c2w[:3, :3]  # 提取旋转部分
#     R_c2w = Rscipy.from_matrix(R.cpu().numpy())
#     q_c2w = R_c2w.as_quat()  # [x, y, z, w]
#     q_c2w = torch.tensor([q_c2w[3], q_c2w[0], q_c2w[1], q_c2w[2]], device=q_old.device)  # 转为 [w, x, y, z]
#     q_c2w = q_c2w.unsqueeze(0).repeat(q_old.shape[0], 1)  # (N, 4)
#     return quaternion_multiply(q_c2w, q_old)

# def transform_and_merge_g2_to_g1(g1, g2, c2w):
#     """
#     输入：
#         g1: [1, N1, 14]，主参考坐标系中的高斯
#         g2: [1, N2, 14]，待变换到 g1 坐标系的高斯
#         c2w: [4, 4]，将 g2 的高斯变换到 g1 坐标系
#     输出：
#         merged: [1, N1 + N2, 14]，融合后的高斯组
#     """
#     g2 = g2.squeeze(0)  # -> (N2, 14)
#     pos = g2[:, 0:3]
#     quat = g2[:, 3:7]
#     scale_opacity_rgb = g2[:, 7:]  # (N2, 7)

#     # 坐标系变换
#     pos_new = transform_positions(pos, c2w)        # (N2, 3)
#     quat_new = transform_quaternions(quat, c2w)    # (N2, 4)

#     g2_transformed = torch.cat([pos_new, quat_new, scale_opacity_rgb], dim=1).unsqueeze(0)  # (1, N2, 14)
#     merged = torch.cat([g1, g2_transformed], dim=1)  # (1, N1 + N2, 14)
#     return merged

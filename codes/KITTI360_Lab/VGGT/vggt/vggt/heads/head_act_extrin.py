import torch
import torch.nn.functional as F


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act=None):
    """
    Activate pose parameters: only [T(3), quaternion(4)] -> output [B, 7]

    Args:
        pred_pose_enc: Tensor [..., 7] → [x, y, z, qx, qy, qz, qw]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: (ignored) kept for API compatibility

    Returns:
        Tensor of shape [..., 7]
    """
    # flatten extra dimension if present, e.g. [B, 1, 7] → [B, 7]
    if pred_pose_enc.dim() == 3 and pred_pose_enc.shape[1] == 1:
        pred_pose_enc = pred_pose_enc[:, 0]

    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    quat = F.normalize(quat, dim=-1)

    return torch.cat([T, quat], dim=-1)  # shape [..., 7]


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor
        act_type: One of ["linear", "inv_log", "exp", "relu"]

    Returns:
        Tensor after activation
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Tensor (B, C, H, W)
        activation: Type of activation for 3D points
        conf_activation: Type of activation for confidence

    Returns:
        Tuple (pts3d, conf): shapes (B, H, W, 3), (B, H, W)
    """
    fmap = out.permute(0, 2, 3, 1)  # (B, H, W, C)

    xyz = fmap[..., :-1]
    conf = fmap[..., -1]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * torch.expm1(torch.abs(y))

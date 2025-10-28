#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from typing import Any, Mapping, Sequence

# ============== 通用：递归迁移到 device / dtype（支持 dict/list/tuple/tensor） ==============

def to_device(obj: Any,
              device: torch.device | str | None = None,
              dtype: torch.dtype | None = None,
              non_blocking: bool = True) -> Any:
    """
    递归地将 obj 中的所有 Tensor 移动到 device/dtype。
    - 支持: Tensor, dict, list, tuple（嵌套均可）
    - 非 Tensor（如 str、int、float 等）原样返回
    """
    if isinstance(obj, torch.Tensor):
        if device is not None or dtype is not None:
            return obj.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return obj
    elif isinstance(obj, Mapping):
        return {k: to_device(v, device, dtype, non_blocking) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [to_device(v, device, dtype, non_blocking) for v in obj]
        return type(obj)(converted)
    else:
        return obj

# ============== 工具函数：归一化 / look-at / 投影到 SO(3) ==============

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

def _project_to_so3(R: torch.Tensor) -> torch.Tensor:
    """
    将任意 (...,3,3) 矩阵投影为最近的 SO(3) 旋转矩阵（正交 & det=+1）
    """
    U, _, Vh = torch.linalg.svd(R)
    R_ortho = U @ Vh
    det = torch.det(R_ortho)
    if det.ndim == 0:
        neg = det < 0
        if bool(neg):
            U = U.clone()
            U[..., :, -1] *= -1
            R_ortho = U @ Vh
        return R_ortho
    neg = det < 0
    if neg.any():
        U = U.clone()
        U[neg, :, -1] *= -1
        R_ortho = U @ Vh
    return R_ortho

def _look_at_c2w(camera_pos: torch.Tensor,
                 target: torch.Tensor,
                 world_up: torch.Tensor | None = None) -> torch.Tensor:
    """
    生成右手系 cam2world（列向量依次为 x=right, y=up, z=forward）。
    camera_pos: (...,3)
    target:     (...,3)
    world_up:   (...,3) 或 None（默认 [0,1,0]）
    返回:       (...,4,4)
    """
    dev = camera_pos.device
    dt = camera_pos.dtype

    if world_up is None:
        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=dt, device=dev).expand_as(camera_pos)

    # z 轴：从相机指向目标（向前）
    z = _normalize(target - camera_pos)  # (...,3)

    # 若 z 与 up 近乎平行，改用另一个 up（防止退化）
    cos_sim = (z * world_up).sum(dim=-1, keepdim=True).abs()
    alt_up = torch.tensor([0.0, 0.0, 1.0], dtype=dt, device=dev).expand_as(world_up)
    up = torch.where(cos_sim > 0.999, alt_up, world_up)

    # x 轴：x = up × z（右手系）
    x = _normalize(torch.cross(up, z, dim=-1))
    # y 轴：y = z × x
    y = torch.cross(z, x, dim=-1)

    # 组装旋转（列为基向量）
    R = torch.stack([x, y, z], dim=-1)  # (...,3,3)
    R = _project_to_so3(R)

    # 组装 4x4
    T = torch.eye(4, dtype=dt, device=dev).expand(R.shape[:-2] + (4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = camera_pos
    return T

# ============== 相机内外参与图像的构造（全部支持 device/dtype） ==============

def _create_intrinsics(batch_size: int, num_views: int, height: int, width: int,
                       fx_scale: float = 0.8, fy_scale: float = 0.8,
                       device: torch.device | str | None = None,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    创建 pinhole 相机内参 (B, V, 3, 3)
    """
    intrinsics = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    fx = width * fx_scale
    fy = height * fy_scale
    cx = width / 2.0
    cy = height / 2.0

    intrinsics[:, :, 0, 0] = fx
    intrinsics[:, :, 1, 1] = fy
    intrinsics[:, :, 0, 2] = cx
    intrinsics[:, :, 1, 2] = cy
    return intrinsics

def _create_images(batch_size: int, num_views: int, height: int, width: int,
                   device: torch.device | str | None = None,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    随机图像，范围 [0,1]，形状 (B, V, C, H, W)
    """
    imgs = torch.randn(batch_size, num_views, 3, height, width, device=device, dtype=dtype)
    return torch.sigmoid(imgs)

def _create_extrinsics_circle(batch_size: int, num_views: int,
                              radius: float = 2.0, z_height: float = 1.0,
                              ang_step_rad: float | None = None,
                              device: torch.device | str | None = None,
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    圆周分布相机，右手系 cam2world，保证 R ∈ SO(3)
    返回: (B, V, 4, 4)
    """
    if ang_step_rad is None:
        ang_step_rad = torch.pi / 4.0  # 每 45°
    else:
        # 转成张量便于跨设备运算
        ang_step_rad = torch.as_tensor(ang_step_rad, device=device, dtype=dtype)

    E = torch.empty(batch_size, num_views, 4, 4, dtype=dtype, device=device)
    target = torch.zeros(3, dtype=dtype, device=device)
    for b in range(batch_size):
        for v in range(num_views):
            angle = (torch.as_tensor(v, device=device, dtype=dtype) * ang_step_rad)
            cam_pos = torch.stack([
                radius * torch.cos(angle),
                radius * torch.sin(angle),
                torch.as_tensor(z_height, device=device, dtype=dtype)
            ], dim=0)
            T = _look_at_c2w(cam_pos, target)
            E[b, v] = T
    return E

def _create_extrinsics_circle_dense(batch_size: int, num_views: int,
                                    center_distance: float = 3.0, z_height: float = 1.0,
                                    device: torch.device | str | None = None,
                                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    更密集的环绕相机，右手系 cam2world，保证 R ∈ SO(3)
    返回: (B, V, 4, 4)
    """
    E = torch.empty(batch_size, num_views, 4, 4, dtype=dtype, device=device)
    target = torch.zeros(3, dtype=dtype, device=device)
    step = (2 * torch.pi) / max(1, num_views)
    step = torch.as_tensor(step, device=device, dtype=dtype)
    for b in range(batch_size):
        for v in range(num_views):
            angle = torch.as_tensor(v, device=device, dtype=dtype) * step
            cam_pos = torch.stack([
                center_distance * torch.cos(angle),
                center_distance * torch.sin(angle),
                torch.as_tensor(z_height, device=device, dtype=dtype)
            ], dim=0)
            T = _look_at_c2w(cam_pos, target)
            E[b, v] = T
    return E

# ============== 对外的三个接口：基础 demo / 打印信息 / 真实 demo（支持 device/dtype） ==============

def create_mvsplat_demo_input(image_height=180,
                              image_width=320,
                              batch_size=1,
                              num_context_views=2,
                              num_target_views=6,
                              device: torch.device | str | None = None,
                              dtype: torch.dtype = torch.float32):
    """
    创建 MVSplat 的完整输入 demo（右手系 cam2world，R ∈ SO(3)）
    """
    # 图像
    context_images = _create_images(batch_size, num_context_views, image_height, image_width, device=device, dtype=dtype)
    target_images  = _create_images(batch_size, num_target_views,  image_height, image_width, device=device, dtype=dtype)

    # 内参
    context_intrinsics = _create_intrinsics(batch_size, num_context_views, image_height, image_width,
                                            fx_scale=0.7, fy_scale=0.7, device=device, dtype=dtype)
    target_intrinsics  = _create_intrinsics(batch_size, num_target_views,  image_height, image_width,
                                            fx_scale=0.7, fy_scale=0.7, device=device, dtype=dtype)

    # 外参（cam2world，环形分布）
    context_extrinsics = _create_extrinsics_circle(batch_size, num_context_views, radius=2.0, z_height=1.0,
                                                   ang_step_rad=float(torch.pi/4), device=device, dtype=dtype)
    target_extrinsics  = _create_extrinsics_circle(batch_size, num_target_views, radius=2.0, z_height=1.0,
                                                   ang_step_rad=float(torch.pi/6), device=device, dtype=dtype)

    # 深度范围
    near_value = 0.1
    far_value  = 1000.0
    context_near = torch.full((batch_size, num_context_views), near_value, dtype=dtype, device=device)
    context_far  = torch.full((batch_size, num_context_views), far_value,  dtype=dtype, device=device)
    target_near  = torch.full((batch_size, num_target_views),  near_value, dtype=dtype, device=device)
    target_far   = torch.full((batch_size, num_target_views),  far_value,  dtype=dtype, device=device)

    # 视图索引（索引保持在 CPU 或放 GPU 都行；给渲染器时注意需要什么设备）
    context_indices = torch.arange(num_context_views, dtype=torch.int64, device=device)
    target_indices  = torch.arange(num_target_views,  dtype=torch.int64, device=device)

    # 组装
    context = {
        "image": context_images,           # (B, V, 3, H, W)
        "intrinsics": context_intrinsics,  # (B, V, 3, 3)
        "extrinsics": context_extrinsics,  # (B, V, 4, 4) cam2world RH
        "near": context_near,              # (B, V)
        "far": context_far,                # (B, V)
        "index": context_indices,          # (V,)
    }
    target = {
        "image": target_images,
        "intrinsics": target_intrinsics,
        "extrinsics": target_extrinsics,
        "near": target_near,
        "far": target_far,
        "index": target_indices,
    }
    batch_example = {
        "context": context,
        "target": target,
        "scene": ["demo_scene_001"]
    }
    return batch_example, context, target

def print_input_info(batch_example, max_print_tensor=False):
    """
    打印输入信息并校验旋转为 SO(3)
    """
    print("=" * 60)
    print("MVSplat 输入Demo信息")
    print("=" * 60)

    context = batch_example["context"]
    target  = batch_example["target"]

    print(f"场景名称: {batch_example['scene']}\n")

    print("Context 信息:")
    print(f"  图像形状: {context['image'].shape} | device={context['image'].device} | dtype={context['image'].dtype}")
    print(f"  内参形状: {context['intrinsics'].shape} | device={context['intrinsics'].device}")
    print(f"  外参形状: {context['extrinsics'].shape} | device={context['extrinsics'].device}")
    print(f"  Near形状: {context['near'].shape}, Far形状: {context['far'].shape} | device={context['near'].device}")
    print(f"  索引形状: {context['index'].shape} | device={context['index'].device}\n")

    print("Target 信息:")
    print(f"  图像形状: {target['image'].shape} | device={target['image'].device} | dtype={target['image'].dtype}")
    print(f"  内参形状: {target['intrinsics'].shape} | device={target['intrinsics'].device}")
    print(f"  外参形状: {target['extrinsics'].shape} | device={target['extrinsics'].device}")
    print(f"  Near形状: {target['near'].shape}, Far形状: {target['far'].shape} | device={target['near'].device}")
    print(f"  索引形状: {target['index'].shape} | device={target['index'].device}\n")

    if max_print_tensor:
        print("相机内参示例 (第一个context视图):")
        print(context['intrinsics'][0, 0])
        print("\n相机外参示例 (第一个context视图):")
        print(context['extrinsics'][0, 0])

    # SO(3) 校验
    with torch.no_grad():
        R_ctx = context["extrinsics"][..., :3, :3]
        I = torch.eye(3, dtype=R_ctx.dtype, device=R_ctx.device)
        ortho_err = (R_ctx.transpose(-1, -2) @ R_ctx - I).abs().amax(dim=(-2, -1))
        det = torch.det(R_ctx)
        print("\n[SO(3) 检查 - Context]")
        print("  ortho_err max/mean:", float(ortho_err.max().cpu()), float(ortho_err.mean().cpu()))
        print("  det   min/mean/max:", float(det.min().cpu()), float(det.mean().cpu()), float(det.max().cpu()))

        R_tgt = target["extrinsics"][..., :3, :3]
        I_tgt = torch.eye(3, dtype=R_tgt.dtype, device=R_tgt.device)
        ortho_err_t = (R_tgt.transpose(-1, -2) @ R_tgt - I_tgt).abs().amax(dim=(-2, -1))
        det_t = torch.det(R_tgt)
        print("[SO(3) 检查 - Target]")
        print("  ortho_err max/mean:", float(ortho_err_t.max().cpu()), float(ortho_err_t.mean().cpu()))
        print("  det   min/mean/max:", float(det_t.min().cpu()), float(det_t.mean().cpu()), float(det_t.max().cpu()))

    print("\n图像数值范围:")
    print(f"  Context图像: [{float(context['image'].min().cpu()):.3f}, {float(context['image'].max().cpu()):.3f}]")
    print(f"  Target图像:  [{float(target['image'].min().cpu()):.3f}, {float(target['image'].max().cpu()):.3f}]")

def create_realistic_demo(device: torch.device | str | None = None,
                          dtype: torch.dtype = torch.float32):
    """
    创建更真实的 demo 数据（右手系 cam2world，R ∈ SO(3)）
    """
    print("创建真实场景的 demo 数据...")

    batch_size = 1
    num_context_views = 2
    num_target_views = 1
    height, width = 180, 320

    # 图像
    context_images = _create_images(batch_size, num_context_views, height, width, device=device, dtype=dtype)
    target_images  = _create_images(batch_size, num_target_views,  height, width, device=device, dtype=dtype)

    # 内参（更接近真实）
    context_intrinsics = _create_intrinsics(batch_size, num_context_views, height, width,
                                            fx_scale=0.8, fy_scale=0.8, device=device, dtype=dtype)
    target_intrinsics  = _create_intrinsics(batch_size, num_target_views,  height, width,
                                            fx_scale=0.8, fy_scale=0.8, device=device, dtype=dtype)

    # 外参（更密集环绕）
    context_extrinsics = _create_extrinsics_circle_dense(batch_size, num_context_views,
                                                         center_distance=3.0, z_height=1.0,
                                                         device=device, dtype=dtype)
    target_extrinsics  = _create_extrinsics_circle_dense(batch_size, num_target_views,
                                                         center_distance=3.0, z_height=1.0,
                                                         device=device, dtype=dtype)

    # 深度范围
    near_value = 0.5
    far_value  = 50.0
    context_near = torch.full((batch_size, num_context_views), near_value, dtype=dtype, device=device)
    context_far  = torch.full((batch_size, num_context_views), far_value,  dtype=dtype, device=device)
    target_near  = torch.full((batch_size, num_target_views),  near_value, dtype=dtype, device=device)
    target_far   = torch.full((batch_size, num_target_views),  far_value,  dtype=dtype, device=device)

    context_indices = torch.arange(num_context_views, dtype=torch.int64, device=device)
    target_indices  = torch.arange(num_target_views,  dtype=torch.int64, device=device)

    context = {
        "image": context_images,
        "intrinsics": context_intrinsics,
        "extrinsics": context_extrinsics,
        "near": context_near,
        "far": context_far,
        "index": context_indices,
    }
    target = {
        "image": target_images,
        "intrinsics": target_intrinsics,
        "extrinsics": target_extrinsics,
        "near": target_near,
        "far": target_far,
        "index": target_indices,
    }
    batch_example = {
        "context": context,
        "target": target,
        "scene": ["realistic_demo_scene"]
    }
    return batch_example

# ============== 主程序（示例：自动选择 GPU；也演示 to_device 的用法） ==============

if __name__ == "__main__":
    print("创建 MVSplat 输入 Demo...")

    # 选择设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float32

    # 基础 demo（直接在目标 device 上创建，避免来回搬）
    batch_example, context, target = create_mvsplat_demo_input(device=device, dtype=dtype)
    print_input_info(batch_example)

    print("\n" + "=" * 60)
    print("创建真实场景 Demo...")
    print("=" * 60)

    realistic_batch = create_realistic_demo(device=device, dtype=dtype)
    print_input_info(realistic_batch)

    # —— 如果你是在 CPU 创建、之后再整体搬迁，也可以这样：
    # cpu_batch, _, _ = create_mvsplat_demo_input(device="cpu")
    # gpu_batch = to_device(cpu_batch, device="cuda:0", dtype=torch.float32)

    print("\nDemo 创建完成！你可以使用这些数据来测试 MVSplat 模型。")

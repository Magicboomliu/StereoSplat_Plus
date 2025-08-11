import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import lpips


@torch.no_grad()
def lpips_vgg_batch(pred_1v3hw: torch.Tensor,
                    gt_1v3hw: torch.Tensor,
                    net: lpips.LPIPS | None = None) -> torch.Tensor:
    """
    Compute LPIPS (VGG16) for each view, batched.
    Args:
        pred_1v3hw: [1, V, 3, H, W], values in [0, 1]
        gt_1v3hw:   [1, V, 3, H, W], values in [0, 1]
        net: optional pre-created lpips.LPIPS(net='vgg')
    Returns:
        lpips_per_view: [V] tensor (float32)
    """
    assert pred_1v3hw.shape == gt_1v3hw.shape, "Pred / GT must have same shape"
    assert pred_1v3hw.ndim == 5 and pred_1v3hw.shape[2] == 3

    B, V, C, H, W = pred_1v3hw.shape
    assert B == 1, "Batch dim must be 1"

    device = pred_1v3hw.device



    # [1, V, 3, H, W] -> [V, 3, H, W]
    pred = pred_1v3hw.squeeze(0)
    gt   = gt_1v3hw.squeeze(0)

    # 归一化到 [-1, 1]
    pred = pred * 2.0 - 1.0
    gt   = gt   * 2.0 - 1.0

    # LPIPS 批量计算 -> [V, 1, 1, 1]
    scores = net(pred, gt).view(-1)
    
    scores = scores.mean()
    
    return scores.float()


class IncrementalGaussianFusion(nn.Module):
    """
    增量式高斯融合模块 - 基于详细流程图的完整实现
    
    实现完整的增量式高斯融合流水线：
    1. FOV判断：分离重叠和遗留的高斯点
    2. 启发式剪枝：去除低质量高斯点
    3. 窗口优化：基于渲染损失的局部优化
    4. 全局优化：基于历史关键帧的全局优化
    """

    
    def __init__(self, 
                 renderer=None,
                 voxel_size: float = 0.05,
                 opacity_threshold: float = 0.01,
                 depth_threshold: float = 0.1,
                 window_optimization_iterations: int = 50,
                 global_optimization_iterations: int = 10,
                 lambda_depth: float = 1.0):
        """
        初始化增量式高斯融合模块
        
        Args:
            renderer: 渲染器实例，用于计算渲染损失
            voxel_size: 体素大小（米）
            opacity_threshold: 透明度阈值
            depth_threshold: 深度阈值
            window_optimization_iterations: 窗口优化迭代次数
            global_optimization_iterations: 全局优化迭代次数
            lambda_depth: 深度损失权重
        """
        super().__init__()
        self.renderer = renderer
        self.voxel_size = voxel_size
        self.opacity_threshold = opacity_threshold
        self.depth_threshold = depth_threshold
        self.window_optimization_iterations = window_optimization_iterations
        self.global_optimization_iterations = global_optimization_iterations
        self.lambda_depth = lambda_depth
        
        self.lpips_net = lpips.LPIPS(net='vgg').cuda().eval()
        
    def incremental_fusion_pipeline(self,
                                  global_gaussians_prev: torch.Tensor,
                                  new_keyframe_gaussians: torch.Tensor,
                                  new_keyframe_pose: torch.Tensor,
                                  new_keyframe_intrinsics: torch.Tensor,
                                  new_keyframe_image_size: Tuple[int, int],
                                  supervision_frames: List[Dict] = None,
                                  lidar_to_world_pose: torch.Tensor = None) -> torch.Tensor:
        """
        完整的增量式高斯融合流水线
        
        Args:
            global_gaussians_prev: 前一步的全局高斯点 [1, N, 14]
            new_keyframe_gaussians: 新关键帧的高斯点 [1, M, 14]
            new_keyframe_pose: 新关键帧的相机姿态 [2, 4, 4] (left/right camera)
            new_keyframe_intrinsics: 新关键帧的相机内参 [2, 3, 3] (left/right camera)
            new_keyframe_image_size: 新关键帧的图像尺寸 (H, W)
            supervision_frames: 监督帧列表（用于窗口优化）
            lidar_to_world_pose: LIDAR到世界坐标系的变换矩阵 [4, 4]，用于坐标变换
            
        Returns:
            updated_global_gaussians: 更新后的全局高斯点 [1, N+M, 14]
        """
        print("Starting incremental gaussian fusion pipeline...")
        
        # 确保数据类型一致性
        dtype = global_gaussians_prev.dtype
        device = global_gaussians_prev.device
        
        # 统一所有输入张量的数据类型
        new_keyframe_gaussians = new_keyframe_gaussians.to(dtype=dtype, device=device)
        new_keyframe_pose = new_keyframe_pose.to(dtype=dtype, device=device)
        new_keyframe_intrinsics = new_keyframe_intrinsics.to(dtype=dtype, device=device)
        
        # Step 1: 将新关键帧的高斯点变换到世界坐标系
        # 使用LIDAR姿态进行坐标变换，如果没有提供则使用相机姿态
        transform_pose = lidar_to_world_pose if lidar_to_world_pose is not None else new_keyframe_pose[0]  # 使用左相机姿态作为备选
        new_gaussians_world = self.transform_gaussians_to_world(
            new_keyframe_gaussians, transform_pose
        )
        

        # Step 2: 基于FOV分离重叠和遗留的高斯点
        print("Step 2: Separating overlapping and legacy gaussians based on FOV...")
        g_overlapped, g_legacy = self.separate_gaussians_by_fov(
            global_gaussians_prev[0],  # [N, 14]
            new_keyframe_pose[0],  # [2, 4, 4] - 左右相机姿态
            new_keyframe_intrinsics,  # [2, 3, 3] - 左右相机内参
            new_keyframe_image_size
        )
        

        
        # Step 3: 启发式剪枝
        print("Step 3: Heuristic pruning of overlapped gaussians...")
        g_overlapped_pruned = self.heuristic_prev_prune(
            g_overlapped, new_gaussians_world[0], new_keyframe_pose[0]  # 使用左相机姿态
        )
        
 
        
        # Step 4: 窗口优化
        print(f"Step 4: Window loss-based optimization ({self.window_optimization_iterations} iterations)...")
        g_overlapped_optimized = self.window_loss_based_optimization(
            g_overlapped_pruned, new_keyframe_pose, new_keyframe_intrinsics,  # 使用完整的左右相机姿态和内参
            new_keyframe_image_size, supervision_frames
        )
        
        
        print(g_overlapped_optimized.shape)
        quit()

        
        
        # Step 5: 融合优化后的重叠高斯点和遗留高斯点
        print("Step 5: Concatenating optimized overlapped and legacy gaussians...")
        g_update_global = torch.cat([g_overlapped_optimized, g_legacy], dim=0)
        
        
        
        
        # Step 6: 全局优化
        print(f"Step 6: Global optimization ({self.global_optimization_iterations} iterations)...")
        g_final_global = self.global_optimization(g_update_global)
        
        print(f"Final global gaussians: {g_final_global.shape[0]}")
        
        return g_final_global.unsqueeze(0)  # [1, N, 14]
    
    def separate_gaussians_by_fov(self,global_gaussians, camera_poses, intrinsics, image_size):
        """
        基于两个相机的 FOV 分离在 FOV 内和 FOV 外的高斯点 (纯 tensor 流版本)
        支持 intrinsics: [2, 3, 3]
        支持 camera_poses: [2, 4, 4]
        """
        # 转成 float tensor，统一 device/dtype
        device = global_gaussians.device
        dtype = global_gaussians.dtype
        global_gaussians = global_gaussians.to(dtype=dtype, device=device)
        camera_poses = camera_poses.to(dtype=dtype, device=device)
        intrinsics = intrinsics.to(dtype=dtype, device=device)

        points = global_gaussians[:, :3]  # [N, 3]
        N = points.shape[0]
        H, W = image_size
        H = torch.tensor(H, dtype=dtype, device=device)
        W = torch.tensor(W, dtype=dtype, device=device)

        in_fov_mask = torch.zeros(N, dtype=torch.bool, device=device)

        for i in range(2):  # 遍历左右相机
            fx = intrinsics[i, 0, 0]  # scalar tensor
            fy = intrinsics[i, 1, 1]

            # 计算 FOV (保持 tensor)
            fov_x = 2 * torch.atan(W / (2.0 * fx))
            fov_y = 2 * torch.atan(H / (2.0 * fy))

            # 世界到相机
            w2c = torch.linalg.inv(camera_poses[i])
            homo_points = torch.cat([points, torch.ones((N, 1), device=device, dtype=dtype)], dim=1)
            cam_points = (w2c @ homo_points.T).T[:, :3]

            x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]

            # 在相机前方
            in_front = z > 0.1

            # 视场角判断
            x_angle = torch.atan2(x, z)
            y_angle = torch.atan2(y, z)
            in_fov = (x_angle.abs() <= fov_x / 2) & (y_angle.abs() <= fov_y / 2)

            # 累积在 FOV 内的点
            in_fov_mask |= (in_front & in_fov)

        g_overlapped = global_gaussians[in_fov_mask]
        g_legacy = global_gaussians[~in_fov_mask]
        return g_overlapped, g_legacy
        
    def heuristic_prev_prune(self,
                            g_overlapped: torch.Tensor,
                            g_new: torch.Tensor,
                            new_keyframe_pose: torch.Tensor) -> torch.Tensor:
        """
        启发式剪枝：去除低质量的高斯点
        
        Args:
            g_overlapped: 重叠的全局高斯点 [N, 14]
            g_new: 新的关键帧高斯点 [M, 14]
            new_keyframe_pose: 新关键帧的相机姿态 [4, 4]
            
        Returns:
            g_pruned: 剪枝后的高斯点 [K, 14]
        """
        print("Applying heuristic pruning...")
        
        # 确保数据类型一致
        dtype = g_overlapped.dtype
        device = g_overlapped.device
        g_new = g_new.to(dtype=dtype, device=device)
        new_keyframe_pose = new_keyframe_pose.to(dtype=dtype, device=device)
        
        # 1. 去除透明度很低的高斯点
        opacity_mask = g_overlapped[:, 6:7] > self.opacity_threshold
        g_filtered = g_overlapped[opacity_mask.squeeze()]
        print(f"After opacity filtering: {g_filtered.shape[0]} gaussians")
        
        # 2. 去除明显的异常值（基于位置和尺度）
        if g_filtered.shape[0] > 0:
            # 计算位置和尺度的统计信息
            positions = g_filtered[:, :3]
            scales = g_filtered[:, 11:14]
            
            # 去除位置异常值（基于3σ原则）
            pos_mean = positions.mean(dim=0)
            pos_std = positions.std(dim=0)
            pos_mask = torch.all(torch.abs(positions - pos_mean) < 3 * pos_std, dim=1)
            
            # 去除尺度异常值
            scale_mean = scales.mean(dim=0)
            scale_std = scales.std(dim=0)
            scale_mask = torch.all(torch.abs(scales - scale_mean) < 3 * scale_std, dim=1)
            
            # 综合mask
            outlier_mask = pos_mask & scale_mask
            g_filtered = g_filtered[outlier_mask]
            print(f"After outlier removal: {g_filtered.shape[0]} gaussians")
        
        # 3. 体素化处理重叠区域
        if g_filtered.shape[0] > 0 and g_new.shape[0] > 0:
            g_voxelized = self.voxel_based_pruning(g_filtered, g_new, new_keyframe_pose)
            print(f"After voxel-based pruning: {g_voxelized.shape[0]} gaussians")
            return g_voxelized
        else:
            return g_filtered
    
    def voxel_based_pruning(self,
                           g_overlapped: torch.Tensor,
                           g_new: torch.Tensor,
                           new_keyframe_pose: torch.Tensor) -> torch.Tensor:
        """
        基于体素的剪枝：在重叠区域保留新的高斯点（完全向量化快速版本）
        
        Args:
            g_overlapped: 重叠的全局高斯点 [N, 14]
            g_new: 新的关键帧高斯点 [M, 14]
            new_keyframe_pose: 新关键帧的相机姿态 [4, 4]
            
        Returns:
            g_pruned: 剪枝后的高斯点 [K, 14]
        """
        if g_overlapped.shape[0] == 0:
            return g_new
        if g_new.shape[0] == 0:
            return g_overlapped
            
        # 确保数据类型一致
        dtype = g_overlapped.dtype
        device = g_overlapped.device
        g_new = g_new.to(dtype=dtype, device=device)
        
        # 体素化重叠区域 - 向量化操作
        all_positions = torch.cat([g_overlapped[:, :3], g_new[:, :3]], dim=0)
        
        # 计算体素索引 - 向量化
        voxel_indices = torch.floor(all_positions / self.voxel_size).long()
        
        # 将体素索引转换为线性索引
        voxel_hash = (voxel_indices[:, 0] * 73856093 + 
                     voxel_indices[:, 1] * 19349663 + 
                     voxel_indices[:, 2] * 83492791) % 1000000007
        
        # 创建点类型标识符：0=旧，1=新
        point_type = torch.cat([
            torch.zeros(g_overlapped.shape[0], dtype=torch.long, device=device),
            torch.ones(g_new.shape[0], dtype=torch.long, device=device)
        ])
        
        # 找到唯一体素和对应的点类型
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_hash, return_inverse=True, return_counts=True
        )
        
        # 对于每个体素，统计新高斯点的数量
        new_counts = torch.zeros(unique_voxels.shape[0], dtype=torch.long, device=device)
        new_counts.scatter_add_(0, inverse_indices, point_type)
        
        # 创建掩码：移除有新高斯点的体素中的旧高斯点
        keep_mask = torch.ones(voxel_hash.shape[0], dtype=torch.bool, device=device)
        old_points_to_remove = (point_type == 0) & (new_counts[inverse_indices] > 0)
        keep_mask[old_points_to_remove] = False
        
        # 返回保留的高斯点
        all_gaussians = torch.cat([g_overlapped, g_new], dim=0)
        return all_gaussians[keep_mask]
    
    def window_loss_based_optimization(self,
                                     gaussians: torch.Tensor,
                                     camera_poses: torch.Tensor,
                                     intrinsics: torch.Tensor,
                                     image_size: Tuple[int, int],
                                     supervision_frames: List[Dict] = None) -> torch.Tensor:
        """
        基于渲染损失的窗口优化（支持左右相机）
        
        Args:
            gaussians: 待优化的高斯点 [N, 14]
            camera_poses: 相机姿态 [2, 4, 4] (left/right camera)
            intrinsics: 相机内参 [2, 3, 3] (left/right camera)
            image_size: 图像尺寸 (H, W)
            supervision_frames: 监督帧列表
            
        Returns:
            optimized_gaussians: 优化后的高斯点 [N, 14]
        """
        if gaussians.shape[0] == 0:
            return gaussians
        
        print(f"Starting window optimization with {gaussians.shape[0]} gaussians")
        
        # 如果没有监督帧，返回原始高斯点
        if not supervision_frames:
            print("No supervision frames provided, returning original gaussians")
            return gaussians
        
        # 确保数据类型一致
        dtype = gaussians.dtype
        device = gaussians.device
        camera_poses = camera_poses.to(dtype=dtype, device=device)
        intrinsics = intrinsics.to(dtype=dtype, device=device)
        
        # 创建可训练的高斯点参数 #[N,14]
        gaussians_optimizable = gaussians.clone().detach().requires_grad_(True)
        
        
        # 优化器
        optimizer = optim.Adam([gaussians_optimizable], lr=0.001)
        
        for iteration in range(self.window_optimization_iterations):
            optimizer.zero_grad()
            total_loss_val = 0.0

            for frame_data in supervision_frames:
                loss = self.compute_rendering_loss_multi_camera(
                    gaussians_optimizable,
                    frame_data,
                    camera_poses[0].detach(),
                    intrinsics.detach(),
                    image_size
                )
                loss.backward()  # 立刻释放该帧的 graph
                total_loss_val += loss.item()

            optimizer.step()

            if iteration % 10 == 0:
                print(f"iter {iteration}, loss: {total_loss_val:.6f}")
            
        print("Window optimization completed")
        return gaussians_optimizable.detach()
    
    def compute_rendering_loss(self, gaussians: torch.Tensor, frame_data: Dict) -> torch.Tensor:
        """
        计算渲染损失（占位符实现）
        
        Args:
            gaussians: 高斯点 [N, 14]
            frame_data: 帧数据
            
        Returns:
            loss: 渲染损失
        """
        # TODO: 实现完整的渲染损失计算
        # 包括图像渲染损失和深度渲染损失
        # loss = |I_render - I_GT| + λ|D_render - D_GT|
        
        # 目前返回一个小的随机损失作为占位符
        return torch.tensor(0.001, device=gaussians.device, requires_grad=True)
    
    def compute_rendering_loss_multi_camera(self, 
                                          gaussians: torch.Tensor, 
                                          frame_data: Dict,
                                          camera_poses: torch.Tensor,
                                          intrinsics: torch.Tensor,
                                          image_size: Tuple[int, int]) -> torch.Tensor:
        """
        计算多相机渲染损失（支持左右相机）
        
        优先使用 psuedo_depth 作为深度损失计算的参考，如果没有则回退到 sparse_gts
        
        Args:
            gaussians: 高斯点 [N, 14]
            frame_data: 帧数据，包含 imgs, psuedo_depth, sparse_gts 等
            camera_poses: 相机姿态 [1,2, 4, 4] (left/right camera)
            intrinsics: 相机内参 [2, 3, 3] (left/right camera)
            image_size: 图像尺寸 (H, W)
            
        Returns:
            loss: 多相机渲染损失 (图像损失 + λ * 深度损失)
        """
        if gaussians.shape[0] == 0:
            return torch.tensor(0.0, device=gaussians.device, requires_grad=True)
    
        # 检查是否有渲染器
        if not hasattr(self, 'renderer') or self.renderer is None:
            print("Warning: No renderer available, returning placeholder loss")
            return torch.tensor(0.001, device=gaussians.device, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=gaussians.device, requires_grad=True)
        
        # compute loss here
        rendered_c2w = frame_data["output"]["c2w"].to(gaussians.device)
        fovxs = frame_data["output"]["fovxs"].to(gaussians.device)
        fovys = frame_data["output"]["fovys"].to(gaussians.device)

        # torch.Size([1, 21254400, 14])
        # torch.Size([1, 2, 4, 4])
        # torch.Size([1, 2])
        # torch.Size([1, 2])
        
        
        
        rendered_results =self.renderer.render(
            gaussians=gaussians.unsqueeze(0),
            c2w=rendered_c2w.unsqueeze(0),
            fovx=fovxs.unsqueeze(0),
            fovy=fovys.unsqueeze(0),
            rays_o=None,
            rays_d=None
        )


        # 提取渲染结果
        rendered_image = rendered_results['image']      # [1, V, 3, H, W]
        rendered_depth = rendered_results['depth']      # [1, V, 1, H, W]
        
        # GT images and the depth information.
        output_rgb_for_supervision = frame_data['output']['imgs'].unsqueeze(0).to(rendered_image.device)
        output_depth_for_supervision = frame_data['output']['psuedo_depth'].unsqueeze(0).to(rendered_image.device)
        
        
        # Loss Function Here
        rgb_l1_loss_value = F.l1_loss(rendered_image, output_rgb_for_supervision)
        
        # lpips_loss_value = lpips_vgg_batch(rendered_image, output_rgb_for_supervision,
        #                                    net=self.lpips_net)
        
        # rgb_loss = rgb_l1_loss_value + lpips_loss_value*0.05
        
        rgb_loss = rgb_l1_loss_value
        
        depth_l1_loss_value = F.l1_loss(rendered_depth, output_depth_for_supervision)
        
        total_loss = rgb_loss + self.lambda_depth * depth_l1_loss_value
        
        
        return total_loss 
        

    
    
    def global_optimization(self, gaussians: torch.Tensor) -> torch.Tensor:
        """
        基于历史关键帧的全局优化
        
        Args:
            gaussians: 待优化的全局高斯点 [N, 14]
            
        Returns:
            optimized_gaussians: 优化后的全局高斯点 [N, 14]
        """
        if gaussians.shape[0] == 0:
            return gaussians
        
        print(f"Starting global optimization with {gaussians.shape[0]} gaussians")
        
        # 创建可训练的高斯点参数
        gaussians_optimizable = gaussians.clone().detach().requires_grad_(True)
        
        # 优化器
        optimizer = optim.Adam([gaussians_optimizable], lr=0.0005)
        
        # 全局优化循环
        for iteration in range(self.global_optimization_iterations):
            optimizer.zero_grad()
            
            # 这里可以实现全局一致性损失
            # 例如：时间一致性、几何一致性等
            global_loss = self.compute_global_consistency_loss(gaussians_optimizable)
            
            # 反向传播
            global_loss.backward()
            optimizer.step()
            
            if iteration % 2 == 0:
                print(f"Global optimization iteration {iteration}, loss: {global_loss.item():.6f}")
        
        print("Global optimization completed")
        return gaussians_optimizable.detach()
    
    def compute_global_consistency_loss(self, gaussians: torch.Tensor) -> torch.Tensor:
        """
        计算全局一致性损失（占位符实现）
        
        Args:
            gaussians: 高斯点 [N, 14]
            
        Returns:
            loss: 全局一致性损失
        """
        # TODO: 实现全局一致性损失
        # 可以包括：
        # 1. 时间一致性损失
        # 2. 几何一致性损失
        # 3. 正则化损失
        
        # 目前返回一个小的随机损失作为占位符
        return torch.tensor(0.001, device=gaussians.device, requires_grad=True)
    
    def transform_gaussians_to_world(self,
                                   gaussians: torch.Tensor,
                                   c2w: torch.Tensor) -> torch.Tensor:
        """
        将高斯点从局部坐标系变换到世界坐标系
        
        Args:
            gaussians: 局部坐标系的高斯点 [1, N, 14]
            c2w: 相机到世界坐标变换 [4, 4]
            
        Returns:
            transformed_gaussians: 世界坐标系的高斯点 [1, N, 14]
        """
        gaussians = gaussians.squeeze(0)  # [N, 14]
        
        # 提取高斯点的各个属性
        mean3D = gaussians[:, 0:3]      # 位置
        rgb = gaussians[:, 3:6]         # 颜色
        opacity = gaussians[:, 6:7]     # 不透明度
        quat = gaussians[:, 7:11]       # 旋转四元数
        scale = gaussians[:, 11:14]     # 尺度
        
        # 位置变换
        mean3D_new = self.transform_positions(mean3D, c2w)
        
        # 旋转四元数变换
        quat_new = self.transform_quaternions(quat, c2w)
        
        # 重新组合高斯点
        transformed_gaussians = torch.cat([
            mean3D_new, rgb, opacity, quat_new, scale
        ], dim=1).unsqueeze(0)  # [1, N, 14]
        
        return transformed_gaussians
    
    def transform_positions(self, positions: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        """变换高斯点的位置"""
        # 确保数据类型一致
        dtype = positions.dtype
        device = positions.device
        c2w = c2w.to(dtype=dtype, device=device)
        
        N = positions.shape[0]
        homo_positions = torch.cat([positions, torch.ones(N, 1, device=device, dtype=dtype)], dim=-1)
        transformed = (c2w @ homo_positions.T).T[:, :3]
        return transformed
    
    def transform_quaternions(self, quat_old: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        """变换高斯点的旋转四元数"""
        # 确保数据类型一致
        dtype = quat_old.dtype
        device = quat_old.device
        c2w = c2w.to(dtype=dtype, device=device)
        
        R = c2w[:3, :3]
        R_c2w = Rscipy.from_matrix(R.cpu().numpy())
        q_c2w = R_c2w.as_quat()
        q_c2w = torch.tensor([q_c2w[3], q_c2w[0], q_c2w[1], q_c2w[2]], 
                            device=device, dtype=dtype)
        q_c2w = q_c2w.unsqueeze(0).repeat(quat_old.shape[0], 1)
        return self.quaternion_multiply(q_c2w, quat_old)
    
    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数乘法"""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=1)



def create_incremental_fusion_pipeline(renderer=None,
                                     voxel_size: float = 0.05,
                                     opacity_threshold: float = 0.01,
                                     depth_threshold: float = 0.1,
                                     window_optimization_iterations: int = 50,
                                     global_optimization_iterations: int = 10,
                                     lambda_depth: float = 1.0) -> IncrementalGaussianFusion:
    """
    创建增量式高斯融合流水线
    
    Args:
        renderer: 渲染器实例，用于计算渲染损失
        voxel_size: 体素大小（米）
        opacity_threshold: 透明度阈值
        depth_threshold: 深度阈值
        window_optimization_iterations: 窗口优化迭代次数
        global_optimization_iterations: 全局优化迭代次数
        lambda_depth: 深度损失权重
        
    Returns:
        IncrementalGaussianFusion: 配置好的增量式高斯融合模块
    """
    return IncrementalGaussianFusion(
        renderer=renderer,
        voxel_size=voxel_size,
        opacity_threshold=opacity_threshold,
        depth_threshold=depth_threshold,
        window_optimization_iterations=window_optimization_iterations,
        global_optimization_iterations=global_optimization_iterations,
        lambda_depth=lambda_depth
    )



import numpy as np
import torch
from plyfile import PlyData, PlyElement

def save_points_to_ply(points, filename="output.ply"):
    """
    将 (N,3) 点云保存为 .ply 文件
    支持 torch.Tensor 或 numpy.ndarray

    Args:
        points: (N, 3) torch.Tensor 或 numpy.ndarray
        filename: 输出文件名
    """
    # 转 numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    elif not isinstance(points, np.ndarray):
        raise TypeError("points 必须是 torch.Tensor 或 numpy.ndarray")

    assert points.ndim == 2 and points.shape[1] == 3, "points 必须是 (N, 3) 形状"

    # 构造 ply 顶点数据
    vertex = np.array([tuple(p) for p in points],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # 保存为 PLY
    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    ply.write(filename)
    print(f"已保存 {points.shape[0]} 个点到 {filename}")
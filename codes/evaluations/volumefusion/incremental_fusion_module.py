from scipy.integrate import qmc_quad
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import lpips
import skimage
from rgb_loss import rgb_loss_l1_dssim_5d

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
                 history_batch_data=None,     
                 opacity_based_pruning_threshold: float = 0.01,
                 gradient_based_pruning_keep_ratio: float = 0.1,
                 geometry_based_pruning_depth_error_threshold: float = 0.1,

                 window_optimization_iterations: int = 50,
                 global_optimization_iterations: int = 20,
                 lambda_depth: float = 1.0,
                 optimiation_options: Dict = None):

        super().__init__()
        self.renderer = renderer
        self.history_batch_data = history_batch_data
        
        self.opacity_based_pruning_threshold = opacity_based_pruning_threshold
        self.gradient_based_pruning_keep_ratio = gradient_based_pruning_keep_ratio
        self.geometry_based_pruning_depth_error_threshold = geometry_based_pruning_depth_error_threshold
        
        self.window_optimization_iterations = window_optimization_iterations
        self.global_optimization_iterations = global_optimization_iterations
        self.lambda_depth = lambda_depth
        
        self.lpips_net = lpips.LPIPS(net='vgg').cuda().eval()
        
        # 恢复原有的优化选项
        if optimiation_options is not None:
            self.optimiation_options = optimiation_options
        else:
            self.optimiation_options = {
                "use_pruning": False,
                "use_opacity_based_pruning": False,
                "use_geometry_based_pruning": False,
                "use_gradient_based_pruning": False,
                "use_window_loss_based_optimization": False,
                "use_global_optimization": True,
            }
        

    
    def store_batch_data_to_history(self, batch_data: Dict):
        self.history_batch_data.append(batch_data)
        
    def random_select_batch_data_from_history(self, num_frames: int):
        if len(self.history_batch_data) < num_frames:
            return self.history_batch_data
        else:
            return random.sample(self.history_batch_data, num_frames)
    
    
    
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
        
        # 将当前关键帧数据添加到历史数据中（用于全局优化）
        if self.history_batch_data is None:
            self.history_batch_data = []
        

        
        # 限制历史数据的大小，避免内存过度增长
        max_history_frames = 20  # 减少到20帧以节省内存
        if len(self.history_batch_data) >= max_history_frames:
            # 移除最旧的数据
            self.history_batch_data.pop(0)
        

        # Step 2: 基于FOV分离重叠和遗留的高斯点
        print("Step 2: Separating overlapping and legacy gaussians based on FOV...")
        g_overlapped, g_legacy = self.separate_gaussians_by_fov(
            global_gaussians_prev[0],  # [N, 14]
            new_keyframe_pose[0],  # [2, 4, 4] - 左右相机姿态
            new_keyframe_intrinsics,  # [2, 3, 3] - 左右相机内参
            new_keyframe_image_size
        )
        
        
        if self.optimiation_options["use_pruning"]:
            # Simple Pruing by Opacity
            if self.optimiation_options["use_opacity_based_pruning"]:
                g_overlapped_pruned = self.filter_by_opacity(g_overlapped, self.opacity_based_pruning_threshold)

                if len(g_overlapped) == 0:
                    g_overlapped_pruned = new_gaussians_world[0][:1,:]
            else:
                g_overlapped_pruned = g_overlapped
                
            # Use Gradient-based Pruning
            if self.optimiation_options["use_gradient_based_pruning"]:
                if len(supervision_frames) > 0:
                    g_overlapped_pruned = self.gradient_based_pruning(g_overlapped_pruned, supervision_frames[0], 
                                                                      self.gradient_based_pruning_keep_ratio)
                    

                    if len(g_overlapped_pruned) == 0:
                        g_overlapped_pruned = new_gaussians_world[0][:1,:]
                else:
                    g_overlapped_pruned = g_overlapped_pruned
            else:
                g_overlapped_pruned = g_overlapped_pruned
                
                
            # # debug here
            
            # rendered_image, rendered_depth = self.rendered_views_using_gaussians(g_overlapped, 
            #                                                                      supervision_frames[0])

            # # skimage.io.imshow(rendered_image[0,0,:,:,:])
            # rendered_image_vis = rendered_image[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            # skimage.io.imsave("min_gradient.png",(rendered_image_vis*255).astype(np.uint8))
            # quit()

                
            # Using Geometry-based Pruning
            if self.optimiation_options["use_geometry_based_pruning"]:
                if len(supervision_frames) > 0:
                    if self.optimiation_options["use_gradient_based_pruning"]:
                        g_overlapped_pruned_geo = self.geometry_based_pruning(g_overlapped, supervision_frames[0], 
                                                                        self.geometry_based_pruning_depth_error_threshold)
                        g_overlapped_pruned = torch.cat([g_overlapped_pruned, g_overlapped_pruned_geo], dim=0)
                    else:
                        g_overlapped_pruned = self.geometry_based_pruning(g_overlapped_pruned, supervision_frames[0], 
                                                                        self.geometry_based_pruning_depth_error_threshold)
                    if len(g_overlapped_pruned) == 0:
                        g_overlapped_pruned = new_gaussians_world[0][:1,:]
                else:
                    g_overlapped_pruned = g_overlapped_pruned
            else:
                    g_overlapped_pruned = g_overlapped_pruned
            
            
        else:
            g_overlapped_pruned = g_overlapped
        
        
        g_overlapped_for_optimization = torch.cat([g_overlapped_pruned, new_gaussians_world[0]], dim=0)

        
        
        
        # Step 4: 窗口优化
        if self.optimiation_options["use_window_loss_based_optimization"]:
            print(f"Step 4: Window loss-based optimization ({self.window_optimization_iterations} iterations)...")
            g_overlapped_optimized = self.window_loss_based_optimization(
                g_overlapped_for_optimization, new_keyframe_pose, new_keyframe_intrinsics,  # 使用完整的左右相机姿态和内参
                new_keyframe_image_size, supervision_frames
            )
        else:
            g_overlapped_optimized = g_overlapped_for_optimization
            
            
        
        # Step 5: 融合优化后的重叠高斯点和遗留高斯点
        print("Step 5: Concatenating optimized overlapped and legacy gaussians...")
        g_update_global = torch.cat([g_overlapped_optimized, g_legacy], dim=0)
        
        # Step 6: 全局优化
        if self.optimiation_options["use_global_optimization"]:
            print(f"Step 6: Global optimization ({self.global_optimization_iterations} iterations)...")
            g_final_global = self.global_optimization(g_update_global,
                                                      window_size=new_keyframe_image_size)
        else:
            g_final_global = g_update_global
        

        
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
            # FXIME
            #g_voxelized = self.voxel_based_pruning(g_filtered, g_new, new_keyframe_pose)
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
    
    def rendered_views_using_gaussians(self, gaussians: torch.Tensor, 
                                       frame_data: Dict) -> torch.Tensor:
        """
        """
        rendered_c2w = frame_data["output"]["c2w"].to(gaussians.device)
        fovxs = frame_data["output"]["fovxs"].to(gaussians.device)
        fovys = frame_data["output"]["fovys"].to(gaussians.device)
        
        
        if len(rendered_c2w.shape) < 4:
            rendered_c2w = rendered_c2w.unsqueeze(0)
        if len(fovxs.shape) < 2:
            fovxs = fovxs.unsqueeze(0)
        if len(fovys.shape) < 2:
            fovys = fovys.unsqueeze(0)
        
        rendered_results = self.renderer.render(
            gaussians=gaussians.unsqueeze(0),
            c2w=rendered_c2w,
            fovx=fovxs,
            fovy=fovys,
            rays_o=None,
            rays_d=None
        )
        
        rendered_image = rendered_results['image']
        rendered_depth = rendered_results['depth']
        
        return rendered_image, rendered_depth


    def gradient_based_pruning(self, gaussians: torch.Tensor, 
                            frame_data: Dict, 
                            keep_ratio: float = 0.1,
                            keep_count: int = None) -> torch.Tensor:
        """
        基于梯度的剪枝：选择最可靠的10%高斯点
        """
        
        if keep_count is None:
            N = gaussians.shape[0]
            target_count = int(N * keep_ratio)
        else:
            target_count = keep_count
        
        #print(f"Starting gradient-based pruning: {N} -> {target_count} gaussians (keep {keep_ratio*100}%)")
        
        # 确保高斯点需要梯度
        gaussians = gaussians.detach().clone().requires_grad_(True)
        
        # 1. 渲染所有高斯点
        rendered_c2w = frame_data["output"]["c2w"].to(gaussians.device)
        fovxs = frame_data["output"]["fovxs"].to(gaussians.device)
        fovys = frame_data["output"]["fovys"].to(gaussians.device)

        if len(rendered_c2w.shape) < 4:
            rendered_c2w = rendered_c2w.unsqueeze(0)
        if len(fovxs.shape) < 2:
            fovxs = fovxs.unsqueeze(0)
        if len(fovys.shape) < 2:
            fovys = fovys.unsqueeze(0)
        
        rendered_results = self.renderer.render(
            gaussians=gaussians.unsqueeze(0),
            c2w=rendered_c2w,
            fovx=fovxs,
            fovy=fovys,
            rays_o=None,
            rays_d=None
        )
        
        # 2. 提取渲染结果
        rendered_image = rendered_results['image']      # [1, V, 3, H, W]
        rendered_depth = rendered_results['depth']      # [1, V, 1, H, W]
        
        # 3. 准备GT数据
        if len(frame_data['output']['imgs'].shape) < 5:    
            output_rgb_for_supervision = frame_data['output']['imgs'].unsqueeze(0).to(rendered_image.device)
        else:
            output_rgb_for_supervision = frame_data['output']['imgs'].to(rendered_image.device)
        
        if len(frame_data['output']['psuedo_depth']) < 4:
            output_depth_for_supervision = frame_data['output']['psuedo_depth'].unsqueeze(0).unsqueeze(0).to(rendered_image.device)
        else:
            output_depth_for_supervision = frame_data['output']['psuedo_depth'].unsqueeze(0).to(rendered_image.device)
        
        # 4. 计算损失
        rgb_l1_loss_value = F.l1_loss(rendered_image, output_rgb_for_supervision)
        depth_l1_loss_value = F.l1_loss(rendered_depth, output_depth_for_supervision)
        total_loss = rgb_l1_loss_value + self.lambda_depth * depth_l1_loss_value
        
        print(f"Total loss: {total_loss.item():.6f}")
        
        # 5. 反向传播，计算梯度
        total_loss.backward()
        
        # 6. 分析每个高斯点的梯度贡献
        gradient_contributions = self._analyze_gaussian_gradients(gaussians)
        
        # 7. 选择梯度最小的点（最可靠的点）
        _, top_indices = torch.topk(-gradient_contributions, k=target_count)  # 负号选择最小的
        
        selected_gaussians = gaussians[top_indices].detach().clone()
        

        # 8. 清理梯度
        gaussians.grad = None
        gaussians.requires_grad_(False)
        
        return selected_gaussians

    def _analyze_gaussian_gradients(self, gaussians: torch.Tensor) -> torch.Tensor:
        """
        分析每个高斯点的梯度贡献
        
        Args:
            gaussians: 高斯点 [N, 14]，需要已经计算过梯度
            
        Returns:
            gradient_contributions: 每个点的梯度贡献 [N]
        """
        if gaussians.grad is None:
            raise ValueError("Gaussians must have gradients computed")
        
        N = gaussians.shape[0]
        device = gaussians.device
        
        # 计算每个高斯点的梯度范数
        gradients = gaussians.grad  # [N, 14]
        
        # 方法1：所有参数的梯度范数
        gradient_norms = torch.norm(gradients, dim=1)  # [N]
        
        # 方法2：只考虑位置和透明度的梯度（更关注几何和外观）
        # position_gradients = gradients[:, :3]  # 位置梯度
        # opacity_gradients = gradients[:, 6:7]  # 透明度梯度
        # scale_gradients = gradients[:, 11:14]  # 尺度梯度
        # 
        # position_norms = torch.norm(position_gradients, dim=1)
        # opacity_norms = torch.abs(opacity_gradients).squeeze()
        # scale_norms = torch.norm(scale_gradients, dim=1)
        # 
        # # 加权组合
        # gradient_norms = 0.5 * position_norms + 0.3 * opacity_norms + 0.2 * scale_norms
        
        return gradient_norms

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
        
        if len(rendered_c2w.shape)<4:
            rendered_c2w = rendered_c2w.unsqueeze(0)
        if len(fovxs.shape)<2:
            fovxs = fovxs.unsqueeze(0)
        if len(fovys.shape)<2:
            fovys = fovys.unsqueeze(0)
        
        
        rendered_results =self.renderer.render(
            gaussians=gaussians.unsqueeze(0),
            c2w=rendered_c2w,
            fovx=fovxs,
            fovy=fovys,
            rays_o=None,
            rays_d=None
        )


        # 提取渲染结果
        rendered_image = rendered_results['image']      # [1, V, 3, H, W]
        rendered_depth = rendered_results['depth']      # [1, V, 1, H, W]
        
        # GT images and the depth information.
        if len(frame_data['output']['imgs'].shape)<5:    
            output_rgb_for_supervision = frame_data['output']['imgs'].unsqueeze(0).to(rendered_image.device)
        else:
            output_rgb_for_supervision = frame_data['output']['imgs'].to(rendered_image.device)
        
        if len(frame_data['output']['psuedo_depth'])<4:
            output_depth_for_supervision = frame_data['output']['psuedo_depth'].unsqueeze(0).unsqueeze(0).to(rendered_image.device)
        else:
            output_depth_for_supervision = frame_data['output']['psuedo_depth'].unsqueeze(0).to(rendered_image.device)
        
        # Loss Function Here
        #rgb_l1_loss_value = F.l1_loss(rendered_image, output_rgb_for_supervision)
        
        rgb_loss_l1_dssim_5d_value, rgb_loss_l1_dssim_5d_dict = rgb_loss_l1_dssim_5d(rendered_image, output_rgb_for_supervision)
        
        # lpips_loss_value = lpips_vgg_batch(rendered_image, output_rgb_for_supervision,
        #                                    net=self.lpips_net)
        
        # rgb_loss = rgb_l1_loss_value + lpips_loss_value*0.05
        
        rgb_loss = rgb_loss_l1_dssim_5d_value
        
        depth_l1_loss_value = F.l1_loss(rendered_depth, output_depth_for_supervision)
        
        total_loss = rgb_loss + self.lambda_depth * depth_l1_loss_value
        
        
        return total_loss 
        

    def global_optimization(self, gaussians: torch.Tensor,
                            window_size: Tuple[int, int]) -> torch.Tensor:
        """
        基于历史关键帧的全局优化
        
        Args:
            gaussians: 待优化的全局高斯点 [N, 14]
            
        Returns:
            optimized_gaussians: 优化后的全局高斯点 [N, 14]
        """
        if gaussians.shape[0] == 0:
            return gaussians
        
        if self.history_batch_data is None or len(self.history_batch_data) == 0:
            print("Warning: No history batch data available, returning original gaussians")
            return gaussians
        
        print(f"Starting global optimization with {gaussians.shape[0]} gaussians")
        print(f"Using {len(self.history_batch_data)} historical frames for optimization")
        
        # 创建可训练的高斯点参数
        gaussians_optimizable = gaussians.clone().detach().requires_grad_(True)
        
        # 优化器
        optimizer = optim.Adam([gaussians_optimizable], lr=0.0005)
        
        # 全局优化循环
        for iteration in range(self.global_optimization_iterations):
            optimizer.zero_grad()
            
            total_loss = 0.0
            
            # 随机选择历史帧进行优化
            num_samples = min(3, len(self.history_batch_data))  # 每次迭代选择3个历史帧
            selected_indices = random.sample(range(len(self.history_batch_data)), num_samples)
            
            for idx in selected_indices:
                frame_data = self.history_batch_data[idx]
                
                
                
                # 计算当前历史帧的渲染损失
                frame_loss = self.compute_rendering_loss_multi_camera(
                    gaussians_optimizable, frame_data, 
                    frame_data["output"]["c2w"][0], 
                    frame_data["input"]["cks"], 
                    window_size,
                )
                
                frame_loss.backward()
                total_loss += frame_loss
                
            # 如果没有成功计算任何损失，使用正则化损失
            if total_loss == 0.0:
                total_loss = torch.tensor(0.001, device=gaussians.device, requires_grad=True)
            
            # 反向传播
            
            optimizer.step()
            
            if iteration % 5 == 0:
                print(f"Global optimization iteration {iteration}, loss: {total_loss.item():.6f}")
        
        print("Global optimization completed")
        return gaussians_optimizable.detach()
    

    def filter_by_opacity(self, gaussians: torch.Tensor, 
                          opacity_threshold: float = 0.01) -> torch.Tensor:
        """
        基于透明度的剪枝
        """
        opacity = gaussians[:, 6]
        mask = opacity > opacity_threshold
        return gaussians[mask]



    def depth_consistency_mask(
        self,
        K: torch.Tensor,                 # (V, 3, 3)
        cam2world: torch.Tensor,         # (V, 4, 4)
        depth_maps: torch.Tensor,        # (V, H, W)
        points_world: torch.Tensor,      # (N, 3)
        threshold: float                 # 标量阈值（同单位：米）
    ) -> torch.Tensor:
        """
        返回: mask (N,)，若某点被任一相机看到且 |z_pred - z_gt| < threshold 则为 True
        """
        device = points_world.device
        dtype  = points_world.dtype
        V, H, W = depth_maps.shape

        # 1) world->cam 变换
        world2cam = torch.linalg.inv(cam2world.to(device=device, dtype=dtype))  # (V, 4, 4)

        # 准备点的齐次坐标，并扩展到 (V, N, 4)
        N = points_world.shape[0]
        ones = torch.ones((N, 1), device=device, dtype=dtype)
        Pw_h = torch.cat([points_world.to(device=device, dtype=dtype), ones], dim=-1)  # (N,4)
        Pw_h = Pw_h.unsqueeze(0).expand(V, -1, -1)                                     # (V,N,4)

        # cam = world2cam @ Pw_h^T
        cam = torch.bmm(world2cam, Pw_h.transpose(1, 2)).transpose(1, 2)  # (V,N,4)
        Xc, Yc, Zc = cam[..., 0], cam[..., 1], cam[..., 2]                 # (V,N)

        # 仅正深度
        z_positive = Zc > 0

        # 2) 像素投影 u = fx*x/z + cx, v = fy*y/z + cy（忽略skew）
        K = K.to(device=device, dtype=dtype)
        fx, fy = K[:, 0, 0].unsqueeze(-1), K[:, 1, 1].unsqueeze(-1)  # (V,1)
        cx, cy = K[:, 0, 2].unsqueeze(-1), K[:, 1, 2].unsqueeze(-1)  # (V,1)

        invZ = torch.where(z_positive, 1.0 / Zc, torch.zeros_like(Zc))
        u = fx * (Xc * invZ) + cx   # (V,N)
        v = fy * (Yc * invZ) + cy   # (V,N)

        # 3) 双线性从深度图采样：需要把(u,v)归一化到[-1,1]
        #    注意 align_corners=True 时，归一化公式为: x_norm = u/(W-1)*2 - 1; y 同理
        x_norm = (u / (W - 1) * 2.0) - 1.0
        y_norm = (v / (H - 1) * 2.0) - 1.0

        # 有效像素范围（不让grid_sample把越界当成0深度误导）
        in_bounds = (
            (x_norm >= -1.0) & (x_norm <= 1.0) &
            (y_norm >= -1.0) & (y_norm <= 1.0)
        )

        # 组织 grid 给 grid_sample: (V, N, 1, 2) ; input: (V,1,H,W)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (V,N,1,2)
        depth_in = depth_maps.to(device=device, dtype=dtype).unsqueeze(1)  # (V,1,H,W)

        # 采样 (V,1,N,1) -> (V,N)
        depth_sampled = torch.nn.functional.grid_sample(
            depth_in, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(1).squeeze(-1)  # (V,N)

        # 4) 构建有效性：正深度 + 在图内 + gt深度>0
        gt_valid = depth_sampled > 0
        valid = z_positive & in_bounds & gt_valid  # (V,N)

        # 5) 误差并聚合到点级别（任一相机满足即可）
        err = (Zc - depth_sampled).abs()  # (V,N)
        ok = valid & (err < float(threshold))

        # 任一相机Ok -> 点可用
        mask_any = ok.any(dim=0)  # (N,)

        return mask_any
    
    # geometry-based pruning
    def geometry_based_pruning(self, gaussians: torch.Tensor, 
                               frame_data: Dict, 
                               depth_error_threshold: float = 0.1) -> torch.Tensor:
        """
        基于几何的剪枝：删除深度误差大于阈值的3DGS
        
        Args:
            gaussians: 高斯点 [N, 14]
            frame_data: 帧数据，包含相机参数和GT深度
            depth_error_threshold: 深度误差阈值，大于此值的点会被删除
            
        Returns:
            selected_gaussians: 选择后的高斯点 [M, 14], M <= N
        """
        gaussians_pcd = gaussians[:, :3]

        gt_depth = frame_data['output']['psuedo_depth'] #(2,H,W)

        mask = self.depth_consistency_mask(frame_data["output"]['cks'], 
                                           frame_data["output"]['c2w'], 
                                           gt_depth, 
                                           gaussians_pcd,depth_error_threshold)
        
        pruned_gaussians = gaussians[mask]
        
        return pruned_gaussians


    def _fallback_simple_selection(self, gaussians: torch.Tensor, target_count: int) -> torch.Tensor:
        """
        回退方案：基于属性的简单选择
        """
        N = gaussians.shape[0]
        
        # 基于透明度和尺度评分
        opacity_scores = gaussians[:, 6]  # 透明度
        
        scales = gaussians[:, 11:14]
        scale_norms = torch.norm(scales, dim=1)
        scale_scores = torch.where(
            (scale_norms >= 0.05) & (scale_norms <= 0.2),
            torch.ones_like(scale_norms),
            0.5 * torch.ones_like(scale_norms)
        )
        
        # 综合评分
        total_scores = 0.7 * opacity_scores + 0.3 * scale_scores
        
        # 选择评分最高的点
        _, top_indices = torch.topk(total_scores, k=target_count)
        selected_gaussians = gaussians[top_indices]
        
        print(f"Fallback selection: {N} -> {target_count} gaussians")
        
        return selected_gaussians
    
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
        # 使用 detach() 避免梯度问题
        R_c2w = Rscipy.from_matrix(R.detach().cpu().numpy())
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
                                     history_batch_data=None,
                                     opacity_based_pruning_threshold: float = 0.01,
                                     gradient_based_pruning_keep_ratio: float = 0.1,
                                     geometry_based_pruning_depth_error_threshold: float = 0.1,
                                     window_optimization_iterations: int = 50,
                                     global_optimization_iterations: int = 20,
                                     lambda_depth: float = 1.0,
                                     optimiation_options: Dict = None) -> IncrementalGaussianFusion:
 

    fusion_module = IncrementalGaussianFusion(
        renderer=renderer,
        history_batch_data=history_batch_data,
        geometry_based_pruning_depth_error_threshold=geometry_based_pruning_depth_error_threshold,
        gradient_based_pruning_keep_ratio=gradient_based_pruning_keep_ratio,
        opacity_based_pruning_threshold=opacity_based_pruning_threshold,
        window_optimization_iterations=window_optimization_iterations,
        global_optimization_iterations=global_optimization_iterations,
        lambda_depth=lambda_depth,
        optimiation_options=optimiation_options
    )
    

    
    return fusion_module


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
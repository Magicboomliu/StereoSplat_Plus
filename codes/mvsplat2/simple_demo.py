import torch
import torch.nn as nn
import numpy as np

def create_simple_mvsplat_demo():
    """
    创建简化的MVSplat输入demo，用于测试
    """
    
    # ==================== 基本参数 ====================
    batch_size = 1
    num_context_views = 2
    height, width = 180, 320
    
    # ==================== 创建Context输入 ====================
    # 图像: [B, V, C, H, W] = [1, 2, 3, 180, 320]
    context_images = torch.randn(batch_size, num_context_views, 3, height, width)
    context_images = torch.sigmoid(context_images)  # 归一化到[0,1]
    
    # 内参: [B, V, 3, 3] = [1, 2, 3, 3]
    context_intrinsics = torch.eye(3, dtype=torch.float32)
    context_intrinsics = context_intrinsics.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    context_intrinsics = context_intrinsics.repeat(batch_size, num_context_views, 1, 1)  # [1, 2, 3, 3]
    
    # 设置合理的焦距和主点
    fx, fy = width * 0.8, height * 0.8
    cx, cy = width / 2.0, height / 2.0
    context_intrinsics[:, :, 0, 0] = fx  # fx
    context_intrinsics[:, :, 1, 1] = fy  # fy  
    context_intrinsics[:, :, 0, 2] = cx  # cx
    context_intrinsics[:, :, 1, 2] = cy  # cy
    
    # 外参 (cam2world): [B, V, 4, 4] = [1, 2, 4, 4]
    context_extrinsics = torch.eye(4, dtype=torch.float32)
    context_extrinsics = context_extrinsics.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    context_extrinsics = context_extrinsics.repeat(batch_size, num_context_views, 1, 1)  # [1, 2, 4, 4]
    
    # 设置两个不同的相机位置
    # 相机1: 位置 (2, 0, 1)
    context_extrinsics[0, 0, :3, 3] = torch.tensor([2.0, 0.0, 1.0])
    # 相机2: 位置 (0, 2, 1) 
    context_extrinsics[0, 1, :3, 3] = torch.tensor([0.0, 2.0, 1.0])
    
    # 设置相机朝向 (都看向原点)
    for v in range(num_context_views):
        camera_pos = context_extrinsics[0, v, :3, 3]
        look_at = torch.tensor([0.0, 0.0, 0.0])
        forward = look_at - camera_pos
        forward = forward / torch.norm(forward)
        
        right = torch.cross(forward, torch.tensor([0.0, 0.0, 1.0]))
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        context_extrinsics[0, v, :3, 0] = right
        context_extrinsics[0, v, :3, 1] = up  
        context_extrinsics[0, v, :3, 2] = forward
    
    # 深度范围: [B, V] = [1, 2]
    near_value = 0.1    # 0.1米
    far_value = 1000.0  # 1000米
    context_near = torch.full((batch_size, num_context_views), near_value, dtype=torch.float32)
    context_far = torch.full((batch_size, num_context_views), far_value, dtype=torch.float32)
    
    # 视图索引: [V] = [2]
    context_indices = torch.arange(num_context_views, dtype=torch.int64)
    
    # ==================== 构建Context字典 ====================
    context = {
        "image": context_images,           # [1, 2, 3, 180, 320]
        "intrinsics": context_intrinsics,  # [1, 2, 3, 3]
        "extrinsics": context_extrinsics,  # [1, 2, 4, 4] - cam2world
        "near": context_near,              # [1, 2]
        "far": context_far,                # [1, 2]
        "index": context_indices,          # [2]
    }
    
    return context

def print_context_info(context):
    """
    打印context的详细信息
    """
    print("=" * 50)
    print("MVSplat Context 输入信息")
    print("=" * 50)
    
    print(f"图像形状: {context['image'].shape}")
    print(f"内参形状: {context['intrinsics'].shape}")
    print(f"外参形状: {context['extrinsics'].shape}")
    print(f"Near形状: {context['near'].shape}, 数值: {context['near']}")
    print(f"Far形状:  {context['far'].shape}, 数值: {context['far']}")
    print(f"索引形状: {context['index'].shape}, 数值: {context['index']}")
    print()
    
    print("图像数值范围:")
    print(f"  Min: {context['image'].min():.3f}")
    print(f"  Max: {context['image'].max():.3f}")
    print()
    
    print("第一个视图的内参:")
    print(context['intrinsics'][0, 0])
    print()
    
    print("第一个视图的外参 (cam2world):")
    print(context['extrinsics'][0, 0])
    print()
    
    print("第二个视图的外参 (cam2world):")
    print(context['extrinsics'][0, 1])

def create_batch_demo():
    """
    创建完整的batch demo (包含context和target)
    """
    batch_size = 1
    num_context_views = 2
    num_target_views = 1
    height, width = 180, 320
    
    # Context数据
    context_images = torch.randn(batch_size, num_context_views, 3, height, width)
    context_images = torch.sigmoid(context_images)
    
    # Target数据  
    target_images = torch.randn(batch_size, num_target_views, 3, height, width)
    target_images = torch.sigmoid(target_images)
    
    # 内参
    def create_intrinsics(batch_size, num_views, height, width):
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
        
        fx, fy = width * 0.8, height * 0.8
        cx, cy = width / 2.0, height / 2.0
        intrinsics[:, :, 0, 0] = fx
        intrinsics[:, :, 1, 1] = fy
        intrinsics[:, :, 0, 2] = cx
        intrinsics[:, :, 1, 2] = cy
        
        return intrinsics
    
    context_intrinsics = create_intrinsics(batch_size, num_context_views, height, width)
    target_intrinsics = create_intrinsics(batch_size, num_target_views, height, width)
    
    # 外参
    def create_extrinsics(batch_size, num_views):
        extrinsics = torch.eye(4, dtype=torch.float32)
        extrinsics = extrinsics.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
        
        # 设置相机位置
        positions = [
            [2.0, 0.0, 1.0],   # 相机1
            [0.0, 2.0, 1.0],   # 相机2
            [1.0, 1.0, 1.0],   # 相机3 (target)
        ]
        
        for v in range(num_views):
            extrinsics[0, v, :3, 3] = torch.tensor(positions[v])
            
            # 设置朝向
            camera_pos = extrinsics[0, v, :3, 3]
            look_at = torch.tensor([0.0, 0.0, 0.0])
            forward = look_at - camera_pos
            forward = forward / torch.norm(forward)
            
            right = torch.cross(forward, torch.tensor([0.0, 0.0, 1.0]))
            right = right / torch.norm(right)
            up = torch.cross(right, forward)
            
            extrinsics[0, v, :3, 0] = right
            extrinsics[0, v, :3, 1] = up
            extrinsics[0, v, :3, 2] = forward
            
        return extrinsics
    
    context_extrinsics = create_extrinsics(batch_size, num_context_views)
    target_extrinsics = create_extrinsics(batch_size, num_target_views)
    
    # 深度范围
    near_value, far_value = 0.1, 1000.0
    
    context_near = torch.full((batch_size, num_context_views), near_value, dtype=torch.float32)
    context_far = torch.full((batch_size, num_context_views), far_value, dtype=torch.float32)
    target_near = torch.full((batch_size, num_target_views), near_value, dtype=torch.float32)
    target_far = torch.full((batch_size, num_target_views), far_value, dtype=torch.float32)
    
    # 索引
    context_indices = torch.arange(num_context_views, dtype=torch.int64)
    target_indices = torch.arange(num_target_views, dtype=torch.int64)
    
    # 构建完整batch
    batch_example = {
        "context": {
            "image": context_images,
            "intrinsics": context_intrinsics,
            "extrinsics": context_extrinsics,
            "near": context_near,
            "far": context_far,
            "index": context_indices,
        },
        "target": {
            "image": target_images,
            "intrinsics": target_intrinsics,
            "extrinsics": target_extrinsics,
            "near": target_near,
            "far": target_far,
            "index": target_indices,
        },
        "scene": ["demo_scene_001"]
    }
    
    return batch_example

if __name__ == "__main__":
    print("创建MVSplat输入Demo...")
    
    # 创建简单的context demo
    context = create_simple_mvsplat_demo()
    print_context_info(context)
    
    print("\n" + "=" * 50)
    print("创建完整Batch Demo...")
    print("=" * 50)
    
    # 创建完整的batch demo
    batch_example = create_batch_demo()
    
    print(f"场景: {batch_example['scene']}")
    print(f"Context图像: {batch_example['context']['image'].shape}")
    print(f"Target图像:  {batch_example['target']['image'].shape}")
    print(f"Context内参: {batch_example['context']['intrinsics'].shape}")
    print(f"Target内参:  {batch_example['target']['intrinsics'].shape}")
    
    print("\nDemo创建完成！")
    print("现在你可以使用这些数据来测试MVSplat模型。")

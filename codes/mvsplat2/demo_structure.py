"""
MVSplat输入Demo - 纯Python版本
展示MVSplat输入数据的结构和格式
"""

import numpy as np

def create_mvsplat_demo_structure():
    """
    创建MVSplat输入数据的结构示例
    """
    
    print("=" * 60)
    print("MVSplat 输入数据结构Demo")
    print("=" * 60)
    
    # ==================== 基本参数 ====================
    batch_size = 1
    num_context_views = 2  # context视图数量
    num_target_views = 1   # target视图数量
    height, width = 180, 320
    
    print(f"基本参数:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Context Views: {num_context_views}")
    print(f"  Target Views: {num_target_views}")
    print(f"  图像尺寸: {height} x {width}")
    print()
    
    # ==================== Context数据结构 ====================
    print("Context 数据结构:")
    print(f"  image: [{batch_size}, {num_context_views}, 3, {height}, {width}]")
    print(f"         - 图像数据，范围[0,1]，RGB格式")
    print(f"  intrinsics: [{batch_size}, {num_context_views}, 3, 3]")
    print(f"             - 相机内参矩阵，标准3x3格式")
    print(f"  extrinsics: [{batch_size}, {num_context_views}, 4, 4]")
    print(f"             - 相机外参矩阵，cam2world格式")
    print(f"  near: [{batch_size}, {num_context_views}]")
    print(f"        - 近平面距离，默认0.1米")
    print(f"  far: [{batch_size}, {num_context_views}]")
    print(f"       - 远平面距离，默认1000.0米")
    print(f"  index: [{num_context_views}]")
    print(f"         - 视图索引，如[0, 1]")
    print()
    
    # ==================== Target数据结构 ====================
    print("Target 数据结构:")
    print(f"  image: [{batch_size}, {num_target_views}, 3, {height}, {width}]")
    print(f"         - 目标图像，范围[0,1]，RGB格式")
    print(f"  intrinsics: [{batch_size}, {num_target_views}, 3, 3]")
    print(f"             - 相机内参矩阵")
    print(f"  extrinsics: [{batch_size}, {num_target_views}, 4, 4]")
    print(f"             - 相机外参矩阵，cam2world格式")
    print(f"  near: [{batch_size}, {num_target_views}]")
    print(f"        - 近平面距离，默认0.1米")
    print(f"  far: [{batch_size}, {num_target_views}]")
    print(f"       - 远平面距离，默认1000.0米")
    print(f"  index: [{num_target_views}]")
    print(f"         - 视图索引，如[2]")
    print()
    
    # ==================== 相机参数示例 ====================
    print("相机内参示例 (标准格式):")
    print("  [[fx,  0, cx],")
    print("   [ 0, fy, cy],")
    print("   [ 0,  0,  1]]")
    print(f"  其中: fx={width*0.8:.1f}, fy={height*0.8:.1f}, cx={width/2:.1f}, cy={height/2:.1f}")
    print()
    
    print("相机外参示例 (cam2world格式):")
    print("  [[R11, R12, R13, tx],")
    print("   [R21, R22, R23, ty],")
    print("   [R31, R32, R33, tz],")
    print("   [ 0,   0,   0,  1]]")
    print("  其中: R是旋转矩阵，t是平移向量")
    print()
    
    # ==================== 完整输入示例 ====================
    print("完整输入示例结构:")
    print("batch_example = {")
    print("    'context': {")
    print("        'image': torch.Tensor([1, 2, 3, 180, 320]),")
    print("        'intrinsics': torch.Tensor([1, 2, 3, 3]),")
    print("        'extrinsics': torch.Tensor([1, 2, 4, 4]),")
    print("        'near': torch.Tensor([1, 2]),")
    print("        'far': torch.Tensor([1, 2]),")
    print("        'index': torch.Tensor([2]),")
    print("    },")
    print("    'target': {")
    print("        'image': torch.Tensor([1, 1, 3, 180, 320]),")
    print("        'intrinsics': torch.Tensor([1, 1, 3, 3]),")
    print("        'extrinsics': torch.Tensor([1, 1, 4, 4]),")
    print("        'near': torch.Tensor([1, 1]),")
    print("        'far': torch.Tensor([1, 1]),")
    print("        'index': torch.Tensor([1]),")
    print("    },")
    print("    'scene': ['demo_scene_001']")
    print("}")
    print()

def create_pytorch_demo_code():
    """
    生成PyTorch代码示例
    """
    print("=" * 60)
    print("PyTorch代码示例")
    print("=" * 60)
    
    code = '''
import torch
import numpy as np

def create_mvsplat_demo():
    """创建MVSplat输入demo"""
    
    # 基本参数
    batch_size = 1
    num_context_views = 2
    num_target_views = 1
    height, width = 180, 320
    
    # 创建图像数据 [B, V, C, H, W]
    context_images = torch.randn(batch_size, num_context_views, 3, height, width)
    context_images = torch.sigmoid(context_images)  # 归一化到[0,1]
    
    target_images = torch.randn(batch_size, num_target_views, 3, height, width)
    target_images = torch.sigmoid(target_images)
    
    # 创建内参矩阵 [B, V, 3, 3]
    def create_intrinsics(batch_size, num_views, height, width):
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
        intrinsics = intrinsics.repeat(batch_size, num_views, 1, 1)
        
        # 设置焦距和主点
        fx, fy = width * 0.8, height * 0.8
        cx, cy = width / 2.0, height / 2.0
        intrinsics[:, :, 0, 0] = fx  # fx
        intrinsics[:, :, 1, 1] = fy  # fy
        intrinsics[:, :, 0, 2] = cx  # cx
        intrinsics[:, :, 1, 2] = cy  # cy
        
        return intrinsics
    
    context_intrinsics = create_intrinsics(batch_size, num_context_views, height, width)
    target_intrinsics = create_intrinsics(batch_size, num_target_views, height, width)
    
    # 创建外参矩阵 (cam2world) [B, V, 4, 4]
    def create_extrinsics(batch_size, num_views):
        extrinsics = torch.eye(4, dtype=torch.float32)
        extrinsics = extrinsics.unsqueeze(0).unsqueeze(0)
        extrinsics = extrinsics.repeat(batch_size, num_views, 1, 1)
        
        # 设置相机位置
        positions = [
            [2.0, 0.0, 1.0],   # 相机1
            [0.0, 2.0, 1.0],   # 相机2
            [1.0, 1.0, 1.0],   # 相机3 (target)
        ]
        
        for v in range(num_views):
            extrinsics[0, v, :3, 3] = torch.tensor(positions[v])
            
            # 设置相机朝向 (看向原点)
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
    
    # 深度范围 [B, V]
    near_value, far_value = 0.1, 1000.0
    context_near = torch.full((batch_size, num_context_views), near_value, dtype=torch.float32)
    context_far = torch.full((batch_size, num_context_views), far_value, dtype=torch.float32)
    target_near = torch.full((batch_size, num_target_views), near_value, dtype=torch.float32)
    target_far = torch.full((batch_size, num_target_views), far_value, dtype=torch.float32)
    
    # 视图索引
    context_indices = torch.arange(num_context_views, dtype=torch.int64)
    target_indices = torch.arange(num_target_views, dtype=torch.int64)
    
    # 构建完整输入
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

# 使用示例
if __name__ == "__main__":
    batch_example = create_mvsplat_demo()
    
    print("Context信息:")
    print(f"  图像: {batch_example['context']['image'].shape}")
    print(f"  内参: {batch_example['context']['intrinsics'].shape}")
    print(f"  外参: {batch_example['context']['extrinsics'].shape}")
    print(f"  Near: {batch_example['context']['near']}")
    print(f"  Far:  {batch_example['context']['far']}")
    
    print("\\nTarget信息:")
    print(f"  图像: {batch_example['target']['image'].shape}")
    print(f"  内参: {batch_example['target']['intrinsics'].shape}")
    print(f"  外参: {batch_example['target']['extrinsics'].shape}")
    print(f"  Near: {batch_example['target']['near']}")
    print(f"  Far:  {batch_example['target']['far']}")
'''
    
    print(code)

def print_usage_guide():
    """
    打印使用指南
    """
    print("=" * 60)
    print("使用指南")
    print("=" * 60)
    
    print("1. 基本用法:")
    print("   - 将demo数据传递给MVSplat的encoder")
    print("   - 确保所有tensor在正确的设备上 (CPU/GPU)")
    print("   - 图像数据范围必须在[0,1]")
    print()
    
    print("2. 关键要点:")
    print("   - 外参是cam2world格式，不是world2cam")
    print("   - 内参是标准的3x3相机内参矩阵")
    print("   - near/far定义深度搜索范围")
    print("   - 图像必须是RGB格式，范围[0,1]")
    print()
    
    print("3. 形状要求:")
    print("   - Context: [B, V_context, C, H, W]")
    print("   - Target:  [B, V_target, C, H, W]")
    print("   - 其中V_context通常=2, V_target通常=1")
    print()
    
    print("4. 数值范围:")
    print("   - 图像: [0, 1] (归一化)")
    print("   - Near: 0.1米 (默认)")
    print("   - Far: 1000.0米 (默认)")
    print("   - 内参: 标准相机参数")
    print("   - 外参: cam2world变换矩阵")

if __name__ == "__main__":
    # 创建结构说明
    create_mvsplat_demo_structure()
    
    # 生成PyTorch代码
    create_pytorch_demo_code()
    
    # 打印使用指南
    print_usage_guide()
    
    print("\n" + "=" * 60)
    print("Demo创建完成！")
    print("=" * 60)
    print("现在你可以使用这些信息来构建MVSplat的输入数据。")

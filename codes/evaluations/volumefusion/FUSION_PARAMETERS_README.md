# Fusion Pipeline Hyperparameters Configuration

## 概述

`baseline_incremental_fusion.py` 现在支持通过命令行参数配置融合管道的所有超参数，无需修改代码即可调整融合策略。

## 可配置的超参数

### 1. 融合核心参数

#### `--voxel_size` (默认: 0.05)
- **类型**: float
- **描述**: 融合管道的体素大小
- **影响**: 较小的值提供更精细的融合，但增加计算开销
- **建议范围**: 0.01 - 0.1

#### `--opacity_threshold` (默认: 0.01)
- **类型**: float
- **描述**: 透明度阈值，用于过滤低质量的高斯点
- **影响**: 较高的值会过滤更多点，减少内存使用但可能降低质量
- **建议范围**: 0.005 - 0.05

#### `--depth_threshold` (默认: 0.1)
- **类型**: float
- **描述**: 深度阈值，用于深度一致性检查
- **影响**: 控制深度融合的严格程度
- **建议范围**: 0.05 - 0.2

### 2. 优化参数

#### `--window_optimization_iterations` (默认: 50)
- **类型**: int
- **描述**: 窗口优化迭代次数
- **影响**: 更多的迭代可能提高质量，但增加计算时间
- **建议范围**: 20 - 100

#### `--global_optimization_iterations` (默认: 30)
- **类型**: int
- **描述**: 全局优化迭代次数
- **影响**: 控制全局融合的精细程度
- **建议范围**: 10 - 50

#### `--lambda_depth` (默认: 0.01)
- **类型**: float
- **描述**: 深度损失权重
- **影响**: 控制深度一致性在融合中的重要性
- **建议范围**: 0.001 - 0.1

### 3. 内存优化参数

#### `--enable_memory_optimization` (默认: True)
- **类型**: flag
- **描述**: 启用内存优化策略
- **影响**: 减少内存使用，但可能略微增加计算时间

#### `--memory_threshold` (默认: 0.8)
- **类型**: float
- **描述**: 内存使用阈值（0.8 = 80%）
- **影响**: 控制何时触发内存优化
- **建议范围**: 0.6 - 0.9

#### `--enable_gradient_checkpointing` (默认: True)
- **类型**: flag
- **描述**: 启用梯度检查点以节省内存
- **影响**: 显著减少内存使用，但增加约20-30%的计算开销

## 使用示例

### 基本使用（使用默认参数）
```bash
python baseline_incremental_fusion.py \
    --config_path config.py \
    --output_folder output \
    --semi_global_map maps \
    --ablation_type incremental_fusion
```

### 调整融合精度
```bash
python baseline_incremental_fusion.py \
    --config_path config.py \
    --output_folder output \
    --semi_global_map maps \
    --ablation_type incremental_fusion \
    --voxel_size 0.03 \
    --opacity_threshold 0.008 \
    --depth_threshold 0.08
```

### 优化性能（减少迭代次数）
```bash
python baseline_incremental_fusion.py \
    --config_path config.py \
    --output_folder output \
    --semi_global_map maps \
    --ablation_type incremental_fusion \
    --window_optimization_iterations 30 \
    --global_optimization_iterations 20
```

### 内存优化配置
```bash
python baseline_incremental_fusion.py \
    --config_path config.py \
    --output_folder output \
    --semi_global_map maps \
    --ablation_type incremental_fusion \
    --memory_threshold 0.7 \
    --enable_gradient_checkpointing
```

### 高质量融合（增加迭代次数）
```bash
python baseline_incremental_fusion.py \
    --config_path config.py \
    --output_folder output \
    --semi_global_map maps \
    --ablation_type incremental_fusion \
    --voxel_size 0.02 \
    --window_optimization_iterations 80 \
    --global_optimization_iterations 50 \
    --lambda_depth 0.02
```

## 参数调优建议

### 内存不足时
1. **降低精度**: 增加 `voxel_size` 到 0.08-0.1
2. **减少迭代**: 降低 `window_optimization_iterations` 到 20-30
3. **启用内存优化**: 使用 `--enable_memory_optimization`
4. **降低内存阈值**: 设置 `--memory_threshold 0.6`

### 质量不足时
1. **提高精度**: 降低 `voxel_size` 到 0.02-0.03
2. **增加迭代**: 提高 `window_optimization_iterations` 到 80-100
3. **调整阈值**: 降低 `opacity_threshold` 到 0.005-0.008
4. **增强深度一致性**: 增加 `lambda_depth` 到 0.02-0.05

### 速度优化
1. **减少迭代**: 降低两个迭代参数
2. **增加体素大小**: 提高 `voxel_size`
3. **提高阈值**: 增加 `opacity_threshold` 和 `depth_threshold`

## 参数组合示例

### 快速测试配置
```bash
--voxel_size 0.08 \
--window_optimization_iterations 20 \
--global_optimization_iterations 15 \
--memory_threshold 0.6
```

### 平衡配置
```bash
--voxel_size 0.05 \
--window_optimization_iterations 50 \
--global_optimization_iterations 30 \
--memory_threshold 0.8
```

### 高质量配置
```bash
--voxel_size 0.02 \
--window_optimization_iterations 100 \
--global_optimization_iterations 60 \
--lambda_depth 0.03 \
--memory_threshold 0.9
```

## 监控和调试

程序会自动打印当前使用的超参数：
```
=== Fusion Pipeline Hyperparameters ===
Voxel size: 0.05
Opacity threshold: 0.01
Depth threshold: 0.1
Window optimization iterations: 50
Global optimization iterations: 30
Lambda depth: 0.01
=====================================
```

## 注意事项

1. **参数依赖**: 某些参数组合可能不兼容，需要实验验证
2. **内存平衡**: 高质量配置通常需要更多内存
3. **时间权衡**: 更多迭代提高质量但增加处理时间
4. **默认值**: 默认参数经过调优，适合大多数场景

## 故障排除

### 如果融合质量差
- 检查 `voxel_size` 是否过大
- 增加迭代次数
- 降低阈值参数

### 如果内存不足
- 增加 `voxel_size`
- 减少迭代次数
- 启用内存优化

### 如果速度太慢
- 减少迭代次数
- 增加 `voxel_size`
- 提高阈值参数

# Memory Optimization for Baseline Incremental Fusion

## 问题描述

在处理第2个key frame时出现CUDA内存不足错误，主要原因是：
1. 增量式融合过程中内存累积
2. Transformer attention计算内存消耗大
3. 缺少主动内存管理

## 解决方案

### 1. 内存管理函数

#### `clear_gpu_cache()`
- 清理GPU缓存
- 同步GPU操作
- 减少内存碎片

#### `clear_intermediate_variables()`
- 删除中间变量
- 主动释放内存
- 防止内存泄漏

### 2. 自适应批处理

#### `adaptive_batch_processing()`
- 监控GPU内存使用率
- 自动跳过内存不足的batch
- 可配置内存阈值（默认80%）

### 3. 内存优化配置

```python
# 环境变量设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# PyTorch设置
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.utils.checkpoint.checkpoint_impl = "reentrant"
```

### 4. 渐进式处理策略

- 每3个key frame强制清理内存
- 异常处理和自动恢复
- 紧急状态保存

## 使用方法

### 基本使用
```bash
python baseline_incremental_fusion.py --config_path your_config.py
```

### 启用内存优化
```bash
python baseline_incremental_fusion.py \
    --config_path your_config.py \
    --enable_memory_optimization \
    --memory_threshold 0.7 \
    --enable_gradient_checkpointing
```

### 参数说明
- `--enable_memory_optimization`: 启用内存优化（默认True）
- `--memory_threshold`: 内存使用阈值（默认0.8，即80%）
- `--enable_gradient_checkpointing`: 启用梯度检查点（默认True）

## 监控和调试

### 内存使用监控
程序会自动打印GPU内存使用情况：
```
GPU Memory: 45.90GB allocated, 47.39GB reserved, 47.54GB total
```

### 内存清理日志
```
Periodic memory cleanup at keyframe 3
Memory optimization enabled
Gradient checkpointing enabled
```

### 异常处理
```
CUDA OOM at keyframe 2. Attempting memory cleanup...
Memory still high after cleanup. Skipping this batch.
```

## 高级优化策略

### 1. 模型分片
- 启用梯度检查点
- CPU卸载支持
- 模型评估模式

### 2. 批处理优化
- 动态批处理大小
- 内存不足时跳过
- 渐进式精度降低

### 3. 内存清理策略
- 周期性清理
- 异常时清理
- 紧急状态保存

## 故障排除

### 如果仍然出现OOM

1. **降低内存阈值**
   ```bash
   --memory_threshold 0.6
   ```

2. **启用更激进的内存优化**
   ```bash
   --enable_gradient_checkpointing
   ```

3. **检查GPU内存**
   ```bash
   nvidia-smi
   ```

4. **监控内存使用**
   - 程序会自动打印内存使用情况
   - 观察内存增长趋势

### 性能影响

- 内存优化会略微增加计算时间
- 梯度检查点会增加约20-30%的计算开销
- 但可以显著减少内存使用

## 最佳实践

1. **定期监控内存使用**
2. **适当设置内存阈值**
3. **启用梯度检查点**
4. **使用周期性内存清理**
5. **保存中间结果用于恢复**

## 注意事项

- 内存优化策略可能会影响模型精度
- 建议在开发阶段启用所有优化
- 生产环境可以根据需求调整参数
- 定期检查内存使用日志

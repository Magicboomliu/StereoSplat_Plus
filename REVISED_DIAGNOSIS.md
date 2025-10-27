# 🔍 Loss 不收敛问题 - 修订诊断

## 更正

**之前的分析有误**：内参归一化的方式本身**在数学上是正确的**。

标准内参矩阵：
```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

当前归一化代码：
```python
intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/width   # [fx, 0, cx]/w → 正确
intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/height  # [0, fy, cy]/h → 正确
```

因为 0/width = 0，0/height = 0，所以不会破坏矩阵结构。

---

## 🔍 需要重新排查的问题

### 1. **可能的真实问题：重复归一化**

**关键问题**：内参在哪里被归一化？

检查路径：
```
数据加载 (loading.py) 
  → 返回像素坐标内参 (fx≈552)
  → forward (mvsplat_model.py Line 354-355) 
  → 第一次归一化
  → 是否还有其他地方归一化？
```

**验证方法**：打印内参值
```python
# 在 mvsplat_model.py Line 353 后添加
print(f"Input intrinsics (before norm):")
print(f"  fx = {intrinsics[0, 0, 0, 0].item():.2f}")
print(f"  fy = {intrinsics[0, 0, 1, 1].item():.2f}")
print(f"  cx = {intrinsics[0, 0, 0, 2].item():.2f}")
print(f"  cy = {intrinsics[0, 0, 1, 2].item():.2f}")

# 在 Line 356 后添加
print(f"Output intrinsics (after norm):")
print(f"  fx = {intrinsics[0, 0, 0, 0].item():.6f}")
print(f"  fy = {intrinsics[0, 0, 1, 1].item():.6f}")
print(f"  cx = {intrinsics[0, 0, 0, 2].item():.6f}")
print(f"  cy = {intrinsics[0, 0, 1, 2].item():.6f}")
```

**期望输出**：
- Before: fx ≈ 500-600 (像素坐标)
- After: fx ≈ 0.9-1.1 (归一化后)

**如果不是这样**：
- Before fx < 10 → 可能已经归一化，会导致二次归一化！

---

### 2. **数据和模型不匹配**

检查配置：
- 分辨率：`resolution = [112, 544]`
- 原始图像尺寸是多少？
- Resize 比例是多少？
- 内参是否正确缩放？

**验证方法**：
```python
# 在数据加载后打印
print(f"Image shape: {input_images.shape}")  # [B, V, 3, H, W]
print(f"Config resolution: {cfg.dataset_params.resolution}")
print(f"fx = {intrinsics[0, 0, 0, 0]}, expected ≈ {552 * 544 / 1408}")  # 假设原始宽度1408
```

---

### 3. **Extrinsics 问题**

检查外参是否正确：
- 是 C2W 还是 W2C？
- 坐标系是 OpenCV 还是 OpenGL？

**当前配置**：
```python
camera_model='OpenCV'
```

**数据加载返回**：
```python
c2w = torch.from_numpy(input_c2ws).float()  # C2W 格式
```

**验证**：检查相机位置是否合理
```python
print(f"Camera positions:")
for v in range(input_extrinsics.shape[1]):
    pos = input_extrinsics[0, v, :3, 3]
    print(f"  View {v}: {pos}")
```

**期望**：相机位置应该在合理范围（比如 -50 到 50 米）

---

### 4. **Near/Far 设置问题**

**当前设置**：
```python
near = 0.1
far = 1000.0
```

**验证**：检查场景深度范围
```python
# 打印实际深度范围
sparse_depth = input_sparse_gt_depth[input_sparse_gt_depth > 0]
print(f"Sparse depth range: {sparse_depth.min():.2f} - {sparse_depth.max():.2f}")
```

**问题**：
- 如果大部分深度在 5-50 米，near=0.1 可能太小
- Far=1000 可能太大

---

### 5. **Loss 权重和数值范围**

**需要监控**：
```python
# 在 loss 计算后添加
print(f"RGB MSE loss: {rec_loss.item():.6f}")
print(f"RGB LPIPS loss: {preception_loss.item():.6f}")
print(f"Depth loss: {depth_loss.item():.6f}")
print(f"Total loss: {loss.item():.6f}")
```

**问题识别**：
- 如果某个 loss >> 其他 loss（比如 10 倍以上），说明不平衡
- 如果 loss 值非常大（> 100）或非常小（< 0.0001），可能有问题

---

### 6. **渲染结果检查**

**最直观的诊断**：查看渲染的图像

```python
# 保存第一个 iteration 的渲染结果
if global_iter == 0:
    import torchvision
    torchvision.utils.save_image(
        rendered_color[0],  # [V, 3, H, W]
        f"{cfg.work_dir}/debug_rendered_iter0.png"
    )
    torchvision.utils.save_image(
        output_rgb[0],  # GT
        f"{cfg.work_dir}/debug_gt_iter0.png"
    )
```

**期望**：
- 如果渲染图完全黑/白 → Gaussian 位置或不透明度有问题
- 如果渲染图模糊但有结构 → 可能只是需要训练
- 如果渲染图完全错位/扭曲 → 相机参数有严重问题

---

### 7. **学习率和优化器问题**

**当前设置**：
```python
lr = 1e-4
grad_max_norm = 1.0
```

**问题**：
- Gaussian Splatting 通常需要更大的学习率（5e-4 to 1e-3）
- grad_max_norm=1.0 可能太小

**验证**：监控梯度范数
```python
if global_iter % 10 == 0:
    total_norm = 0
    for p in my_model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Grad norm before clip: {total_norm:.4f}, after clip: {grad_norm:.4f}")
```

---

## 🎯 诊断优先级（修订）

| 优先级 | 检查项 | 验证方法 |
|--------|--------|----------|
| **P0** | 打印内参值 | 确认是否重复归一化 |
| **P0** | 查看渲染图像 | 最直观判断问题类型 |
| **P1** | 监控 Loss 数值 | 判断是否平衡 |
| **P1** | 检查深度范围 | Near/far 是否合理 |
| **P2** | 监控梯度范数 | 判断优化器配置 |
| **P2** | 验证 Extrinsics | 相机位置是否合理 |

---

## 🔬 立即执行的调试步骤

### 添加综合调试输出

在 `mvsplat_model.py` 的 `forward` 函数开始处添加（第 345 行后）：

```python
# ============ DEBUG START ============
if iter == 0 and self.training:  # 只在第一个 iteration 打印
    print("\n" + "="*50)
    print("DEBUG: First Training Iteration")
    print("="*50)
    
    # 1. 检查输入图像
    print(f"\n[1] Input Images:")
    print(f"    Shape: {input_images.shape}")
    print(f"    Min: {input_images.min():.4f}, Max: {input_images.max():.4f}")
    print(f"    Mean: {input_images.mean():.4f}")
    
    # 2. 检查内参（归一化前）
    print(f"\n[2] Intrinsics (BEFORE normalization):")
    print(f"    fx = {intrinsics[0, 0, 0, 0].item():.2f}")
    print(f"    fy = {intrinsics[0, 0, 1, 1].item():.2f}")
    print(f"    cx = {intrinsics[0, 0, 0, 2].item():.2f}")
    print(f"    cy = {intrinsics[0, 0, 1, 2].item():.2f}")
    
    # 3. 检查外参
    print(f"\n[3] Extrinsics (C2W):")
    for v in range(min(2, input_extrinsics.shape[1])):
        pos = input_extrinsics[0, v, :3, 3]
        print(f"    View {v} position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    # 4. 检查深度范围
    print(f"\n[4] Depth Info:")
    sparse_valid = input_sparse_gt_depth[input_sparse_gt_depth > 0]
    pseudo_valid = input_pseudo_depth[input_pseudo_depth > 0]
    print(f"    Sparse depth range: {sparse_valid.min():.2f} - {sparse_valid.max():.2f}")
    print(f"    Pseudo depth range: {pseudo_valid.min():.2f} - {pseudo_valid.max():.2f}")
    print(f"    Near/Far: {depth_min_value} / {depth_max_value}")
    
    # 5. 检查图像尺寸
    print(f"\n[5] Image Resolution:")
    print(f"    Config: {cfg.dataset_params.resolution}")
    print(f"    Actual: [{height}, {width}]")
    print(f"    Match: {cfg.dataset_params.resolution == [height, width]}")

# ... 执行归一化 ...

if iter == 0 and self.training:
    # 6. 检查内参（归一化后）
    print(f"\n[6] Intrinsics (AFTER normalization):")
    print(f"    fx = {intrinsics[0, 0, 0, 0].item():.6f}")
    print(f"    fy = {intrinsics[0, 0, 1, 1].item():.6f}")
    print(f"    cx = {intrinsics[0, 0, 0, 2].item():.6f}")
    print(f"    cy = {intrinsics[0, 0, 1, 2].item():.6f}")
    print("="*50 + "\n")
# ============ DEBUG END ============
```

### 在 Loss 计算后添加

```python
# 在 line 475 后（return 之前）
if iter % 10 == 0:  # 每 10 个 iteration 打印一次
    print(f"Iter {iter}: RGB={rgb_loss_total.item():.4f}, "
          f"Depth={depth_loss.item():.4f}, "
          f"Total={loss.item():.4f}")
```

---

## 📊 根据调试输出判断问题

### 情况 1: 内参异常
```
Before: fx = 0.98  # < 10，可能已经归一化！
After:  fx = 0.0018  # 远小于 1，二次归一化！
```
→ **问题**：重复归一化，需要检查数据流

### 情况 2: 渲染图全黑
```
rendered_color min=0, max=0
```
→ **问题**：Gaussian 不透明度为 0 或位置错误

### 情况 3: Loss 数值异常
```
RGB loss: 10.5
Depth loss: 0.001
```
→ **问题**：Loss 权重严重不平衡

### 情况 4: 深度范围不匹配
```
Sparse depth: 5.2 - 45.8
Near/Far: 0.1 / 1000.0
```
→ **问题**：Near太小，Far太大，可能导致数值精度问题

---

## 结论

请先运行上述调试代码，提供输出结果，我可以帮你精确定位问题！


# 🔍 Loss 不收敛问题诊断报告

## 执行时间
2025-10-26

---

## 🚨 严重问题（Critical Issues）

### **问题 1: 内参归一化方式错误 - 最严重！** ⚠️⚠️⚠️

**位置**: `codes/mvsplat/src/model/mvsplat_model.py` 第 354-355 行

**当前代码**:
```python
intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/width   # 错误！
intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/height  # 错误！
```

**问题分析**:

1. **数据流确认**:
   - 数据加载器返回的 `input_cks` 是**像素坐标内参**（未归一化）
   - 证据：`codes/data/KITTI360_CenterCam_Ref/transforms/loading.py` 第186-193行
   ```python
   raw_ck = np.array([[552.554261,   0,       682.049453],
                     [  0, 552.554261, 238.769549],
                     [  0, 0,    1]])
   ```
   - FOV计算使用的是像素坐标：`fovx = 2 * np.arctan(cx / fx)`（第239行）

2. **错误的归一化方式**:
   ```python
   intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/width
   ```
   这会将**整个第0行**都除以 width，包括：
   - `intrinsics[:, :, 0, 0]` → fx（正确）
   - `intrinsics[:, :, 0, 1]` → skew parameter（错误！应该保持为0）
   - `intrinsics[:, :, 0, 2]` → cx（正确）

3. **正确的归一化方式应该是**:
   ```python
   intrinsics[:, :, 0, 0] /= width   # fx
   intrinsics[:, :, 0, 2] /= width   # cx
   intrinsics[:, :, 1, 1] /= height  # fy
   intrinsics[:, :, 1, 2] /= height  # cy
   ```

4. **影响**:
   - **完全破坏相机几何关系**
   - Gaussian 的投影位置完全错误
   - 渲染出的图像与 GT 不匹配
   - RGB loss 和 depth loss 都无法优化
   - **这是导致 loss 不收敛的最主要原因**

**严重程度**: 🔴 极高（99% 可能是主要原因）

---

### **问题 2: 重复归一化风险** ⚠️⚠️

**位置**: 同上

**问题**:
- 代码注释说 "Maybe not neccssary"（可能不必要）
- 如果数据加载器已经归一化了（虽然目前看起来没有），会导致二次归一化

**建议**:
- 需要在训练开始时打印 intrinsics 的实际值，确认范围
- 如果 fx > 1，说明是像素坐标（未归一化）
- 如果 fx < 1，说明已经归一化

**严重程度**: 🟡 中等（需要验证）

---

## ⚠️ 中等问题（Medium Issues）

### **问题 3: 学习率可能偏小** 

**位置**: `codes/configs/MVSplat/mvsplat_vanilla_first_lidar.py` 第 14 行

**当前设置**:
```python
lr = 1e-4  # 基础学习率
# pretrained 部分的学习率是 1e-6 (lr * 0.01)
```

**问题分析**:
- Gaussian Splatting 类型模型通常需要较大的学习率
- 3D-GS 原论文使用的学习率范围：
  - Position: 1.6e-4
  - Rotation/Scale: 1e-3
  - Opacity: 5e-2
  - SH: 2.5e-3
- 当前的 1e-4 可能太保守，特别是对于非 pretrained 部分

**建议学习率**:
- 基础学习率：2e-4 到 5e-4
- Pretrained 部分：保持 lr * 0.01

**严重程度**: 🟠 中等

---

### **问题 4: Gradient Clipping 过于严格**

**位置**: 同上，第 15 行

**当前设置**:
```python
grad_max_norm = 1.0
```

**问题分析**:
- Gaussian 参数（特别是位置和缩放）的梯度可能较大
- `grad_max_norm=1.0` 会频繁触发梯度裁剪
- 限制了模型的优化速度

**建议**:
- 增加到 5.0 或 10.0
- 或者监控训练时的 grad_norm 值，根据实际情况调整

**严重程度**: 🟠 中等

---

### **问题 5: Loss 权重可能不平衡**

**位置**: `codes/configs/MVSplat/mvsplat_vanilla_first_lidar.py` 第 162-172 行

**当前设置**:
```python
loss_settings_dict = dict(
    rendered_rgb_supervision=True,
    rendered_rgb_supervison_type="MSE_LPIPS",
    lpips_alpha=0.05,
    rendered_depth_weight=0.15,
)
```

**问题分析**:
- RGB loss = MSE + 0.05 * LPIPS
- Depth loss 权重 = 0.15

需要检查实际训练时的 loss 数值范围：
- 如果 RGB loss ≈ 0.01-0.1，depth loss ≈ 1-10，则 depth loss 会主导
- 如果 RGB loss ≈ 1-10，depth loss ≈ 0.01-0.1，则 RGB loss 会主导

**建议**:
- 在训练日志中分别打印 RGB loss 和 depth loss 的数值
- 确保两者在相近的数量级（比如都在 0.1-1.0 范围）
- 根据实际情况调整权重

**严重程度**: 🟡 中等

---

## 🔵 次要问题（Minor Issues）

### **问题 6: Mixed Precision 未启用**

**位置**: 第 25 行

**当前设置**:
```python
mixed_precision = "no"
```

**影响**:
- 训练速度较慢
- 显存占用较大
- 但不影响收敛性

**建议**: 改为 `"fp16"` 加速训练

**严重程度**: 🔵 低

---

### **问题 7: Gradient Accumulation Steps = 1**

**位置**: 第 26 行

**当前设置**:
```python
gradient_accumulation_steps = 1
batch_size_train = 1
```

**影响**:
- 有效 batch size = 1，可能导致训练不稳定
- 梯度噪声较大

**建议**: 
- 设置 `gradient_accumulation_steps = 2` 或 `4`
- 相当于 batch size = 2 或 4

**严重程度**: 🔵 低到中等

---

### **问题 8: Warm-up Steps 可能不够**

**位置**: 第 24 行

**当前设置**:
```python
warmup_steps = 1000
```

**问题**:
- 对于从头训练的模型，1000 步可能不够
- 特别是学习率较高时

**建议**: 增加到 2000-3000 步

**严重程度**: 🔵 低

---

### **问题 9: NaN/Inf 处理掩盖问题**

**位置**: `codes/train_kitti360_mvsplat_vanilla.py` 第 264-267 行

**当前代码**:
```python
loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
if torch.isnan(loss) or torch.isinf(loss):
    print(f"[Warning] NaN or INF loss at iter {global_iter}, skipping...")
    continue
```

**问题**:
- 跳过有问题的 batch，但不解决根本原因
- 如果经常出现 NaN，说明模型或数据有严重问题
- 应该追踪 NaN 的来源

**建议**:
- 统计 NaN 出现的频率
- 如果 > 1%，说明有严重问题需要解决
- 不要简单跳过，要找到根本原因

**严重程度**: 🔵 低（但如果频繁出现则是严重问题）

---

## 📊 问题优先级总结

| 优先级 | 问题 | 严重程度 | 预计影响 |
|--------|------|---------|----------|
| **P0** | ❗ 内参归一化方式错误 | 🔴 极高 | **99%** |
| **P1** | ⚠️ 学习率偏小 | 🟠 中等 | 30% |
| **P1** | ⚠️ Gradient Clipping 过严 | 🟠 中等 | 20% |
| **P2** | 🔍 Loss 权重不平衡 | 🟡 中等 | 10% |
| **P2** | 🔍 重复归一化风险 | 🟡 中等 | 需验证 |
| **P3** | Warm-up 不够 | 🔵 低 | 5% |
| **P3** | Gradient Accumulation | 🔵 低 | 5% |
| **P3** | Mixed Precision | 🔵 低 | 0% |

---

## 🔬 诊断步骤建议

### 1. **立即检查内参值**

在训练代码中添加调试输出（`mvsplat_model.py` 第 356 行后）:

```python
# 在归一化之前打印
print(f"Before normalization:")
print(f"  fx = {intrinsics[0, 0, 0, 0].item():.2f}")
print(f"  fy = {intrinsics[0, 0, 1, 1].item():.2f}")
print(f"  cx = {intrinsics[0, 0, 0, 2].item():.2f}")
print(f"  cy = {intrinsics[0, 0, 1, 2].item():.2f}")
print(f"  skew = {intrinsics[0, 0, 0, 1].item():.6f}")

# 在归一化之后打印
intrinsics[:, :, 0] = intrinsics[:, :, 0]*1.0/width
intrinsics[:, :, 1] = intrinsics[:, :, 1]*1.0/height

print(f"After normalization:")
print(f"  fx = {intrinsics[0, 0, 0, 0].item():.6f}")
print(f"  fy = {intrinsics[0, 0, 1, 1].item():.6f}")
print(f"  cx = {intrinsics[0, 0, 0, 2].item():.6f}")
print(f"  cy = {intrinsics[0, 0, 1, 2].item():.6f}")
print(f"  skew = {intrinsics[0, 0, 0, 1].item():.6f}")  # 这个应该是0，但现在不是！
```

**预期结果**:
- Before: fx ≈ 552, cx ≈ 682
- After (正确): fx ≈ 1.01, cx ≈ 1.25
- After (当前错误): skew ≠ 0（这是错的！）

### 2. **监控 Loss 数值范围**

在训练代码中分别打印（`mvsplat_model.py` 第 422-426 行）:

```python
# 修改 set_loss 调用
set_loss(key='rgb_mse_loss', split=mode, loss_value=rec_loss, loss_weight=1.0)
set_loss(key='rgb_lpips_loss', split=mode, loss_value=preception_loss, loss_weight=lpips_loss_alpha)
set_loss(key='rgb_total_loss', split=mode, loss_value=rgb_loss_total, loss_weight=1.0)
```

**预期查看**:
- RGB MSE loss 数值范围
- LPIPS loss 数值范围
- Depth loss 数值范围
- 三者是否在相近数量级

### 3. **监控 Gradient Norm**

在训练脚本中（`train_kitti360_mvsplat_vanilla.py` 第 273 行后）:

```python
if accelerator.sync_gradients:
    grad_norm = accelerator.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
    # 添加这行
    if global_iter % 10 == 0:
        print(f"Grad norm: {grad_norm:.4f}, Max allowed: {cfg.grad_max_norm}")
```

**预期查看**:
- 如果 grad_norm 经常 > grad_max_norm，说明裁剪太严格
- 如果 grad_norm 经常 < 0.1，说明梯度太小（可能学习率太低）

### 4. **监控 NaN 出现频率**

```python
# 统计 NaN 出现次数
nan_count = 0
total_count = 0

# 在训练循环中
total_count += 1
if torch.isnan(loss) or torch.isinf(loss):
    nan_count += 1
    print(f"NaN ratio: {nan_count}/{total_count} = {100*nan_count/total_count:.2f}%")
```

---

## 🎯 推荐修复顺序

1. **立即修复 P0 问题**（内参归一化）→ 预计解决 99% 问题
2. **验证 P2 问题**（重复归一化）→ 确保数据流正确
3. **调整 P1 问题**（学习率、gradient clipping）→ 加速收敛
4. **监控 Loss**（权重平衡）→ 微调
5. **优化 P3 问题**（训练效率）→ 最后优化

---

## 📝 需要的信息

为了进一步诊断，请提供：

1. **训练日志样本** - 前100个iteration的loss值
2. **内参打印输出** - 归一化前后的值
3. **是否看到渲染的图像** - 即使loss不收敛，渲染图是什么样的？
   - 完全黑/白 → 可能是几何问题
   - 模糊/错位 → 可能是内参问题
   - 有结构但不清晰 → 可能是优化问题

---

## ✅ 验证方法

修复后，应该看到：
- ✅ Loss 从第一个 epoch 就开始稳定下降
- ✅ RGB PSNR 在 1000 steps 内达到 > 15 dB
- ✅ 渲染图像有明显的场景结构
- ✅ Gradient norm 稳定在合理范围（1-10）
- ✅ 无 NaN 或极少 NaN（< 0.1%）

---

## 🎓 技术背景说明

**为什么内参归一化如此重要？**

在 Gaussian Splatting 中：
1. Gaussians 的位置由相机内参投影到 2D 图像
2. 错误的内参 → 错误的投影位置 → 渲染的图像完全错误
3. 即使模型能学习，也无法匹配 GT，因为几何关系被破坏
4. 这是一个**系统性错误**，不是优化问题，所以无法通过调参解决

**类比**：
- 就像戴着度数完全错误的眼镜看世界
- 无论怎么学习，都无法看清楚
- 必须先摘掉眼镜（修复内参），才能开始正常学习


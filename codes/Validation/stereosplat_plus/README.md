# `stereosplat_plus` 说明

这个文件夹主要用于做 `StereoSplat Plus` 相关的验证实验。核心目标是比较三类结果：

- `G_base`：只使用基础渲染结果
- `G_plus`：加入增强/补全后的结果
- `G_fusion`：尝试将 `base` 和 `plus` 融合后的结果

这些脚本大多是“同一套验证框架 + 不同融合策略”的对比实验。也就是说，它们的主体流程基本类似，差别主要在于 `VolumeFusionRevision` 里调用的融合函数不同。

## 文件说明

### `stereosplat_baseline_preserving_fusion_oracle.py`

实验意图：

- 作为一个**上界参考**。
- 使用 GT 误差来决定每个像素应该选 `base` 还是 `plus`。
- 这个方法在真实推理时不可用，因为它依赖 GT，但可以用来回答一个问题：
  如果像素级选择器足够理想，`base` 和 `plus` 的理论融合上限能到哪里。

适合用途：

- 作为所有可实现融合策略的 ceiling / upper bound。
- 判断“融合这件事本身是否值得做”。

### `stereosplat_baseline_perserving_fusion_simple_alpha.py`

实验意图：

- 用一个**最简单、最直接**的策略测试 `alpha` 是否已经足够作为融合依据。
- 通常是将 `base` 的 alpha 当作连续权重，做 soft fusion。
- 重点是验证：不引入复杂几何或纹理判断时，仅靠渲染 alpha 是否就能带来收益。

适合用途：

- 作为轻量级 baseline。
- 和 oracle / depth-consistency / texture-based 方法做对照。

### `stereosplat_baseline_perserving_fusion_hard_alpha_fusion.py`

实验意图：

- 测试一个比 simple alpha 更“硬”的版本。
- 不做连续加权，而是直接比较 `alpha_base` 和 `alpha_plus`，例如：
  `alpha_plus > alpha_base` 时选 `plus`，否则保留 `base`。
- 这个实验关注的是：
  alpha 更适合作为 soft weight，还是更适合作为 hard selector。

适合用途：

- 分析 alpha 的判别性到底够不够强。
- 对比 soft alpha 与 hard alpha 的行为差异。

### `stereosplat_baseline_perserving_fusion_depth_consistency.py`

实验意图：

- 引入**几何一致性**来辅助融合。
- 典型思路是：利用 first stereo 的深度和相机位姿，把 depth 投影到其他视角，再与 `plus` 或 `base` 的深度进行比较。
- 想回答的问题是：
  在遮挡区域、出视野区域、深度不一致区域，是否应该更倾向于 `plus`。

适合用途：

- 做比 alpha 更有几何依据的融合。
- 重点观察遮挡边界、外扩区域、新出现区域的质量变化。

### `stereosplat_baseline_preserving_fusion_canny.py`

实验意图：

- 用**边缘 / 高频 / texture-rich 区域**来决定是否偏向 `plus`。
- 一般会假设：
  `plus` 在边界、纹理、细节恢复上更强；
  `base` 在平坦区域、稳定区域更可信。
- 因此这个脚本本质上是在验证：
  是否可以把融合重点放在 edges / textures 上，而不是全图统一处理。

适合用途：

- 观察细节锐度、边界清晰度是否改善。
- 和深度一致性方法对比“几何先验 vs 图像细节先验”。

### `utils.py`

实验意图：

- 放一些调试可视化和辅助函数。
- 当前主要用于保存 debug 结果，例如：
  projected depth、valid mask、relative error、consistency mask 等。
- 它本身不是一个主实验入口，而是给上面这些融合实验提供辅助分析能力。

适合用途：

- 快速排查融合失败的原因。
- 观察某个 mask / projected depth / consistency 规则是否符合预期。

## 这些脚本之间的关系

可以把它们理解成一组逐步增强的融合假设：

1. `oracle`
   先看理论最优情况能到哪里。
2. `simple_alpha`
   看最简单的 alpha soft fusion 是否有效。
3. `hard_alpha_fusion`
   看 alpha 做 hard selection 是否更合适。
4. `depth_consistency`
   引入几何一致性，处理遮挡 / 外扩 / 新区域。
5. `fusion_canny`
   引入边缘与高频先验，强调细节区域的 `plus` 优势。

## 目录命名说明

目录里有些文件名写成了 `perserving`，有些写成了 `preserving`。  
这里更像是历史命名遗留，不影响当前实验逻辑；如果后续要整理目录，建议统一成 `preserving`。


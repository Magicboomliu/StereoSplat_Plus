## Docker（容器内用 Pixi 装环境）

Base image：`pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`

### Build

在仓库根目录执行：

```bash
docker build -t stereosplat-plus:cu118 -f docker/Dockerfile .
```

### Run（GPU）

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace/StereoSplat_Plus \
  stereosplat-plus:cu118
```

### 在容器里进入 Pixi 环境

```bash
pixi shell --manifest-path stereosplat/pyproject.toml --environment cu118
```

或不进入 shell，直接跑命令：

```bash
pixi run --manifest-path stereosplat/pyproject.toml --environment cu118 \
  python -c "import torch; print(torch.__version__, torch.version.cuda)"
```


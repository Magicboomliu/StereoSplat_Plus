## Docker (Pixi environment inside the container)

Base image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`

For training and evaluation, see **[stereosplat_conf/README.md](../stereosplat_conf/README.md)** and **[stereosplat_conf/eval/README.md](../stereosplat_conf/eval/README.md)**.

### Build

From the repository root:

```bash
docker build -t stereosplat-plus:cu118 -f docker/Dockerfile .
```

### Run (GPU)

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace/StereoSplat_Plus \
  stereosplat-plus:cu118
```

### Enter the Pixi environment in the container

```bash
pixi shell --manifest-path stereosplat_conf/pyproject.toml --environment cu118
```

Or run a command without entering the shell:

```bash
pixi run --manifest-path stereosplat_conf/pyproject.toml --environment cu118 \
  python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

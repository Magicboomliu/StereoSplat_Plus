# StereoSplat (Pixi)

## Requirements

- Linux (`linux-64`)
- `git`

## Install Pixi

Install Pixi (official installer), then verify it is on your `PATH`.

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi --version
```

## Build the environment

Create/sync the Pixi environment for this project, then run the one-shot setup (installs PyTorch/CUDA deps, builds rasterizer, and installs Python requirements).

```bash
cd stereosplat
pixi install
pixi run setup
```
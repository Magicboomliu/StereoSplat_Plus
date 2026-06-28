FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    bash \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pixi (static binary) into /usr/local/bin
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    /root/.pixi/bin/pixi --version && \
    ln -s /root/.pixi/bin/pixi /usr/local/bin/pixi

WORKDIR /workspace/StereoSplat_Plus

# Copy repo
COPY . .

# IMPORTANT: run pixi tasks from the stereosplat_conf project directory,
# because tasks like `pip install -e .` are relative to CWD.
WORKDIR /workspace/StereoSplat_Plus/stereosplat_conf

# Install pixi environment (cu118) and run project setup tasks.
# `--locked` uses pixi.lock for reproducible builds (if present).
RUN pixi install --locked --environment cu118 && \
    pixi run --environment cu118 setup

# Back to repo root for interactive work
WORKDIR /workspace/StereoSplat_Plus

# Default: interactive shell (enter pixi via `pixi shell ...`)
CMD ["bash"]


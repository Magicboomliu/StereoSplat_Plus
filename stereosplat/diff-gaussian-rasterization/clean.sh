#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning build artifacts and pixi environment..."

# compiled extension
rm -f diff_gaussian_rasterization/_C*.so

# Python caches
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null || true

# setuptools / pip build artifacts
rm -rf build/ dist/ *.egg-info diff_gaussian_rasterization/*.egg-info

# pytest cache
rm -rf .pytest_cache

# pixi environment
rm -rf .pixi

echo "Done."

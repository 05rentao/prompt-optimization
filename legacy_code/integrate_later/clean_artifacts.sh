#!/usr/bin/env bash
# deletes all the python cache, build artifacts, and other common tool caches


set -euo pipefail

# Run from repo root:
#   bash scripts/clean_artifacts.sh

echo "Cleaning Python cache and build artifacts..."

# Python bytecode / caches
find . -name "__pycache__" -type d -prune -print -exec rm -rf {} +
find . \( -name "*.pyc" -o -name "*.pyo" \) -print -delete

# Build artifacts
rm -rf build/ dist/ wheels/ *.egg-info

# Common tool caches (pytest, mypy, notebooks)
rm -rf .pytest_cache .mypy_cache
find . -name ".ipynb_checkpoints" -type d -prune -print -exec rm -rf {} +

# Optional: uncomment if you want to clear logs/temp files
# find . -name "*.log" -print -delete
# find . \( -name "*.tmp" -o -name "*.temp" \) -print -delete

echo "Done."


#!/usr/bin/env bash

# Exit on error, treat unset variables as an error, and catch pipeline failures
set -euo pipefail

# ==========================================
# CONFIGURATION
# ==========================================
DATA_EXPORT="data/export.json"
DATA_TEAM="data/teaminfo.json"
OUTPUT_DIR="outputs"

VERSION="v321"
RUNS=1000
YEAR=2013
SEED=69

# Get the directory where this script actually lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==========================================
# BUILD STEP (Out-of-source & Safe)
# ==========================================
cmake -B build -S .

# Build using all available CPU cores for speed
cmake --build build --parallel "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"

cp build/progbox .

# ==========================================
# EXECUTION STEP
# ==========================================
# Ensure the output directory exists before running the binary
mkdir -p "$OUTPUT_DIR"

./progbox "$DATA_EXPORT" "$DATA_TEAM" "$OUTPUT_DIR" \
  -v "$VERSION" \
  -r "$RUNS" \
  -y "$YEAR" \
  -s "$SEED"
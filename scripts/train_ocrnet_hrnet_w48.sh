#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.train configs/ocrnet_hrnet_w48.py "$@"

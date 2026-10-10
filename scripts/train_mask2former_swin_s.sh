#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.train configs/mask2former_swin_s.py "$@"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.train configs/pspnet_r101.py "$@"

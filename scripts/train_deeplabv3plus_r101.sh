#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.train configs/deeplabv3plus_r101.py "$@"

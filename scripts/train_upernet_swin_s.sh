#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.train configs/upernet_swin_s.py "$@"

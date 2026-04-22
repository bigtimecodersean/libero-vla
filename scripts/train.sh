#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Usage: ./scripts/train.sh [extra args forwarded to train.py]
#   ./scripts/train.sh --wandb --max-steps 15000
python -m src.training.train --config config.yaml "$@"

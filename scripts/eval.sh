#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.training.evaluate --config config.yaml "$@"

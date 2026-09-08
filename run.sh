#!/usr/bin/env bash
# run.sh — Entry point for the token_wrapper benchmark submission
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash run.sh                              # wrapped run only
#   bash run.sh --baseline                  # wrapped + baseline comparison
#   bash run.sh --model claude-3-5-haiku-20241022  # change model
#   bash run.sh --benchmark benchmark_hidden.json  # run hidden benchmark
#
# All arguments are forwarded to benchmark_runner.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check for API key ──────────────────────────────────────────────────────
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "Export it with: export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

# ── Install dependencies (idempotent) ──────────────────────────────────────
echo "[setup] Installing dependencies..."
pip install -q -r requirements.txt

# ── Run benchmark ──────────────────────────────────────────────────────────
echo "[run] Starting benchmark runner..."
python benchmark_runner.py \
    --benchmark benchmark_sample.json \
    --output results.json \
    "$@"

echo ""
echo "[done] Results written to results.json"

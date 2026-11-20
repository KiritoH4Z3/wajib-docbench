#!/usr/bin/env bash
# AIPerf load test against a running OpenAI-compatible server, at concurrency 1, 4 and 16.
#
#   perf/run_aiperf.sh <model_slug> <host:port> <served_model_alias>
#
# One-time setup (Python 3.12, separate from the project venv):
#   uv venv --python 3.12 .venv-aiperf
#   uv pip install --python .venv-aiperf/bin/python aiperf==0.13.0
#
# Flags were checked against the aiperf 0.13.0 source (aiperf/config/flags/cli_config.py).
# --concurrency, --request-count and --warmup-request-count are per run, so the loop below
# makes one run per level. --use-server-token-count adds stream_options.include_usage.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <model_slug> <host:port> <served_model_alias>" >&2
  exit 2
fi
SLUG="$1"; URL="$2"; ALIAS="$3"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AIPERF="$ROOT/.venv-aiperf/bin/aiperf"
if [ ! -x "$AIPERF" ]; then
  echo "AIPerf is not installed. See the setup lines at the top of $0." >&2
  exit 1
fi

IN="$ROOT/perf/aiperf/$SLUG"
OUT="$ROOT/results/perf/$SLUG/aiperf-artifacts"
(cd "$ROOT" && uv run python -m perf.aiperf build "$SLUG")

for N in 1 4 16; do
  "$AIPERF" profile \
    --model "$ALIAS" \
    --endpoint-type chat \
    --streaming \
    --url "$URL" \
    --input-file "$IN/prompts.jsonl" \
    --custom-dataset-type single_turn \
    --system-prompt-file "$IN/system_prompt.txt" \
    --concurrency "$N" \
    --request-count 120 \
    --warmup-request-count 8 \
    --random-seed 42 \
    --extra-inputs temperature:0 max_tokens:1024 \
    --tokenizer builtin \
    --use-server-token-count \
    --artifact-dir "$OUT/c$N" \
    --ui simple
done

(cd "$ROOT" && uv run python -m perf.aiperf parse "$SLUG")

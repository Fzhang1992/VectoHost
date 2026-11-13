#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# src layout, so all `python -m vh3.*` invocations resolve.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec bash scripts/run_all.sh "${@:-}"

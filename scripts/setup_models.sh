#!/usr/bin/env bash
# Install official LucaVirusTasks inference source and dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mode="${1:---install}"
[[ "$mode" == --install || "$mode" == --fetch-checkpoints ]] || {
  echo 'Usage: bash scripts/setup_models.sh [--install|--fetch-checkpoints]' >&2; exit 2; }
[[ -f config.env ]] || { echo 'First copy config.example.env to config.env' >&2; exit 2; }
# shellcheck disable=SC1091
source config.env
repo="${LUCAVIRUS_TASKS:-external/LucaVirusTasks}"
if [[ ! -d "$repo/.git" ]]; then
  if [[ -e "$repo" ]]; then echo "Existing non-git directory at $repo; refusing to replace it" >&2; exit 2; fi
  mkdir -p "$(dirname "$repo")"
  git clone https://github.com/LucaOne/LucaVirusTasks.git "$repo"
fi
if [[ -n "${LUCAVIRUS_COMMIT:-}" ]]; then git -C "$repo" checkout --detach "$LUCAVIRUS_COMMIT"; fi
rev="$(git -C "$repo" rev-parse HEAD)"
echo "[UPSTREAM] LucaVirusTasks commit $rev"
if [[ "$mode" == --fetch-checkpoints ]]; then
  script="$repo/src/fetch_checkpoints.py"
  [[ -f "$script" ]] || { echo "Upstream fetcher missing: $script" >&2; exit 2; }
  echo '[UPSTREAM] Fetching official checkpoints; this may download large files.'
  (cd "$repo/src" && python fetch_checkpoints.py)
  exit 0
fi
# Install upstream dependencies first; reinstall locally constrained research packages last.
python -m pip install -r "$repo/requirements.txt"
python -m pip install -r requirements.txt
python - <<'PY'
import esm, torch, numpy, pandas, sklearn, imblearn, joblib
print('[OK] Dependencies import:', 'torch',torch.__version__,'sklearn',sklearn.__version__)
PY
printf '%s\n' "$rev" > "$repo/.vh3_checked_commit"
echo '[DONE] Upstream source and dependencies ready; model weights are retrieved separately or by upstream inference.'

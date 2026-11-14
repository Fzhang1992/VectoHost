#!/usr/bin/env bash
# Full reproducible research workflow: new FASTA -> NEW features -> 96 CV -> evaluation -> predictors.
# All completed outputs are checked before reuse. Component training is per feature/model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:---run}"
if [[ "$MODE" != --check && "$MODE" != --run ]]; then
  echo 'Usage: bash run_all.sh --check|--run' >&2; exit 2
fi
[[ -f config.env ]] || { echo 'Missing config.env. First: cp config.example.env config.env' >&2; exit 2; }
# shellcheck disable=SC1091
source config.env
for key in INSECT_FASTA PLANT_FASTA OTHER_FASTA; do
  val="${!key:-}"
  [[ -n "$val" && -s "$val" ]] || { echo "Missing/empty $key: $val" >&2; exit 2; }
done
if [[ "${SLICING_MODE:-regenerate}" == regenerate && -z "${SLICE_SEED:-}" ]]; then
  echo 'Set SLICE_SEED to a documented non-placeholder value. Original seed is necessary for exact historical slicing.' >&2; exit 2
fi
if [[ "${SLICING_MODE:-regenerate}" == historical && ! -d "${HISTORICAL_FRAGMENTS:-}" ]]; then
  echo 'historical mode requires HISTORICAL_FRAGMENTS' >&2; exit 2
fi
LUCA="${LUCAVIRUS_TASKS:-external/LucaVirusTasks}"
[[ -f "$LUCA/src/llm/lucavirus/get_embedding.py" ]] || { echo "Missing upstream LucaVirusTasks at $LUCA. Run bash scripts/setup_models.sh" >&2; exit 2; }
if [[ -n "${LUCAVIRUS_COMMIT:-}" ]]; then
  actual=$(git -C "$LUCA" rev-parse HEAD)
  [[ "$actual" == "$LUCAVIRUS_COMMIT" ]] || { echo "Pinned upstream commit differs: $actual != $LUCAVIRUS_COMMIT" >&2; exit 2; }
fi
if [[ "${STRICT_REFERENCE:-0}" == 1 ]]; then
  [[ -n "${LUCAVIRUS_COMMIT:-}" && -n "${LUCAVIRUS_CHECKPOINT_PATH:-}" && -n "${LUCAVIRUS_CHECKPOINT_SHA256:-}" ]] || {
    echo 'STRICT_REFERENCE requires pinned original upstream commit plus checkpoint path and SHA256.' >&2; exit 2; }
fi
if [[ -n "${LUCAVIRUS_CHECKPOINT_PATH:-}" ]]; then
  [[ -s "$LUCAVIRUS_CHECKPOINT_PATH" ]] || { echo 'Specified LucaVirus checkpoint does not exist' >&2; exit 2; }
  checksum=$(sha256sum "$LUCAVIRUS_CHECKPOINT_PATH" | cut -d' ' -f1)
  if [[ -n "${LUCAVIRUS_CHECKPOINT_SHA256:-}" && "$checksum" != "$LUCAVIRUS_CHECKPOINT_SHA256" ]]; then
    echo 'Specified LucaVirus checkpoint SHA256 mismatch' >&2; exit 2
  fi
  echo "[MODEL] LucaVirus checkpoint SHA256: $checksum"
fi
python -c 'import numpy,pandas,sklearn,imblearn,torch,esm,joblib; print("[CHECK] Python dependencies import")'
if [[ "$MODE" == --check ]]; then
  python -m vh3.prepare --help >/dev/null
  python -m vh3.embedding_esm2 --help >/dev/null
  python -m vh3.embedding_lucavirus --help >/dev/null
  python -m vh3.merge_parts --help >/dev/null
  python -m pytest -q tests
  echo '[CHECK PASS] Code/dependencies present. This is not a real inference or checkpoint validation.'
  exit 0
fi
mkdir -p work logs
python -m vh3.run_identity --config config.env --insect "$INSECT_FASTA" --plant "$PLANT_FASTA" \
  --other "$OTHER_FASTA" --luca "$LUCA" --output work/run_identity.json
# Input provenance, written only on initial run and locked by run_identity.json.
if [[ ! -f work/source_fasta_sha256.tsv ]]; then
  {
    printf 'class\tsha256\n'
    for cls in Insect Plant Other; do
      case "$cls" in Insect) path="$INSECT_FASTA";; Plant) path="$PLANT_FASTA";; Other) path="$OTHER_FASTA";; esac
      printf '%s\t%s\n' "$cls" "$(sha256sum "$path" | cut -d' ' -f1)"
    done
  } > work/source_fasta_sha256.tsv
fi
CHECK=(python -m vh3.check --sequences work/sequences)
if [[ "${EXPECT_PUBLISHED:-1}" == 1 ]]; then CHECK+=(--expect-published); fi
if [[ -f work/sequences/sequence_manifest.tsv ]]; then
  echo '[RESUME] Validate and reuse sequence dataset'
  "${CHECK[@]}"
else
  if [[ -e work/sequences ]]; then
    recovery="work/recovery/sequences_interrupted_$(date +%Y%m%d_%H%M%S)"
    mkdir -p work/recovery
    mv work/sequences "$recovery"
    echo "[RECOVERY] Moved incomplete sequence outputs to $recovery"
  fi
  PREP=(python -u -m vh3.prepare --insect "$INSECT_FASTA" --plant "$PLANT_FASTA" \
        --other "$OTHER_FASTA" --output work/sequences --mode "${SLICING_MODE:-regenerate}")
  if [[ "${SLICING_MODE:-regenerate}" == regenerate ]]; then
    PREP+=(--slice-seed "$SLICE_SEED")
  else
    PREP+=(--historical-dir "$HISTORICAL_FRAGMENTS")
  fi
  if [[ "${EXPECT_PUBLISHED:-1}" == 1 ]]; then PREP+=(--expect-published); fi
  echo '[STEP] Build four length-specific sequence datasets'; "${PREP[@]}"
  "${CHECK[@]}"
fi
features=(1mer 2mer 3mer 4mer ESM2 LucaVirus)
models=(RandomForest KNN SVM GaussianNB)
for length in 1000bp 3000bp 5000bp full_length; do
  echo "========== $length =========="
  for k in 1 2 3 4; do
    python -u -m vh3.features --sequences work/sequences --features-dir work/features --length "$length" --k "$k"
  done
  for cls in Insect Plant Other; do
    python -u -m vh3.embedding_esm2 --sequences work/sequences --features-dir work/features \
      --length "$length" --class-name "$cls" --batch-size "${ESM2_BATCH_SIZE:-16}" --device "${ESM2_DEVICE:-auto}"
    python -u -m vh3.embedding_lucavirus --sequences work/sequences --features-dir work/features \
      --length "$length" --class-name "$cls" --lucavirus-tasks "$LUCA" \
      --gpu-id "${LUCAVIRUS_GPU_ID:--1}" --chunk-size "${LUCAVIRUS_CHUNK_SIZE:-1000}"
  done
  for feature in "${features[@]}"; do
    for model in "${models[@]}"; do
      out="work/results_parts/$length/${feature}_${model}"
      if [[ -d "$out" ]]; then
        python -m vh3.merge_parts check-one --directory "$out" --length "$length" \
          --feature "$feature" --model "$model" --features-dir work/features >/dev/null
        echo "[RESUME] Verified $length $feature+$model"
      else
        cmd=(python -u -m vh3.train --features-dir work/features --length "$length" --output "$out" \
              --features "$feature" --models "$model" --mode legacy)
        if [[ "${EXPECT_PUBLISHED:-1}" == 1 ]]; then cmd+=(--expect-published); fi
        "${cmd[@]}"
        python -m vh3.merge_parts check-one --directory "$out" --length "$length" \
          --feature "$feature" --model "$model" --features-dir work/features >/dev/null
      fi
    done
  done
  merged="work/results/$length"
  if [[ -d "$merged" ]]; then
    python -m vh3.merge_parts check-merged --directory "$merged" --length "$length"
    echo "[RESUME] Verified $length merged OOF"
  else
    python -m vh3.merge_parts merge --parts-root work/results_parts --length "$length" \
      --output "$merged" --features-dir work/features
  fi
done
if [[ -d work/final ]]; then
  python - <<'PY'
from pathlib import Path
import pandas as pd
p=Path('work/final/feature_model_summary.tsv')
if not p.is_file():raise SystemExit('Incomplete previous work/final: inspect manually')
df=pd.read_csv(p,sep='\t')
if len(df)!=96 or len(df[['length','Feature','Model']].drop_duplicates())!=96:
    raise SystemExit('Previous work/final does not contain exactly 96 unique combinations')
print('[RESUME] 96 summary combinations verified')
PY
else
  args=(python -m vh3.evaluate --results-dir work/results --output work/final)
  if [[ "${EXPECT_PUBLISHED:-1}" == 1 ]]; then args+=(--expect-published); fi
  "${args[@]}"
fi
if [[ -f work/final/Figure5_reference_comparison.tsv ]]; then
  echo '[RESUME] Previous Figure 5 comparison found.'
  if [[ "${STRICT_REFERENCE:-0}" == 1 ]]; then
    python - <<'PY'
import pandas as pd
p=pd.read_csv('work/final/Figure5_reference_comparison.tsv',sep='\t')
if len(p)!=4 or set(p.status)!={'MATCH'}:raise SystemExit('Strict Figure 5 reference comparison not fully matching')
PY
  fi
else
  ref=(python -m vh3.verify_reference --final-dir work/final)
  if [[ "${STRICT_REFERENCE:-0}" == 1 ]]; then ref+=(--strict); fi
  "${ref[@]}"
fi
if [[ "${FIT_DEPLOY_MODELS:-1}" == 1 ]]; then
  for length in 1000bp 3000bp 5000bp full_length; do
    out="work/models/$length/LucaVirus_SVM"
    if [[ -d "$out" ]]; then
      python - <<PY
import json
from pathlib import Path
from vh3.common import sha256sum
p=Path('$out')
manifest=json.loads((p/'model_manifest.json').read_text())
assert manifest['length']=='$length' and manifest['feature']=='LucaVirus' and manifest['algorithm']=='SVM'
assert sha256sum(p/'model.joblib')==manifest['model_sha256']
print('[RESUME] Verified saved model:',p)
PY
    else
      fit=(python -u -m vh3.final_model --features-dir work/features --length "$length" \
        --feature LucaVirus --algorithm SVM --output "$out" --window-mode "${SLICING_MODE:-regenerate}")
      if [[ "${SLICING_MODE:-regenerate}" == regenerate ]]; then fit+=(--slice-seed "$SLICE_SEED"); fi
      "${fit[@]}"
    fi
  done
fi
echo '[COMPLETE] All 96 pooled OOF combinations evaluated; saved deployable models under work/models.'

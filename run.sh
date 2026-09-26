#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CMD="${1:-help}"

case "$CMD" in
  download)
    python scripts/download_model.py --output models/sd35-medium
    ;;
  infer)
    INPUT="${2:-data/private}"
    OUTPUT="${3:-outputs/private}"
    python scripts/run_inference.py --config configs/default.yaml --input "$INPUT" --output "$OUTPUT" --mode coarse_fine
    ;;
  exhaustive)
    INPUT="${2:-data/Task10_Colon/imagesTr}"
    OUTPUT="${3:-outputs/msd}"
    python scripts/run_inference.py --config configs/default.yaml --input "$INPUT" --output "$OUTPUT" --mode exhaustive
    ;;
  prepare-msd)
    python scripts/prepare_msd.py --root data/MSD
    ;;
  msd)
    python scripts/prepare_msd.py --root data/MSD
    INPUT="$(find data/MSD -type d -name imagesTr -print -quit)"
    test -n "$INPUT"
    python scripts/run_inference.py --dataset msd --config configs/default.yaml --input "$INPUT" --output outputs/msd --mode exhaustive
    ;;
  msi)
    python scripts/run_inference.py --dataset msi --config configs/default.yaml --input data/MSI --output outputs/msi --mode coarse_fine
    ;;
  eval-msd)
    LABELS="$(find data/MSD -type d -name labelsTr -print -quit)"
    test -n "$LABELS"
    python scripts/evaluate_msd.py --dataset msd --pred-dir outputs/msd --gt-dir "$LABELS" --output outputs/msd_metrics.json
    ;;
  eval-msi)
    python scripts/evaluate_msd.py --dataset msi --pred-dir outputs/msi --gt-dir data/MSI --output outputs/msi_metrics.json
    ;;
  eval)
    PRED="${2:-outputs/msd}"
    GT="${3:-data/Task10_Colon/labelsTr}"
    OUT="${4:-outputs/msd_metrics.json}"
    python scripts/evaluate_msd.py --pred-dir "$PRED" --gt-dir "$GT" --output "$OUT"
    ;;
  *)
    cat <<'HELP'
Usage:
  bash run.sh download
  bash run.sh infer [input_dir] [output_dir]
  bash run.sh exhaustive [imagesTr_dir] [output_dir]
  bash run.sh eval [pred_dir] [labelsTr_dir] [metrics_json]
  bash run.sh prepare-msd
  bash run.sh msd
  bash run.sh msi
  bash run.sh eval-msd
  bash run.sh eval-msi
HELP
    ;;
esac

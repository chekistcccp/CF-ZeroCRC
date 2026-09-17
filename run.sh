#!/usr/bin/env bash
set -euo pipefail

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
HELP
    ;;
esac

#!/usr/bin/env bash
# Layer ablation: runs the distances stage across layers 6,8,10,12,16 for all
# four categories. Timestamps are reused from the original category directories.
# Results written to results/layer_ablation/layer{N}/{category_suffix}/distances/
# Output → /tmp/layer_ablation.log
# Run with: nohup bash scripts/run_layer_ablation.sh &
set -e
cd "$(dirname "$0")/.."

LOG=/tmp/layer_ablation.log
exec > >(tee -a "$LOG") 2>&1

LAYERS=(6 8 10 12 16)

cat_suffix() {
    case "$1" in
        personal-introduction)  echo "phonological_distance" ;;
        personal-commitment)    echo "phonological_distance_commitment" ;;
        financial-product)      echo "phonological_distance_financial" ;;
        client-disagreement)    echo "phonological_distance_disagreement" ;;
    esac
}

echo "=== $(date) === Starting layer ablation ==="

for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "════ Layer $LAYER ════"
    for CAT in personal-introduction personal-commitment financial-product client-disagreement; do
        SUFFIX="$(cat_suffix "$CAT")"
        DIST_OUT="results/layer_ablation/layer${LAYER}/${SUFFIX}/distances"
        echo "  [$CAT] $(date)"
        python3.10 scripts/phonological_distance_pipeline.py \
            --stage distances \
            --category "$CAT" \
            --layer "$LAYER" \
            --dist-out "$DIST_OUT"
        echo "  [done]  $(date)"
    done
done

echo ""
echo "=== $(date) === ALL DONE ==="

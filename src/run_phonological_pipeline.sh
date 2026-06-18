#!/usr/bin/env bash
# XLS-R phonological-distance pipeline over all four scripted hiring categories.
# Set SOURCE below, then run:  bash src/run_phonological_pipeline.sh
#
#   SOURCE=human      (default) — audio pulled from the HuggingFace corpus
#   SOURCE=synthetic            — audio from the ElevenLabs set (SYNTH_PATH)
#
# Each category runs extract -> transcribe -> distances. Resumable.
set -e
cd "$(dirname "$0")/.."

SOURCE="${SOURCE:-human}"
SYNTH_PATH="${SYNTH_PATH:-audio_samples/elevenlabs}"
SCRIPT_TYPE="${SCRIPT_TYPE:-improved}"   # synthetic only: improved | disfluent
PYTHON="${PYTHON:-python3.10}"

EXTRA=""
if [ "$SOURCE" = "synthetic" ]; then
    EXTRA="--source synthetic --synthetic-path $SYNTH_PATH --script-type $SCRIPT_TYPE"
fi

echo "=== $(date) === phonological pipeline (source=$SOURCE) ==="
for CAT in personal-introduction personal-commitment financial-product client-disagreement; do
    echo ""; echo "──── $CAT ────"
    for STAGE in extract transcribe distances; do
        echo "  [$STAGE] $(date)"
        "$PYTHON" src/phonological_distance_pipeline.py --stage "$STAGE" --category "$CAT" $EXTRA
    done
done
echo ""; echo "=== $(date) === ALL DONE ==="

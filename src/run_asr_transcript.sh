#!/bin/bash
#SBATCH --job-name=asr_transcript
#SBATCH --partition=gpu-a40
#SBATCH --account=argon
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --gpus=2
#SBATCH --output="slurm/asr_transcript-%J.out"

# -------------------------------------------------------
# Full-transcript ASR (WER) on the scripted human hiring corpus.
# Set MODEL below, then run. One file per model; resumable.
#
#   API models (run locally):  bash src/run_asr_transcript.sh
#   Local models (GPU):        sbatch src/run_asr_transcript.sh
#
# MODEL options:
#   API:   gemini-2.5-flash, gemini-2.5-pro, gpt-audio-1.5
#   Local: qwen
# -------------------------------------------------------

cd "$(dirname "$0")/.."

MODEL="${MODEL:-gemini-2.5-flash}"
OUT="results/asr_transcript/${MODEL}_asr_transcript.csv"
# Local models don't need rate limiting; API models do.
case "$MODEL" in qwen|voxtral|flamingo) RATE=0.0 ;; *) RATE=1.0 ;; esac
PYTHON="${PYTHON:-python3.10}"

mkdir -p slurm results/asr_transcript

"$PYTHON" src/run_asr_transcript.py \
    --model "$MODEL" \
    --output_path "$OUT" \
    --rate_limit "$RATE"

echo ""
echo "=== Quick summary ($MODEL) ==="
"$PYTHON" - "$OUT" <<'PYEOF'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
df = df[df["wer"] != "FAILED"].copy()
df["wer"] = df["wer"].astype(float)
print("Median WER by accent:")
print(df.groupby("accent")["wer"].median().round(3).sort_values().to_string())
print(f"\nOverall: n={len(df)}  median WER={df['wer'].median():.3f}")
PYEOF

#!/bin/bash
#SBATCH --job-name=speecheval
#SBATCH --partition=gpu-a40
#SBATCH --account=argon
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=360G
#SBATCH --time=50:00:00
#SBATCH --gpus=4
#SBATCH --output="slurm/slurm-%J.out"

cat $0
echo "--------------------"

# -------------------------------------------------------
# Usage: sbatch scripts/run_evaluation.sh
#
# Set MODEL and EVAL_TYPE below, then submit.
# Resubmitting with the same OUTPUT_PATH will skip already-completed
# rows and only run new or previously FAILED ones.
#
# EVAL_TYPE options:
#   synthetic  — ElevenLabs file-based dataset
#   corpus     — HuggingFace human speaker dataset
#
# MODEL options:
#   API-based (set api key below): gemini-2.5-flash, gemini-2.5-pro, gpt-4o-audio-preview
#   Local (GPU required):          qwen, voxtral, flamingo
# -------------------------------------------------------

MODEL="${MODEL:-gpt-4o-audio-preview}"
EVAL_TYPE="${EVAL_TYPE:-corpus}"
if [ -z "${OUTPUT_PATH:-}" ]; then
    if [ "$EVAL_TYPE" = "corpus" ]; then
        OUTPUT_PATH="results/hiring_corpus/${MODEL}_hiring_corpus.csv"
    else
        OUTPUT_PATH="results/hiring_synthetic/${MODEL}_hiring_synthetic.csv"
    fi
fi

API_KEY="${API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"
PYTHON="${PYTHON:-python3.10}"
PROMPT_FILES="${PROMPT_FILES:-critical=audio_samples/elevenlabs_dataset4/prompts_two_part_rating_critical.csv ideal=audio_samples/elevenlabs_dataset4/prompts_two_part_rating_ideal.csv native=audio_samples/elevenlabs_dataset4/prompts_two_part_rating_native.csv}"

export HF_TOKEN

if [ "$EVAL_TYPE" = "corpus" ]; then
    "$PYTHON" scripts/run_evaluation.py \
        --eval_type corpus \
        --model "$MODEL" \
        --output_path "$OUTPUT_PATH" \
        --prompt_files \
            critical=audio_samples/elevenlabs_dataset4/prompts_two_part_rating_critical.csv \
            ideal=audio_samples/elevenlabs_dataset4/prompts_two_part_rating_ideal.csv \
            native=audio_samples/elevenlabs_dataset4/prompts_two_part_rating_native.csv
else
    "$PYTHON" scripts/run_evaluation.py \
        --eval_type synthetic \
        --model "$MODEL" \
        --dataset_path "audio_samples/elevenlabs_dataset4" \
        --prompt_files $PROMPT_FILES \
        --output_path "$OUTPUT_PATH" \
        ${API_KEY:+--api_key "$API_KEY"}
fi

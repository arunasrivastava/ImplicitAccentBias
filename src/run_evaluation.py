import argparse
import csv
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

MAX_RETRIES = 6
RETRY_BACKOFF = 15       # seconds × attempt for generic errors
RATE_LIMIT_BACKOFF = 60  # seconds × attempt for 429 / quota errors


def _is_rate_limit(msg: str) -> bool:
    s = msg.lower()
    return any(x in s for x in (
        "ratelimiterror", "rate_limit", "rate limit",
        "429", "resource_exhausted", "quota", "too many requests",
    ))

CORPUS_CATEGORY_MAP = {
    "personal-introduction": "Personal-Introduction",
    "personal-commitment": "Personal-Commitment",
    "financial-product": "Financial-Product",
    "client-disagreement": "Client-Disagreement",
    "disagreement-unscripted": "Disagreement-Unscripted",
    "professional-introduction-unscripted": "Professional-Introduction-Unscripted",
}

CORPUS_PREFIX_MAP = {
    "personal-introduction": "pi",
    "personal-commitment": "pc",
    "financial-product": "fp",
    "client-disagreement": "cd",
    "disagreement-unscripted": "du",
    "professional-introduction-unscripted": "piu",
}

CORPUS_META_COLS = [
    "file_path", "category", "speaker_name_from_file", "name",
    "consent_18_plus", "data_sharing_consent", "age_range",
    "english_acquired_location", "native_language", "other_languages",
    "age_learned_english", "countries_lived", "gender", "uw_affiliation",
    "recording_device", "language_background_notes", "accent_nationality_origin",
    "audio_quality_rating", "quality_notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run accent evaluation with a specified model.")
    parser.add_argument('--model', type=str, required=True, help="Model name (e.g. gemini-2.5-flash, gpt-audio-1.5, qwen, voxtral, flamingo)")
    parser.add_argument('--eval_type', type=str, choices=["synthetic", "corpus"], default="synthetic", help="Eval mode: synthetic (ElevenLabs) or corpus (HuggingFace)")
    parser.add_argument('--output_path', type=str, required=True, help="Path to results CSV (reuse same path to resume/retry failed rows)")
    parser.add_argument('--api_key', type=str, default=None, help="API key for gemini or gpt (optional, falls back to env var)")
    parser.add_argument('--rate_limit', type=float, default=1.5, help="Seconds between API calls (default 1.5; use 0 for local models)")
    # synthetic-only args
    parser.add_argument('--dataset_path', type=str, help="[synthetic] Path to dataset folder (contains metadata CSV and speaker audio subdirs)")
    parser.add_argument('--prompts_file', type=str, help="[synthetic] Path to one prompts CSV file")
    # prompt args
    parser.add_argument('--hf_dataset', type=str, default="multispeak/hiring-accent-corpus", help="[corpus] HuggingFace dataset name")
    parser.add_argument('--hf_split', type=str, default="train", help="[corpus] HuggingFace dataset split")
    parser.add_argument('--prompt_files', type=str, nargs='+', help="Prompt CSVs as KEY=PATH pairs, e.g. critical=path/to/critical.csv ideal=path/to/ideal.csv")
    return parser.parse_args()


def load_model(model_name):
    if "gemini" in model_name:
        from models.gemini import run_gemini
        return None, run_gemini
    elif "gpt" in model_name:
        from models.gpt import run_gpt
        return None, run_gpt
    elif "qwen" in model_name:
        from models.run_qwen import load_qwen3_model, do_qwen3_inference
        model, processor = load_qwen3_model()
        return (model, processor), do_qwen3_inference
    elif "voxtral" in model_name:
        from models.run_voxtral import load_voxtral_model, do_voxtral_inference
        model, processor = load_voxtral_model()
        return (model, processor), do_voxtral_inference
    elif "flamingo" in model_name:
        from models.flamingo import load_flamingo_model, do_flamingo_inference
        model, processor = load_flamingo_model()
        return (model, processor), do_flamingo_inference
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_inference(model_name, model_state, infer_fn, prompt, audio_file, api_key=None):
    if "gemini" in model_name or "gpt" in model_name:
        from utils.audio import audio_file_to_array
        audio = audio_file_to_array(audio_file)
        output = infer_fn(audio, prompt, model=model_name, api_key=api_key)
        return str(output).strip() if output else ""
    elif "voxtral" in model_name or "flamingo" in model_name:
        model, processor = model_state
        output = infer_fn(model, processor, prompt, audio_file)
        return str(output).strip() if output else ""
    else:  # qwen
        model, processor = model_state
        _, answer, _ = infer_fn(model, processor, prompt, audio_file)
        return str(answer).strip() if answer else ""


def load_completed(results_csv, key_cols):
    if not os.path.isfile(results_csv):
        return set()
    with open(results_csv, newline="") as f:
        return {
            tuple(r[col] for col in key_cols)
            for r in csv.DictReader(f)
            if all(col in r for col in key_cols)
            if r["model_output"] not in ("FAILED", "", None)
            and not str(r["model_output"]).startswith("Error analyzing:")
        }


def infer_with_retry(infer_callable, label=""):
    for attempt in range(MAX_RETRIES):
        try:
            raw = infer_callable()
            output = str(raw).strip().replace("\n", " ").replace("\r", "") if raw else ""
            if not output:
                pass  # fall through to backoff
            elif _is_rate_limit(output):
                wait = RATE_LIMIT_BACKOFF * (attempt + 1)
                print(f"Rate limit — {label} (attempt {attempt+1}), waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            elif output.startswith("Error analyzing:"):
                print(f"Attempt {attempt+1} API error — {label}: {output}", flush=True)
            else:
                return output
        except Exception as e:
            err = str(e)
            if _is_rate_limit(err):
                wait = RATE_LIMIT_BACKOFF * (attempt + 1)
                print(f"Rate limit — {label} (attempt {attempt+1}), waiting {wait}s: {e}", flush=True)
                time.sleep(wait)
                continue
            print(f"Attempt {attempt+1} failed — {label}: {e}", flush=True)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    return ""


def load_prompt_files(prompt_files):
    prompt_rows = []
    for prompt_type, path in prompt_files.items():
        with open(path) as f:
            for r in csv.DictReader(f):
                prompt_rows.append({**r, "prompt_type": prompt_type})
    return prompt_rows


def get_corpus_meta(sample):
    meta = {col: sample.get(col, "") for col in CORPUS_META_COLS}
    category = sample["category"]
    speaker = sample.get("speaker_name_from_file", "")
    prefix = CORPUS_PREFIX_MAP.get(category, category[:2])
    meta["file_path"] = f"{category}/{prefix} - {speaker}.wav"
    return meta


def normalize_audio(audio_array):
    if audio_array.dtype != np.int16:
        return (audio_array * 32767).clip(-32768, 32767).astype(np.int16)
    return audio_array


def run_eval_synthetic(model_name, dataset_path, prompt_rows, output_path, api_key=None, rate_limit=1.5):
    metadata_df = pd.read_csv(os.path.join(dataset_path, "elevenlabs_metadata.csv"))
    prompt_df = pd.DataFrame(prompt_rows)
    fieldnames = list(metadata_df.columns) + list(prompt_df.columns) + ["model_version", "model_output"]
    completed = load_completed(output_path, ["speaker_name", "script_filename", "prompt_type", "prompt"])
    model_state, infer_fn = load_model(model_name)
    print(f"Loaded model: {model_name}", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_header = not os.path.isfile(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for row in metadata_df.itertuples():
            audio_path = os.path.join(dataset_path, row.speaker_name, row.script_filename)
            if not os.path.isfile(audio_path):
                print(f"Missing audio: {audio_path}", flush=True)
                continue

            filtered_prompts = prompt_df[prompt_df["category"] == row.category]
            if filtered_prompts.empty:
                print(f"No prompts found for category: {row.category}")
                continue

            print(f"Processing: {row.speaker_name} | {row.category}", flush=True)
            for prompt_row in filtered_prompts.itertuples():
                if (row.speaker_name, row.script_filename, prompt_row.prompt_type, prompt_row.prompt) in completed:
                    continue

                output = infer_with_retry(
                    lambda: run_inference(model_name, model_state, infer_fn, prompt_row.prompt, audio_path, api_key),
                    label=f"{row.speaker_name} / {prompt_row.prompt[:40]}"
                )
                if not output:
                    print(f"FAILED: {row.speaker_name} / {prompt_row.prompt[:40]}", flush=True)
                    continue

                writer.writerow({**row._asdict(), **prompt_row._asdict(), "model_version": model_name, "model_output": output})
                f.flush()
                if rate_limit > 0:
                    time.sleep(rate_limit)

    print(f"Saved results to {output_path}", flush=True)


def run_eval_corpus(model_name, hf_dataset, prompt_files, output_path, api_key=None, rate_limit=1.5):
    """
    prompt_files: dict mapping prompt_type label -> path to prompts CSV
                  e.g. {"critical": "...", "ideal": "...", "native": "..."}
    """
    prompt_rows = load_prompt_files(prompt_files)
    fieldnames = CORPUS_META_COLS + ["prompt_type", "script_type", "prompt", "model_version", "model_output"]
    completed = load_completed(output_path, ["file_path", "prompt_type", "script_type"])
    model_state, infer_fn = load_model(model_name)
    print(f"Loaded model: {model_name}", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_header = not os.path.isfile(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for sample in hf_dataset:
            category = sample["category"]
            if category not in CORPUS_CATEGORY_MAP:
                continue

            meta = get_corpus_meta(sample)
            audio_array = normalize_audio(sample["audio"]["array"])
            matching_prompts = [p for p in prompt_rows if p["category"] == CORPUS_CATEGORY_MAP[category]]

            for p in matching_prompts:
                if (meta["file_path"], p["prompt_type"], p["script_type"]) in completed:
                    continue

                output = infer_with_retry(
                    lambda: infer_fn(audio_array, p["prompt"], model=model_name, api_key=api_key),
                    label=f"{meta['name']} / {p['prompt_type']} / {p['script_type']}"
                )
                if not output:
                    print(f"FAILED: {meta['name']} / {p['prompt_type']} / {p['script_type']}", flush=True)
                    output = "FAILED"

                writer.writerow({**meta, "prompt_type": p["prompt_type"], "script_type": p["script_type"], "prompt": p["prompt"], "model_version": model_name, "model_output": output})
                f.flush()
                print(f"Done: {meta['name']} | {category} | {p['prompt_type']} | {p['script_type']}", flush=True)
                if rate_limit > 0:
                    time.sleep(rate_limit)

    print(f"Saved results to {output_path}", flush=True)


def _load_corpus_from_arrow(hf_dataset_name):
    """
    Fallback loader that reads the local pyarrow IPC cache directly.
    Used when load_dataset fails due to a LocalFileSystem cache incompatibility.
    Yields dicts with sample["audio"]["array"] as a float32 numpy array at 16 kHz.
    """
    import pyarrow.ipc as ipc
    import soundfile as sf
    import io
    from pathlib import Path

    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    # HF caches parquet datasets under <org>___parquet/<org>--<name>-<hash>/0.0.0/<hash>/
    # Use a broad recursive search to locate the arrow file regardless of exact hash.
    matches = sorted(cache_root.glob("*/*/0.0.0/*/parquet-train.arrow"))
    if not matches:
        sys.exit(f"Arrow cache not found under {cache_root}. "
                 "Run load_dataset once with a working datasets version first.")
    arrow_path = matches[-1]
    print(f"Loading corpus from local arrow cache: {arrow_path}", flush=True)

    tables = []
    with open(arrow_path, "rb") as fh:
        reader = ipc.open_stream(fh)
        for batch in reader:
            tables.append(batch.to_pydict())

    merged = {}
    for t in tables:
        for k, v in t.items():
            merged.setdefault(k, []).extend(v)

    n = len(next(iter(merged.values())))
    for i in range(n):
        row = {k: merged[k][i] for k in merged}
        audio_data = row.get("audio") or {}
        raw_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else None
        if raw_bytes:
            arr, sr = sf.read(io.BytesIO(raw_bytes))
            if arr.ndim > 1:
                arr = arr[:, 0]
            row["audio"] = {"array": arr.astype(np.float32), "sampling_rate": sr}
        yield row


def main():
    """
    See run_evaluation.sh for example usage and SLURM configuration.
    """
    args = parse_args()
    if args.eval_type == "corpus":
        prompt_files = dict(kv.split("=", 1) for kv in args.prompt_files)
        try:
            from datasets import load_dataset
            hf_dataset = load_dataset(args.hf_dataset, split=args.hf_split)
        except Exception as e:
            print(f"load_dataset failed ({e}), falling back to local arrow cache.", flush=True)
            hf_dataset = _load_corpus_from_arrow(args.hf_dataset)
        run_eval_corpus(args.model, hf_dataset, prompt_files, args.output_path, args.api_key, args.rate_limit)
    else:
        if args.prompt_files:
            prompt_rows = load_prompt_files(dict(kv.split("=", 1) for kv in args.prompt_files))
        elif args.prompts_file:
            prompt_rows = load_prompt_files({"single": args.prompts_file})
        else:
            sys.exit("Synthetic eval requires --prompt_files or --prompts_file")
        run_eval_synthetic(args.model, args.dataset_path, prompt_rows, args.output_path, args.api_key, args.rate_limit)


if __name__ == "__main__":
    main()

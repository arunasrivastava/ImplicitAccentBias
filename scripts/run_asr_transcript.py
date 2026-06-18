#!/usr/bin/env python3
"""
Full-transcript ASR experiment on the scripted human hiring corpus.

For each scripted audio clip (4 categories × all speakers):
  1. Send the full audio to the model for free transcription
  2. Strip filler words (um, uh, ah, er, …) from the returned text
  3. Compute WER against the known reference script

Normalization: both the reference script and the model hypothesis are
lowercased and stripped of punctuation via normalize_tokens() before the
Levenshtein edit distance is computed. Filler removal is applied to the
model output only (speakers stutter naturally; we don't penalize the model
for faithfully transcribing those).

Audio source (automatic fallback):
  1. HuggingFace (multispeak/hiring-accent-corpus) — tried first; no local
     files needed. Use --local to skip this step.
  2. Local directory (.data/speech_hiring_data_processed or --audio_root DIR)
     — used automatically if HuggingFace is unavailable.

Usage (Gemini):
    python3.10 scripts/run_asr_transcript.py \\
        --model gemini-2.5-flash \\
        --output_path results/asr_transcript/gemini-2.5-flash_asr_transcript.csv

Usage (Qwen on Hyak):
    python3.10 scripts/run_asr_transcript.py \\
        --model qwen \\
        --output_path results/asr_transcript/qwen_asr_transcript.csv

Resumable: re-running with the same --output_path skips completed rows.
Log: <output_path>.log
"""

import argparse
import csv
import os
import re
import sys
import time
import tempfile
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

SAMPLE_RATE = 16_000
MAX_RETRIES = 6
RETRY_BACKOFF = 15   # seconds × attempt for generic errors
RATE_LIMIT_BACKOFF = 60  # seconds × attempt for 429 / quota errors


def _is_rate_limit(msg: str) -> bool:
    s = msg.lower()
    return any(x in s for x in (
        "ratelimiterror", "rate_limit", "rate limit",
        "429", "resource_exhausted", "quota", "too many requests",
    ))

SCRIPTED_CATEGORIES = [
    "personal-introduction",
    "personal-commitment",
    "financial-product",
    "client-disagreement",
]

REFERENCE_SCRIPTS = {
    "personal-introduction": (
        "I graduated about three years ago and I've been working in customer service "
        "and client-facing roles since then. Most recently I was at a financial services "
        "firm helping onboard clients and answer their questions about accounts. I really "
        "enjoy being that connection between the technical side and clients helping people "
        "understand their options. I think I'm a strong candidate because I have solid "
        "experience with customers I'm getting more familiar with financial products and "
        "I'm pretty detail-oriented which helps me catch mistakes early. I'm looking to "
        "take on more responsibility in this kind of role."
    ),
    "personal-commitment": (
        "Earlier this year I committed to finishing a client report by Thursday for a "
        "Friday meeting but on Wednesday my manager needed help with an urgent compliance "
        "request that had a same-day deadline. I realized I couldn't do both well so I "
        "talked to my manager about prioritizing the compliance issue and then I reached "
        "out to the client directly to let them know the report would be a day late. I "
        "apologized and we rescheduled the meeting to Monday. The client appreciated the "
        "heads-up and I got the report to them Friday afternoon. I learned it's way better "
        "to communicate early when priorities shift rather than just missing a deadline."
    ),
    "financial-product": (
        "I had a client who wanted to open a retirement account but didn't understand the "
        "difference between a traditional IRA and a Roth IRA. I asked them some questions "
        "about their tax situation and timeline first then explained it as basically pay "
        "taxes now or pay taxes later. I used a simple example with actual numbers to show "
        "how each might work for them. I made sure to mention I wasn't giving tax advice "
        "just explaining the options. They had more questions so I walked through a "
        "comparison and suggested they talk to a tax professional too. They opened a Roth "
        "and told me later they appreciated that I explained it clearly without pushing "
        "one option."
    ),
    "client-disagreement": (
        "I had a client who wanted to put a lot of their portfolio in one tech stock "
        "because they worked in that industry and felt confident about it. I recommended "
        "more diversification based on their risk tolerance. They pushed back because they "
        "felt they had good insight. Instead of just agreeing or arguing I asked them to "
        "walk me through their thinking so I could understand it better. Then we talked "
        "through some scenarios like what would happen if that stock dropped. I acknowledged "
        "they knew the industry well but explained diversification was about managing risk. "
        "We ended up compromising they kept a bigger position than I suggested but not as "
        "concentrated as they wanted. They appreciated that I listened."
    ),
}

FILLER_WORDS = {
    "um", "uh", "ah", "er", "hmm", "hm", "mm", "uhh", "umm", "ahh",
    "erm", "eh", "mhm", "mmm", "uhhh",
}

ASR_PROMPT = (
    "Transcribe this audio clip exactly. "
    "Return only the spoken words — no punctuation, no labels, no commentary."
)

HF_DATASET = "multispeak/hiring-accent-corpus"
CORPUS_CSV_DEFAULT = "results/hiring_corpus/gemini-2.5-flash_hiring_corpus.csv"
AUDIO_ROOT_DEFAULT = ".data/speech_hiring_data_processed"


# ── Text helpers ──────────────────────────────────────────────────────────────

def normalize_tokens(text: str) -> list[str]:
    """Lowercase + strip punctuation + whitespace-tokenize."""
    return re.sub(r"[^a-z\s]", " ", text.lower()).split()


def remove_fillers(text: str) -> str:
    """Strip filler words from model output (after normalizing)."""
    return " ".join(t for t in normalize_tokens(text) if t not in FILLER_WORDS)


# ── WER ───────────────────────────────────────────────────────────────────────

def token_edit_distance(ref: list[str], hyp: list[str]) -> int:
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
        prev = cur
    return prev[n]


def compute_wer(reference: str, hypothesis: str) -> tuple:
    """Returns (wer, edit_ops, ref_length). Both sides are normalize_tokens'd."""
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)
    if not ref:
        return 0.0, 0, 0
    ops = token_edit_distance(ref, hyp)
    return ops / len(ref), ops, len(ref)


# ── Model inference ───────────────────────────────────────────────────────────

def get_api_key(model: str):
    if "gemini" in model:
        return os.getenv("GEMINI_API_KEY")
    if "gpt" in model:
        return os.getenv("OPENAI_API_KEY")
    return None


def run_model(model: str, audio_input, api_key) -> str:
    """
    audio_input: str (file path) or np.ndarray of int16 at 16 kHz.
    For Qwen, an array is written to a temp WAV file before inference.
    """
    # Resolve to numpy array
    if isinstance(audio_input, str):
        arr, _sr = sf.read(audio_input, dtype="int16")
        if arr.ndim > 1:
            arr = arr[:, 0]
        audio_path = audio_input
    else:
        arr = audio_input
        audio_path = None  # HF path — Qwen needs a temp file

    if "gemini" in model:
        from models.gemini import run_gemini
        return run_gemini(arr, ASR_PROMPT, model=model, api_key=api_key) or ""

    elif "gpt" in model:
        from models.gpt import run_gpt
        return run_gpt(arr, ASR_PROMPT, model=model, api_key=api_key) or ""

    elif "qwen" in model:
        from models.run_qwen import run_qwen
        if audio_path:
            return run_qwen(audio_path, ASR_PROMPT) or ""
        # Write array to temp WAV for Qwen
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, arr, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            return run_qwen(tmp.name, ASR_PROMPT) or ""

    raise ValueError(f"Unknown model: {model}")


# ── Manifest builders ─────────────────────────────────────────────────────────

def build_manifest_local(corpus_csv: Path, audio_root: Path):
    """Returns (manifest_df, hf_index=None) for local-file mode."""
    corpus = pd.read_csv(corpus_csv)
    manifest = (
        corpus[
            corpus["category"].isin(SCRIPTED_CATEGORIES) &
            (corpus["script_type"] == "delivery")
        ]
        .drop_duplicates("file_path")
        .reset_index(drop=True)
        .copy()
    )
    manifest["full_path"] = manifest["file_path"].apply(lambda p: str(audio_root / p))
    return manifest, None


def build_manifest_hf():
    """
    Returns (manifest_df, hf_index) where hf_index maps file_path → dataset row.
    Audio arrays come from the HuggingFace dataset; no local files needed.
    """
    from datasets import load_dataset
    print("Loading HuggingFace dataset (first run may download audio)…", flush=True)
    ds = load_dataset(HF_DATASET, split="train")

    # Build file_path → index lookup (audio decoded lazily per access)
    hf_index = {row["file_path"]: i for i, row in enumerate(ds)}

    # Build metadata-only manifest (drop the audio column to keep memory light)
    rows = []
    for ex in ds:
        if (ex.get("category") in SCRIPTED_CATEGORIES and
                ex.get("script_type") == "delivery"):
            rows.append({k: v for k, v in ex.items() if k != "audio"})
    manifest = (
        pd.DataFrame(rows)
        .drop_duplicates("file_path")
        .reset_index(drop=True)
    )

    # Normalize accent column name to match local mode
    if "accent_nationality_origin" not in manifest.columns and "accent" in manifest.columns:
        manifest["accent_nationality_origin"] = manifest["accent"]
    if "name" not in manifest.columns and "speaker_name" in manifest.columns:
        manifest["name"] = manifest["speaker_name"]

    return manifest, (ds, hf_index)


# ── Resumability ──────────────────────────────────────────────────────────────

def load_completed(output_path: str) -> set:
    if not os.path.isfile(output_path):
        return set()
    with open(output_path, newline="") as f:
        return {
            (r["file_path"], r["model"])
            for r in csv.DictReader(f)
            if r.get("raw_transcription") not in ("FAILED", "", None)
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Model: gemini-2.5-flash or qwen")
    parser.add_argument("--output_path", required=True,
                        help="Output CSV path (resumable)")
    parser.add_argument("--local", action="store_true",
                        help="Skip HuggingFace and load audio from local --audio_root instead.")
    parser.add_argument("--corpus_csv", default=CORPUS_CSV_DEFAULT,
                        help="Hiring corpus CSV (local mode only, for manifest metadata)")
    parser.add_argument("--audio_root", default=AUDIO_ROOT_DEFAULT,
                        help="Root dir for audio WAV files (local fallback)")
    parser.add_argument("--rate_limit", type=float, default=1.0,
                        help="Seconds between calls (default 1.0; use 0.0 for local models)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing output and start from scratch")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    log_path = args.output_path.replace(".csv", ".log")

    if args.fresh:
        for p in [args.output_path, log_path]:
            if os.path.isfile(p):
                os.remove(p)

    log_fh = open(log_path, "a")

    def log(msg):
        s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(s, flush=True)
        print(s, file=log_fh, flush=True)

    log("=== ASR transcript experiment started ===")
    log(f"model={args.model}  output={args.output_path}  local_only={args.local}")

    api_key = get_api_key(args.model)
    if "gemini" in args.model and not api_key:
        log("ERROR: no GEMINI_API_KEY found in .env")
        sys.exit(1)
    if "gpt" in args.model and not api_key:
        log("ERROR: no OPENAI_API_KEY found in .env")
        sys.exit(1)
    if api_key:
        log(f"API key: ...{api_key[-6:]}")

    # Build manifest — try HuggingFace first, fall back to local
    hf_ds = hf_index = None
    use_hf = False

    if not args.local:
        try:
            log(f"Trying HuggingFace dataset ({HF_DATASET})…")
            manifest, hf_bundle = build_manifest_hf()
            hf_ds, hf_index = hf_bundle
            use_hf = True
            log("HuggingFace dataset loaded successfully.")
        except Exception as e:
            log(f"HuggingFace unavailable ({e}), falling back to local audio.")

    if not use_hf:
        corpus_csv = (Path(args.corpus_csv) if os.path.isabs(args.corpus_csv)
                      else root / args.corpus_csv)
        audio_root = (Path(args.audio_root) if os.path.isabs(args.audio_root)
                      else root / args.audio_root)
        log(f"Audio source: local ({audio_root})")
        manifest, _ = build_manifest_local(corpus_csv, audio_root)

        missing = manifest[~manifest["full_path"].apply(os.path.isfile)]
        if len(missing):
            log(f"WARNING: {len(missing)} audio files not found at audio_root.")
            for p in missing["full_path"].head(3):
                log(f"  missing: {p}")

    log(f"Scripted clips in manifest: {len(manifest)}")
    log(f"Accent breakdown: {manifest['accent_nationality_origin'].value_counts().to_dict()}")

    completed = load_completed(args.output_path)
    total = len(manifest)
    n_skip = sum(1 for _, r in manifest.iterrows()
                 if (r["file_path"], args.model) in completed)
    log(f"Already completed: {n_skip}  Remaining: {total - n_skip}")

    fieldnames = [
        "file_path", "category", "speaker_id", "accent", "model",
        "raw_transcription", "clean_transcription",
        "ref_word_count", "hyp_word_count", "edit_ops", "wer",
    ]
    write_header = (
        not os.path.isfile(args.output_path) or
        os.path.getsize(args.output_path) == 0
    )

    n_done = n_skip
    wer_sum = 0.0
    wer_count = 0
    t_start = time.time()

    with open(args.output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for _, row in manifest.iterrows():
            key = (row["file_path"], args.model)
            if key in completed:
                continue

            # Resolve audio input
            if use_hf:
                fp = row["file_path"]
                if fp not in hf_index:
                    log(f"  SKIP (not in HF index): {fp}")
                    continue
                hf_ex = hf_ds[hf_index[fp]]
                audio_info = hf_ex["audio"]
                arr = np.array(audio_info["array"])
                if arr.dtype != np.int16:
                    arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
                audio_input = arr
            else:
                audio_input = row["full_path"]
                if not os.path.isfile(audio_input):
                    log(f"  SKIP (missing): {audio_input}")
                    continue

            raw = ""
            for attempt in range(MAX_RETRIES):
                try:
                    result = run_model(args.model, audio_input, api_key)
                    if isinstance(result, (list, tuple)):
                        result = " ".join(str(x) for x in result)
                    raw = result.strip().replace("\n", " ") if result else ""
                    if not raw:
                        raise RuntimeError("empty response")
                    if _is_rate_limit(raw):
                        wait = RATE_LIMIT_BACKOFF * (attempt + 1)
                        log(f"  Rate limit [{row['file_path']}] attempt {attempt+1}, waiting {wait}s")
                        time.sleep(wait)
                        raw = ""
                        continue
                    if raw.startswith("Error analyzing"):
                        raise RuntimeError(raw)
                    break
                except Exception as e:
                    err = str(e)
                    if _is_rate_limit(err):
                        wait = RATE_LIMIT_BACKOFF * (attempt + 1)
                        log(f"  Rate limit [{row['file_path']}] attempt {attempt+1}, waiting {wait}s")
                        time.sleep(wait)
                        raw = ""
                        continue
                    log(f"  Attempt {attempt+1} failed [{row['file_path']}]: {e}")
                    raw = ""
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF * (attempt + 1))

            if not raw:
                log(f"  FAILED after {MAX_RETRIES} attempts: {row['file_path']}")
                raw = "FAILED"

            if raw != "FAILED":
                clean = remove_fillers(raw)
                ref = REFERENCE_SCRIPTS[row["category"]]
                wer_val, edit_ops, ref_len = compute_wer(ref, clean)
                hyp_len = len(normalize_tokens(clean))
                wer_sum += wer_val
                wer_count += 1
            else:
                clean, wer_val, edit_ops, ref_len, hyp_len = "FAILED", "FAILED", -1, -1, -1

            writer.writerow({
                "file_path":           row["file_path"],
                "category":            row["category"],
                "speaker_id":          row["name"],
                "accent":              row["accent_nationality_origin"],
                "model":               args.model,
                "raw_transcription":   raw,
                "clean_transcription": clean,
                "ref_word_count":      ref_len,
                "hyp_word_count":      hyp_len,
                "edit_ops":            edit_ops,
                "wer":                 round(wer_val, 4) if isinstance(wer_val, float) else wer_val,
            })
            f.flush()
            completed.add(key)

            n_done += 1
            pct = 100 * n_done / total
            eta_s = (time.time() - t_start) / max(n_done - n_skip, 1) * (total - n_done)
            running_wer = wer_sum / wer_count if wer_count else 0
            wer_str = f"WER={wer_val:.3f}" if isinstance(wer_val, float) else "FAILED"
            log(f"  [{n_done}/{total} {pct:.1f}% ETA {eta_s/60:.1f}m] "
                f"{row['accent_nationality_origin']:10s} "
                f"{str(row['name'])[:28]:28s} "
                f"{row['category']:25s} "
                f"{wer_str}  (running mean {running_wer:.3f})")

            if args.rate_limit > 0:
                time.sleep(args.rate_limit)

    log("")
    log("=== DONE ===")
    if wer_count:
        log(f"Mean WER across {wer_count} clips: {wer_sum/wer_count:.3f}")
    log(f"Results saved to {args.output_path}")
    log_fh.close()


if __name__ == "__main__":
    main()

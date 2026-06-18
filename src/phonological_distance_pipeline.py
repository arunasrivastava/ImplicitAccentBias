"""
Phonological distance pipeline for accent analysis.

Stages:
  1. extract   – pull scripted audio from HF dataset → 16kHz mono WAV
  2. transcribe – Whisper word-level timestamps per speaker
  3. distances  – XLS-R layer-14 embeddings + DTW per word vs. American English reference

Phonological-feature categorization (consonant cluster / schwa / ...) is done in the
analysis notebook (notebooks/Figures_Phonolgical_Distance.ipynb), not here.

Run a single stage:
    python3.10 src/phonological_distance_pipeline.py --stage extract --category personal-introduction
    python3.10 src/phonological_distance_pipeline.py --stage transcribe --category personal-introduction
    python3.10 src/phonological_distance_pipeline.py --stage distances --category personal-introduction

Other categories: personal-commitment, financial-product, client-disagreement
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent

# ── Scripts ──────────────────────────────────────────────────────────────────
# The verbatim scripted prompts live in data/human_hiring_corpus/scripted_scripts.csv
# (one row per category). They are loaded here only to build the script vocabulary
# used to keep in-script words from the Whisper transcription. Phonological-feature
# categorization (consonant cluster / schwa / ...) is NOT done here — it lives in
# the analysis notebook (notebooks/Figures_Phonolgical_Distance.ipynb).

def _load_scripts():
    with open(ROOT / "data" / "human_hiring_corpus" / "scripted_scripts.csv", newline="") as f:
        return {row["category"]: row["script"] for row in csv.DictReader(f)}

SCRIPTS = _load_scripts()

# ── Category configurations ───────────────────────────────────────────────────

CATEGORY_CONFIGS = {
    "personal-introduction": {
        "hf_category": "personal-introduction",
        "script": SCRIPTS["personal-introduction"],
        "out": ROOT / "results" / "phonological_distance",
    },
    "personal-commitment": {
        "hf_category": "personal-commitment",
        "script": SCRIPTS["personal-commitment"],
        "out": ROOT / "results" / "phonological_distance_commitment",
    },
    "financial-product": {
        "hf_category": "financial-product",
        "script": SCRIPTS["financial-product"],
        "out": ROOT / "results" / "phonological_distance_financial",
    },
    "client-disagreement": {
        "hf_category": "client-disagreement",
        "script": SCRIPTS["client-disagreement"],
        "out": ROOT / "results" / "phonological_distance_disagreement",
    },
}
OUT      = ROOT / "results" / "phonological_distance"
WAV_DIR  = OUT / "wav"
TS_DIR   = OUT / "timestamps"
DIST_DIR = OUT / "distances"


def get_script_vocab(script):
    tokens = re.sub(r"[—\-]", " ", script)
    tokens = re.sub(r"[^a-zA-Z' ]", "", tokens)
    return {w.lower() for w in tokens.split() if w}


def normalize_word(word):
    # split on hyphens first so "detail-oriented" → "detail", "client-facing" → "client"
    first = re.split(r"[-–—]", word)[0]
    return re.sub(r"[^a-zA-Z']", "", first).lower()


# ---------------------------------------------------------------------------
# Stage 1a: extract audio — human corpus (HuggingFace arrow cache)
# ---------------------------------------------------------------------------

# Human corpus lives in a gated HF dataset (anonymized speaker IDs). Requires
# `huggingface_hub` login with an account that has been granted access.
HUMAN_DATASET = "multispeak/hiring-accent-speech-human-voices"


def stage_extract(cfg):
    """Pull scripted human audio for one category from the gated HF dataset -> 16 kHz mono WAVs."""
    from datasets import load_dataset
    import soundfile as sf

    print(f"Loading {HUMAN_DATASET} (gated -- requires HF access)...", flush=True)
    ds = load_dataset(HUMAN_DATASET, split="train")
    rows = [r for r in ds if r["category"] == cfg["hf_category"]]
    print(f"Found {len(rows)} {cfg['hf_category']} rows")

    TARGET_SR = 16000
    out_dir = cfg["out"]
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for r in rows:
        speaker_id = r["speaker_id"]                     # already anonymized in the dataset
        accent     = r["accent_nationality_origin"]
        audio      = r["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr  = audio["sampling_rate"]
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        out_path = wav_dir / f"{speaker_id}.wav"
        sf.write(str(out_path), arr.astype(np.float32), TARGET_SR)
        meta_rows.append({"speaker_id": speaker_id, "accent": accent, "wav": str(out_path)})
        print(f"  Saved {out_path.name}  ({accent})")

    meta_path = out_dir / "speakers.json"
    with open(meta_path, "w") as f:
        json.dump(meta_rows, f, indent=2)
    print(f"\nSaved {len(meta_rows)} WAV files. Metadata -> {meta_path}")


# ---------------------------------------------------------------------------
# Stage 1b: extract audio — ElevenLabs synthetic corpus
# ---------------------------------------------------------------------------

def stage_extract_synthetic(cfg, synthetic_path, script_type="improved"):
    """
    Copy & resample ElevenLabs WAV files into the same wav/ + speakers.json
    layout that stage_transcribe and stage_distances expect.

    synthetic_path : root directory of the ElevenLabs dataset
                     (contains elevenlabs_metadata.csv and one sub-dir per speaker)
    script_type    : "improved" (default) or "disfluent"
    """
    import soundfile as sf
    import pandas as pd

    synthetic_path = Path(synthetic_path)
    meta_csv = synthetic_path / "elevenlabs_metadata.csv"
    if not meta_csv.exists():
        sys.exit(f"elevenlabs_metadata.csv not found at {meta_csv}")

    meta = pd.read_csv(meta_csv)

    # Map pipeline category key → ElevenLabs metadata category label
    # e.g. "personal-introduction" → "Personal-Introduction"
    el_category = cfg["hf_category"].title()          # "Personal-Introduction"
    rows = meta[
        (meta["category"] == el_category) &
        (meta["script_type"] == script_type)
    ].reset_index(drop=True)

    if rows.empty:
        sys.exit(
            f"No rows found for category='{el_category}', script_type='{script_type}'. "
            f"Available: {meta['category'].unique()}, {meta['script_type'].unique()}"
        )

    print(f"Found {len(rows)} rows  (category={el_category}, script_type={script_type})")
    print(rows["accent"].value_counts().to_string())

    TARGET_SR = 16000
    out_dir = cfg["out"]
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for _, row in rows.iterrows():
        speaker_id = re.sub(r"[^a-zA-Z0-9_]", "_", row["speaker_name"]).lower()
        accent     = row["accent"]
        src_wav    = synthetic_path / row["speaker_name"] / row["script_filename"]

        if not src_wav.exists():
            print(f"  SKIP {speaker_id}: file not found ({src_wav})")
            continue

        arr, sr = sf.read(str(src_wav))
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

        arr = arr.astype(np.float32)
        out_path = wav_dir / f"{speaker_id}.wav"
        sf.write(str(out_path), arr, TARGET_SR)
        meta_rows.append({"speaker_id": speaker_id, "accent": accent, "wav": str(out_path)})
        print(f"  Saved {out_path.name}  ({accent})")

    meta_path = out_dir / "speakers.json"
    output = {
        "run_config": {
            "source":      "synthetic_elevenlabs",
            "script_type": script_type,
            "note": (
                "Clean scripted speech with no disfluencies (ElevenLabs 'improved' variant)."
                if script_type == "improved"
                else "Disfluent scripted speech with filler words and hesitations (ElevenLabs 'disfluent' variant)."
            ),
            "category":    el_category,
            "synthetic_path": str(synthetic_path),
        },
        "speakers": meta_rows,
    }
    with open(meta_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSaved {len(meta_rows)} WAV files. Metadata → {meta_path}")


# ---------------------------------------------------------------------------
# Stage 2: transcribe
# ---------------------------------------------------------------------------

def stage_transcribe(cfg):
    import transformers

    out_dir   = cfg["out"]
    ts_dir    = out_dir / "timestamps"
    meta_path = out_dir / "speakers.json"

    if not meta_path.exists():
        sys.exit("Run --stage extract first.")

    with open(meta_path) as f:
        _meta = json.load(f)
    speakers = _meta["speakers"] if isinstance(_meta, dict) else _meta

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Whisper on {device} …")

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device=device,
        return_timestamps="word",
        generate_kwargs={"language": "en", "task": "transcribe"},
    )

    script_vocab = get_script_vocab(cfg["script"])
    ts_dir.mkdir(parents=True, exist_ok=True)

    for sp in speakers:
        out_path = ts_dir / f"{sp['speaker_id']}.tsv"
        if out_path.exists():
            print(f"  SKIP {sp['speaker_id']} (already transcribed)")
            continue

        print(f"  Transcribing {sp['speaker_id']} ({sp['accent']}) …", flush=True)
        try:
            result = pipe(sp["wav"])
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        chunks = result.get("chunks", [])
        rows = []
        for ch in chunks:
            word = normalize_word(ch["text"])
            if not word:
                continue
            ts = ch.get("timestamp", (None, None))
            t_start, t_end = ts if ts else (None, None)
            if t_start is None or t_end is None:
                continue
            if word in script_vocab:
                rows.append({"word": word, "start": round(t_start, 3), "end": round(t_end, 3)})

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["word", "start", "end"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        print(f"    → {len(rows)} script words found (of {len(chunks)} total tokens)")

    print(f"\nTimestamps saved to {ts_dir}")


# ---------------------------------------------------------------------------
# Stage 3: compute acoustic distances
# ---------------------------------------------------------------------------

def _load_xlsr_model(layer=14):
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    model_id = "facebook/wav2vec2-xls-r-300m"
    print(f"Loading {model_id} …")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id, output_hidden_states=True)
    model = model.to(device).eval()
    return model, feature_extractor, device, layer


def _get_embedding(model, feature_extractor, device, layer, audio_array, sr=16000):
    """Return (T, D) hidden-state tensor at the given layer for an audio segment."""
    inputs = feature_extractor(
        audio_array, sampling_rate=sr, return_tensors="pt", padding=True
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    # hidden_states: tuple of (batch=1, T, D), one per layer (including embedding layer)
    hidden = outputs.hidden_states[layer]  # (1, T, D)
    return hidden.squeeze(0).cpu().float().numpy()  # (T, D)


def _dtw_cosine(emb1, emb2):
    """DTW distance using cosine metric between two (T, D) embedding arrays."""
    from scipy.spatial.distance import cdist

    # cosine similarity → distance
    cost = cdist(emb1, emb2, metric="cosine")
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


FRAME_RATE = 50.0  # XLS-R encoder ~50 frames/sec (used for duration normalization)


def stage_distances(cfg, xlsr_layer=14, dist_dir_override=None):
    out_dir   = cfg["out"]
    ts_dir    = out_dir / "timestamps"
    dist_dir  = Path(dist_dir_override) if dist_dir_override else out_dir / "distances"
    meta_path = out_dir / "speakers.json"

    if not meta_path.exists():
        sys.exit("Run --stage extract first.")
    with open(meta_path) as f:
        _meta = json.load(f)
    speakers = _meta["speakers"] if isinstance(_meta, dict) else _meta

    # index by speaker_id
    sp_map = {sp["speaker_id"]: sp for sp in speakers}

    # separate reference (American) from targets
    american_ids = [sp["speaker_id"] for sp in speakers if sp["accent"] == "American"]
    target_groups = {}
    for sp in speakers:
        a = sp["accent"]
        if a not in target_groups:
            target_groups[a] = []
        target_groups[a].append(sp["speaker_id"])

    print(f"Reference (American): {len(american_ids)} speakers")
    for acc, ids in target_groups.items():
        print(f"  {acc}: {len(ids)} speakers")

    model, feature_extractor, device, layer = _load_xlsr_model(xlsr_layer)

    def load_audio_segment(wav_path, start, end, sr=16000):
        arr, file_sr = sf.read(wav_path)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        s = int(start * file_sr)
        e = int(end   * file_sr)
        segment = arr[s:e].astype(np.float32)
        if len(segment) < 400:  # < 25ms → too short
            return None
        return segment

    def load_timestamps(speaker_id):
        path = ts_dir / f"{speaker_id}.tsv"
        if not path.exists():
            return {}
        words = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                w = row["word"]
                if w not in words:  # keep first occurrence
                    words[w] = (float(row["start"]), float(row["end"]))
        return words

    # precompute reference embeddings per word
    # stored as (speaker_id, embedding) to enable leave-one-out for American speakers
    print("\nPrecomputing reference (American) embeddings …")
    ref_embeddings = {}   # word → list of (ref_speaker_id, np array)
    am_word_durs   = {}   # word → list of American speaker durations (for normalization)
    for ref_id in american_ids:
        ts = load_timestamps(ref_id)
        wav_path = sp_map[ref_id]["wav"]
        for word, (t0, t1) in ts.items():
            am_word_durs.setdefault(word, []).append(t1 - t0)
            seg = load_audio_segment(wav_path, t0, t1)
            if seg is None:
                continue
            emb = _get_embedding(model, feature_extractor, device, layer, seg)
            if word not in ref_embeddings:
                ref_embeddings[word] = []
            ref_embeddings[word].append((ref_id, emb))
        print(f"  {ref_id}: {len(ts)} words")

    am_dur_mean = {w: float(np.mean(durs)) for w, durs in am_word_durs.items()}
    print(f"Reference vocab coverage: {len(ref_embeddings)} unique words")

    # compute per-word distances for all target speakers
    # Leave-one-out for American speakers: each American speaker is excluded from
    # their own reference pool, removing the self-comparison zero that would otherwise
    # deflate the American baseline by ~1/n_ref (~9%).
    dist_dir.mkdir(parents=True, exist_ok=True)
    rows_all = []

    for accent, sp_ids in target_groups.items():
        print(f"\nProcessing {accent} ({len(sp_ids)} speakers) …")
        for sp_id in sp_ids:
            ts = load_timestamps(sp_id)
            wav_path = sp_map[sp_id]["wav"]
            for word, (t0, t1) in ts.items():
                if word not in ref_embeddings:
                    continue  # word not produced by any American reference speaker
                seg = load_audio_segment(wav_path, t0, t1)
                if seg is None:
                    continue
                try:
                    tgt_emb = _get_embedding(model, feature_extractor, device, layer, seg)
                except Exception as e:
                    print(f"    embedding error {sp_id}/{word}: {e}")
                    continue

                # LOO: American speakers exclude their own embedding from the reference pool
                if accent == "American":
                    pool = [emb for (rid, emb) in ref_embeddings[word] if rid != sp_id]
                else:
                    pool = [emb for (_, emb) in ref_embeddings[word]]

                if not pool:
                    continue  # no references after LOO (only happens if n_ref was 1)

                dists = []
                for ref_emb in pool:
                    try:
                        d = _dtw_cosine(tgt_emb, ref_emb)
                    except Exception:
                        continue
                    dists.append(d)

                if not dists:
                    continue

                mean_dist    = float(np.mean(dists))
                speaker_dur  = t1 - t0
                am_dur       = am_dur_mean.get(word)
                if am_dur is not None:
                    longer_dur = max(speaker_dur, am_dur)
                    norm_dist  = mean_dist / (longer_dur * FRAME_RATE)
                else:
                    norm_dist  = None

                rows_all.append({
                    "speaker_id":  sp_id,
                    "accent":      accent,
                    "word":        word,
                    "distance":    mean_dist,
                    "n_ref":       len(dists),
                    "speaker_dur": round(speaker_dur, 5),
                    "am_dur":      round(am_dur, 5) if am_dur is not None else None,
                    "norm_dist":   round(norm_dist, 7) if norm_dist is not None else None,
                })

            n_words = sum(1 for r in rows_all if r["speaker_id"] == sp_id)
            print(f"  {sp_id}: {n_words} words computed")

    out_csv = dist_dir / "word_distances.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["speaker_id", "accent", "word", "distance", "n_ref",
                        "speaker_dur", "am_dur", "norm_dist"],
        )
        writer.writeheader()
        writer.writerows(rows_all)

    print(f"\nSaved {len(rows_all)} rows → {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["extract", "transcribe", "distances"], required=True)
    parser.add_argument("--layer", type=int, default=14, help="XLS-R hidden layer index (default 14)")
    parser.add_argument("--dist-out", type=str, default=None,
                        help="Override directory for distances output (reads timestamps from normal cfg dir)")
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_CONFIGS.keys()),
        default="personal-introduction",
        help="Which scripted prompt category to process (default: personal-introduction)",
    )
    parser.add_argument(
        "--source",
        choices=["hf", "synthetic"],
        default="hf",
        help="Audio source: 'hf' = HuggingFace human corpus (default), 'synthetic' = ElevenLabs TTS",
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        default="audio_samples/elevenlabs",
        help="Root directory of the ElevenLabs dataset (only used with --source synthetic)",
    )
    parser.add_argument(
        "--script-type",
        choices=["improved", "disfluent"],
        default="improved",
        help="ElevenLabs script variant (only used with --source synthetic, default: improved)",
    )
    args = parser.parse_args()

    cfg = dict(CATEGORY_CONFIGS[args.category])   # shallow copy so we can mutate

    # Redirect output directory for synthetic runs
    if args.source == "synthetic":
        suffix = "" if args.script_type == "improved" else f"_{args.script_type}"
        base   = cfg["out"].name                              # e.g. "phonological_distance"
        cfg["out"] = ROOT / "results" / f"{base}_synthetic{suffix}"

    if args.stage == "extract":
        if args.source == "synthetic":
            stage_extract_synthetic(cfg, args.synthetic_path, args.script_type)
        else:
            stage_extract(cfg)
    elif args.stage == "transcribe":
        stage_transcribe(cfg)
    elif args.stage == "distances":
        stage_distances(cfg, xlsr_layer=args.layer, dist_dir_override=args.dist_out)

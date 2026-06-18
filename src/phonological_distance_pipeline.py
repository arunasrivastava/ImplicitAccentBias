"""
Phonological distance pipeline for accent analysis.

Stages:
  1. extract   – pull scripted audio from HF dataset → 16kHz mono WAV
  2. transcribe – Whisper word-level timestamps per speaker
  3. distances  – XLS-R layer-14 embeddings + DTW per word vs. American English reference
  4. analyze    – phonological category grouping (called from notebook)

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

# ── Script texts ──────────────────────────────────────────────────────────────

_PERSONAL_INTRO_SCRIPT = (
    "I graduated about three years ago and I've been working in customer service "
    "and client-facing roles since then. Most recently I was at a financial services "
    "firm helping onboard clients and answer their questions about accounts. I really "
    "enjoy being that connection between the technical side and clients helping people "
    "understand their options. I think I'm a strong candidate because I have solid "
    "experience with customers I'm getting more familiar with financial products and "
    "I'm pretty detail-oriented which helps me catch mistakes early. I'm looking to "
    "take on more responsibility in this kind of role."
)

_PERSONAL_COMMITMENT_SCRIPT = (
    "Earlier this year I committed to finishing a client report by Thursday for a "
    "Friday meeting, but on Wednesday my manager needed help with an urgent compliance "
    "request that had a same-day deadline. I realized I couldn't do both well, so I "
    "talked to my manager about prioritizing the compliance issue, and then I reached "
    "out to the client directly to let them know the report would be a day late. I "
    "apologized and we rescheduled the meeting to Monday. The client appreciated the "
    "heads-up, and I got the report to them Friday afternoon. I learned it's way better "
    "to communicate early when priorities shift rather than just missing a deadline."
)

_FINANCIAL_PRODUCT_SCRIPT = (
    "I had a client who wanted to open a retirement account but didn't understand the "
    "difference between a traditional IRA and a Roth IRA. I asked them some questions "
    "about their tax situation and timeline first, then explained it as basically pay "
    "taxes now or pay taxes later. I used a simple example with actual numbers to show "
    "how each might work for them. I made sure to mention I wasn't giving tax advice, "
    "just explaining the options. They had more questions so I walked through a "
    "comparison and suggested they talk to a tax professional too. They opened a Roth "
    "and told me later they appreciated that I explained it clearly without pushing "
    "one option."
)

_CLIENT_DISAGREEMENT_SCRIPT = (
    "I had a client who wanted to put a lot of their portfolio in one tech stock because "
    "they worked in that industry and felt confident about it. I recommended more "
    "diversification based on their risk tolerance. They pushed back because they felt "
    "they had good insight. Instead of just agreeing or arguing, I asked them to walk "
    "me through their thinking so I could understand it better. Then we talked through "
    "some scenarios like what would happen if that stock dropped. I acknowledged they "
    "knew the industry well, but explained diversification was about managing risk. We "
    "ended up compromising, they kept a bigger position than I suggested but not as "
    "concentrated as they wanted. They appreciated that I listened."
)

# ── Phonological categories (documentation; analysis uses notebook CATEGORIES) ─
# Each word is listed under its primary phonological challenge category.
# A word may appear in multiple categories if it exhibits several features.

_PERSONAL_INTRO_PHON = {
    # /aɪ/ or /eɪ/ diphthongs
    "diphthong": ["i've", "really", "client", "clients", "onboard", "side", "options",
                  "client-facing", "enjoy", "oriented"],
    # unstressed vowels reduced to schwa in American stress-timed speech
    "schwa_reduction": ["about", "been", "between", "because", "their", "pretty",
                        "familiar", "understand", "customer", "customers"],
    # onset or coda consonant clusters absent in CV-phonotactic L1s (e.g., Mandarin)
    "consonant_cluster": ["graduated", "clients", "financial", "products", "detail",
                          "services", "strength", "oriented", "experience", "recently"],
    "content": ["graduated", "financial", "services", "technical", "mistakes",
                "responsibility", "candidate", "experience", "customers", "products"],
    "function": ["i", "a", "the", "and", "in", "of", "to", "at", "that", "with",
                 "which", "this", "kind", "on", "more"],
}

_PERSONAL_COMMITMENT_PHON = {
    # /aɪ/ + /eɪ/ diphthongs: Friday (/fr.aɪ.deɪ/), deadline (/dɛd.laɪn/),
    #   realized (/rɪ.ə.laɪzd/), monday (/mʌn.deɪ/)
    "diphthong": ["friday", "deadline", "realized", "monday"],
    # initial or medial schwa: about (/ə.baʊt/), manager (/-dʒər/),
    #   apologized (/ə.pɒl.ə.dʒaɪzd/), communicate (/kə.mjuː./), better (/bɛt.ər/)
    "schwa_reduction": ["about", "manager", "apologized", "communicate", "better"],
    # onset/coda clusters: compliance (/pl/), appreciated (/pr/),
    #   priorities (/pr/), directly (/kt/ coda), request (/kw/ + /st/ coda),
    #   afternoon (/ft/ coda-to-onset adjacency)
    "consonant_cluster": ["compliance", "appreciated", "priorities", "directly",
                          "request", "afternoon"],
    "function": ["and", "then", "but", "than"],
}

_FINANCIAL_PRODUCT_PHON = {
    # /aɪ/ or /eɪ/ diphthongs: retirement (/taɪ/), timeline (/taɪ/ twice),
    #   situation (/eɪ/), basically (/eɪ/), advice (/aɪ/)
    "diphthong": ["retirement", "timeline", "situation", "basically", "advice"],
    # schwa: about (/ə.baʊt/), difference (/dɪf.ər.əns/), account (/ə.kaʊnt/),
    #   comparison (/kəm.ˈpær.ɪ.sən/), options (/ˈɒp.ʃənz/)
    "schwa_reduction": ["about", "difference", "account", "comparison", "options"],
    # clusters: understand (/st/ onset + /nd/ coda), questions (/kw/ onset),
    #   explained (/spl/ onset), professional (/pr/), suggested (/dʒ/ + /st/),
    #   traditional (/tr/)
    "consonant_cluster": ["understand", "questions", "explained", "professional",
                          "suggested", "traditional"],
    "function": ["and", "then", "between", "but"],
}

_CLIENT_DISAGREEMENT_PHON = {
    # /oʊ/ and /aɪ/ diphthongs: portfolio (/oʊ/ twice), insight (/aɪ/),
    #   appreciated (/eɪ/), scenarios (/oʊ/)
    "diphthong": ["portfolio", "insight", "appreciated", "scenarios"],
    # schwa: about, because (/bɪ.kəz/), tolerance (/tɒl.ər.əns/),
    #   position (/pə.zɪʃ.ən/), better (/bɛt.ər/)
    "schwa_reduction": ["about", "because", "tolerance", "position", "better"],
    # clusters: compromising (/mpr/), concentrated (/tr/ + /nts/),
    #   acknowledged (/kn/ onset), industry (/nd/ + /str/),
    #   confident (/nf/ + /nt/ coda), explained (/spl/)
    "consonant_cluster": ["compromising", "concentrated", "acknowledged", "industry",
                          "confident", "explained"],
    "function": ["and", "but", "then", "than"],
}

# ── Category configurations ───────────────────────────────────────────────────

CATEGORY_CONFIGS = {
    "personal-introduction": {
        "hf_category": "personal-introduction",
        "script": _PERSONAL_INTRO_SCRIPT,
        "phonological_categories": _PERSONAL_INTRO_PHON,
        "out": ROOT / "results" / "phonological_distance",
    },
    "personal-commitment": {
        "hf_category": "personal-commitment",
        "script": _PERSONAL_COMMITMENT_SCRIPT,
        "phonological_categories": _PERSONAL_COMMITMENT_PHON,
        "out": ROOT / "results" / "phonological_distance_commitment",
    },
    "financial-product": {
        "hf_category": "financial-product",
        "script": _FINANCIAL_PRODUCT_SCRIPT,
        "phonological_categories": _FINANCIAL_PRODUCT_PHON,
        "out": ROOT / "results" / "phonological_distance_financial",
    },
    "client-disagreement": {
        "hf_category": "client-disagreement",
        "script": _CLIENT_DISAGREEMENT_SCRIPT,
        "phonological_categories": _CLIENT_DISAGREEMENT_PHON,
        "out": ROOT / "results" / "phonological_distance_disagreement",
    },
}

# Backward-compatible module-level aliases (personal-introduction defaults)
PERSONAL_INTRO_SCRIPT   = _PERSONAL_INTRO_SCRIPT
PHONOLOGICAL_CATEGORIES = _PERSONAL_INTRO_PHON
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

def stage_extract(cfg):
    import pyarrow.ipc as ipc

    ARROW = Path.home() / ".cache/huggingface/datasets/multispeak___parquet" / \
            "multispeak--hiring-accent-corpus-f16810540282c8f8/0.0.0" / \
            "2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec" / \
            "parquet-train.arrow"

    if not ARROW.exists():
        sys.exit(f"Arrow cache not found at {ARROW}. Run load_dataset first.")

    import io
    import pyarrow as pa

    with open(ARROW, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()

    df = table.to_pandas()
    category_rows = df[df["category"] == cfg["hf_category"]].reset_index(drop=True)
    print(f"Found {len(category_rows)} {cfg['hf_category']} rows")
    print(category_rows["accent_nationality_origin"].value_counts().to_string())

    TARGET_SR = 16000
    out_dir = cfg["out"]
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for _, row in category_rows.iterrows():
        audio = row["audio"]
        speaker_id = re.sub(r"[^a-zA-Z0-9_]", "_", row["name"]).lower()
        accent     = row["accent_nationality_origin"]

        if isinstance(audio, dict):
            raw_bytes = audio.get("bytes")
            if raw_bytes is None:
                print(f"  SKIP {speaker_id}: no audio bytes")
                continue
            import soundfile as sf
            import io
            try:
                arr, sr = sf.read(io.BytesIO(raw_bytes))
            except Exception as e:
                print(f"  SKIP {speaker_id}: {e}")
                continue
        else:
            print(f"  SKIP {speaker_id}: unexpected audio type {type(audio)}")
            continue

        # mono + resample to 16kHz
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
    with open(meta_path, "w") as f:
        json.dump(meta_rows, f, indent=2)
    print(f"\nSaved {len(meta_rows)} WAV files. Metadata → {meta_path}")


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

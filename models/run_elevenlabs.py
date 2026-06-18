"""
ElevenLabs text-to-speech: generate the synthetic voice set used in the study.

Each voice in `data/synthetic_voices/speaker_ids.csv` reads every script in
`data/synthetic_voices/scripts.csv`; audio is written per-speaker and a metadata CSV
(`elevenlabs_metadata.csv`, the same schema shipped in `data/synthetic_voices/`) is
produced alongside it.

    python models/run_elevenlabs.py            # regenerate the full set
    python models/run_elevenlabs.py --help     # override inputs / output dir

Requires ELEVENLABS_API_KEY in .env (https://elevenlabs.io/app/developers/api-keys).
"""
import os
import sys
from pathlib import Path

from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.audio import audio_array_to_bytes, audio_file_to_array

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16_000

API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    raise ValueError(
        "Missing ELEVENLABS_API_KEY. Add it to .env at the repo root "
        "(get one at https://elevenlabs.io/app/developers/api-keys)."
    )
CLIENT = ElevenLabs(api_key=API_KEY)


def run_elevenlabs(text="Hello, this is a test sentence.",
                   voice_id="ZUrEGyu8GFMwnHbvLhv2",
                   output_path=None):
    """Synthesize `text` with `voice_id`; optionally write a 16 kHz WAV."""
    response = CLIENT.text_to_speech.convert(
        text=text,
        voice_id=voice_id,  # browse voice ids at https://elevenlabs.io/app/voice-library
        output_format="pcm_16000",
    )
    audio_bytes = b"".join(chunk for chunk in response)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        wavfile.write(output_path, SAMPLE_RATE, audio_array)
    return audio_array


def generate_elevenlabs_audio(voice_ids_df, scripts_df, output_dir):
    """
    Generate audio for every (voice x script) pair and write a metadata CSV.

    voice_ids_df: rows of name, accent, language, gender, voice_id (see speaker_ids.csv)
    scripts_df:   rows of category, filename, script            (see scripts.csv)
    output_dir:   per-speaker WAVs + elevenlabs_metadata.csv are written here
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "elevenlabs_metadata.csv")

    for _, sample in voice_ids_df.iterrows():
        voice_dir = os.path.join(output_dir, sample["name"])
        os.makedirs(voice_dir, exist_ok=True)
        metadata = []
        for _, script in scripts_df.iterrows():
            filename = script["filename"] + ".wav"
            output_path = os.path.join(voice_dir, filename)
            if os.path.exists(output_path):
                print(f"Skipping existing file: {output_path}")
                continue
            run_elevenlabs(text=script["script"], voice_id=sample["voice_id"],
                           output_path=output_path)
            metadata.append({
                "speaker_name": sample["name"],
                "gender": sample["gender"],
                "voice_id": sample["voice_id"],
                "accent": sample["accent"],
                "language": sample["language"],
                "script_filename": filename,
                "script_text": script["script"],
                "category": script["category"],
            })
        if metadata:
            df = pd.DataFrame(metadata)
            header = not os.path.exists(metadata_path)
            df.to_csv(metadata_path, mode="a", header=header, index=False)


def clone_elevenlabs_audio(file_paths, voice_name):
    """Create an instant voice clone from a list of audio files; returns the new voice_id."""
    audio_bytes_array = [audio_array_to_bytes(audio_file_to_array(p)) for p in file_paths]
    voice = CLIENT.voices.ivc.create(name=voice_name, files=audio_bytes_array)
    print(f"Voice ID: {voice.voice_id} - pass this to run_elevenlabs() to synthesize with the clone.")
    return voice.voice_id


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate the ElevenLabs synthetic voice set.")
    ap.add_argument("--speaker_ids", type=Path,
                    default=ROOT / "data" / "synthetic_voices" / "speaker_ids.csv",
                    help="CSV of voices (name, accent, language, gender, voice_id).")
    ap.add_argument("--scripts", type=Path,
                    default=ROOT / "data" / "synthetic_voices" / "scripts.csv",
                    help="CSV of scripts (category, filename, script).")
    ap.add_argument("--output_dir", type=Path,
                    default=ROOT / "audio_samples" / "elevenlabs",
                    help="Destination for per-speaker WAVs + elevenlabs_metadata.csv.")
    args = ap.parse_args()

    generate_elevenlabs_audio(
        voice_ids_df=pd.read_csv(args.speaker_ids),
        scripts_df=pd.read_csv(args.scripts),
        output_dir=str(args.output_dir),
    )

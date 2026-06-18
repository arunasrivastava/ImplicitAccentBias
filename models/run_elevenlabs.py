import os, sys
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.audio import audio_array_to_bytes, audio_file_to_array

load_dotenv()

API_KEY = os.getenv("ELEVENLABS_API_KEY_OREVA_2")
if not API_KEY:
    raise ValueError(
        "Missing API key. Set the environment variable `ELEVENLABS_API_KEY` at https://elevenlabs.io/app/developers/api-keys in .env at the ROOT level."
    )
CLIENT = ElevenLabs(api_key=API_KEY)


def run_elevenlabs(
    text="Hello, this is a test sentence.",
    voice_id="ZUrEGyu8GFMwnHbvLhv2",
    output_path=None,
):
    response = CLIENT.text_to_speech.convert(
        text=text,
        voice_id=voice_id,  # login and browse voice_ids: https://elevenlabs.io/app/voice-library
        output_format="pcm_16000",
    )
    audio_bytes = b"".join(chunk for chunk in response)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        wavfile.write(output_path, 16000, audio_array)
    return audio_array


def generate_elevenlabs_audio(
    voice_ids_df,
    scripts_df,
    output_dir="../audio_samples/",
):
    """
    Generates set of audio samples using ElevenLabs TTS for multiple voices and scripts.
    pass in dataframes for voice ids and scripts (see ../audio_samples/elevenlabs_dataset1).
    """
    for idx, sample in voice_ids_df.iterrows():
        metadata = []
        voice_id = sample["voice_id"]
        speaker_name = sample["name"]
        accent = sample["accent"]
        language = sample["language"]
        gender = sample["gender"]

        voice_dir = os.path.join(output_dir, speaker_name)
        os.mkdir(voice_dir) if not os.path.exists(voice_dir) else None
        for idx, script in scripts_df.iterrows():
            script_text = script["script"]
            filename = script["filename"] + ".wav"
            category = script["category"]
            output_path = os.path.join(voice_dir, filename)
            if os.path.exists(output_path):
                print(f"Skipping existing file: {output_path}")
                continue
            run_elevenlabs(text=script_text, voice_id=voice_id, output_path=output_path)
            metadata.append(
                {
                    "speaker_name": speaker_name,
                    "gender": gender,
                    "voice_id": voice_id,
                    "accent": accent,
                    "language": language,
                    "script_filename": filename,
                    "script_text": script_text,
                    "category": category,
                }
            )
        metadata_df = pd.DataFrame(metadata)
        metadata_path = os.path.join(output_dir, "elevenlabs_metadata.csv")
        if not os.path.exists(metadata_path):
            metadata_df.to_csv(metadata_path, index=False)
        else:
            with open(metadata_path, "a") as f:
                metadata_df.to_csv(f, header=False, index=False)


def clone_elevenlabs_audio(file_paths: list[str], voice_name: str):
    audio_bytes_array = [
        audio_array_to_bytes(audio_file_to_array(path)) for path in file_paths
    ]
    voice = CLIENT.voices.ivc.create(
        name=voice_name,
        files=audio_bytes_array,  # The more files you add, the better the clone will be.
    )
    print(
        f"Voice ID: {voice.voice_id}, call run_elevenlabs with this voice_id to generate audio with the cloned voice."
    )
    return voice.voice_id


if __name__ == "__main__":
    # generating elevenlabs dataset
    generate_elevenlabs_audio(
        voice_ids_df=pd.read_csv(
            "../audio_samples/elevenlabs_dataset4/speaker_ids.csv"
        ),
        scripts_df=pd.read_csv("../audio_samples/elevenlabs_dataset4/scripts.csv"),
        output_dir="../audio_samples/elevenlabs_dataset4",
    )
    # generating a single clone voice from a set of audio files
    directory = f"../audio_samples/personal_speech_processed/demo/"
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    cloned_voice_id = clone_elevenlabs_audio(file_paths=files, voice_name=f"Test_Clone")
    demo_clone_audio = run_elevenlabs(
        text=f"Hello, this is a clone of a voice! Can we capture the speakers voice? ",
        voice_id=cloned_voice_id,
        output_path=f"test_clone.wav",
    )

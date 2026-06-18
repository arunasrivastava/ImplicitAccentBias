from google import genai
from google.genai.types import Part, GenerateContentConfig
import sys, os
import numpy as np
import io
import soundfile as sf
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()
PROJECT_ID = os.getenv("GEMINI_PROJECT_ID")
SAMPLE_RATE = 16_000


API_KEY = os.getenv("GEMINI_API_KEY_LUCY")


def run_gemini(audio_input, prompt, model="gemini-2.0-flash-001", temp=0, api_key=None):
    assert isinstance(
        audio_input, np.ndarray
    ), "audio_input must be a NumPy array, is type {}".format(
        type(audio_input)
    )  # Debug: print actual d
    assert audio_input.dtype == np.int16, "audio_input must be of type np.int16"

    client = genai.Client(api_key=api_key or API_KEY) if (api_key or API_KEY) else genai.Client(vertexai=True, project=PROJECT_ID)
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_input, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_bytes = buffer.getvalue()

        audio_part = Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        response = client.models.generate_content(
            model=model,
            contents=[prompt, audio_part],
            config=GenerateContentConfig(
                temperature=temp,
                seed=42,
            ),
        )
        if model == "gemini-3-flash-preview":
            for p in response.candidates[0].content.parts:
                return p.text
        else:
            return response.text if response.text is not None else "[inaudible]"

    except Exception as e:
        return f"Error analyzing: {e}"


# Optional test: python models/gemini.py
if __name__ == "__main__":
    audio_file = "./audio_samples/accent_archive_samples/arabic63_english19_merged.wav"
    audio, sr = sf.read(audio_file, dtype="int16")
    prompt = "Describe the speakers of the audio. How many speakers are there? Do they sound happy, angry, sad?"
    result = run_gemini(audio, prompt, model="gemini-3-flash-preview")
    print(result)

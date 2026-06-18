from openai import OpenAI
import sys, os
import numpy as np
import io
import base64
import soundfile as sf
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

SAMPLE_RATE = 16_000
API_KEY = os.getenv("OPENAI_API_KEY")


def run_gpt(audio_input, prompt, model="gpt-4o-audio-preview", temp=0, api_key=None):
    """
    Run GPT with audio input.
    
    Args:
        audio_input: NumPy array of audio data (int16)
        prompt: Text prompt for the model
        model: Model name (default: gpt-4o-audio-preview)
        temp: Temperature for generation (default: 0)
    
    Returns:
        str: Model response text (from the transcript field)
    """
    assert isinstance(
        audio_input, np.ndarray
    ), "audio_input must be a NumPy array, is type {}".format(type(audio_input))
    assert audio_input.dtype == np.int16, "audio_input must be of type np.int16"
    
    client = OpenAI(api_key=api_key or API_KEY)
    try:
        # Convert audio to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_input, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_bytes = buffer.getvalue()
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Create the API request
        response = client.chat.completions.create(
            model=model,
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ],
            temperature=temp,
            seed=42
        )
        
        # Extract the transcript from the audio response
        message = response.choices[0].message
        
        if message.content:
            return message.content
        else:
            return "No response received"
        
    except Exception as e:
        return f"Error analyzing: {e}"


# Optional test: python models/gpt4o.py
if __name__ == "__main__":
    audio_file = "./audio_samples/accent_archive_samples/arabic63_english19_merged.wav"
    audio, sr = sf.read(audio_file, dtype="int16")
    
    prompt = "Describe the speakers of the audio. How many speakers are there? Do they sound happy, angry, sad?"
    result = run_gpt4o(audio, prompt, model="gpt-4o-audio-preview")
    print(result)
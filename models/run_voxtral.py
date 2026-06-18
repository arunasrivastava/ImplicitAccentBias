import os
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor

# ----------------- CONFIG -----------------
MODEL_PATH = "mistralai/Voxtral-Small-24B-2507"
max_tokens = 1024

# ----------------- MODEL LOADING -----------------
def load_voxtral_model():
    print(f"{torch.cuda.device_count()} GPUs detected.", flush=True)
    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Loading model …")
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, processor

# ----------------- INFERENCE FUNCTION -----------------
def do_voxtral_inference(model, processor, question: str, audio_file: str) -> str:
    assert os.path.isfile(audio_file), f"Audio file not found: {audio_file}"

    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "path": audio_file},
            {"type": "text", "text": question},
        ],
    }]
    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(model.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    decoded = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return decoded[0].strip()


# ─────────────────────────────────────────────────────
# Wrapper function to match the calling convention in hiring_eval.py
# ─────────────────────────────────────────────────────
_voxtral_model = None
_voxtral_processor = None


def run_voxtral(audio_path: str, prompt: str) -> str:
    """
    Run Voxtral inference on an audio file with the given prompt.
    Caches the model and processor globally to avoid reloading on every call.
    """
    global _voxtral_model, _voxtral_processor

    if _voxtral_model is None or _voxtral_processor is None:
        _voxtral_model, _voxtral_processor = load_voxtral_model()

    return do_voxtral_inference(_voxtral_model, _voxtral_processor, prompt, audio_path)


# Optional test: python models/run_voxtral.py
if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio_samples/example.wav"

    prompt = "Describe the speaker's accent, clarity, and delivery."

    result = run_voxtral(audio_file, prompt)
    print(result)

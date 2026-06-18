import torch
import os
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"
MAX_NEW_TOKENS = 1024


# ----------------- MODEL LOADING -----------------
def load_flamingo_model():
    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model …")
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, processor


# ----------------- INFERENCE FUNCTION -----------------
def do_flamingo_inference(model, processor, question: str, audio_path: str) -> str:
    assert os.path.isfile(audio_path), f"Audio file not found: {audio_path}"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "audio", "path": audio_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    inputs = {
        k: v.to(model.dtype) if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    decoded = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return decoded[0].strip()


# ─────────────────────────────────────────────────────
# Wrapper function to match the calling convention in hiring_eval.py
# ─────────────────────────────────────────────────────
_flamingo_model = None
_flamingo_processor = None


def run_flamingo(audio_path: str, prompt: str) -> str:
    """
    Run Audio Flamingo 3 inference on an audio file with the given prompt.
    Caches the model and processor globally to avoid reloading on every call.
    """
    global _flamingo_model, _flamingo_processor
    
    if _flamingo_model is None or _flamingo_processor is None:
        _flamingo_model, _flamingo_processor = load_flamingo_model()
    
    return do_flamingo_inference(_flamingo_model, _flamingo_processor, prompt, audio_path)


# Optional test: python models/flamingo.py
if __name__ == "__main__":
    audio_file = "./audio_samples/accent_archive_samples/arabic63_english19_merged.wav"

    prompt = "Describe the speakers of the audio. How many speakers are there? Do they sound happy, angry, sad?"

    result = run_flamingo(audio_file, prompt)
    print(result)


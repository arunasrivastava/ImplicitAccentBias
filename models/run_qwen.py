import os
import re
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# ----------------- CONFIG -----------------
MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
max_tokens = 1024

# ----------------- HELPER FUNCTIONS -----------------
def extract_thinks(text):
    if text is None:
        return []
    return "".join([t.strip() for t in re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)])

def extract_answer(text):
    if text is None:
        return []
    parts = re.split(r"</think>", text, flags=re.DOTALL)
    if len(parts) > 1:
        return "".join([parts[-1].strip()])
    return []

# ----------------- MODEL LOADING -----------------
def load_qwen3_model():
    print(f"{torch.cuda.device_count()} GPUs detected.", flush=True)
    print("Loading processor …")
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    print("Loading model …")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"#,
        #attn_implementation="flash_attention_2",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, processor

# ----------------- INFERENCE FUNCTION -----------------
def do_qwen3_inference(model, processor, question: str, audio_file: str) -> tuple:
    assert os.path.isfile(audio_file), f"Audio file not found: {audio_file}"

    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_file},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios if audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        text_ids, _ = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_new_tokens=max_tokens,
        )
    decoded = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = decoded[0]
    return extract_thinks(response), extract_answer(response), response


# ─────────────────────────────────────────────────────
# Wrapper function to match the calling convention in hiring_eval.py
# ─────────────────────────────────────────────────────
_qwen_model = None
_qwen_processor = None


def run_qwen(audio_path: str, prompt: str) -> str:
    """
    Run Qwen3-Omni inference on an audio file with the given prompt.
    Caches the model and processor globally to avoid reloading on every call.
    Returns only the final answer (post-</think> text).
    """
    global _qwen_model, _qwen_processor

    if _qwen_model is None or _qwen_processor is None:
        _qwen_model, _qwen_processor = load_qwen3_model()

    _, answer, _ = do_qwen3_inference(_qwen_model, _qwen_processor, prompt, audio_path)
    return answer


# Optional test: python models/run_qwen.py
if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio_samples/example.wav"

    prompt = "Describe the speaker's accent, clarity, and delivery."

    thinks, answer, full_response = do_qwen3_inference(
        *load_qwen3_model(), prompt, audio_file
    )
    print("=== Thinking ===")
    print(thinks)
    print("\n=== Answer ===")
    print(answer)
